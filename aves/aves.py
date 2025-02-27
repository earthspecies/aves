"""AVES: Audio and Visual Event Detection with Self-supervised learning"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torchaudio.models import wav2vec2_model


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as ff:
        obj = json.load(ff)
    return obj


class AVESTorchaudioWrapper(nn.Module):
    def __init__(self, config_path: str | Path, model_path: str | Path = None, device: str = "cuda"):
        super().__init__()

        self.config = load_config(str(config_path))

        print("Loading Hubert model (without AVES weights)...")
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        if model_path is not None:
            print("Loading AVES model weights from", model_path)
            self.model.load_state_dict(torch.load(str(model_path), weights_only=True))

        self.device = device

    def forward(self, inputs: torch.Tensor, layers: list[int] | int | None = -1) -> torch.Tensor | list[torch.Tensor]:
        """For training, use the forward method to get the output of the model.

        Args:
            inputs (torch.Tensor): Input tensor
            layers (list[int] | int | None, optional): Layer(s) to extract features from. Defaults to -1 (last layer).

        Returns:
            torch.Tensor | list[torch.Tensor]: Output tensor(s) from the model
        """
        out = self.model.extract_features(inputs)[0]

        if layers is not None and isinstance(layers, int):
            return out[layers]

        if layers and isinstance(layers, list):
            return [out[layer] for layer in layers]

        # return all layers
        return out

    @torch.no_grad()
    def extract_features(
        self,
        inputs: torch.Tensor,
        layers: list[int] | int | None = -1,
    ) -> torch.Tensor:
        """For inference, use the extract_features method to get the output of the model.

        Args:
            inputs (torch.Tensor): Input tensor
            layers (list[int] | int | None, optional): Layer(s) to extract features from. Defaults to -1 (last layer).

        Returns:
            torch.Tensor | list[torch.Tensor]: Output tensor
        """
        if inputs.ndim == 1:
            inputs = inputs.unsqueeze(0)
        return self.forward(inputs, layers)


def load_feature_extractor(
    config_path: str | Path, model_path: str | Path = None, device: str = "cuda", for_inference: bool = True
) -> AVESTorchaudioWrapper:
    """Load the AVES feature extractor model

    Args:
        config_path (str | Path): Path to the model configuration file
        model_path (str | Path): Path to the model weights file
        device (str, optional): Device to run the model on. Defaults to "cuda".
        for_inference (bool, optional): Whether to set the underlying feature extractor to inference mode. Defaults to True.

    Returns:
        AVESTorchaudioWrapper: The AVES feature extractor model
    """
    device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
    if for_inference:
        return AVESTorchaudioWrapper(config_path, model_path, device).to(device).eval()

    return AVESTorchaudioWrapper(config_path, model_path, device).to(device)


class AVESClassifier(nn.Module):
    """A classifier model using AVES as a feature extractor

    Args:
        config_path (str | Path): Path to the model configuration file
        model_path (str | Path): Path to the model weights file
        num_classes (int): Number of target classes
        freeze_feature_extractor (bool, optional): Whether to freeze the feature extractor. Defaults to True.
        for_inference (bool, optional): Whether to set the underlying feature extractor to inference mode. Defaults to False.
    """

    def __init__(
        self,
        config_path: str | Path,
        num_classes: int,
        model_path: str | Path = None,  # or a pre-trained model path
        freeze_feature_extractor: bool = True,
        for_inference: bool = False,
        device: str = "cuda",
    ):
        super().__init__()

        self.model = load_feature_extractor(config_path, model_path, for_inference=for_inference, device=device)
        embeddings_dim = self.model.config.get("encoder_embed_dim", 768)
        self.head = nn.Linear(in_features=embeddings_dim, out_features=num_classes)

        device = "cuda" if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.device = device  # you still have to move the model to the device you want to use
        self.head.to(device)

        if freeze_feature_extractor:
            print("Freezing feature extractor, it will NOT be updated during training!")
            self.model.requires_grad_(False)

        if num_classes == 1:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = None) -> tuple[torch.Tensor | None, torch.Tensor]:
        out = self.model.forward(inputs, layers=-1)
        out = out.mean(dim=1)  # mean pooling over time dimension
        logits = self.head(out)

        loss = None
        if labels is not None:
            loss = self.loss_func(logits, labels)

        return loss, logits


if __name__ == "__main__":
    # test

    # Initialize an AVES classifier with 10 target classes
    print("Initializing AVES classifier to CPU...")
    model = AVESClassifier(
        config_path="../config/birdaves-biox-large.json",
        model_path="birdaves-biox-large.torchaudio.pt",
        num_classes=10,
        for_inference=True,
        freeze_feature_extractor=True,
        device="cpu",
    )

    print("Running forward pass...")
    # Create a batch of 2 1-second random sound
    x = torch.rand(2, 16000)
    y = torch.tensor([0, 1])

    # Run the forward pass
    loss, logits = model(inputs=x, labels=y)

    print("Loss:", loss)
    print("Logits shape = ", logits.shape)
