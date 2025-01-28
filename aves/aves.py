import json
from pathlib import Path
import torch
import torch.nn as nn
from torchaudio.models import wav2vec2_model


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as ff:
        obj = json.load(ff)
    return obj


class AvesTorchaudioWrapper(nn.Module):
    def __init__(self, config_path: str | Path, model_path: str | Path = None):
        super().__init__()

        self.config = load_config(str(config_path))
        print("Loading Hubert model (without AVES weights)...")
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        if model_path is not None:
            print("Loading AVES model weights from", model_path)
            self.model.load_state_dict(torch.load(str(model_path), weights_only=True))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """For training, use the forward method to get the output of the model"""
        # extract_feature in the sorchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(inputs)[0][-1]

        return out

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """For inference, use the extract_features method to get the output of the model"""
        return self.forward(inputs)


def load_feature_extractor(
    config_path: str | Path, model_path: str | Path = None, device: str = "cuda", for_inference: bool = True
) -> AvesTorchaudioWrapper:
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    if for_inference:
        return AvesTorchaudioWrapper(config_path, model_path).to(device).eval()
    return AvesTorchaudioWrapper(config_path, model_path).to(device)


class AvesClassifier(nn.Module):
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
    ):
        super().__init__()

        self.model = load_feature_extractor(config_path, model_path, for_inference=for_inference)
        embeddings_dim = self.model.config.get("encoder_embed_dim", 768)
        self.head = nn.Linear(in_features=embeddings_dim, out_features=num_classes)

        if freeze_feature_extractor:
            print("Freezing feature extractor, it will NOT be updated during training!")
            self.model.requires_grad_(False)

        if num_classes == 1:
            self.loss_func = nn.BCEWithLogitsLoss()
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs: torch.Tensor, labels: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model.forward(inputs)
        out = out.mean(dim=1)  # mean pooling
        logits = self.head(out)

        loss = None
        if labels is not None:
            loss = self.loss_func(logits, labels)

        return loss, logits


if __name__ == "__main__":
    # test

    # Initialize an AVES classifier with 10 target classes
    print("Initializing AVES classifier...")
    model = AvesClassifier(
        config_path="../config/birdaves-biox-large.json",  # "./aves_all.json",
        model_path="birdaves-biox-large.torchaudio.pt",  # ./aves-base-all.torchaudio.pt",
        num_classes=10,
        for_inference=True,
        freeze_feature_extractor=True,
    )

    print("Running forward pass...")
    # Create a 1-second random sound
    x = torch.rand(2, 16000)
    y = torch.tensor([0, 1])

    # Run the forward pass
    loss, logits = model(inputs=x, labels=y)

    print("Loss:", loss)
    print("Logits shape = ", logits.shape)
