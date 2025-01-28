"""Unit tests for the aves package.

IMPORTANT! This assumes that you have downloaded these two checkpoints:2
1. birdaves-biox-large.torchaudio.pt
2. birdaves-biox-large.onnx

and placed them in the ../aves folder.
"""

import torch
from aves import AvesClassifier, load_feature_extractor, AvesOnnxModel


def test_feature_extractor_loader():
    # Test loading feature extractor
    model = load_feature_extractor(
        config_path="config/default_cfg_aves-base-all.json", device="cpu", for_inference=True
    )
    assert model is not None
    assert model.config.get("encoder_embed_dim") == 768


def test_aves_feature_extractor():
    # Test AVES feature extractor
    model = load_feature_extractor(
        config_path="config/default_cfg_aves-base-all.json", device="cpu", for_inference=True
    )
    embeddings = model.extract_features(torch.rand(2, 16000))
    print("Embeddings shape", embeddings.shape)
    assert embeddings.shape == (2, 49, 768)


def test_birdaves_feature_extractor():
    # Test BirdAVES feature extractor
    model = load_feature_extractor(
        config_path="config/default_cfg_birdaves-biox-large.json", device="cpu", for_inference=True
    )
    embeddings = model.extract_features(torch.rand(2, 16000))
    print("Embeddings shape", embeddings.shape)
    assert embeddings.shape == (2, 49, 1024)


def test_aves_classifier():
    # Initialize an AVES classifier with 10 target classes
    print("Initializing AVES classifier...")
    model = AvesClassifier(
        config_path="config/default_cfg_birdaves-biox-large.json",
        model_path="aves/birdaves-biox-large.torchaudio.pt",
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

    assert loss is not None
    assert logits.shape[0] == 2
    assert logits.shape[1] == 10


def test_aves_onnx():
    # Initialize an AVES ONNX model
    model = AvesOnnxModel("aves/birdaves-biox-large.onnx")
    # Create two 1-second random sounds
    x = torch.rand(2, 16000, dtype=torch.float32).numpy()
    # Run the forward pass
    outputs = model(x)
    assert outputs.shape == (2, 49, 1024)


def test_onnx_vs_torchaudio_output():
    # Initialize an AVES ONNX model
    onnx_model = AvesOnnxModel("aves/birdaves-biox-large.onnx")

    # BirdAVES feature extractor
    model = load_feature_extractor(
        config_path="config/default_cfg_birdaves-biox-large.json",
        model_path="aves/birdaves-biox-large.torchaudio.pt",
        device="cpu",
        for_inference=True,
    )

    # Create a 1-second random sound
    x = torch.rand(2, 16000, dtype=torch.float32).numpy()

    # Run the forward pass
    onnx_outputs = onnx_model(x)
    torchaudio_outputs = model.extract_features(torch.from_numpy(x))

    assert onnx_outputs.shape == torchaudio_outputs.shape
    # !!Fails to produce identical outputs!!
    assert torch.allclose(torch.from_numpy(onnx_outputs), torchaudio_outputs, atol=1e-5)
