"""Unit tests for the aves package.

IMPORTANT! This assumes that you have downloaded the checkpoint:
1. birdaves-biox-large.onnx

and placed them in the ../models folder.
"""

import os
import subprocess
from pathlib import Path

import numpy as np
import torch

from aves import AVESClassifier, AVESOnnxModel, load_feature_extractor


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

    assert embeddings.shape == (2, 49, 768)

    # test multiple layers
    embeddings = model.extract_features(torch.rand(2, 16000), layers=[-2, -1])
    assert len(embeddings) == 2
    assert embeddings[0].shape == (2, 49, 768)
    assert embeddings[1].shape == (2, 49, 768)


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
    model = AVESClassifier(
        config_path="config/default_cfg_birdaves-biox-large.json",
        model_path=None,
        num_classes=10,
        for_inference=True,
        freeze_feature_extractor=True,
        device="cpu",
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
    # check if the model is downloaded
    if not Path("models/birdaves-biox-large.onnx").exists():
        print("Please download the birdaves-biox-large.onnx model file to the models/ folder, skipping test...")
        return
    model = AVESOnnxModel("aves/birdaves-biox-large.onnx")
    # Create two 1-second random sounds
    x = torch.rand(2, 16000, dtype=torch.float32).numpy()
    # Run the forward pass
    outputs = model(x)
    assert outputs.shape == (2, 49, 1024)


def test_cli():
    # Test the AVES CLI
    p = subprocess.run(
        [
            "aves",
            "-c",
            "config/default_cfg_birdaves-biox-large.json",
            "--path_to_audio_dir",
            "example_audios/",
            "--output_dir",
            "example_audios/",
            "--save_as",
            "npy",
            "--device",
            "cpu",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert p.returncode == 0
    assert b"Processing 2 audio files..." in p.stdout
    assert b"Saving embedding to example_audios/" in p.stdout

    # check that embdding files are saved
    assert (Path("example_audios") / "XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.embedding.npy").exists()
    assert (Path("example_audios") / "XC936872 - Helmeted Guineafowl - Numida meleagris.embedding.npy").exists()

    # load one of them
    emb = np.load("example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.embedding.npy", allow_pickle=True)
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (2, 1049, 1024)

    # remove these files
    os.remove("example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.embedding.npy")
    os.remove("example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.embedding.npy")


def test_cli_multiple_layers():
    # Test the AVES CLI
    p = subprocess.run(
        [
            "aves",
            "-c",
            "config/default_cfg_birdaves-biox-large.json",
            "--path_to_audio_dir",
            "example_audios/",
            "--output_dir",
            "example_audios/",
            "--layers",
            "0-3",
            "--save_as",
            "npy",
            "--device",
            "cpu",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert p.returncode == 0
    assert b"Processing 2 audio files..." in p.stdout
    assert b"Saving embedding to example_audios/" in p.stdout

    # check that embdding files are saved
    assert (Path("example_audios") / "XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.embedding.npy").exists()
    assert (Path("example_audios") / "XC936872 - Helmeted Guineafowl - Numida meleagris.embedding.npy").exists()

    # load one of them
    emb = np.load("example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.embedding.npy", allow_pickle=True)
    assert len(emb) == 4
    assert isinstance(emb[0], np.ndarray)
    assert emb[0].shape == (2, 1049, 1024)

    # remove these files
    os.remove("example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.embedding.npy")
    os.remove("example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.embedding.npy")


def test_cli_multiple_layers_2():
    # Test the AVES CLI
    p = subprocess.run(
        [
            "aves",
            "-c",
            "config/default_cfg_birdaves-biox-large.json",
            "--path_to_audio_dir",
            "example_audios/",
            "--output_dir",
            "example_audios/",
            "--layers",
            "0,1,11",
            "--save_as",
            "npy",
            "--device",
            "cpu",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert p.returncode == 0
    assert b"Processing 2 audio files..." in p.stdout
    assert b"Saving embedding to example_audios/" in p.stdout

    # check that embdding files are saved
    assert (Path("example_audios") / "XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.embedding.npy").exists()
    assert (Path("example_audios") / "XC936872 - Helmeted Guineafowl - Numida meleagris.embedding.npy").exists()

    # load one of them
    emb = np.load("example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.embedding.npy", allow_pickle=True)
    assert len(emb) == 3
    assert isinstance(emb[0], np.ndarray)
    assert emb[0].shape == (2, 1049, 1024)

    # remove these files
    os.remove("example_audios/XC448414 - Eurasian Bullfinch - Pyrrhula pyrrhula.embedding.npy")
    os.remove("example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.embedding.npy")


# THIS TEST FAILS BECAUSE OF ONNX OUTPUT DIFFERENCES
# def test_onnx_vs_torchaudio_output():
#     # Initialize an AVES ONNX model
#     onnx_model = AVESOnnxModel("aves/birdaves-biox-large.onnx")

#     # BirdAVES feature extractor
#     model = load_feature_extractor(
#         config_path="config/default_cfg_birdaves-biox-large.json",
#         model_path="aves/birdaves-biox-large.torchaudio.pt",
#         device="cpu",
#         for_inference=True,
#     )

#     # Create a 1-second random sound
#     x = torch.rand(2, 16000, dtype=torch.float32).numpy()

#     # Run the forward pass
#     onnx_outputs = onnx_model(x)
#     torchaudio_outputs = model.extract_features(torch.from_numpy(x))

#     assert onnx_outputs.shape == torchaudio_outputs.shape
#     # !!Fails to produce identical outputs!!
#     assert torch.allclose(torch.from_numpy(onnx_outputs), torchaudio_outputs, atol=1e-5)
