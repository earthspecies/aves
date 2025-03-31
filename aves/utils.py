"""Utility functions for AVES."""

import os
import logging
from typing import Literal
from pathlib import Path

import numpy as np
import torch
import torchaudio

logger = logging.getLogger("aves")


AUDIO_FILE_EXTENSIONS = ["wav", "mp3", "flac", "ogg", "m4a"]
TARGET_SR = 16000  # AVES works with 16kHz audio
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_audio_file_paths(audio_dir: str, audio_file_extension: str | None = None) -> list[Path]:
    """Parse audio file paths from a directory or a list of paths.

    Arguments
    ---------
    audio_dir: str
        Path to the directory containing the audio files.
    audio_file_extension: str, optional
        Extension of the audio files to process. Defaults to None.

    Returns
    -------
    list[Path]
        List of Path objects for the audio files.

    Examples
    --------
    >>> files = parse_audio_file_paths("../example_audios")
    >>> str(files[0])
    '../example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.wav'
    >>> parse_audio_file_paths("../example_audios", "wav")
    [PosixPath('../example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.wav')]
    """
    audio_dir = Path(audio_dir)
    if audio_file_extension is None:
        audio_files = []
        logger.info(f"Searching for audio files in {audio_dir} with extensions: {AUDIO_FILE_EXTENSIONS}")
        for ext in AUDIO_FILE_EXTENSIONS:
            audio_files.extend(list(audio_dir.glob(f"*.{ext}")))

        logger.info(f"Fetched {len(audio_files)} audio files from {audio_dir}")
    else:
        # check if the provided extension is supported
        audio_files = list(audio_dir.glob(f"*.{audio_file_extension}"))

    return audio_files


def save_embedding(embedding: torch.Tensor | list[torch.Tensor], output_path: str | Path, save_as: str) -> None:
    """Save the AVES embedding to a file.

    Arguments
    ---------
    embedding: torch.Tensor | list[torch.Tensor]
        The embedding to save.
    output_path: str | Path
        Path to save the embedding.
    save_as: str
        Format to save the embedding, either 'pt' or 'npy'.

    Examples
    --------
    >>> embedding = torch.randn(1, 128)
    >>> save_embedding(embedding, Path("embedding"), "pt")
    >>> import pathlib; pathlib.Path("embedding.pt").exists()
    True
    >>> pathlib.Path("embedding.pt").unlink()
    """
    if isinstance(output_path, Path):
        output_path = str(output_path)

    if save_as == "pt":
        torch.save(embedding, output_path + ".pt")

    elif save_as == "npy":
        if isinstance(embedding, list):
            np_embedding = [e.cpu().numpy() for e in embedding]
        else:
            np_embedding = embedding.cpu().numpy()

        # numpy adds extension .npy automatically
        np.save(output_path, np_embedding)
    else:
        raise ValueError(f"Unsupported save_as format: {save_as}. Use 'pt' for pytorch or 'npy' for numpy")

    logger.info(f"Saving embedding to {output_path}")


def load_audio(audio_file: str | os.PathLike | Path, mono: bool = False, mono_avg: bool = False) -> torch.Tensor:
    """Load audio file and resample if necessary.

    Arguments
    ---------
    audio_file (str | os.PathLike | Path):
        Path to the audio file.
    mono (bool, optional): Whether to load the audio as mono. Defaults to False.
        If True, the audio will be converted to mono by averaging the channels.
    mono_avg (bool, optional): Whether to average the channels for mono conversion. Defaults to False.
        False = keep the first channel.

    Returns
    -------
        audio (torch.Tensor): Audio tensor

    Examples
    --------
    >>> audio = load_audio("../example_audios/XC936872 - Helmeted Guineafowl - Numida meleagris.wav")
    """
    audio, sr = torchaudio.load(audio_file)

    if sr != TARGET_SR:
        logger.warning(f"CAUTION: Resampling audio from {sr} to {TARGET_SR} required by AVES models!")
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        audio = resampler(audio)

    if audio.ndim > 1:
        channel_dim = int(np.argmin(audio.shape))
        if channel_dim != 0:
            audio = audio.transpose(0, channel_dim)

    if mono and audio.ndim > 1:
        if audio.shape[0] > 1:
            # stereo or more channels
            if mono_avg:
                audio = audio.mean(dim=0, keepdim=False)
            else:
                audio = audio[0]

    # AVES models expect (C, N) shape. Stereo audio is (2, N).
    # else its (1, N) and if (N,) a single channel dim will be added as batch dimension by AVES
    audio = audio.to(torch.float32)  # require by AVES model weights
    return audio


def parse_layers_argument(layers: str, max_layers: int = 24) -> list[int] | int | None:
    """Parse the layers argument from the command line.
    Supports "all", single layer, comma-separated list, and range of layers like "0-3".

    Arguments
    ---------
    layers: str
        Layers argument from the command line

    Returns
    -------
    list[int] | int | None
        List of layers to extract features from. If None, extract from all layers.

    Examples
    --------
    >>> parse_layers_argument("all") is None
    True
    >>> parse_layers_argument("0")
    0
    >>> parse_layers_argument("-1")
    -1
    >>> parse_layers_argument("0-3")
    [0, 1, 2, 3]
    >>> parse_layers_argument("0,2,4")
    [0, 2, 4]
    >>> parse_layers_argument("25")
    Traceback (most recent call last):
    ...
    ValueError: Need 'layers' argument to obey -24 <= layer numbers <= 23
    >>> parse_layers_argument("0,2,-25")
    Traceback (most recent call last):
    ...
    ValueError: Need 'layers' argument to obey -24 <= layer numbers <= 23
    """

    if layers == "all":
        return None

    try:
        layers = int(layers)
    except Exception:
        pass

    if isinstance(layers, int):
        if layers >= max_layers or layers < -max_layers:
            raise ValueError(f"Need 'layers' argument to obey -{max_layers} <= layer numbers <= {max_layers - 1}")
        return layers

    if "," in layers:
        # comma separated list ?
        layers = [int(layer) for layer in layers.split(",")]
        if any(layer >= max_layers or layer < -max_layers for layer in layers):
            raise ValueError(f"Need 'layers' argument to obey -{max_layers} <= layer numbers <= {max_layers - 1}")
        return layers

    if "-" in layers:
        layers = [int(layer) for layer in layers.split("-")]
        layers = [i for i in range(layers[0], layers[1] + 1)]
        if any(layer >= max_layers or layer < -max_layers for layer in layers):
            raise ValueError(f"Need 'layers' argument to obey -{max_layers} <= layer numbers <= {max_layers - 1}")
        return layers
