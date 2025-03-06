"""Utility functions for Aves."""

import os
from pathlib import Path

import numpy as np
import torch
import torchaudio

ALLOWED_AUDIO_FILE_EXTENSIONS = ["wav", "mp3", "flac", "ogg", "m4a"]
TARGET_SR = 16000  # Aves works with 16kHz audio


def parse_audio_file_paths(audio_dir: str, audio_file_extension: str | None = None) -> list[Path]:
    """Parse audio file paths from a directory or a list of paths.

    Args:
        audio_dir (str): Path to the directory containing the audio files.
        audio_file_extension (str, optional): Extension of the audio files to process. Defaults to None.

    Returns:
        list[Path]: List of Path objects for the audio files.
    """
    audio_dir = Path(audio_dir)
    if audio_file_extension is None:
        audio_files = []
        print(f"Searching for audio files in {audio_dir} with extensions: {ALLOWED_AUDIO_FILE_EXTENSIONS}")
        for ext in ALLOWED_AUDIO_FILE_EXTENSIONS:
            audio_files.extend(list(audio_dir.glob(f"*.{ext}")))

        print(f"Fetched {len(audio_files)} audio files from {audio_dir}")
    else:
        # check if the provided extension is supported
        if audio_file_extension not in ALLOWED_AUDIO_FILE_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio file extension: {audio_file_extension}. Supported extensions are: {ALLOWED_AUDIO_FILE_EXTENSIONS}"
            )
        audio_files = list(audio_dir.glob(f"*.{audio_file_extension}"))

    return audio_files


def save_embedding(embedding: torch.Tensor | list[torch.Tensor], output_file: Path, save_as: str) -> None:
    """Save the embedding to a file.

    Args:
        embedding (torch.Tensor): The embedding to save.
        output_file (Path): Path to save the embedding.
        save_as (str): Format to save the embedding, either 'pt' or 'npy'.
    """

    if save_as == "pt":
        torch.save(embedding, output_file + ".pt")

    elif save_as == "npy":
        if isinstance(embedding, list):
            np_embedding = [e.cpu().numpy() for e in embedding]
        else:
            np_embedding = embedding.cpu().numpy()

        # numpy adds extension .npy automatically
        np.save(output_file, np_embedding)

    else:
        raise ValueError(f"Unsupported save_as format: {save_as}")

    print(f"Saving embedding to {output_file}...")


def load_audio(audio_file: str | os.PathLike | Path, mono: bool = False, mono_avg: bool = False) -> torch.Tensor:
    """Load audio file and resample if necessary.

    Args:
        audio_file (str | os.PathLike | Path): Path to the audio file.
        mono (bool, optional): Whether to load the audio as mono. Defaults to False.
            If True, the audio will be converted to mono by averaging the channels.
        mono_avg (bool, optional): Whether to average the channels for mono conversion. Defaults to False.
            False = keep the first channel.

    Returns:
        audio (torch.Tensor): Audio tensor
    """
    audio, sr = torchaudio.load(audio_file)
    if sr != TARGET_SR:
        print(f"CAUTION: Resampling audio from {sr} to {TARGET_SR} required by AVES models...")
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        audio = resampler(audio)

    if mono and audio.shape[0] > 1:
        if mono_avg:
            audio = audio.mean(dim=0, keepdim=False)
        else:
            audio = audio[0]

    return audio
