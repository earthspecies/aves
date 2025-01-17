"""Main entry point for the aves package."""

from typing import Literal
import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio

from .aves import load_feature_extractor


ALLOWED_AUDIO_FILE_EXTENSIONS = ["wav", "mp3", "flac", "ogg", "m4a"]
TARGET_SR = 16000  # Aves works with 16kHz audio


def parse_audio_file_paths(audio_dir: str, audio_file_extension: str | None = None) -> list[Path]:
    audio_dir = Path(audio_dir)
    if audio_file_extension is None:
        audio_files = []
        for ext in ALLOWED_AUDIO_FILE_EXTENSIONS:
            audio_files.extend(list(audio_dir.glob(f"*.{ext}")))
    else:
        audio_files = list(audio_dir.glob(f"*.{audio_file_extension}"))

    return audio_files


def save_embedding(embedding: torch.Tensor, output_file: Path, save_as: str) -> None:
    print(f"Saving embedding to {output_file}...")
    if save_as == "pt":
        torch.save(embedding, output_file)
    elif save_as == "npy":
        np_embedding = embedding.cpu().numpy()
        np.save(output_file, np_embedding)
    else:
        raise ValueError(f"Unsupported save_as format: {save_as}")


def load_audio(audio_file: Path) -> torch.Tensor:
    audio, sr = torchaudio.load(audio_file)
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
        audio = resampler(audio)
    return audio


def main():
    """Run the AVES model on a set of audio file paths"""

    parser = argparse.ArgumentParser(description="Run the AVES model on a set of audio file paths")
    parser.add_argument("-c", "--config_path", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument("-m", "--model_path", type=str, required=True, help="Path to the model weights file")
    parser.add_argument(
        "-a", "--audio_paths", type=str, default=None, nargs="+", help="Paths to the audio files to process"
    )
    parser.add_argument(
        "--path_to_audio_dir",
        type=str,
        default=None,
        help="Path to the directory containing the audio files to process",
    )
    parser.add_argument(
        "--audio_file_extension", type=str, default=None, help="Extension of the audio files to process"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on (default: cuda)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the output embedding files")
    parser.add_argument(
        "--save_as",
        type=str,
        default="pt",
        help="Format to save the output embeddings, either 'pt' or 'npy'",
    )

    args = parser.parse_args()

    print("Loading AVES model...")
    model = load_feature_extractor(args.config_path, args.model_path, args.device, for_inference=True)

    # if audio_dir fetch all audio files in the directory with extention audio_file_extension
    if args.audio_paths is None:
        assert args.path_to_audio_dir is not None, "Either audio_paths or path_to_audio_dir must be provided"
        audio_files = parse_audio_file_paths(args.path_to_audio_dir, args.audio_file_extension)
    else:
        audio_files = [Path(audio_path) for audio_path in args.audio_paths]

    print(f"Processing {len(audio_files)} audio files...")
    for audio_file in audio_files:
        print(f"==== Processing {audio_file} ====")

        audio = load_audio(audio_file)
        embedding = model.extract_features(audio)
        save_embedding(embedding, Path(args.output_dir) / f"{audio_file.stem}.embedding", args.save_as)
