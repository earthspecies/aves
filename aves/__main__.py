"""Main entry point for the aves package."""

import argparse
from pathlib import Path

from .aves import load_feature_extractor
from .utils import load_audio, parse_audio_file_paths, save_embedding


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

    parser.add_argument(
        "--layers",
        type=str,
        default="-1",
        help="""Layers to extract features from.
        You can either say "all" for all layers, or provide a range like "0-3" for layers 0 to 3 (inclusive),
        or just a single layer like "2" for layer 2, starting at 0.
        Default is the last layer ("-1").
        """,
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

    # parse layers argument
    if args.layers == "all":
        layers = None
    elif args.layers[0] == "-":
        layers = int(args.layers)
    else:
        layers = [int(layer) for layer in args.layers.split("-")]
        if len(layers) == 1:
            layers = layers[0]
        else:
            layers = [i for i in range(layers[0], layers[1] + 1)]

    print(f"Processing {len(audio_files)} audio files...")
    for audio_file in audio_files:
        print(f"==== Processing {audio_file} ====")

        audio = load_audio(audio_file)
        embedding = model.extract_features(audio.to(args.device), layers)
        save_embedding(embedding, Path(args.output_dir) / f"{audio_file.stem}.embedding", args.save_as)
