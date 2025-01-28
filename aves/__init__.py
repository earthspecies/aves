"""AVES package for audio feature extraction."""

from .aves import (
    AvesTorchaudioWrapper,
    AvesClassifier,
    load_feature_extractor,
)
from .aves_onnx import AvesOnnxModel

__all__ = [
    "AvesTorchaudioWrapper",
    "AvesClassifier",
    "load_feature_extractor",
    "AvesOnnxModel",
]
