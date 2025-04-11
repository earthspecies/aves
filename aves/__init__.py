"""AVES package for audio feature extraction."""

from .aves import (
    AVESTorchaudioWrapper,
    AVESClassifier,
    load_feature_extractor,
)
from .aves_onnx import AVESOnnxModel

__all__ = [
    "AVESTorchaudioWrapper",
    "AVESClassifier",
    "load_feature_extractor",
    "AVESOnnxModel",
]
