"""AVES package for audio feature extraction."""

from .aves import (
    AvesTorchaudioWrapper,
    AvesClassifier,
    load_feature_extractor,
)

__all__ = ["AvesTorchaudioWrapper", "AvesClassifier", "load_feature_extractor"]
