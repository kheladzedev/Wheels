"""Custom production model pipeline (MobileNetV2-skipless wheel pose detector).

Scope of this subpackage: direct torch/torchvision imports are allowed
here per the dependency rule in CLAUDE.md. Outside `src/models/` the
project remains ultralytics-only.

The model architecture, loss design, and training recipe are specified
in `docs/MODEL_ARCHITECTURE_PROPOSAL.md` (§2-§3). This module is the
implementation of that proposal.
"""

from src.models.mobilenetv2_skipless_pose import (
    MobileNetV2SkiplessPose,
    WheelPoseHead,
    N_KEYPOINTS,
    FEATURE_STRIDE,
)

__all__ = [
    "MobileNetV2SkiplessPose",
    "WheelPoseHead",
    "N_KEYPOINTS",
    "FEATURE_STRIDE",
]
