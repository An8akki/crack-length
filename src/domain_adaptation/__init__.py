"""
Domain adaptation package for cross-bridge semantic segmentation.
"""
from .deep_coral import (
    compute_covariance,
    coral_loss,
    DeepCORALUNet
)

__all__ = [
    "compute_covariance",
    "coral_loss",
    "DeepCORALUNet"
]
