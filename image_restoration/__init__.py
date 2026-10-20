"""
image_restoration
=================

Deep civil infrastructure image restoration module using RestorationNet.
Provides denoising, contrast enhancement, super-resolution, and sharpness diagnostics.
"""

from .restoration_model import RestorationNet, get_restoration_model
from .restoration_pipeline import (
    restore_image,
    compute_restoration_metrics,
    calculate_image_sharpness,
    calculate_image_entropy,
)

__all__ = [
    "RestorationNet",
    "get_restoration_model",
    "restore_image",
    "compute_restoration_metrics",
    "calculate_image_sharpness",
    "calculate_image_entropy",
]

__version__ = "1.0.0"
