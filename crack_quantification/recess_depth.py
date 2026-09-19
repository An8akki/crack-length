"""
recess_depth.py
===============

Integration bridge connecting the crack_quantification package with the
crack_depth relative recess estimation engine.
"""

from typing import Union, Optional, Dict, Any
import numpy as np
from crack_depth import FinalCrackPredictor, FinalPredictionResult


_PREDICTOR_INSTANCE: Optional[FinalCrackPredictor] = None


def get_depth_predictor() -> FinalCrackPredictor:
    """Singleton getter for FinalCrackPredictor."""
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        _PREDICTOR_INSTANCE = FinalCrackPredictor()
    return _PREDICTOR_INSTANCE


def estimate_crack_recess(
    image_rgb: Union[str, np.ndarray],
    mask: Union[str, np.ndarray],
    depth_map: Union[str, np.ndarray],
) -> FinalPredictionResult:
    """
    Estimates relative crack crevice recess (ΔZ) in mm.

    Parameters:
        image_rgb: 3-channel RGB image (H, W, 3) or file path.
        mask: Binary crack mask (H, W) or file path.
        depth_map: 8-bit depth map (H, W) or file path.

    Returns:
        FinalPredictionResult with predicted_relative_depth, conformal intervals,
        and explainable reliability classification.
    """
    predictor = get_depth_predictor()
    return predictor.predict(rgb_input=image_rgb, depth_input=depth_map, mask_input=mask)
