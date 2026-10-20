"""
Tests for Integrated Relative Crack Depth & Recess Estimation Engine
InfraVision AI / crack-length Platform.
"""

import os
import sys
import numpy as np
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from crack_depth import FinalCrackPredictor, FinalPredictionResult, CalibrationState, NotCalibratedError
from crack_quantification import estimate_crack_recess


@pytest.fixture(scope="module")
def predictor():
    return FinalCrackPredictor()


@pytest.fixture(scope="module")
def synthetic_rgbd():
    """Generates a synthetic high-resolution RGB-D crack sample."""
    h, w = 1080, 1920
    rgb = np.full((h, w, 3), 160, dtype=np.uint8)
    depth = np.full((h, w), 120, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Draw synthetic crack across center
    for x in range(300, 1600):
        y = int(540 + 40 * np.sin(x / 100.0))
        # Crack width ~20 px
        y_min, y_max = max(0, y - 10), min(h, y + 10)
        mask[y_min:y_max, x] = 1
        rgb[y_min:y_max, x] = [40, 40, 40]  # dark crevice
        depth[y_min:y_max, x] = 90          # recessed optical depth

    return rgb, depth, mask


def test_predictor_initialization_and_config(predictor):
    """Verifies model, scaler, and configuration files load cleanly."""
    assert predictor.model is not None
    assert predictor.scaler is not None
    assert len(predictor.feature_names) == 25
    assert predictor.calibration_state == CalibrationState.UNCALIBRATED


def test_synthetic_sample_inference(predictor, synthetic_rgbd):
    """Tests end-to-end inference on synthetic RGB-D crack."""
    rgb, depth, mask = synthetic_rgbd
    result = predictor.predict(rgb, depth, mask)

    assert isinstance(result, FinalPredictionResult)
    assert isinstance(result.predicted_relative_depth, float)
    assert result.unit == "mm"
    assert result.calibration_state == "UNCALIBRATED"
    assert result.reliability in ["HIGH", "RELIABLE", "BORDERLINE", "UNRELIABLE", "INVALID"]
    assert result.median_crack_width_px > 0
    assert result.crack_area_px > 0


def test_conformal_interval_properties(predictor, synthetic_rgbd):
    """Verifies that conformal prediction intervals expand monotonically."""
    rgb, depth, mask = synthetic_rgbd
    res = predictor.predict(rgb, depth, mask)

    w80 = res.interval_80[1] - res.interval_80[0]
    w90 = res.interval_90[1] - res.interval_90[0]
    w95 = res.interval_95[1] - res.interval_95[0]

    assert 0 < w80 < w90 < w95
    assert res.interval_95[0] <= res.interval_90[0] <= res.interval_80[0]
    assert res.interval_80[1] <= res.interval_90[1] <= res.interval_95[1]
    assert res.interval_80[0] <= res.predicted_relative_depth <= res.interval_80[1]


def test_crack_quantification_adapter(synthetic_rgbd):
    """Tests the estimate_crack_recess bridge in crack_quantification."""
    rgb, depth, mask = synthetic_rgbd
    res = estimate_crack_recess(rgb, mask, depth)

    assert isinstance(res, FinalPredictionResult)
    assert res.predicted_relative_depth > 0
    assert res.unit == "mm"


def test_invalid_input_rejection(predictor, synthetic_rgbd):
    """Verifies robust rejection of empty masks and dimension mismatches."""
    rgb, depth, mask = synthetic_rgbd

    # Empty mask
    empty_mask = np.zeros_like(mask)
    with pytest.raises(ValueError, match="empty"):
        predictor.predict(rgb, depth, empty_mask)

    # Dimension mismatch
    wrong_depth = depth[:500, :500]
    with pytest.raises(ValueError, match="Dimension mismatch"):
        predictor.predict(rgb, wrong_depth, mask)


def test_calibration_safety_lock(predictor):
    """Enforces permanent software-only safety: predict_depth_mm() is disabled."""
    assert predictor.calibration_state == CalibrationState.UNCALIBRATED
    with pytest.raises(NotCalibratedError):
        predictor.predict_depth_mm()
