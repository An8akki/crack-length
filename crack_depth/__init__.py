"""
crack_depth
===========
Relative Concrete Crack Recess & Depth Estimation Package
Integrated into InfraVision AI / crack-length platform.
"""

from .predictor import FinalCrackPredictor, FinalPredictionResult
from .core.calibration_schema import CalibrationState
from .core.calibration_models import NotCalibratedError

__all__ = [
    "FinalCrackPredictor",
    "FinalPredictionResult",
    "CalibrationState",
    "NotCalibratedError",
]

__version__ = "1.0.0"
