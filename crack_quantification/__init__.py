"""
crack_quantification
====================

A self-contained, publishable module for quantifying cracks from a binary
segmentation mask.

Pipeline (all stages operate on a single binary mask, obtained once from a
segmentation detector):

    Binary Mask
        |
        v
    [skeleton]  -> 1-pixel-wide crack centerline
        |
        v
    [length]    -> crack length in pixels (graph traversal, 1 / sqrt(2) steps)
        |
        v
    [width]     -> width at every skeleton point via distance transform
        |
        v
    [area]      -> number of white pixels (crack area in px^2)
        |
        v
    [calibration] -> px <-> real-world units (mm, mm^2)
        |
        v
    [severity]  -> severity score, orientation, branch count
        |
        v
    Final Report
"""

from .skeleton import skeletonize_mask
from .length import compute_length
from .width import compute_width_profile
from .area import compute_area
from .calibration import (
    Calibration,
    scale_from_mm_per_pixel,
    scale_from_aruco_marker,
    scale_from_camera_intrinsics,
)
from .severity import assess_severity, compute_orientation_and_branches
from .graph_analysis import analyze_skeleton
from .reference_scale import (
    scale_from_reference_segment,
    homography_from_reference,
    polyline_length_on_reference_plane,
)
from .depth_3d import crack_length_from_depth
from .recess_depth import estimate_crack_recess
from crack_depth import FinalCrackPredictor, FinalPredictionResult

__all__ = [
    "skeletonize_mask",
    "compute_length",
    "compute_width_profile",
    "compute_area",
    "Calibration",
    "scale_from_mm_per_pixel",
    "scale_from_aruco_marker",
    "scale_from_camera_intrinsics",
    "assess_severity",
    "compute_orientation_and_branches",
    "analyze_skeleton",
    "scale_from_reference_segment",
    "homography_from_reference",
    "polyline_length_on_reference_plane",
    "crack_length_from_depth",
    "estimate_crack_recess",
    "FinalCrackPredictor",
    "FinalPredictionResult",
]

__version__ = "1.1.0"
