"""Planar Calibration & Homography Quantification Package.

Provides perspective rectification, metric calibration from planar references,
and physical crack measurement alongside 2D and 3D methods.
"""

from .homography import (
    estimate_homography,
    validate_quadrilateral,
    compute_reprojection_error,
    assess_perspective_distortion,
)
from .marker_detection import detect_aruco_marker, detect_planar_marker
from .reference_detection import PlanarReference, resolve_planar_reference
from .rectification import rectify_image_and_mask, create_calibration_overlay
from .metric_transform import (
    MetricCoordinateTransform,
    compute_skeleton_metric_length,
)
from .planar_measurement import quantify_planar_crack
from .planar_validation import (
    assess_planarity_heuristic,
    run_measurement_sanity_checks,
    compute_research_comparison,
)

__all__ = [
    "estimate_homography",
    "validate_quadrilateral",
    "compute_reprojection_error",
    "assess_perspective_distortion",
    "detect_aruco_marker",
    "detect_planar_marker",
    "PlanarReference",
    "resolve_planar_reference",
    "rectify_image_and_mask",
    "create_calibration_overlay",
    "MetricCoordinateTransform",
    "compute_skeleton_metric_length",
    "quantify_planar_crack",
    "assess_planarity_heuristic",
    "run_measurement_sanity_checks",
    "compute_research_comparison",
]
