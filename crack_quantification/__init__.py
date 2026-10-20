"""
crack_quantification
====================

A self-contained, publishable module for physical crack quantification from
binary segmentation masks, calibrated depth maps, and temporal inspections.

Pipeline stages:
    Binary Mask (+ Depth Map / Reference / Timestamps)
        |
        v
    [skeleton]    -> 1-pixel-wide crack centerline (Guo-Hall / Zhang-Suen / Morphological)
        |
        v
    [length]      -> 2D geodesic & MST network graph length, 3D arc-length (mm)
        |
        v
    [width]       -> Orthogonal normal-ray profiles & EDT distance transform (mm),
                     boundary reliability under restoration, ACI 224R classification
        |
        v
    [area]        -> Calibrated surface area (mm²) with boundary uncertainty bounds (±Δ mm²),
                     planar homography perspective-corrected area
        |
        v
    [volume]      -> 3D crevice recess volume (mm³ & cm³) via surface baseline integration
        |
        v
    [temporal]    -> Uncertainty-aware temporal progression & growth velocity (mm/year)
        |
        v
    [severity]    -> Structural damage score & branching topology
        |
        v
    Unified Engineering Report
"""

from .skeleton import skeletonize_mask
from .length import compute_length
from .width import (
    compute_width_profile,
    compute_normal_width_profile,
    compute_advanced_width_profile,
    assess_width_reliability,
    classify_crack_width,
    evaluate_width_accuracy,
)
from .area import (
    compute_area,
    area_px_to_mm2,
    compute_area_uncertainty,
    compute_perspective_corrected_area,
    evaluate_area_accuracy,
)
from .volume import (
    compute_crack_volume,
    compute_volume_uncertainty,
    evaluate_volume_accuracy,
    estimate_surface_baseline,
)
from .temporal import (
    analyze_temporal_progression,
    compute_spatial_progression_map,
    evaluate_temporal_progression_accuracy,
)
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
from .quantify import quantify_crack, format_report
from .visualize_markers import (
    mark_crack_comprehensive,
    mark_width_diagnostic,
    mark_volume_diagnostic,
    mark_severity_diagnostic,
    mark_restoration_comparison,
    create_all_in_one_dashboard,
)

__all__ = [
    # Centerline & Length
    "skeletonize_mask",
    "compute_length",
    "analyze_skeleton",
    "crack_length_from_depth",
    # Candidate A: Width Estimation
    "compute_width_profile",
    "compute_normal_width_profile",
    "compute_advanced_width_profile",
    "assess_width_reliability",
    "classify_crack_width",
    "evaluate_width_accuracy",
    # Candidate C: Area Quantification
    "compute_area",
    "area_px_to_mm2",
    "compute_area_uncertainty",
    "compute_perspective_corrected_area",
    "evaluate_area_accuracy",
    # Candidate B: Volume Estimation
    "compute_crack_volume",
    "compute_volume_uncertainty",
    "evaluate_volume_accuracy",
    "estimate_surface_baseline",
    # Candidate D: Temporal Progression
    "analyze_temporal_progression",
    "compute_spatial_progression_map",
    "evaluate_temporal_progression_accuracy",
    # Calibration & Scale
    "Calibration",
    "scale_from_mm_per_pixel",
    "scale_from_aruco_marker",
    "scale_from_camera_intrinsics",
    "scale_from_reference_segment",
    "homography_from_reference",
    "polyline_length_on_reference_plane",
    # Severity & Orchestration
    "assess_severity",
    "compute_orientation_and_branches",
    "quantify_crack",
    "format_report",
    # Visual Markings & Engineering HUD
    "mark_crack_comprehensive",
    "mark_width_diagnostic",
    "mark_volume_diagnostic",
    "mark_severity_diagnostic",
    "mark_restoration_comparison",
    "create_all_in_one_dashboard",
]

__version__ = "2.0.0"
