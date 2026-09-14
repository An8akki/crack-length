"""Factorial performance matrix breakdown and failure mode analysis."""

from typing import Dict, List, Optional
import numpy as np
from .real_specimen_schema import SpecimenRecord


def generate_performance_matrix(records: List[SpecimenRecord]) -> Dict:
    """Generate condition-stratified performance matrix contrasting 2D, Planar, and 3D."""
    matrix = {
        "by_angle": {},
        "by_distance": {},
        "by_surface": {},
        "by_lighting": {},
    }

    def _add_to_group(group_dict, key, rec):
        group = group_dict.setdefault(key, {"count": 0, "2d_errs": [], "planar_errs": [], "3d_errs": []})
        group["count"] += 1
        if rec.error_2d_pct is not None: group["2d_errs"].append(rec.error_2d_pct)
        if rec.error_planar_pct is not None: group["planar_errs"].append(rec.error_planar_pct)
        if rec.error_3d_pct is not None: group["3d_errs"].append(rec.error_3d_pct)

    for r in records:
        # Group by angle
        if r.camera_yaw_deg < 10.0:
            a_key = "0° (Frontal)"
        elif r.camera_yaw_deg < 25.0:
            a_key = "15°-20° (Moderate Oblique)"
        elif r.camera_yaw_deg < 40.0:
            a_key = "30°-35° (High Oblique)"
        else:
            a_key = "45°+ (Extreme Oblique)"
        _add_to_group(matrix["by_angle"], a_key, r)

        # Group by distance
        if r.camera_distance_mm <= 750.0:
            d_key = "0.5m (Close-up)"
        elif r.camera_distance_mm <= 1250.0:
            d_key = "1.0m (Standard)"
        elif r.camera_distance_mm <= 1750.0:
            d_key = "1.5m (Mid-range)"
        else:
            d_key = "2.0m+ (Far inspection)"
        _add_to_group(matrix["by_distance"], d_key, r)

        # Group by surface
        _add_to_group(matrix["by_surface"], r.surface_type, r)

        # Group by lighting
        _add_to_group(matrix["by_lighting"], r.lighting_condition, r)

    # Compute averages for each slice
    result_matrix = {}
    for factor, groups in matrix.items():
        result_matrix[factor] = []
        for key, g in groups.items():
            result_matrix[factor].append({
                "condition": key,
                "trials": g["count"],
                "mean_error_2d_pct": round(float(np.mean(g["2d_errs"])), 2) if g["2d_errs"] else None,
                "mean_error_planar_pct": round(float(np.mean(g["planar_errs"])), 2) if g["planar_errs"] else None,
                "mean_error_3d_pct": round(float(np.mean(g["3d_errs"])), 2) if g["3d_errs"] else None,
            })

    return result_matrix


def identify_failure_modes(records: List[SpecimenRecord], error_threshold_pct: float = 15.0) -> List[Dict]:
    """Diagnose specific empirical conditions causing measurement breakdown."""
    failure_reports = []

    # 1. Evaluate 2D failure boundary
    high_angle_2d = [r.error_2d_pct for r in records if r.camera_yaw_deg >= 25.0 and r.error_2d_pct is not None]
    if high_angle_2d and np.mean(high_angle_2d) > error_threshold_pct:
        failure_reports.append({
            "method": "2D Global Scale",
            "condition": "Viewing Angle >= 25°",
            "mean_error_pct": round(float(np.mean(high_angle_2d)), 1),
            "failure_mechanism": "Uncompensated perspective foreshortening contracts horizontal pixel distances.",
            "recommendation": "Switch to Planar Homography or 3D Reconstruction for oblique viewpoints."
        })

    # 2. Evaluate Planar failure boundary on non-planar surfaces
    non_planar_homog = [r.error_planar_pct for r in records if not r.is_planar and r.error_planar_pct is not None]
    if non_planar_homog and np.mean(non_planar_homog) > error_threshold_pct:
        failure_reports.append({
            "method": "Planar Homography",
            "condition": "Non-Planar Surface (Cylindrical / Irregular)",
            "mean_error_pct": round(float(np.mean(non_planar_homog)), 1),
            "failure_mechanism": "Projective homography assumes flat surface; out-of-plane curvature causes metric distortion.",
            "recommendation": "Switch to 3D surface reconstruction when surface planarity is LOW."
        })

    # 3. Evaluate 3D failure boundary under adverse lighting
    lighting_3d = [r.error_3d_pct for r in records if r.lighting_condition in ("glare", "low_light") and r.error_3d_pct is not None]
    if lighting_3d and np.mean(lighting_3d) > error_threshold_pct:
        failure_reports.append({
            "method": "3D Reconstruction",
            "condition": "Adverse Illumination (Glare / Low Light)",
            "mean_error_pct": round(float(np.mean(lighting_3d)), 1),
            "failure_mechanism": "Photometric gradient degradation impairs monocular depth relief and edge localization.",
            "recommendation": "Employ localized contrast equalization and verify with planar homography if reference target is present."
        })

    return failure_reports
