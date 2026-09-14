"""Planarity heuristic assessment, measurement sanity checks, and research ground-truth evaluation."""

from typing import Dict, List, Optional, Tuple
import numpy as np


def assess_planarity_heuristic(
    homography_data: Optional[Dict],
    depth_heatmap: Optional[np.ndarray] = None,
    crack_mask: Optional[np.ndarray] = None
) -> Dict:
    """Assess whether the inspected surface can reasonably be treated as planar.

    Note: Clearly labeled as a heuristic assessment based on perspective condition
    and local depth gradient variance across the crack region.

    Returns:
        dict with:
            'score': float (0 - 100%)
            'category': 'HIGH' | 'MEDIUM' | 'LOW'
            'label': 'HIGH PLANARITY' | 'MEDIUM PLANARITY' | 'LOW PLANARITY'
            'recommended_method': 'PLANAR' | '3D'
            'assessment_type': 'heuristic'
            'explanation': str
    """
    base_score = 90.0

    # Factor 1: Perspective distortion penalty from homography
    if homography_data is not None:
        distort_score = homography_data.get("distortion_score", 0.0)
        reproj_err = homography_data.get("reprojection_error_px", 0.0)
        base_score -= min(35.0, distort_score * 0.35)
        base_score -= min(15.0, reproj_err * 2.5)

    # Factor 2: Depth map gradient variance along crack pixels if available
    if depth_heatmap is not None and crack_mask is not None and crack_mask.any():
        try:
            if depth_heatmap.ndim == 3:
                depth_gray = np.mean(depth_heatmap, axis=2)
            else:
                depth_gray = depth_heatmap.astype(np.float32)

            crack_depths = depth_gray[crack_mask > 0]
            if len(crack_depths) > 10:
                depth_std = float(np.std(crack_depths))
                # Standard deviation in normalized 0-255 depth
                if depth_std > 35.0:
                    base_score -= 25.0
                elif depth_std > 20.0:
                    base_score -= 10.0
        except Exception:
            pass

    score = max(20.0, min(98.0, base_score))

    if score >= 80.0:
        category = "HIGH"
        label = "HIGH PLANARITY"
        rec_method = "PLANAR"
        explanation = "Surface exhibits uniform geometric plane alignment with low relief distortion."
    elif score >= 55.0:
        category = "MEDIUM"
        label = "MEDIUM PLANARITY"
        rec_method = "PLANAR (Verify 3D)"
        explanation = "Moderate perspective or depth relief variance detected. Planar homography is valid but verify with 3D."
    else:
        category = "LOW"
        label = "LOW PLANARITY"
        rec_method = "3D"
        explanation = "Significant non-planar curvature or extreme perspective detected. 3D surface reconstruction is recommended."

    return {
        "score": round(score, 1),
        "category": category,
        "label": label,
        "recommended_method": rec_method,
        "assessment_type": "heuristic",
        "explanation": explanation,
    }


def run_measurement_sanity_checks(
    length_2d_mm: Optional[float],
    length_planar_mm: Optional[float],
    length_3d_mm: Optional[float],
    homography_status: str = "VALID",
    crack_in_plane_ratio: float = 1.0,
    discrepancy_threshold: float = 1.35
) -> List[Dict[str, str]]:
    """Detect suspicious discrepancies and anomalies between measurement methods.

    Returns:
        list of warning dicts: [{'level': 'warning'|'danger'|'info', 'title': str, 'message': str}]
    """
    warnings = []

    # 1. 3D vs Planar Discrepancy
    if length_3d_mm is not None and length_planar_mm is not None and length_planar_mm > 0:
        ratio_3d_planar = length_3d_mm / length_planar_mm
        if ratio_3d_planar > discrepancy_threshold or ratio_3d_planar < (1.0 / discrepancy_threshold):
            warnings.append({
                "level": "warning",
                "title": "Large Discrepancy: 3D vs Planar",
                "message": (
                    f"3D estimate ({length_3d_mm:.1f} mm) deviates from planar estimate ({length_planar_mm:.1f} mm) "
                    f"by a factor of {ratio_3d_planar:.2f}x. Check camera calibration distance, 3D scale, "
                    "or planar surface assumptions."
                )
            })

    # 2. Planar vs 2D Discrepancy
    if length_planar_mm is not None and length_2d_mm is not None and length_2d_mm > 0:
        ratio_p_2d = length_planar_mm / length_2d_mm
        if ratio_p_2d > 1.6 or ratio_p_2d < 0.5:
            warnings.append({
                "level": "info",
                "title": "Substantial Perspective Correction",
                "message": (
                    f"Planar homography shifted length from 2D {length_2d_mm:.1f} mm to {length_planar_mm:.1f} mm "
                    f"({(ratio_p_2d - 1.0)*100:+.1f}%). This indicates significant oblique perspective in the raw photo."
                )
            })

    # 3. Reference Extrapolation Warning
    if crack_in_plane_ratio < 0.4:
        warnings.append({
            "level": "warning",
            "title": "Crack Outside Reference Plane",
            "message": (
                f"Only {crack_in_plane_ratio * 100:.0f}% of crack pixels lie within the 4-point reference quadrilateral. "
                "Portions of the crack are measured via geometric plane extrapolation."
            )
        })

    # 4. Homography Status Warnings
    if homography_status == "WARNING":
        warnings.append({
            "level": "warning",
            "title": "Planar Calibration Warning",
            "message": "High perspective distortion or reprojection residual detected in reference polygon."
        })
    elif homography_status == "INVALID":
        warnings.append({
            "level": "danger",
            "title": "Invalid Planar Calibration",
            "message": "The reference quadrilateral is self-intersecting, degenerate, or has invalid dimensions."
        })

    return warnings


def compute_research_comparison(
    measurements: Dict[str, Optional[float]],
    ground_truth_mm: Optional[float] = None
) -> Dict:
    """Compare 2D, Planar, and 3D measurements against ground truth length if provided.

    Args:
        measurements: dict like {'2d': float, 'planar': float, '3d': float}
        ground_truth_mm: Optional ground-truth physical length in mm

    Returns:
        dict with comparison table and accuracy analysis
    """
    rows = []
    has_gt = ground_truth_mm is not None and ground_truth_mm > 0

    best_method = None
    min_err = float("inf")

    for method_name, est_len in measurements.items():
        if est_len is None or est_len <= 0:
            continue

        row = {
            "method": method_name.upper(),
            "estimated_mm": round(float(est_len), 2),
        }

        if has_gt:
            abs_err = abs(est_len - ground_truth_mm)
            pct_err = (abs_err / ground_truth_mm) * 100.0
            row["ground_truth_mm"] = round(float(ground_truth_mm), 2)
            row["absolute_error_mm"] = round(abs_err, 2)
            row["percentage_error"] = round(pct_err, 2)

            if pct_err < min_err:
                min_err = pct_err
                best_method = method_name.upper()

        rows.append(row)

    return {
        "has_ground_truth": has_gt,
        "ground_truth_mm": round(float(ground_truth_mm), 2) if has_gt else None,
        "comparison_rows": rows,
        "best_performing_method": best_method,
        "best_error_pct": round(min_err, 2) if has_gt and min_err != float("inf") else None,
    }
