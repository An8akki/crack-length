"""Adaptive measurement method selector and engineering reliability gatekeeper."""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class AdaptiveMethodRecommendation:
    """Engineering method recommendation with scientific justification."""
    recommended_method: str              # 'PLANAR' | '3D' | '2D' | 'UNRELIABLE'
    reliability: str                     # 'HIGH' | 'MEDIUM' | 'LOW' | 'UNRELIABLE'
    is_measurement_acceptable: bool      # True if engineering tolerance is satisfied
    justification: str                   # Scientific explanation for the recommendation
    warnings: list                       # Active warning flags

    def to_dict(self) -> Dict:
        return {
            "recommended_method": self.recommended_method,
            "reliability": self.reliability,
            "is_measurement_acceptable": self.is_measurement_acceptable,
            "justification": self.justification,
            "warnings": self.warnings,
        }


def recommend_measurement_method(diagnostics: Dict) -> AdaptiveMethodRecommendation:
    """Analyze observable geometric and calibration diagnostics to recommend the optimal method."""
    warnings = []

    planarity_score = float(diagnostics.get("planarity_score", 85.0))
    planarity_label = str(diagnostics.get("planarity_label", "HIGH")).upper()
    is_planar = bool(diagnostics.get("is_planar", True))

    reproj_err = diagnostics.get("reprojection_error_px")
    distortion = str(diagnostics.get("perspective_distortion", "LOW")).upper()
    homog_status = str(diagnostics.get("homography_status", "NOT_CONFIGURED")).upper()

    has_valid_planar = homog_status in ("VALID", "WARNING") and reproj_err is not None
    is_calibrated_2d = bool(diagnostics.get("is_physical_scale_calibrated", False))
    yaw = abs(float(diagnostics.get("camera_yaw_deg", 0.0)))

    len_2d = diagnostics.get("length_2d_mm")
    len_planar = diagnostics.get("length_planar_mm")
    len_3d = diagnostics.get("length_3d_mm")

    # Discrepancy checks
    if len_3d and len_planar and len_planar > 0:
        ratio = len_3d / len_planar
        if ratio > 1.4 or ratio < 0.7:
            warnings.append(f"Significant divergence ({ratio:.2f}x) between 3D and planar estimates.")

    # 1. Check for Complete Unreliability Condition
    if not has_valid_planar and not is_calibrated_2d and yaw > 35.0:
        return AdaptiveMethodRecommendation(
            recommended_method="UNRELIABLE",
            reliability="UNRELIABLE",
            is_measurement_acceptable=False,
            justification="Physical crack length cannot be reliably estimated from this image. Viewing perspective is highly oblique (>= 35°) with no metric reference or calibrated scale.",
            warnings=warnings
        )

    # 2. Check for Non-Planar Surface -> Recommend 3D
    if not is_planar or planarity_score < 60.0 or planarity_label == "LOW":
        return AdaptiveMethodRecommendation(
            recommended_method="3D",
            reliability="MEDIUM" if len_3d else "LOW",
            is_measurement_acceptable=bool(len_3d),
            justification="Significant out-of-plane surface curvature or irregular elevation profile detected. 3D surface arc-length projection is required.",
            warnings=warnings
        )

    # 3. Check for Valid Planar Reference on Planar Surface -> Recommend Planar Homography
    if has_valid_planar and planarity_score >= 70.0:
        rel = "HIGH" if (reproj_err < 1.5 and distortion != "HIGH") else "MEDIUM"
        return AdaptiveMethodRecommendation(
            recommended_method="PLANAR",
            reliability=rel,
            is_measurement_acceptable=True,
            justification="High surface planarity verified with valid metric reference target. Fronto-parallel projective rectification eliminates perspective foreshortening.",
            warnings=warnings
        )

    # 4. Frontal View with Calibrated 2D Scale
    if is_calibrated_2d and yaw < 15.0:
        return AdaptiveMethodRecommendation(
            recommended_method="2D",
            reliability="HIGH",
            is_measurement_acceptable=True,
            justification="Near-orthogonal frontal perspective (< 15°) with user-calibrated physical scale.",
            warnings=warnings
        )

    # 5. Default Fallback
    if len_3d:
        return AdaptiveMethodRecommendation(
            recommended_method="3D",
            reliability="LOW",
            is_measurement_acceptable=True,
            justification="Planar reference unavailable; using 3D depth-based back-projection with unconstrained camera assumptions.",
            warnings=warnings
        )

    return AdaptiveMethodRecommendation(
        recommended_method="UNRELIABLE",
        reliability="UNRELIABLE",
        is_measurement_acceptable=False,
        justification="No reliable physical calibration reference detected. Additional reference imagery is required for engineering confidence.",
        warnings=warnings
    )
