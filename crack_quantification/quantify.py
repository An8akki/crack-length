"""
quantify.py
===========

Top-level orchestrator. Given a binary crack mask (and optional calibration, depth map,
and planar homography), runs the complete crack quantification pipeline and produces
a unified structural report.

Integrated Pipeline Features:
1. Multi-method skeletonization & network length (Guo-Hall / Zhang-Suen / Morphological)
2. Advanced Crack Width: Normal-ray cross-sections, EDT radius, boundary sensitivity & ACI 224R code
3. Defensible Crack Area: Calibrated metric area with boundary uncertainty confidence bounds (± Δ mm²)
4. 3D Crack Volume: Physical crevice recess volume (mm³ & cm³) via depth map surface integration
5. Severity & Structural Code Assessment: ACI 224R engineering guidance and branching topology
"""

from typing import Optional
import numpy as np

from .skeleton import skeletonize_mask
from .length import compute_length
from .width import (
    compute_width_profile,
    compute_advanced_width_profile,
    classify_crack_width,
)
from .area import compute_area, compute_area_uncertainty, compute_perspective_corrected_area
from .calibration import Calibration
from .severity import assess_severity, compute_orientation_and_branches
from .volume import compute_crack_volume, compute_volume_uncertainty


def quantify_crack(
    mask: np.ndarray,
    mm_per_px: Optional[float] = None,
    depth_map_mm: Optional[np.ndarray] = None,
    H_homography: Optional[np.ndarray] = None,
    ref_width_mm: Optional[float] = None,
    ref_height_mm: Optional[float] = None,
    return_advanced: bool = True,
) -> dict:
    """
    Run the full crack quantification pipeline on a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary crack mask, shape (H, W) or (H, W, 1). Values 0/1 or 0/255.
    mm_per_px : float, optional
        Calibration scale (mm per pixel). If None, pixel-only units are returned.
    depth_map_mm : np.ndarray, optional
        Calibrated 2D depth / heightmap array in mm for 3D recess volume estimation.
    H_homography : np.ndarray, optional
        3x3 planar homography matrix for perspective-corrected area calculation.
    ref_width_mm, ref_height_mm : float, optional
        Reference physical plane dimensions for homography metric scaling.
    return_advanced : bool
        If True, includes normal-ray profiles, boundary uncertainty, and volume models.

    Returns
    -------
    dict
        Unified report dictionary containing length, width, area, volume, severity,
        and civil structural engineering descriptors.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    binary = mask.astype(bool)

    # ---- 1. Skeleton ----
    skeleton = skeletonize_mask(binary)

    # ---- 2. Length ----
    length = compute_length(skeleton)

    # ---- 3. Width (Basic + Advanced) ----
    width_basic = compute_width_profile(binary, skeleton)
    width_adv = None
    if return_advanced:
        try:
            width_adv = compute_advanced_width_profile(binary, skeleton, mm_per_px=mm_per_px)
        except Exception:
            width_adv = None

    # ---- 4. Area (Basic + Uncertainty + Perspective) ----
    area = compute_area(binary)
    area_unc = compute_area_uncertainty(binary, mm_per_px=mm_per_px)
    perspective_area = None
    if H_homography is not None and ref_width_mm and ref_height_mm:
        try:
            perspective_area = compute_perspective_corrected_area(
                binary,
                H_px=H_homography,
                rectified_width_mm=ref_width_mm,
                rectified_height_mm=ref_height_mm,
            )
        except Exception:
            perspective_area = None

    # ---- 5. Calibration / physical units ----
    cal = Calibration(mm_per_px) if mm_per_px else None

    def conv_l(px):
        return cal.px_to_mm(px) if cal else px

    def conv_a(px2):
        return cal.px2_to_mm2(px2) if cal else px2

    length_mm = conv_l(length["length_px"])
    mean_w = conv_l(width_basic["width_stats"]["mean_width_px"])
    max_w = conv_l(width_basic["width_stats"]["max_width_px"])
    min_w = conv_l(width_basic["width_stats"]["min_width_px"])
    area_mm2 = conv_a(area["area_px"])

    # ---- 6. 3D Volume Integration (Candidate B) ----
    volume_res = None
    volume_unc = None
    if depth_map_mm is not None and mm_per_px and mm_per_px > 0:
        try:
            est_surface = bool(np.nanmedian(depth_map_mm[binary]) > 100.0) if binary.any() else True
            volume_res = compute_crack_volume(binary, depth_map_mm, mm_per_px=mm_per_px, estimate_surface=est_surface)
            volume_unc = compute_volume_uncertainty(binary, depth_map_mm, mm_per_px=mm_per_px)
        except Exception as e:
            volume_res = {"error": str(e), "volume_mm3": 0.0, "volume_cm3": 0.0}

    # ---- 7. Severity + geometric descriptors + ACI code classification ----
    if cal:
        severity = assess_severity(max_w, mean_w)
        aci_class = classify_crack_width(max_w)
    else:
        severity = {
            "severity": "N/A",
            "based_on": "no_calibration",
            "max_width_mm": None,
            "average_width_mm": None,
            "note": "Provide mm_per_px to enable severity assessment.",
        }
        aci_class = classify_crack_width(None)

    orientation = compute_orientation_and_branches(skeleton)

    report = {
        # Core backward-compatible fields:
        "length_px": length["length_px"],
        "length_mm": length_mm,
        "components": length["components"],
        "width_stats_px": width_basic["width_stats"],
        "mean_width_mm": mean_w,
        "max_width_mm": max_w,
        "min_width_mm": min_w,
        "area_px": area["area_px"],
        "area_mm2": area_mm2,
        "severity": severity,
        "orientation_deg": orientation["orientation_deg"],
        "branches": orientation["branches"],
        "endpoints": orientation["endpoints"],
        "calibration_mm_per_px": cal.mm_per_px if cal else None,

        # Candidate A: Enhanced Width Estimation:
        "width_advanced": width_adv,
        "aci_classification": aci_class,

        # Candidate C: Area Uncertainty & Perspective:
        "area_uncertainty": area_unc,
        "perspective_area": perspective_area,

        # Candidate B: Volume Estimation:
        "volume": volume_res,
        "volume_uncertainty": volume_unc,
    }

    return report


def format_report(report: dict, image_id: str = "Image 001") -> str:
    """
    Format a report dict into a human-readable engineering inspection summary.
    """
    unit = "mm" if report["calibration_mm_per_px"] else "px"
    unit2 = "mm^2" if report["calibration_mm_per_px"] else "px^2"

    lines = [
        image_id,
        "=" * 45,
        f"Length          : {report['length_mm']:.1f} {unit}",
        f"Average Width   : {report['mean_width_mm']:.2f} {unit}",
        f"Maximum Width   : {report['max_width_mm']:.2f} {unit}",
        f"Minimum Width   : {report['min_width_mm']:.2f} {unit}",
        f"Area            : {report['area_mm2']:.1f} {unit2}",
    ]

    # Add Area Uncertainty if available
    area_unc = report.get("area_uncertainty")
    if area_unc and area_unc.get("nominal_area_mm2") is not None:
        delta = area_unc["uncertainty_delta_mm2"]
        lines.append(f"Area Bounds (95%): {area_unc['nominal_area_mm2']:.1f} +/- {delta:.1f} mm^2 (+/-{area_unc['relative_uncertainty_percent']}%)")

    # Add Volume if available
    vol = report.get("volume")
    if vol and vol.get("volume_mm3", 0) > 0:
        lines.append(f"Recess Volume   : {vol['volume_mm3']:.1f} mm^3 ({vol['volume_cm3']:.3f} cm^3)")
        lines.append(f"Mean Depth      : {vol['mean_recess_depth_mm']:.2f} mm (Max: {vol['max_recess_depth_mm']:.2f} mm)")

    # Add ACI Classification if available
    aci = report.get("aci_classification")
    if aci and aci.get("category") and aci["category"] != "Unknown":
        lines.append(f"ACI Code Class  : {aci['category']} ({aci.get('code_standard', 'ACI 224R')})")
        lines.append(f"Action Advised  : {aci.get('action', '')}")
    else:
        lines.append(f"Severity        : {report['severity']['severity']}")

    lines.extend([
        f"Orientation     : {report['orientation_deg']:.0f} deg",
        f"Branches        : {report['branches']}",
        f"Endpoints       : {report['endpoints']}",
    ])

    return "\n".join(lines)
