"""
quantify.py
===========

Top-level orchestrator. Given a binary crack mask (and an optional calibration),
run the full pipeline and produce a single report dict.

The detector is only used once to produce the mask. Everything downstream is
handled here.
"""

import numpy as np

from .skeleton import skeletonize_mask
from .length import compute_length
from .width import compute_width_profile
from .area import compute_area, area_px_to_mm2
from .calibration import Calibration
from .severity import assess_severity, compute_orientation_and_branches


def quantify_crack(mask: np.ndarray, mm_per_px: float = None) -> dict:
    """
    Run the full crack quantification pipeline on a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary crack mask, shape (H, W) or (H, W, 1). Values 0/1 or 0/255.
    mm_per_px : float, optional
        Calibration scale. If None, the report is returned in pixel units and
        the severity assessment is skipped (since it requires millimetres).

    Returns
    -------
    dict
        A report with length, width, area, severity, orientation and branches.
    """
    mask = np.asarray(mask)
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]

    binary = mask.astype(bool)

    # ---- 1. Skeleton ----
    skeleton = skeletonize_mask(binary)

    # ---- 2. Length ----
    length = compute_length(skeleton)

    # ---- 3. Width ----
    width = compute_width_profile(binary, skeleton)

    # ---- 4. Area ----
    area = compute_area(binary)

    # ---- 5. Calibration / physical units ----
    cal = Calibration(mm_per_px) if mm_per_px else None

    def conv_l(px):
        return cal.px_to_mm(px) if cal else px

    def conv_a(px2):
        return cal.px2_to_mm2(px2) if cal else px2

    length_mm = conv_l(length["length_px"])
    mean_w = conv_l(width["width_stats"]["mean_width_px"])
    max_w = conv_l(width["width_stats"]["max_width_px"])
    min_w = conv_l(width["width_stats"]["min_width_px"])
    area_mm2 = conv_a(area["area_px"])

    # ---- 6. Severity + geometric descriptors ----
    if cal:
        severity = assess_severity(max_w, mean_w)
    else:
        severity = {
            "severity": "N/A",
            "based_on": "no_calibration",
            "max_width_mm": None,
            "average_width_mm": None,
            "note": "Provide mm_per_px to enable severity assessment.",
        }

    orientation = compute_orientation_and_branches(skeleton)

    report = {
        "length_px": length["length_px"],
        "length_mm": length_mm,
        "components": length["components"],
        "width_stats_px": width["width_stats"],
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
    }

    return report


def format_report(report: dict, image_id: str = "Image 001") -> str:
    """
    Format a report dict into the human-readable output block.

    Parameters
    ----------
    report : dict
        Output of `quantify_crack`.
    image_id : str
        Label for the image (e.g. 'Image 001').

    Returns
    -------
    str
        Multi-line report string.
    """
    unit = "mm" if report["calibration_mm_per_px"] else "px"
    unit2 = "mm²" if report["calibration_mm_per_px"] else "px²"

    lines = [
        image_id,
        f"Length          : {report['length_mm']:.1f} {unit}",
        f"Average Width   : {report['mean_width_mm']:.2f} {unit}",
        f"Maximum Width   : {report['max_width_mm']:.2f} {unit}",
        f"Minimum Width   : {report['min_width_mm']:.2f} {unit}",
        f"Area            : {report['area_mm2']:.1f} {unit2}",
        f"Severity        : {report['severity']['severity']}",
        f"Orientation     : {report['orientation_deg']:.0f}°",
        f"Branches        : {report['branches']}",
    ]
    return "\n".join(lines)
