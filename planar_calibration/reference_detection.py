"""Planar reference resolution combining manual 4-point inputs and automatic marker detection."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from .homography import (
    estimate_homography,
    order_quadrilateral_corners,
    validate_quadrilateral,
)
from .marker_detection import detect_aruco_marker


@dataclass
class PlanarReference:
    """Resolved planar reference geometry."""
    mode: str                          # 'manual_4point' | 'marker_aruco' | 'none'
    is_valid: bool
    status: str                        # 'VALID' | 'WARNING' | 'INVALID'
    message: str
    corners_px: Optional[np.ndarray]   # (4, 2) array
    ref_width_mm: float
    ref_height_mm: float
    homography_data: Optional[Dict]
    marker_id: Optional[int] = None
    crack_in_plane_ratio: float = 1.0


def check_crack_inside_reference(
    crack_mask: np.ndarray,
    corners_px: np.ndarray
) -> float:
    """Compute the fraction of crack pixels that lie inside the calibrated polygon.

    Returns:
        float ratio in [0.0, 1.0]
    """
    if crack_mask is None or not crack_mask.any() or corners_px is None:
        return 1.0

    try:
        import cv2
        poly = np.asarray(corners_px, dtype=np.int32).reshape((-1, 1, 2))
        h, w = crack_mask.shape[:2]
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], 1)

        total_crack = np.count_nonzero(crack_mask)
        if total_crack == 0:
            return 1.0

        inside_crack = np.count_nonzero((crack_mask > 0) & (poly_mask > 0))
        return float(inside_crack / total_crack)
    except Exception:
        return 1.0


def resolve_planar_reference(
    image: np.ndarray,
    crack_mask: Optional[np.ndarray] = None,
    mode: str = "auto",
    manual_corners: Optional[Union[List, np.ndarray]] = None,
    ref_width_mm: Optional[float] = None,
    ref_height_mm: Optional[float] = None,
    marker_size_mm: float = 100.0,
) -> PlanarReference:
    """Resolve a planar reference from manual corners or automatic marker detection.

    Args:
        image: RGB input image
        crack_mask: Optional binary crack mask
        mode: 'manual_4point', 'marker_aruco', 'auto', or 'none'
        manual_corners: 4 points if mode is manual
        ref_width_mm: Physical width of reference in mm
        ref_height_mm: Physical height of reference in mm
        marker_size_mm: Physical marker side length in mm

    Returns:
        PlanarReference instance
    """
    # 1. Check if user provided manual 4 points
    if mode == "manual_4point" or (mode == "auto" and manual_corners is not None and len(manual_corners) == 4):
        if manual_corners is None or len(manual_corners) != 4:
            return PlanarReference(
                mode="manual_4point",
                is_valid=False,
                status="UNCONFIGURED",
                message="Manual 4-point calibration requires 4 corner points to be selected on the image. Click 4 points or click 'Auto 4 Corners'.",
                corners_px=None,
                ref_width_mm=float(ref_width_mm) if (ref_width_mm and ref_width_mm > 0) else 0.0,
                ref_height_mm=float(ref_height_mm) if (ref_height_mm and ref_height_mm > 0) else 0.0,
                homography_data=None,
            )

        if ref_width_mm is None or ref_width_mm <= 0 or ref_height_mm is None or ref_height_mm <= 0:
            return PlanarReference(
                mode="manual_4point",
                is_valid=False,
                status="INVALID",
                message="Physical reference dimensions (width and height in mm) must be provided and > 0.",
                corners_px=None,
                ref_width_mm=0.0,
                ref_height_mm=0.0,
                homography_data=None,
            )

        try:
            corners = np.asarray(manual_corners, dtype=np.float32).reshape(4, 2)
        except Exception as e:
            return PlanarReference(
                mode="manual_4point",
                is_valid=False,
                status="INVALID",
                message=f"Failed to parse corner points into (4, 2) array: {e}",
                corners_px=None,
                ref_width_mm=float(ref_width_mm),
                ref_height_mm=float(ref_height_mm),
                homography_data=None,
            )

        homog = estimate_homography(corners, ref_width_mm, ref_height_mm)

        if not homog["is_valid"]:
            return PlanarReference(
                mode="manual_4point",
                is_valid=False,
                status="INVALID",
                message=homog["message"],
                corners_px=corners,
                ref_width_mm=ref_width_mm,
                ref_height_mm=ref_height_mm,
                homography_data=homog,
            )

        crack_ratio = check_crack_inside_reference(crack_mask, homog["corners_ordered"])
        status = homog["status"]
        msg = homog["message"]
        if crack_ratio < 0.3:
            status = "WARNING"
            msg += f" (Note: only {crack_ratio*100:.0f}% of crack lies within reference polygon; extrapolation will be applied)"

        return PlanarReference(
            mode="manual_4point",
            is_valid=True,
            status=status,
            message=msg,
            corners_px=np.array(homog["corners_ordered"], dtype=np.float32),
            ref_width_mm=float(ref_width_mm),
            ref_height_mm=float(ref_height_mm),
            homography_data=homog,
            crack_in_plane_ratio=crack_ratio,
        )

    # 2. Try Automatic Marker Detection (ArUco)
    if mode in ("marker_aruco", "auto"):
        marker_res = detect_aruco_marker(image, marker_size_mm=marker_size_mm)
        if marker_res["detected"] and marker_res["corners_px"] is not None:
            corners = marker_res["corners_px"]
            w_mm = marker_res["marker_width_mm"]
            h_mm = marker_res["marker_height_mm"]

            homog = estimate_homography(corners, w_mm, h_mm)
            if homog["is_valid"]:
                crack_ratio = check_crack_inside_reference(crack_mask, homog["corners_ordered"])
                return PlanarReference(
                    mode="marker_aruco",
                    is_valid=True,
                    status=homog["status"],
                    message=f"{homog['message']} - Marker ID {marker_res['marker_id']}",
                    corners_px=np.array(homog["corners_ordered"], dtype=np.float32),
                    ref_width_mm=w_mm,
                    ref_height_mm=h_mm,
                    homography_data=homog,
                    marker_id=marker_res["marker_id"],
                    crack_in_plane_ratio=crack_ratio,
                )

    # 3. No valid planar reference found or requested
    return PlanarReference(
        mode="none",
        is_valid=False,
        status="INVALID",
        message="No planar calibration reference provided or detected.",
        corners_px=None,
        ref_width_mm=0.0,
        ref_height_mm=0.0,
        homography_data=None,
    )
