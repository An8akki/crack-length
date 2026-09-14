"""Homography estimation, polygon validation, and perspective distortion evaluation."""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np


def order_quadrilateral_corners(points: Union[List, np.ndarray]) -> np.ndarray:
    """Order four 2D corner points in clockwise order starting from top-left.

    Returns:
        np.ndarray with shape (4, 2) in order:
        [0]: Top-Left (TL)
        [1]: Top-Right (TR)
        [2]: Bottom-Right (BR)
        [3]: Bottom-Left (BL)
    """
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    # Sum of coordinates: TL will have smallest sum, BR will have largest sum
    s = pts.sum(axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]

    # Difference (y - x): TR will have smallest difference, BL will have largest difference
    diff = pts[:, 1] - pts[:, 0]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


def validate_quadrilateral(corners_px: np.ndarray) -> Tuple[bool, str]:
    """Validate that 4 points form a non-degenerate, strictly convex quadrilateral.

    Args:
        corners_px: (4, 2) numpy array of points.

    Returns:
        (is_valid, reason_str)
    """
    pts = np.asarray(corners_px, dtype=np.float64)
    if pts.shape != (4, 2):
        return False, f"Expected 4 points of shape (4, 2), got {pts.shape}"

    # Check for NaNs or Infinities
    if not np.all(np.isfinite(pts)):
        return False, "Corner points contain NaN or Infinite values"

    # Check for distinct points
    for i in range(4):
        for j in range(i + 1, 4):
            dist = np.linalg.norm(pts[i] - pts[j])
            if dist < 1.0:
                return False, f"Points {i} and {j} are too close (distance={dist:.2f}px)"

    # Compute cross products of adjacent edges to verify strict convexity and no self-intersection
    signs = []
    for i in range(4):
        p1 = pts[i]
        p2 = pts[(i + 1) % 4]
        p3 = pts[(i + 2) % 4]
        v1 = p2 - p1
        v2 = p3 - p2
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-4:
            return False, f"Collinear vertices detected at vertex {i}"
        signs.append(np.sign(cross))

    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return False, "Quadrilateral is concave or self-intersecting (hourglass shape)"

    # Check polygon area
    area = 0.5 * abs(
        pts[0, 0] * (pts[1, 1] - pts[3, 1]) +
        pts[1, 0] * (pts[2, 1] - pts[0, 1]) +
        pts[2, 0] * (pts[3, 1] - pts[1, 1]) +
        pts[3, 0] * (pts[0, 1] - pts[2, 1])
    )
    if area < 100.0:
        return False, f"Quadrilateral area is too small ({area:.1f} sq pixels)"

    return True, "Valid strictly convex quadrilateral"


def compute_reprojection_error(
    src_points: np.ndarray,
    dst_points: np.ndarray,
    H: np.ndarray
) -> float:
    """Calculate Root Mean Square (RMS) reprojection error in pixels."""
    src = np.asarray(src_points, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_points, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    if n == 0 or H is None:
        return 0.0

    # Project src using H
    ones = np.ones((n, 1), dtype=np.float64)
    src_homo = np.hstack([src, ones])  # (n, 3)
    proj_homo = (H @ src_homo.T).T      # (n, 3)
    proj = proj_homo[:, :2] / np.maximum(np.abs(proj_homo[:, 2:3]), 1e-9)

    errors = np.linalg.norm(proj - dst, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return rmse


def assess_perspective_distortion(corners_px: np.ndarray) -> Tuple[str, float]:
    """Assess the degree of perspective distortion of the quadrilateral.

    Compares side length ratios (top vs bottom, left vs right) and aspect ratio.
    Returns:
        (category: 'LOW' | 'MEDIUM' | 'HIGH', distortion_score: float in [0, 100])
    """
    pts = np.asarray(corners_px, dtype=np.float64)
    if pts.shape != (4, 2):
        return "HIGH", 100.0

    # Ordered: TL=0, TR=1, BR=2, BL=3
    top_len = np.linalg.norm(pts[1] - pts[0])
    right_len = np.linalg.norm(pts[2] - pts[1])
    bottom_len = np.linalg.norm(pts[3] - pts[2])
    left_len = np.linalg.norm(pts[0] - pts[3])

    if min(top_len, bottom_len, left_len, right_len) < 1e-6:
        return "HIGH", 100.0

    ratio_h = max(top_len, bottom_len) / min(top_len, bottom_len)
    ratio_v = max(left_len, right_len) / min(left_len, right_len)

    # Diagonal ratio
    diag1 = np.linalg.norm(pts[2] - pts[0])
    diag2 = np.linalg.norm(pts[3] - pts[1])
    ratio_diag = max(diag1, diag2) / max(min(diag1, diag2), 1e-6)

    # Combined distortion index
    max_ratio = max(ratio_h, ratio_v, ratio_diag)
    distortion_score = min(100.0, float((max_ratio - 1.0) * 50.0))

    if max_ratio < 1.15:
        category = "LOW"
    elif max_ratio < 1.45:
        category = "MEDIUM"
    else:
        category = "HIGH"

    return category, round(distortion_score, 1)


def estimate_homography(
    corners_px: np.ndarray,
    ref_width_mm: float,
    ref_height_mm: float,
    target_px_per_mm: Optional[float] = None,
    max_rectified_dim: int = 1400
) -> Dict:
    """Estimate perspective transformation and metric mapping from 4 reference points.

    Args:
        corners_px: 4 image coordinates (TL, TR, BR, BL)
        ref_width_mm: Physical width of reference region in mm
        ref_height_mm: Physical height of reference region in mm
        target_px_per_mm: Optional desired output resolution. If None, auto-calculated.
        max_rectified_dim: Maximum size in pixels for rectified image buffer.

    Returns:
        dict containing:
            'is_valid': bool
            'status': 'VALID' | 'WARNING' | 'INVALID'
            'message': str
            'H_px': (3, 3) matrix mapping image px -> rectified px
            'H_px_inv': (3, 3) matrix mapping rectified px -> image px
            'H_metric': (3, 3) matrix mapping image px -> metric mm coordinates
            'rectified_width_px': int
            'rectified_height_px': int
            'mm_per_px_x': float
            'mm_per_px_y': float
            'mean_mm_per_px': float
            'reprojection_error_px': float
            'distortion_rating': 'LOW' | 'MEDIUM' | 'HIGH'
            'distortion_score': float
            'corners_ordered': (4, 2) ndarray
    """
    if ref_width_mm <= 0 or ref_height_mm <= 0:
        return {
            "is_valid": False,
            "status": "INVALID",
            "message": f"Reference dimensions must be positive (got {ref_width_mm}x{ref_height_mm} mm)",
        }

    corners = order_quadrilateral_corners(corners_px)
    valid_quad, quad_reason = validate_quadrilateral(corners)
    if not valid_quad:
        return {
            "is_valid": False,
            "status": "INVALID",
            "message": f"Invalid calibration polygon: {quad_reason}",
            "corners_ordered": corners.tolist(),
        }

    # Assess perspective distortion
    distortion_cat, distortion_score = assess_perspective_distortion(corners)

    # Average side lengths in source image
    top_w = np.linalg.norm(corners[1] - corners[0])
    bot_w = np.linalg.norm(corners[2] - corners[3])
    avg_w_px = 0.5 * (top_w + bot_w)

    left_h = np.linalg.norm(corners[3] - corners[0])
    right_h = np.linalg.norm(corners[2] - corners[1])
    avg_h_px = 0.5 * (left_h + right_h)

    # Determine rectified pixel dimensions
    aspect_ratio = ref_width_mm / ref_height_mm
    if target_px_per_mm is not None and target_px_per_mm > 0:
        rect_w = int(round(ref_width_mm * target_px_per_mm))
        rect_h = int(round(ref_height_mm * target_px_per_mm))
    else:
        # Scale to match image pixel density while respecting physical aspect ratio
        base_w = avg_w_px
        rect_w = int(round(base_w))
        rect_h = int(round(base_w / aspect_ratio))

    # Clamp rectified dimensions to prevent out-of-memory
    if max(rect_w, rect_h) > max_rectified_dim:
        scale_down = max_rectified_dim / max(rect_w, rect_h)
        rect_w = max(64, int(round(rect_w * scale_down)))
        rect_h = max(64, int(round(rect_h * scale_down)))

    # Ensure minimum reasonable size
    rect_w = max(64, rect_w)
    rect_h = max(64, rect_h)

    # Destination points in rectified pixel space: (TL, TR, BR, BL)
    dst_px = np.array([
        [0.0, 0.0],
        [float(rect_w), 0.0],
        [float(rect_w), float(rect_h)],
        [0.0, float(rect_h)]
    ], dtype=np.float32)

    # Destination points in physical metric space (mm): (TL, TR, BR, BL)
    dst_metric = np.array([
        [0.0, 0.0],
        [float(ref_width_mm), 0.0],
        [float(ref_width_mm), float(ref_height_mm)],
        [0.0, float(ref_height_mm)]
    ], dtype=np.float32)

    try:
        import cv2
        H_px = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_px)
        H_metric = cv2.getPerspectiveTransform(corners.astype(np.float32), dst_metric)
        H_px_inv = np.linalg.inv(H_px)
    except Exception as e:
        return {
            "is_valid": False,
            "status": "INVALID",
            "message": f"Perspective transform computation failed: {str(e)}",
        }

    # Compute reprojection error (RMSE in pixels)
    reproj_err = compute_reprojection_error(corners, dst_px, H_px)

    mm_per_px_x = float(ref_width_mm / rect_w)
    mm_per_px_y = float(ref_height_mm / rect_h)
    mean_mm_per_px = float(0.5 * (mm_per_px_x + mm_per_px_y))

    # Determine status
    if reproj_err > 4.0 or distortion_cat == "HIGH":
        status = "WARNING"
        status_msg = f"Valid homography computed with high perspective or reprojection error ({reproj_err:.2f}px)"
    else:
        status = "VALID"
        status_msg = f"High quality homography (reprojection error: {reproj_err:.2f}px)"

    return {
        "is_valid": True,
        "status": status,
        "message": status_msg,
        "H_px": H_px,
        "H_px_inv": H_px_inv,
        "H_metric": H_metric,
        "rectified_width_px": rect_w,
        "rectified_height_px": rect_h,
        "ref_width_mm": float(ref_width_mm),
        "ref_height_mm": float(ref_height_mm),
        "mm_per_px_x": mm_per_px_x,
        "mm_per_px_y": mm_per_px_y,
        "mean_mm_per_px": mean_mm_per_px,
        "reprojection_error_px": round(reproj_err, 3),
        "distortion_rating": distortion_cat,
        "distortion_score": distortion_score,
        "corners_ordered": corners.tolist(),
    }
