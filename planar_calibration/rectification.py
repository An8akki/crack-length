"""Perspective rectification of images and crack masks onto a fronto-parallel metric plane."""

from typing import Dict, Optional, Tuple
import numpy as np


def rectify_image_and_mask(
    image_rgb: np.ndarray,
    crack_mask: Optional[np.ndarray],
    H_px: np.ndarray,
    rect_width_px: int,
    rect_height_px: int
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Warp the original RGB image and crack mask to the rectified plane.

    Args:
        image_rgb: Input image (H, W, 3)
        crack_mask: Optional binary crack mask (H, W)
        H_px: 3x3 homography matrix mapping (x, y) -> (u, v)
        rect_width_px: Rectified image width in pixels
        rect_height_px: Rectified image height in pixels

    Returns:
        (rectified_image, rectified_mask)
    """
    try:
        import cv2
    except ImportError as e:
        raise ImportError("OpenCV is required for perspective warping") from e

    dsize = (int(rect_width_px), int(rect_height_px))

    # Warp image with linear interpolation
    rectified_img = cv2.warpPerspective(
        image_rgb, H_px, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )

    rectified_mask = None
    if crack_mask is not None and crack_mask.any():
        # Warp mask with nearest neighbor to preserve clean binary edges
        mask_uint8 = (crack_mask > 0).astype(np.uint8) * 255
        warped_m = cv2.warpPerspective(
            mask_uint8, H_px, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        rectified_mask = (warped_m > 127).astype(np.uint8)

    return rectified_img, rectified_mask


def create_calibration_overlay(
    image_rgb: np.ndarray,
    corners_px: np.ndarray,
    ref_width_mm: float,
    ref_height_mm: float,
    status: str = "VALID",
    crack_mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """Create an annotated visualization on the original image showing the calibrated plane.

    Highlights:
        - Translucent reference quadrilateral
        - Corner points labeled TL, TR, BR, BL with pixel coordinates
        - Metric dimensions (W x H mm) along top and left edges
        - Crack contour if mask is provided
    """
    try:
        import cv2
    except ImportError:
        return image_rgb.copy()

    vis = image_rgb.copy()
    h, w = vis.shape[:2]
    pts = np.asarray(corners_px, dtype=np.int32).reshape((-1, 1, 2))

    # 1. Draw translucent polygon fill
    overlay = vis.copy()
    fill_color = (2, 132, 199) if status == "VALID" else (245, 158, 11)  # Sky blue or Amber
    cv2.fillPoly(overlay, [pts], fill_color)
    vis = cv2.addWeighted(vis, 0.78, overlay, 0.22, 0)

    # 2. Draw border
    border_color = (56, 189, 248) if status == "VALID" else (251, 191, 36)
    cv2.polylines(vis, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

    # 3. Label corners
    labels = ["P1 (TL)", "P2 (TR)", "P3 (BR)", "P4 (BL)"]
    corner_colors = [(0, 255, 255), (0, 255, 0), (255, 128, 0), (255, 0, 255)]

    for idx, (pt, label) in enumerate(zip(corners_px, labels)):
        x, y = int(round(pt[0])), int(round(pt[1]))
        cv2.circle(vis, (x, y), 6, corner_colors[idx], -1, lineType=cv2.LINE_AA)
        cv2.circle(vis, (x, y), 8, (0, 0, 0), 1, lineType=cv2.LINE_AA)

        # Smart label offset
        ox = 10 if x < w - 80 else -70
        oy = -10 if y > 30 else 20
        cv2.putText(vis, label, (x + ox, y + oy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, label, (x + ox, y + oy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, corner_colors[idx], 1, cv2.LINE_AA)

    # 4. Draw dimension text on polygon center
    center_x = int(np.mean(corners_px[:, 0]))
    center_y = int(np.mean(corners_px[:, 1]))
    dim_txt = f"{ref_width_mm:.0f} x {ref_height_mm:.0f} mm"
    cv2.putText(vis, dim_txt, (center_x - 55, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, dim_txt, (center_x - 55, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    # 5. Overlay crack if provided
    if crack_mask is not None and crack_mask.any():
        colored_crack = np.zeros_like(vis)
        colored_crack[crack_mask > 0] = [236, 72, 153]  # Pink crack mask
        vis = cv2.addWeighted(vis, 0.85, colored_crack, 0.35, 0)

    return vis


def create_rectified_crack_overlay(
    rectified_img: np.ndarray,
    rectified_mask: Optional[np.ndarray],
    rectified_skel: Optional[np.ndarray],
    ref_width_mm: float,
    ref_height_mm: float
) -> np.ndarray:
    """Create a presentation overlay of the rectified plane with crack mask and centerline."""
    try:
        import cv2
    except ImportError:
        return rectified_img.copy()

    vis = rectified_img.copy()
    rh, rw = vis.shape[:2]

    # Crack mask in translucent cyan/magenta
    if rectified_mask is not None and rectified_mask.any():
        colored_m = np.zeros_like(vis)
        colored_m[rectified_mask > 0] = [168, 85, 247]  # Purple mask
        vis = cv2.addWeighted(vis, 0.8, colored_m, 0.3, 0)

    # Skeleton centerline in bright green
    if rectified_skel is not None and rectified_skel.any():
        dil = cv2.dilate(rectified_skel.astype(np.uint8), np.ones((2, 2), np.uint8))
        vis[dil > 0] = [34, 197, 94]  # Emerald green centerline

    # Add metric scale bar on bottom right
    bar_len_mm = max(10.0, round(ref_width_mm * 0.2 / 10.0) * 10.0)  # ~20% of width
    px_per_mm = rw / ref_width_mm
    bar_len_px = int(round(bar_len_mm * px_per_mm))

    margin_x = 24
    margin_y = 24
    bar_x2 = rw - margin_x
    bar_x1 = max(10, bar_x2 - bar_len_px)
    bar_y = rh - margin_y

    cv2.rectangle(vis, (bar_x1 - 8, bar_y - 24), (bar_x2 + 8, bar_y + 12), (15, 23, 42), -1)
    cv2.line(vis, (bar_x1, bar_y), (bar_x2, bar_y), (255, 255, 255), 3, lineType=cv2.LINE_AA)
    cv2.line(vis, (bar_x1, bar_y - 4), (bar_x1, bar_y + 4), (255, 255, 255), 2)
    cv2.line(vis, (bar_x2, bar_y - 4), (bar_x2, bar_y + 4), (255, 255, 255), 2)

    bar_label = f"{bar_len_mm:.0f} mm"
    cv2.putText(vis, bar_label, (bar_x1 + 4, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

    # Title watermark
    cv2.putText(vis, f"RECTIFIED METRIC PLANE ({ref_width_mm:.0f}x{ref_height_mm:.0f} mm)", (16, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 2, cv2.LINE_AA)

    return vis
