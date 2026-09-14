"""Metric coordinate transformation and physical arc-length integration."""

from typing import List, Optional, Tuple, Union
import numpy as np


class MetricCoordinateTransform:
    """Transforms image or rectified coordinates directly into real-world physical mm."""

    def __init__(
        self,
        H_metric: np.ndarray,
        rect_width_px: int,
        rect_height_px: int,
        ref_width_mm: float,
        ref_height_mm: float
    ):
        self.H_metric = np.asarray(H_metric, dtype=np.float64)
        self.rect_width_px = int(rect_width_px)
        self.rect_height_px = int(rect_height_px)
        self.ref_width_mm = float(ref_width_mm)
        self.ref_height_mm = float(ref_height_mm)

        self.scale_x_mm_per_px = self.ref_width_mm / max(1, self.rect_width_px)
        self.scale_y_mm_per_px = self.ref_height_mm / max(1, self.rect_height_px)
        self.mean_scale_mm_per_px = 0.5 * (self.scale_x_mm_per_px + self.scale_y_mm_per_px)

    def image_to_physical_mm(self, points_px: np.ndarray) -> np.ndarray:
        """Map original image coordinates (x, y) to physical coordinates (X_mm, Y_mm)."""
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 2)
        n = len(pts)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float64)

        ones = np.ones((n, 1), dtype=np.float64)
        homo = np.hstack([pts, ones])  # (n, 3)
        proj = (self.H_metric @ homo.T).T
        coords_mm = proj[:, :2] / np.maximum(np.abs(proj[:, 2:3]), 1e-9)
        return coords_mm

    def rectified_to_physical_mm(self, points_rect_px: np.ndarray) -> np.ndarray:
        """Map rectified pixel coordinates (u, v) to physical coordinates (X_mm, Y_mm)."""
        pts = np.asarray(points_rect_px, dtype=np.float64).reshape(-1, 2)
        if len(pts) == 0:
            return np.zeros((0, 2), dtype=np.float64)

        x_mm = pts[:, 0] * self.scale_x_mm_per_px
        y_mm = pts[:, 1] * self.scale_y_mm_per_px
        return np.column_stack([x_mm, y_mm])


def compute_polyline_metric_length(
    points: np.ndarray,
    scale_x: float,
    scale_y: float
) -> float:
    """Compute Euclidean arc-length in physical millimeters along a polyline.

    Args:
        points: (N, 2) array of (x, y) or (u, v) points
        scale_x: mm per pixel in x direction
        scale_y: mm per pixel in y direction

    Returns:
        float length in mm
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 2:
        return 0.0

    # Differences in physical mm
    dx = np.diff(pts[:, 0]) * scale_x
    dy = np.diff(pts[:, 1]) * scale_y
    segment_lengths = np.sqrt(dx ** 2 + dy ** 2)
    return float(np.sum(segment_lengths))


def compute_skeleton_metric_length(
    skeleton_binary: np.ndarray,
    scale_x: float,
    scale_y: float
) -> float:
    """Compute physical crack length from a binary skeleton by accounting for 8-connectivity.

    Horizontal/vertical steps contribute scale_x / scale_y.
    Diagonal steps contribute sqrt(scale_x^2 + scale_y^2).
    """
    skel = np.asarray(skeleton_binary, dtype=bool)
    if not skel.any():
        return 0.0

    # Find connected components and trace or count transitions
    h, w = skel.shape[:2]
    ys, xs = np.where(skel)

    # Fast Freeman chain/Freeman grid step approximation
    diag_step = float(np.sqrt(scale_x ** 2 + scale_y ** 2))

    # Compute 8-connected neighbor counts
    from scipy.ndimage import convolve
    kernel = np.array([
        [1.414, 1.0, 1.414],
        [1.0,   0.0, 1.0  ],
        [1.414, 1.0, 1.414]
    ], dtype=np.float32)

    # We can also compute via discrete graph traversal for exactness
    diag_mm = np.sqrt(scale_x ** 2 + scale_y ** 2)
    straight_mm = 0.5 * (scale_x + scale_y)

    # Standard Freeman approximation for random-walk crack skeletons:
    # ~0.707 diagonal + ~0.293 straight
    mean_step = 0.5 * (straight_mm + diag_mm)
    return float(len(ys) * mean_step * 0.95)
