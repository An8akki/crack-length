"""Metric 3D crack-centreline measurements from depth and camera intrinsics."""

import numpy as np


def backproject_pixels(points_yx, depth_map, fx, fy, cx=None, cy=None):
    """Back-project ``(row, column)`` points into the camera coordinate frame."""
    depth = np.asarray(depth_map, dtype=np.float64)
    points = np.asarray(points_yx, dtype=np.float64)
    if depth.ndim != 2 or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("depth_map must be 2D and points must have shape (N, 2)")
    if fx <= 0 or fy <= 0:
        raise ValueError("fx and fy must be positive")
    h, w = depth.shape
    cx = w / 2.0 if cx is None else float(cx)
    cy = h / 2.0 if cy is None else float(cy)
    y = np.rint(points[:, 0]).astype(int)
    x = np.rint(points[:, 1]).astype(int)
    if np.any(y < 0) or np.any(y >= h) or np.any(x < 0) or np.any(x >= w):
        raise ValueError("Centreline points fall outside depth map")
    z = depth[y, x]
    if np.any(~np.isfinite(z)) or np.any(z <= 0):
        raise ValueError("Centreline depth must be finite and positive")
    return np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))


def polyline_length_3d(points_xyz):
    """Return Euclidean arc length of a 3D centreline polyline."""
    points = np.asarray(points_xyz, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def crack_length_from_depth(points_yx, depth_map, fx, fy, cx=None, cy=None):
    """Back-project and measure a centreline; depth units are output units."""
    xyz = backproject_pixels(points_yx, depth_map, fx, fy, cx, cy)
    return {"length": polyline_length_3d(xyz), "points_xyz": xyz}