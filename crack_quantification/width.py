"""
width.py
========

Compute the crack width at every point along its centerline (skeleton).

Approach
--------
1. Compute the Euclidean distance transform of the binary mask:
   each background pixel gets its distance to the nearest crack pixel, and
   each crack pixel gets its distance to the nearest background pixel.
   We use `scipy.ndimage.distance_transform_edt` on the *inverted* mask, so
   every crack pixel carries the distance to the nearest boundary.
2. For every skeleton pixel, the local radius = distance transform value.
   Width = 2 * radius  (full width across the crack perpendicular to the
   centerline).

This yields a per-point width along the crack, from which average, maximum and
minimum width are derived, plus the full width profile.
"""

import numpy as np

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False


def _distance_transform(mask: np.ndarray) -> np.ndarray:
    """
    Euclidean distance transform of the crack mask.

Returns an array, same shape as mask, where each foreground (crack) pixel
    holds its distance (in pixels) to the nearest background pixel.
    """
    binary = mask.astype(bool)
    if not SCIPY_AVAILABLE:
        # Very small fallback: brute-force nearest background distance.
        h, w = binary.shape
        idx = np.argwhere(~binary)
        dist = np.zeros_like(binary, dtype=np.float32)
        if len(idx) == 0:
            return dist
        fg = np.argwhere(binary)
        for (y, x) in fg:
            d = idx - np.array([y, x])
            dist[y, x] = np.sqrt((d ** 2).sum(axis=1)).min()
        return dist

    # distance_transform_edt(binary) computes, for every foreground (True)
    # pixel, its Euclidean distance to the nearest background (False) pixel.
    # That is exactly the "radius" we need for the local width.
    return distance_transform_edt(binary)


def compute_width_profile(mask: np.ndarray, skeleton: np.ndarray) -> dict:
    """
    Compute width statistics and a per-point width profile along the skeleton.

    Parameters
    ----------
    mask : np.ndarray
        Binary crack mask, shape (H, W).
    skeleton : np.ndarray
        Boolean skeleton (1-pixel centerline), shape (H, W).

    Returns
    -------
    dict
        {
            'distance_transform': np.ndarray (radius per crack pixel),
            'width_profile': np.ndarray (width per skeleton pixel),
            'width_stats': {
                'mean_width_px': float,
                'max_width_px': float,
                'min_width_px': float,
                'median_width_px': float,
                'std_width_px': float,
            },
        }
    """
    mask = np.asarray(mask)
    skeleton = np.asarray(skeleton)

    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if skeleton.ndim == 3 and skeleton.shape[2] == 1:
        skeleton = skeleton[:, :, 0]

    if mask.shape != skeleton.shape:
        raise ValueError(
            f"mask {mask.shape} and skeleton {skeleton.shape} must have the same shape"
        )

    binary = mask.astype(bool)
    skeleton = skeleton.astype(bool)

    if not binary.any():
        return {
            "distance_transform": np.zeros_like(binary, dtype=np.float32),
            "width_profile": np.zeros_like(binary, dtype=np.float32),
            "width_stats": {
                "mean_width_px": 0.0,
                "max_width_px": 0.0,
                "min_width_px": 0.0,
                "median_width_px": 0.0,
                "std_width_px": 0.0,
            },
        }

    dist = _distance_transform(binary)

    # Width at each skeleton pixel = 2 * radius
    width_profile = np.zeros_like(binary, dtype=np.float32)
    width_profile[skeleton] = 2.0 * dist[skeleton]

    sp = width_profile[skeleton]
    if sp.size == 0:
        stats = {
            "mean_width_px": 0.0,
            "max_width_px": 0.0,
            "min_width_px": 0.0,
            "median_width_px": 0.0,
            "std_width_px": 0.0,
        }
    else:
        stats = {
            "mean_width_px": float(sp.mean()),
            "max_width_px": float(sp.max()),
            "min_width_px": float(sp.min()),
            "median_width_px": float(np.median(sp)),
            "std_width_px": float(sp.std()),
        }

    return {
        "distance_transform": dist,
        "width_profile": width_profile,
        "width_stats": stats,
    }
