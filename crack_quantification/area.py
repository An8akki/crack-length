"""
area.py
=======

Compute the crack area from a binary mask.

Area = number of white (foreground) pixels.

To convert to real-world units, multiply by the pixel area (mm_per_px ** 2).
"""

import numpy as np


def compute_area(mask: np.ndarray) -> dict:
    """
    Compute the crack area in pixels from a binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, shape (H, W) or (H, W, 1). Values 0/1 or 0/255.

    Returns
    -------
    dict
        {
            'area_px': total number of foreground pixels,
            'area_px2': same as area_px (alias for clarity),
            'pixel_count': same as area_px,
        }
    """
    mask = np.asarray(mask)

    # Squeeze single channel
    if mask.ndim == 3:
        if mask.shape[2] == 1:
            mask = mask[:, :, 0]
        else:
            raise ValueError(f"Expected single-channel mask, got {mask.shape}")

    if mask.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {mask.ndim}D")

    binary = mask.astype(np.uint8) > 0
    count = int(binary.sum())

    return {
        "area_px": count,
        "area_px2": count,
        "pixel_count": count,
    }


def area_px_to_mm2(area_px: float, mm_per_px: float) -> float:
    """
    Convert a pixel count to square millimetres.

    Each pixel spans mm_per_px millimetres, so its area is mm_per_px**2.

    Parameters
    ----------
    area_px : float
        Crack area in pixels.
    mm_per_px : float
        Real-world calibration (mm per pixel). Must be > 0.

    Returns
    -------
    float
        Crack area in square millimetres.
    """
    if mm_per_px <= 0:
        raise ValueError("mm_per_px must be positive")
    return float(area_px) * (mm_per_px ** 2)
