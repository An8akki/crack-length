"""
severity.py
===========

Assess crack severity from physical measurements and compute geometric
descriptors (orientation and branch count).

Severity is currently rule-based on crack width (in millimetres), and can be
replaced later with an ML model.

    Width < 0.2 mm              -> Minor
    0.2 mm <= Width <= 0.5 mm   -> Moderate
    Width > 0.5 mm              -> Severe

Orientation is the dominant angle of the skeleton (via the principal
component / PCA), reported in degrees. Branches are counted as skeleton
junctions (pixels with 3 or more neighbours).
"""

import numpy as np


# Public severity thresholds (mm) - easy to tune / replace with ML later.
SEVERITY_THRESHOLDS = [
    (0.2, "Minor"),
    (0.5, "Moderate"),
    (float("inf"), "Severe"),
]


def severity_from_width(max_width_mm: float) -> str:
    """
    Map a crack width (mm) to a severity label.

    Parameters
    ----------
    max_width_mm : float
        Maximum crack width in millimetres.

    Returns
    -------
    str
        'Minor', 'Moderate', or 'Severe'.
    """
    for threshold, label in SEVERITY_THRESHOLDS:
        if max_width_mm < threshold:
            return label
    return "Severe"


def severity_from_avg_width(avg_width_mm: float) -> str:
    """Map an average crack width (mm) to a severity label."""
    return severity_from_width(avg_width_mm)


def compute_orientation_and_branches(skeleton: np.ndarray) -> dict:
    """
    Compute the dominant orientation and branch count of the crack skeleton.

    Parameters
    ----------
    skeleton : np.ndarray
        Boolean skeleton array, shape (H, W).

    Returns
    -------
    dict
        {
            'orientation_deg': float,  # dominant angle in degrees (0-180)
            'branches': int,           # number of branch/junction pixels
            'endpoints': int,          # number of skeleton endpoints
        }
    """
    skeleton = np.asarray(skeleton)
    if skeleton.ndim == 3 and skeleton.shape[2] == 1:
        skeleton = skeleton[:, :, 0]
    skeleton = skeleton.astype(bool)

    if not skeleton.any():
        return {"orientation_deg": 0.0, "branches": 0, "endpoints": 0}

    ys, xs = np.nonzero(skeleton)

    # ---- Orientation via PCA of skeleton point cloud ----
    points = np.column_stack([xs, ys]).astype(np.float64)
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    if cov.size == 0:
        eigvec = np.array([1.0, 0.0])
    else:
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Principal axis = eigenvector of the largest eigenvalue
        eigvec = eigvecs[:, np.argmax(eigvals)]

    angle_rad = np.arctan2(eigvec[1], eigvec[0])
    angle_deg = np.degrees(angle_rad) % 180.0

    # ---- Branch / endpoint counting ----
    padded = np.pad(skeleton, pad_width=1, mode="constant", constant_values=False)
    h, w = padded.shape
    branches = 0
    endpoints = 0

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if not padded[y, x]:
                continue
            # Count 8-connected neighbours
            neigh = int(
                padded[y - 1, x - 1] + padded[y - 1, x] + padded[y - 1, x + 1]
                + padded[y, x - 1] + padded[y, x + 1]
                + padded[y + 1, x - 1] + padded[y + 1, x] + padded[y + 1, x + 1]
            )
            if neigh >= 3:
                branches += 1
            elif neigh == 1:
                endpoints += 1

    return {
        "orientation_deg": float(angle_deg),
        "branches": int(branches),
        "endpoints": int(endpoints),
    }


def assess_severity(max_width_mm: float, avg_width_mm: float = None) -> dict:
    """
    Convenience wrapper producing a severity report.

    Parameters
    ----------
    max_width_mm : float
        Maximum crack width in millimetres.
    avg_width_mm : float, optional
        Average crack width in millimetres.

    Returns
    -------
    dict
        {
            'severity': str,
            'based_on': 'max_width' or 'average_width',
            'max_width_mm': float,
            'average_width_mm': float,
        }
    """
    if avg_width_mm is not None:
        label_from_avg = severity_from_avg_width(avg_width_mm)
        label_from_max = severity_from_width(max_width_mm)
        # Report the more severe of the two.
        order = {"Minor": 0, "Moderate": 1, "Severe": 2}
        if order[label_from_max] >= order[label_from_avg]:
            label = label_from_max
            based = "max_width"
        else:
            label = label_from_avg
            based = "average_width"
    else:
        label = severity_from_width(max_width_mm)
        based = "max_width"

    return {
        "severity": label,
        "based_on": based,
        "max_width_mm": float(max_width_mm),
        "average_width_mm": float(avg_width_mm) if avg_width_mm is not None else None,
    }
