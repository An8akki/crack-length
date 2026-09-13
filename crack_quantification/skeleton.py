"""
skeleton.py
===========

Reduce a binary crack mask to a 1-pixel-wide centerline (skeleton).

The skeleton preserves the topology (connectivity and branching) of the crack
while removing its thickness. It is the foundation for computing accurate
crack length and per-point width.

Three skeletonization algorithms are provided and can be compared:

1. `zhang_suen`  - Classic Zhang-Suen fast parallel thinning.
2. `guo_hall`    - Guo-Hall thinning (often produces more connected, thinner
                   skeletons with fewer spurious branches).
3. `skeletonize` - scikit-image `skeletonize` (Zhang-Suen / Rutowitz based),
                   the default when scikit-image is available.

Use `skeletonize_mask(mask, method=...)` to choose an algorithm. A benchmark
(`compare_skeletons.py`) is provided to compare their length accuracy.
"""

import numpy as np

try:
    from skimage.morphology import skeletonize as _sk_skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:  # pragma: no cover
    SKIMAGE_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Zhang-Suen thinning
# --------------------------------------------------------------------------- #
def _zhang_suen_thinning(binary: np.ndarray) -> np.ndarray:
    """Zhang-Suen fast parallel thinning algorithm (Zhang & Suen 1984)."""
    img = (binary > 0).astype(np.uint8)
    img = np.pad(img, pad_width=1, mode="constant", constant_values=0)
    prev = np.zeros_like(img)

    while not np.array_equal(img, prev):
        prev = img.copy()
        for subiter in (1, 2):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            p1 = img[1:-1, 1:-1]

            b = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9
            a = (
                ((1 - p2) & p3)
                + ((1 - p3) & p4)
                + ((1 - p4) & p5)
                + ((1 - p5) & p6)
                + ((1 - p6) & p7)
                + ((1 - p7) & p8)
                + ((1 - p8) & p9)
                + ((1 - p9) & p2)
            )

            if subiter == 1:
                m1 = p2 * p4 * p6
                m2 = p4 * p6 * p8
            else:
                m1 = p2 * p4 * p8
                m2 = p2 * p6 * p8

            delete = (
                (p1 == 1)
                & (2 <= b)
                & (b <= 6)
                & (a == 1)
                & (m1 == 0)
                & (m2 == 0)
            )
            img[1:-1, 1:-1][delete] = 0

    return img[1:-1, 1:-1].astype(bool)


# --------------------------------------------------------------------------- #
# Guo-Hall thinning
# --------------------------------------------------------------------------- #
def _guo_hall_thinning(binary: np.ndarray) -> np.ndarray:
    """Guo-Hall fast parallel thinning algorithm (Guo & Hall 1989)."""
    img = (binary > 0).astype(np.uint8)
    img = np.pad(img, pad_width=1, mode="constant", constant_values=0)
    prev = np.zeros_like(img)

    while not np.array_equal(img, prev):
        prev = img.copy()
        for subiter in (1, 2):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            p1 = img[1:-1, 1:-1]

            c = (
                ((1 - p2) & (p3 | p4))
                + ((1 - p4) & (p5 | p6))
                + ((1 - p6) & (p7 | p8))
                + ((1 - p8) & (p9 | p2))
            )

            n1 = (p9 | p2) + (p3 | p4) + (p5 | p6) + (p7 | p8)
            n2 = (p2 | p3) + (p4 | p5) + (p6 | p7) + (p8 | p9)
            n = np.minimum(n1, n2)

            if subiter == 1:
                m = (p2 | p3 | (1 - p5)) & p4
            else:
                m = (p6 | p7 | (1 - p9)) & p8

            delete = (p1 == 1) & (c == 1) & (2 <= n) & (n <= 3) & (m == 0)
            img[1:-1, 1:-1][delete] = 0

    return img[1:-1, 1:-1].astype(bool)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def skeletonize_mask(mask: np.ndarray, method: str = "skeletonize") -> np.ndarray:
    """Compute the 1-pixel-wide skeleton of a binary crack mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask. Can be 2D (H, W) or 3D (H, W, 1) with values 0/1 or 0/255.
    method : str
        One of 'skeletonize' (scikit-image, default when available),
        'zhang_suen', or 'guo_hall'.

    Returns
    -------
    skeleton : np.ndarray
        Boolean array, shape (H, W). True = skeleton pixel.
    """
    if mask is None:
        raise ValueError("mask cannot be None")

    binary = np.asarray(mask)

    # Squeeze single-channel 3D arrays
    if binary.ndim == 3:
        if binary.shape[2] == 1:
            binary = binary[:, :, 0]
        else:
            raise ValueError(
                f"Expected a single-channel mask, got shape {binary.shape}"
            )

    if binary.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got {binary.ndim}D")

    # Normalize to boolean safely (handles boolean, float probabilities [0, 1], and [0, 255])
    if np.issubdtype(binary.dtype, np.floating):
        binary = (binary > 0.5) if binary.max() <= 1.0 else (binary > 127)
    else:
        binary = binary > 0

    if not binary.any():
        return np.zeros_like(binary, dtype=bool)

    method = method.lower()
    if method not in ("skeletonize", "zhang_suen", "guo_hall"):
        raise ValueError(
            f"Unknown method '{method}'. Choose from 'skeletonize', "
            "'zhang_suen', or 'guo_hall'."
        )

    if method == "zhang_suen":
        skeleton = _zhang_suen_thinning(binary)
    elif method == "guo_hall":
        skeleton = _guo_hall_thinning(binary)
    else:  # 'skeletonize'
        if SKIMAGE_AVAILABLE:
            from skimage.morphology import skeletonize as sk_skeletonize

            skeleton = sk_skeletonize(binary)
        else:
            skeleton = _zhang_suen_thinning(binary)

    return skeleton.astype(bool)
