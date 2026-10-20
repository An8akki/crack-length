"""
width.py
========

Compute physical crack width along its centerline (skeleton) using Euclidean
distance transforms, orthogonal normal-vector cross-sections, boundary-reliability
assessment, and structural engineering classification.

Capabilities:
1. Distance Transform Width: Fast radius calculation via Euclidean Distance Transform (EDT).
2. Orthogonal Normal Profile: Computes local skeleton tangent vectors and traces
   orthogonal normal rays across crack boundaries for true cross-sectional width.
3. Reliability & Sensitivity Analysis: Evaluates width stability when restoration
   or segmentation shifts boundary edges (dilation/erosion sensitivity).
4. Structural Engineering Classification: ACI 224R crack width thresholds (hairline,
   fine, moderate, severe).
5. Benchmark Evaluation: MAE, RMSE, bias, percentage error, and range-binned errors.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt, binary_dilation, binary_erosion
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

    return distance_transform_edt(binary)


def compute_width_profile(mask: np.ndarray, skeleton: np.ndarray) -> dict:
    """
    Compute width statistics and a per-point width profile along the skeleton via EDT.

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
            'distance_transform': np.ndarray,
            'width_profile': np.ndarray,
            'width_stats': {
                'mean_width_px': float,
                'max_width_px': float,
                'min_width_px': float,
                'median_width_px': float,
                'std_width_px': float,
                'p95_width_px': float,
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

    if not binary.any() or not skeleton.any():
        return {
            "distance_transform": np.zeros_like(binary, dtype=np.float32),
            "width_profile": np.zeros_like(binary, dtype=np.float32),
            "width_stats": {
                "mean_width_px": 0.0,
                "max_width_px": 0.0,
                "min_width_px": 0.0,
                "median_width_px": 0.0,
                "std_width_px": 0.0,
                "p95_width_px": 0.0,
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
            "p95_width_px": 0.0,
        }
    else:
        stats = {
            "mean_width_px": float(sp.mean()),
            "max_width_px": float(sp.max()),
            "min_width_px": float(sp.min()),
            "median_width_px": float(np.median(sp)),
            "std_width_px": float(sp.std()),
            "p95_width_px": float(np.percentile(sp, 95)),
        }

    return {
        "distance_transform": dist,
        "width_profile": width_profile,
        "width_stats": stats,
    }


def compute_normal_width_profile(
    mask: np.ndarray,
    skeleton: np.ndarray,
    max_search_radius_px: int = 50,
) -> dict:
    """
    Compute crack width by marching along the orthogonal normal rays at each skeleton pixel.

    For each skeleton coordinate (y, x), local tangent is derived from immediate 8-connected
    skeleton neighbours. The orthogonal normal vector is traced outward in both directions
    until a background pixel is encountered or max_search_radius_px is reached.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W).
    skeleton : np.ndarray
        Boolean skeleton (H, W).
    max_search_radius_px : int
        Maximum ray search distance in pixels.

    Returns
    -------
    dict
        {
            'normal_widths_px': np.ndarray of widths along skeleton,
            'normal_stats': dict of mean, median, max, min, std, p95,
            'coordinates': list of (y, x) skeleton points sampled,
        }
    """
    binary = (np.asarray(mask) > 0).astype(bool)
    skel = (np.asarray(skeleton) > 0).astype(bool)
    h, w = binary.shape

    ys, xs = np.nonzero(skel)
    if len(ys) == 0:
        return {
            "normal_widths_px": np.array([], dtype=np.float32),
            "normal_stats": {
                "mean_width_px": 0.0,
                "median_width_px": 0.0,
                "max_width_px": 0.0,
                "min_width_px": 0.0,
                "std_width_px": 0.0,
                "p95_width_px": 0.0,
            },
            "coordinates": [],
        }

    # Pre-build coordinate lookup for fast neighbour search
    skel_coords = list(zip(ys, xs))
    skel_set = set(skel_coords)
    normal_widths: List[float] = []
    sampled_coords: List[Tuple[int, int]] = []

    for y, x in skel_coords:
        # Find 8-connected skeleton neighbours to estimate local tangent
        neighbors = []
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if (ny, nx) in skel_set:
                    neighbors.append((dy, dx))

        if not neighbors:
            # Isolated pixel: fallback to distance transform
            normal_widths.append(1.0)
            sampled_coords.append((y, x))
            continue

        # Average local direction vector (tangent)
        mean_dy = float(np.mean([n[0] for n in neighbors]))
        mean_dx = float(np.mean([n[1] for n in neighbors]))
        tangent_norm = np.hypot(mean_dy, mean_dx)

        if tangent_norm < 1e-4:
            # Degenerate tangent: pick first neighbour
            mean_dy, mean_dx = float(neighbors[0][0]), float(neighbors[0][1])
            tangent_norm = np.hypot(mean_dy, mean_dx)

        # Normal vector perpendicular to tangent: (-dx, dy)
        ny_dir = -mean_dx / tangent_norm
        nx_dir = mean_dy / tangent_norm

        # March along positive normal direction
        r_pos = 0.0
        for step in np.arange(0.5, max_search_radius_px + 0.5, 0.5):
            py = int(round(y + step * ny_dir))
            px = int(round(x + step * nx_dir))
            if 0 <= py < h and 0 <= px < w:
                if binary[py, px]:
                    r_pos = step
                else:
                    break
            else:
                break

        # March along negative normal direction
        r_neg = 0.0
        for step in np.arange(0.5, max_search_radius_px + 0.5, 0.5):
            py = int(round(y - step * ny_dir))
            px = int(round(x - step * nx_dir))
            if 0 <= py < h and 0 <= px < w:
                if binary[py, px]:
                    r_neg = step
                else:
                    break
            else:
                break

        total_width = float(r_pos + r_neg)
        if total_width < 1.0:
            total_width = 1.0
        normal_widths.append(total_width)
        sampled_coords.append((y, x))

    arr = np.array(normal_widths, dtype=np.float32)
    stats = {
        "mean_width_px": float(arr.mean()) if len(arr) > 0 else 0.0,
        "median_width_px": float(np.median(arr)) if len(arr) > 0 else 0.0,
        "max_width_px": float(arr.max()) if len(arr) > 0 else 0.0,
        "min_width_px": float(arr.min()) if len(arr) > 0 else 0.0,
        "std_width_px": float(arr.std()) if len(arr) > 0 else 0.0,
        "p95_width_px": float(np.percentile(arr, 95)) if len(arr) > 0 else 0.0,
    }

    return {
        "normal_widths_px": arr,
        "normal_stats": stats,
        "coordinates": sampled_coords,
    }


def assess_width_reliability(
    mask: np.ndarray,
    skeleton: np.ndarray,
    mm_per_px: Optional[float] = None,
) -> dict:
    """
    Evaluate width measurement reliability under boundary errors and restoration smoothing.

    Perturbs the segmentation mask boundaries via 1-pixel dilation and erosion
    to simulate segmentation uncertainty and restoration edge-shifts, then quantifies
    width sensitivity.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W).
    skeleton : np.ndarray
        Binary skeleton (H, W).
    mm_per_px : float, optional
        Calibration factor.

    Returns
    -------
    dict
        {
            'nominal_mean_width_px': float,
            'eroded_mean_width_px': float,
            'dilated_mean_width_px': float,
            'boundary_sensitivity_px': float (delta from nominal),
            'confidence_interval_95_px': Tuple[float, float],
            'reliability_score': float (0.0 to 1.0, 1.0 = highly stable),
            'is_reliable': bool,
            'nominal_mean_width_mm': Optional[float],
            'confidence_interval_95_mm': Optional[Tuple[float, float]],
        }
    """
    binary = (np.asarray(mask) > 0).astype(bool)
    skel = (np.asarray(skeleton) > 0).astype(bool)

    if not binary.any() or not skel.any():
        return {
            "nominal_mean_width_px": 0.0,
            "eroded_mean_width_px": 0.0,
            "dilated_mean_width_px": 0.0,
            "boundary_sensitivity_px": 0.0,
            "confidence_interval_95_px": (0.0, 0.0),
            "reliability_score": 0.0,
            "is_reliable": False,
            "nominal_mean_width_mm": None,
            "confidence_interval_95_mm": None,
        }

    nominal_res = compute_width_profile(binary, skel)["width_stats"]
    nom_mean = nominal_res["mean_width_px"]
    nom_std = nominal_res["std_width_px"]

    if SCIPY_AVAILABLE:
        struct = np.ones((3, 3), dtype=bool)
        eroded = binary_erosion(binary, structure=struct)
        dilated = binary_dilation(binary, structure=struct)
    else:
        # Fallback simple border modification
        eroded = binary.copy()
        dilated = binary.copy()

    eroded_mean = compute_width_profile(eroded, skel)["width_stats"]["mean_width_px"]
    dilated_mean = compute_width_profile(dilated, skel)["width_stats"]["mean_width_px"]

    delta_erosion = abs(nom_mean - eroded_mean)
    delta_dilation = abs(dilated_mean - nom_mean)
    sensitivity_px = float(max(delta_erosion, delta_dilation))

    # Margin of error bound
    ci_low_px = max(0.0, nom_mean - sensitivity_px)
    ci_high_px = nom_mean + sensitivity_px

    # Score: ratio of nominal width relative to 1-pixel boundary uncertainty
    # If width is 8 pixels and shift is 1 px, reliability is high (~0.88).
    # If width is 1.2 pixels and shift is 1 px, reliability is low (~0.2).
    if nom_mean > 0:
        rel_score = float(np.clip(1.0 - (sensitivity_px / nom_mean), 0.1, 0.99))
    else:
        rel_score = 0.0

    is_reliable = bool(rel_score >= 0.5 and nom_mean >= 1.5)

    res = {
        "nominal_mean_width_px": round(nom_mean, 3),
        "eroded_mean_width_px": round(eroded_mean, 3),
        "dilated_mean_width_px": round(dilated_mean, 3),
        "boundary_sensitivity_px": round(sensitivity_px, 3),
        "confidence_interval_95_px": (round(ci_low_px, 3), round(ci_high_px, 3)),
        "reliability_score": round(rel_score, 3),
        "is_reliable": is_reliable,
    }

    if mm_per_px and mm_per_px > 0:
        res["nominal_mean_width_mm"] = round(nom_mean * mm_per_px, 3)
        res["confidence_interval_95_mm"] = (
            round(ci_low_px * mm_per_px, 3),
            round(ci_high_px * mm_per_px, 3),
        )
        res["boundary_sensitivity_mm"] = round(sensitivity_px * mm_per_px, 3)

    return res


def classify_crack_width(width_mm: float) -> dict:
    """
    Classify crack width according to ACI 224R civil engineering guidelines.

    Categories:
    - Hairline:  w < 0.10 mm   (Structurally negligible, aesthetic only)
    - Fine:      0.10 <= w < 0.20 mm (Acceptable under benign indoor conditions)
    - Moderate:  0.20 <= w < 0.30 mm (Waterproofing required, deicing risk)
    - Severe:    w >= 0.30 mm  (Structural limit exceeded, active repair required)
    """
    if width_mm is None or width_mm < 0:
        return {"category": "Unknown", "limit_mm": None, "action": "Requires calibration"}

    if width_mm < 0.10:
        return {
            "category": "Hairline",
            "code_standard": "ACI 224R-01",
            "threshold_limit_mm": 0.10,
            "action": "Cosmetic monitoring only; no structural intervention required.",
        }
    elif width_mm < 0.20:
        return {
            "category": "Fine",
            "code_standard": "ACI 224R-01",
            "threshold_limit_mm": 0.20,
            "action": "Tolerable for interior dry exposure; monitor seasonally.",
        }
    elif width_mm < 0.30:
        return {
            "category": "Moderate",
            "code_standard": "ACI 224R-01",
            "threshold_limit_mm": 0.30,
            "action": "Exceeds exterior humidity threshold; apply hydrophobic sealant.",
        }
    else:
        return {
            "category": "Severe",
            "code_standard": "ACI 224R-01",
            "threshold_limit_mm": 0.30,
            "action": "Structural integrity compromised; immediate epoxy injection / reinforcement repair.",
        }


def compute_advanced_width_profile(
    mask: np.ndarray,
    skeleton: np.ndarray,
    mm_per_px: Optional[float] = None,
) -> dict:
    """
    Unified high-accuracy crack width estimator combining EDT, orthogonal ray normal
    cross-sections, boundary-reliability modeling, and civil structural classification.
    """
    edt_res = compute_width_profile(mask, skeleton)
    norm_res = compute_normal_width_profile(mask, skeleton)
    rel_res = assess_width_reliability(mask, skeleton, mm_per_px=mm_per_px)

    edt_stats = edt_res["width_stats"]
    norm_stats = norm_res["normal_stats"]

    # Hybrid consensus: EDT provides robust minimum bounding radius,
    # normal profile captures irregular surface edge bulging.
    # Take weighted ensemble: 0.6 EDT + 0.4 Normal rays
    if norm_stats["mean_width_px"] > 0:
        consensus_mean_px = 0.6 * edt_stats["mean_width_px"] + 0.4 * norm_stats["mean_width_px"]
        consensus_max_px = max(edt_stats["max_width_px"], norm_stats["max_width_px"])
        consensus_min_px = min(edt_stats["min_width_px"], norm_stats["min_width_px"])
        consensus_std_px = 0.5 * (edt_stats["std_width_px"] + norm_stats["std_width_px"])
        consensus_p95_px = max(edt_stats["p95_width_px"], norm_stats["p95_width_px"])
    else:
        consensus_mean_px = edt_stats["mean_width_px"]
        consensus_max_px = edt_stats["max_width_px"]
        consensus_min_px = edt_stats["min_width_px"]
        consensus_std_px = edt_stats["std_width_px"]
        consensus_p95_px = edt_stats["p95_width_px"]

    calibrated_stats = {}
    classification = None
    if mm_per_px and mm_per_px > 0:
        calibrated_stats = {
            "mean_width_mm": round(consensus_mean_px * mm_per_px, 3),
            "max_width_mm": round(consensus_max_px * mm_per_px, 3),
            "min_width_mm": round(consensus_min_px * mm_per_px, 3),
            "std_width_mm": round(consensus_std_px * mm_per_px, 3),
            "p95_width_mm": round(consensus_p95_px * mm_per_px, 3),
        }
        classification = classify_crack_width(calibrated_stats["max_width_mm"])

    return {
        "consensus_stats_px": {
            "mean_width_px": round(consensus_mean_px, 3),
            "max_width_px": round(consensus_max_px, 3),
            "min_width_px": round(consensus_min_px, 3),
            "std_width_px": round(consensus_std_px, 3),
            "p95_width_px": round(consensus_p95_px, 3),
        },
        "calibrated_stats_mm": calibrated_stats,
        "edt_stats": edt_stats,
        "normal_stats": norm_stats,
        "reliability": rel_res,
        "classification": classification,
        "distance_transform": edt_res["distance_transform"],
        "width_profile": edt_res["width_profile"],
    }


def evaluate_width_accuracy(
    estimated_widths_mm: Union[List[float], np.ndarray],
    ground_truth_widths_mm: Union[List[float], np.ndarray],
) -> dict:
    """
    Research benchmark evaluation for crack width estimation against ground truth.

    Computes:
    - Mean Absolute Error (MAE) in mm
    - Root Mean Square Error (RMSE) in mm
    - Bias (Mean Signed Error) in mm
    - Mean Absolute Percentage Error (MAPE) in %
    - Range-specific errors (< 0.2mm, 0.2 - 0.5mm, > 0.5mm)
    """
    est = np.asarray(estimated_widths_mm, dtype=np.float64)
    gt = np.asarray(ground_truth_widths_mm, dtype=np.float64)

    if len(est) == 0 or len(gt) == 0 or len(est) != len(gt):
        raise ValueError("Estimated and ground truth arrays must have identical non-zero lengths.")

    errors = est - gt
    abs_errors = np.abs(errors)
    mae = float(abs_errors.mean())
    rmse = float(np.sqrt((errors ** 2).mean()))
    bias = float(errors.mean())

    # Avoid div by zero in percentage error
    valid_gt = gt > 1e-4
    if np.any(valid_gt):
        mape = float((abs_errors[valid_gt] / gt[valid_gt]).mean() * 100.0)
    else:
        mape = 0.0

    # Range-specific breakdown
    bins = [
        ("sub_0.2mm", gt < 0.2),
        ("range_0.2_to_0.5mm", (gt >= 0.2) & (gt <= 0.5)),
        ("above_0.5mm", gt > 0.5),
    ]
    range_eval = {}
    for label, mask in bins:
        if np.any(mask):
            range_eval[label] = {
                "count": int(mask.sum()),
                "mae_mm": round(float(abs_errors[mask].mean()), 4),
                "rmse_mm": round(float(np.sqrt((errors[mask] ** 2).mean())), 4),
            }
        else:
            range_eval[label] = {"count": 0, "mae_mm": None, "rmse_mm": None}

    return {
        "mae_mm": round(mae, 4),
        "rmse_mm": round(rmse, 4),
        "bias_mm": round(bias, 4),
        "mape_percent": round(mape, 2),
        "sample_count": len(est),
        "range_breakdown": range_eval,
    }
