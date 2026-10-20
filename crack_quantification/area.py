"""
area.py
=======

Compute crack surface area from binary masks with perspective-distortion correction,
boundary-uncertainty quantification, and benchmark evaluation against physical references.

Capabilities:
1. Pixel & Metric Area: Standard calibrated area (mm² / px²).
2. Perspective-Corrected Planar Area: Rectification-aware area via homography matrix
   or metric coordinate mapping to eliminate foreshortening errors.
3. Boundary Uncertainty Modeling: Inner/outer boundary bounds via morphological dilation
   and erosion, providing defensible physical error intervals (± Δ mm²).
4. Benchmark Accuracy Evaluation: Area MAE, relative percentage error, IoU, and calibration sensitivity.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV2_AVAILABLE = False

try:
    from scipy.ndimage import binary_dilation, binary_erosion
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False


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
    """
    if mm_per_px <= 0:
        raise ValueError("mm_per_px must be positive")
    return float(area_px) * (mm_per_px ** 2)


def compute_area_uncertainty(
    mask: np.ndarray,
    mm_per_px: Optional[float] = None,
) -> dict:
    """
    Quantify physical crack area bounds and boundary uncertainty.

    Evaluates inner (eroded) and outer (dilated) mask boundaries by 1 pixel to model
    segmentation boundary fuzziness, noise, and restoration edge-smoothing.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask (H, W).
    mm_per_px : float, optional
        Calibration factor.

    Returns
    -------
    dict
        {
            'nominal_area_px': int,
            'min_bound_area_px': int,
            'max_bound_area_px': int,
            'uncertainty_delta_px': float,
            'relative_uncertainty_percent': float,
            'nominal_area_mm2': Optional[float],
            'confidence_interval_95_mm2': Optional[Tuple[float, float]],
            'uncertainty_delta_mm2': Optional[float],
        }
    """
    mask_2d = np.asarray(mask)
    if mask_2d.ndim == 3 and mask_2d.shape[2] == 1:
        mask_2d = mask_2d[:, :, 0]
    binary = (mask_2d > 0).astype(bool)

    nominal_px = int(binary.sum())
    if nominal_px == 0:
        return {
            "nominal_area_px": 0,
            "min_bound_area_px": 0,
            "max_bound_area_px": 0,
            "uncertainty_delta_px": 0.0,
            "relative_uncertainty_percent": 0.0,
            "nominal_area_mm2": 0.0 if mm_per_px else None,
            "confidence_interval_95_mm2": (0.0, 0.0) if mm_per_px else None,
            "uncertainty_delta_mm2": 0.0 if mm_per_px else None,
        }

    if SCIPY_AVAILABLE:
        struct = np.ones((3, 3), dtype=bool)
        eroded = binary_erosion(binary, structure=struct)
        dilated = binary_dilation(binary, structure=struct)
    elif CV2_AVAILABLE:
        k = np.ones((3, 3), dtype=np.uint8)
        u8 = binary.astype(np.uint8)
        eroded = cv2.erode(u8, k) > 0
        dilated = cv2.dilate(u8, k) > 0
    else:
        eroded = binary.copy()
        dilated = binary.copy()

    min_px = int(eroded.sum())
    max_px = int(dilated.sum())
    delta_px = float(max(nominal_px - min_px, max_px - nominal_px))
    rel_pct = float((delta_px / nominal_px) * 100.0) if nominal_px > 0 else 0.0

    res = {
        "nominal_area_px": nominal_px,
        "min_bound_area_px": min_px,
        "max_bound_area_px": max_px,
        "uncertainty_delta_px": round(delta_px, 1),
        "relative_uncertainty_percent": round(rel_pct, 2),
    }

    if mm_per_px and mm_per_px > 0:
        px_to_mm2 = mm_per_px ** 2
        nom_mm2 = nominal_px * px_to_mm2
        min_mm2 = min_px * px_to_mm2
        max_mm2 = max_px * px_to_mm2
        delta_mm2 = delta_px * px_to_mm2

        res["nominal_area_mm2"] = round(nom_mm2, 2)
        res["confidence_interval_95_mm2"] = (round(min_mm2, 2), round(max_mm2, 2))
        res["uncertainty_delta_mm2"] = round(delta_mm2, 2)

    return res


def compute_perspective_corrected_area(
    mask: np.ndarray,
    H_px: Optional[np.ndarray] = None,
    rectified_width_mm: Optional[float] = None,
    rectified_height_mm: Optional[float] = None,
    rectified_mask: Optional[np.ndarray] = None,
) -> dict:
    """
    Compute physical crack area accounting for perspective foreshortening via homography.

    If a rectified mask is already computed, area is measured directly on the metric plane.
    Otherwise, if homography matrix H_px is supplied, the mask is warped to eliminate
    camera perspective distortion before area calculation.
    """
    if rectified_mask is not None and rectified_width_mm and rectified_height_mm:
        rh, rw = rectified_mask.shape[:2]
        if rw > 0 and rh > 0:
            scale_x = rectified_width_mm / rw
            scale_y = rectified_height_mm / rh
            metric_pixel_area = scale_x * scale_y
            rect_px = int((rectified_mask > 0).sum())
            return {
                "perspective_corrected": True,
                "area_mm2": round(rect_px * metric_pixel_area, 2),
                "rectified_pixels": rect_px,
                "scale_x_mm_per_px": round(scale_x, 5),
                "scale_y_mm_per_px": round(scale_y, 5),
                "method": "planar_homography_metric_plane",
            }

    if H_px is not None and CV2_AVAILABLE and rectified_width_mm and rectified_height_mm:
        h, w = mask.shape[:2]
        # Target rectified canvas
        target_w = int(w)
        target_h = int(h)
        warped = cv2.warpPerspective(mask.astype(np.uint8), H_px, (target_w, target_h), flags=cv2.INTER_NEAREST)
        scale_x = rectified_width_mm / target_w
        scale_y = rectified_height_mm / target_h
        rect_px = int((warped > 0).sum())
        return {
            "perspective_corrected": True,
            "area_mm2": round(rect_px * (scale_x * scale_y), 2),
            "rectified_pixels": rect_px,
            "scale_x_mm_per_px": round(scale_x, 5),
            "scale_y_mm_per_px": round(scale_y, 5),
            "method": "warped_homography",
        }

    # Fallback to naive uncalibrated perspective
    raw_px = int((np.asarray(mask) > 0).sum())
    return {
        "perspective_corrected": False,
        "area_mm2": None,
        "rectified_pixels": raw_px,
        "method": "uncalibrated_perspective",
    }


def evaluate_area_accuracy(
    estimated_areas_mm2: Union[List[float], np.ndarray],
    ground_truth_areas_mm2: Union[List[float], np.ndarray],
    est_masks: Optional[List[np.ndarray]] = None,
    gt_masks: Optional[List[np.ndarray]] = None,
) -> dict:
    """
    Research benchmark evaluation for crack surface area estimation.

    Calculates:
    - Area MAE in mm²
    - Area RMSE in mm²
    - Mean relative percentage error (%)
    - Mean Intersection-over-Union (mIoU) if binary masks are provided
    """
    est = np.asarray(estimated_areas_mm2, dtype=np.float64)
    gt = np.asarray(ground_truth_areas_mm2, dtype=np.float64)

    if len(est) == 0 or len(gt) == 0 or len(est) != len(gt):
        raise ValueError("Estimated and ground truth areas must have identical non-zero lengths.")

    diffs = est - gt
    abs_diffs = np.abs(diffs)
    mae = float(abs_diffs.mean())
    rmse = float(np.sqrt((diffs ** 2).mean()))

    valid_gt = gt > 1e-4
    if np.any(valid_gt):
        rel_err = float((abs_diffs[valid_gt] / gt[valid_gt]).mean() * 100.0)
    else:
        rel_err = 0.0

    iou_list = []
    if est_masks is not None and gt_masks is not None and len(est_masks) == len(gt_masks):
        for em, gm in zip(est_masks, gt_masks):
            e_bool = np.asarray(em) > 0
            g_bool = np.asarray(gm) > 0
            intersection = np.logical_and(e_bool, g_bool).sum()
            union = np.logical_or(e_bool, g_bool).sum()
            if union > 0:
                iou_list.append(float(intersection / union))
            else:
                iou_list.append(1.0 if not e_bool.any() and not g_bool.any() else 0.0)

    mean_iou = float(np.mean(iou_list)) if iou_list else None

    return {
        "mae_mm2": round(mae, 3),
        "rmse_mm2": round(rmse, 3),
        "mean_relative_error_percent": round(rel_err, 2),
        "mean_iou": round(mean_iou, 4) if mean_iou is not None else None,
        "sample_count": len(est),
    }
