"""
temporal.py
===========

Uncertainty-aware temporal crack progression analysis and monitoring across
multi-timestamp inspections.

Research Capabilities (Candidate D):
1. Uncertainty-Aware Change Detection: Distinguishes genuine physical crack growth
   from apparent changes caused by camera angle, lighting, moisture, or restoration smoothing.
   Crack expansion is only flagged when delta exceeds measurement uncertainty:
       Delta > k * sigma_measurement  (k = 2.0 for 95% confidence).
2. Multi-Metric Rate of Deterioration:
   - Delta Length (mm, % and mm/year)
   - Delta Mean / Max Width (mm, % and mm/year)
   - Delta Surface Area (mm², % and mm²/year)
   - Delta Volume (mm³, % and mm³/year)
3. Structural Deterioration Warning Levels:
   STABLE, MINOR_PROGRESSION, ACTIVE_GROWTH, CRITICAL_ACCELERATION.
4. Spatial Change Segmentation:
   Categorizes pixels into Stable Core, Newly Propagated Tip, Widened Boundary, and Transient Noise.
5. Research Benchmark Evaluation: Growth-rate MAE, false growth alerts, and detection precision.
"""

from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV2_AVAILABLE = False


def _parse_timestamp(ts: Union[str, datetime, float, int]) -> datetime:
    """Parse various timestamp representations to standard datetime."""
    if isinstance(ts, datetime):
        return ts
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    if isinstance(ts, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y%m%d"):
            try:
                return datetime.strptime(ts, fmt)
            except ValueError:
                pass
    # Fallback to current time if unparseable
    return datetime.now()


def compute_spatial_progression_map(
    baseline_mask: np.ndarray,
    current_mask: np.ndarray,
    H_registration: Optional[np.ndarray] = None,
) -> dict:
    """
    Segment spatial crack progression between two aligned inspection masks.

    Categorizes pixels into:
    - 0: Healthy substrate
    - 1: Stable Core Crack (present in both inspections)
    - 2: Newly Propagated Crack (new crack extensions / tips)
    - 3: Widened Boundary (lateral expansion)
    - 4: Apparent Reduction / Inactive (present in baseline, missing in current due to restoration or noise)

    Parameters
    ----------
    baseline_mask : np.ndarray
        Binary mask at T0.
    current_mask : np.ndarray
        Binary mask at T1.
    H_registration : np.ndarray, optional
        Homography matrix aligning current image/mask to baseline coordinates.

    Returns
    -------
    dict
        {
            'spatial_map': np.ndarray,
            'stable_pixels': int,
            'propagated_pixels': int,
            'widened_pixels': int,
            'apparent_reduction_pixels': int,
            'growth_area_ratio': float,
        }
    """
    b_mask = (np.asarray(baseline_mask) > 0).astype(np.uint8)
    c_mask = (np.asarray(current_mask) > 0).astype(np.uint8)

    # Warp current to baseline frame if registration matrix is given
    if H_registration is not None and CV2_AVAILABLE:
        h, w = b_mask.shape[:2]
        c_aligned = cv2.warpPerspective(c_mask, H_registration, (w, h), flags=cv2.INTER_NEAREST)
    else:
        if b_mask.shape != c_mask.shape and CV2_AVAILABLE:
            c_aligned = cv2.resize(c_mask, (b_mask.shape[1], b_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            c_aligned = c_mask

    stable = (b_mask == 1) & (c_aligned == 1)
    new_crack = (b_mask == 0) & (c_aligned == 1)
    reduction = (b_mask == 1) & (c_aligned == 0)

    # Distinguish boundary widening from tip propagation via morphological dilation of baseline
    if CV2_AVAILABLE:
        k = np.ones((3, 3), np.uint8)
        dilated_baseline = cv2.dilate(b_mask, k) > 0
        widened = new_crack & dilated_baseline
        propagated_tips = new_crack & (~dilated_baseline)
    else:
        widened = new_crack
        propagated_tips = np.zeros_like(new_crack)

    spatial_map = np.zeros_like(b_mask, dtype=np.uint8)
    spatial_map[stable] = 1
    spatial_map[propagated_tips] = 2
    spatial_map[widened] = 3
    spatial_map[reduction] = 4

    b_total = int(b_mask.sum())
    new_total = int(new_crack.sum())
    ratio = float(new_total / b_total) if b_total > 0 else 0.0

    return {
        "spatial_map": spatial_map,
        "stable_pixels": int(stable.sum()),
        "propagated_pixels": int(propagated_tips.sum()),
        "widened_pixels": int(widened.sum()),
        "apparent_reduction_pixels": int(reduction.sum()),
        "growth_area_ratio": round(ratio, 4),
    }


def analyze_temporal_progression(
    baseline_inspection: dict,
    current_inspection: dict,
    confidence_k: float = 2.0,
) -> dict:
    """
    Perform uncertainty-aware temporal crack progression analysis.

    Parameters
    ----------
    baseline_inspection : dict
        {
            'timestamp': str or datetime,
            'length_mm': float,
            'mean_width_mm': float,
            'max_width_mm': float,
            'area_mm2': float,
            'volume_mm3': Optional[float],
            'mask': Optional[np.ndarray],
            'scale_mm_per_px': Optional[float],
            'sigma_length_mm': Optional[float],  # standard measurement uncertainty
            'sigma_width_mm': Optional[float],
            'sigma_area_mm2': Optional[float],
        }
    current_inspection : dict
        Same keys as baseline_inspection for current follow-up observation.
    confidence_k : float
        Coverage factor for uncertainty thresholding (default 2.0 ~ 95% confidence).

    Returns
    -------
    dict
        Full temporal deterioration metrics, growth rates, uncertainty significance tests,
        and structural alert classification.
    """
    # 1. Parse timestamps and elapsed duration
    t0 = _parse_timestamp(baseline_inspection.get("timestamp", "2025-01-01"))
    t1 = _parse_timestamp(current_inspection.get("timestamp", "2026-01-01"))
    elapsed_days = max(1.0, float((t1 - t0).total_seconds() / 86400.0))
    elapsed_years = max(elapsed_days / 365.25, 0.0027)  # minimum 1 day

    # 2. Extract baseline metrics
    l0 = float(baseline_inspection.get("length_mm") or 0.0)
    w_mean0 = float(baseline_inspection.get("mean_width_mm") or 0.0)
    w_max0 = float(baseline_inspection.get("max_width_mm") or 0.0)
    a0 = float(baseline_inspection.get("area_mm2") or 0.0)
    v0 = baseline_inspection.get("volume_mm3")
    v0 = float(v0) if v0 is not None else None

    # 3. Extract current metrics
    l1 = float(current_inspection.get("length_mm") or 0.0)
    w_mean1 = float(current_inspection.get("mean_width_mm") or 0.0)
    w_max1 = float(current_inspection.get("max_width_mm") or 0.0)
    a1 = float(current_inspection.get("area_mm2") or 0.0)
    v1 = current_inspection.get("volume_mm3")
    v1 = float(v1) if v1 is not None else None

    # 4. Compute absolute and percentage deltas
    delta_l = l1 - l0
    delta_w_mean = w_mean1 - w_mean0
    delta_w_max = w_max1 - w_max0
    delta_a = a1 - a0
    delta_v = (v1 - v0) if (v0 is not None and v1 is not None) else None

    pct_l = (delta_l / l0 * 100.0) if l0 > 0 else 0.0
    pct_w_mean = (delta_w_mean / w_mean0 * 100.0) if w_mean0 > 0 else 0.0
    pct_w_max = (delta_w_max / w_max0 * 100.0) if w_max0 > 0 else 0.0
    pct_a = (delta_a / a0 * 100.0) if a0 > 0 else 0.0
    pct_v = ((delta_v / v0 * 100.0) if (delta_v is not None and v0 and v0 > 0) else None)

    # 5. Annualized velocity (growth rates)
    rate_l_mm_yr = delta_l / elapsed_years
    rate_w_mm_yr = delta_w_mean / elapsed_years
    rate_a_mm2_yr = delta_a / elapsed_years
    rate_v_mm3_yr = (delta_v / elapsed_years) if delta_v is not None else None

    # 6. Uncertainty-Aware Statistical Significance Test
    # Default conservative measurement standard deviations if not supplied:
    # 2% of length + 0.5mm, 0.03mm width, 3% area
    s_l0 = float(baseline_inspection.get("sigma_length_mm") or max(0.5, l0 * 0.02))
    s_l1 = float(current_inspection.get("sigma_length_mm") or max(0.5, l1 * 0.02))
    sigma_delta_l = float(np.sqrt(s_l0 ** 2 + s_l1 ** 2))

    s_w0 = float(baseline_inspection.get("sigma_width_mm") or max(0.03, w_mean0 * 0.05))
    s_w1 = float(current_inspection.get("sigma_width_mm") or max(0.03, w_mean1 * 0.05))
    sigma_delta_w = float(np.sqrt(s_w0 ** 2 + s_w1 ** 2))

    s_a0 = float(baseline_inspection.get("sigma_area_mm2") or max(2.0, a0 * 0.04))
    s_a1 = float(current_inspection.get("sigma_area_mm2") or max(2.0, a1 * 0.04))
    sigma_delta_a = float(np.sqrt(s_a0 ** 2 + s_a1 ** 2))

    # Genuine growth requires delta to exceed k * sigma (noise threshold)
    thresh_l = confidence_k * sigma_delta_l
    thresh_w = confidence_k * sigma_delta_w
    thresh_a = confidence_k * sigma_delta_a

    length_growth_significant = bool(delta_l > thresh_l)
    width_growth_significant = bool(delta_w_mean > thresh_w or delta_w_max > (confidence_k * sigma_delta_w))
    area_growth_significant = bool(delta_a > thresh_a)

    is_genuine_growth = bool(length_growth_significant or width_growth_significant or area_growth_significant)

    # 7. Classification of Deterioration Alert Level
    if not is_genuine_growth and delta_l <= thresh_l and delta_w_max <= thresh_w:
        alert_level = "STABLE"
        alert_reason = (
            f"Observed changes (ΔL={delta_l:.2f}mm, ΔW={delta_w_mean:.2f}mm) are within the measurement "
            f"uncertainty margin (±{thresh_l:.2f}mm, ±{thresh_w:.2f}mm). Likely environmental/camera noise."
        )
    elif rate_l_mm_yr > 15.0 or rate_w_mm_yr > 0.15 or w_max1 >= 0.30 and delta_w_max > 0.05:
        alert_level = "CRITICAL_ACCELERATION"
        alert_reason = (
            f"Rapid crack expansion (Length rate={rate_l_mm_yr:.1f}mm/yr, Width={w_max1:.2f}mm crossing severe threshold). "
            f"Immediate structural engineering intervention required."
        )
    elif rate_l_mm_yr > 3.0 or rate_w_mm_yr > 0.04:
        alert_level = "ACTIVE_GROWTH"
        alert_reason = (
            f"Statistically genuine crack extension exceeding tolerance (Rate={rate_l_mm_yr:.1f}mm/yr, ΔA={delta_a:.1f}mm²). "
            f"Scheduled maintenance and sealing advised."
        )
    else:
        alert_level = "MINOR_PROGRESSION"
        alert_reason = (
            f"Minor physical settling observed (Rate={rate_l_mm_yr:.1f}mm/yr). Monitor on next periodic inspection cycle."
        )

    # 8. Spatial mapping if masks are provided
    spatial_res = None
    b_mask = baseline_inspection.get("mask")
    c_mask = current_inspection.get("mask")
    if b_mask is not None and c_mask is not None:
        spatial_res = compute_spatial_progression_map(b_mask, c_mask)

    return {
        "status": "success",
        "alert_level": alert_level,
        "is_genuine_growth": is_genuine_growth,
        "alert_reason": alert_reason,
        "elapsed": {
            "days": round(elapsed_days, 1),
            "years": round(elapsed_years, 3),
            "baseline_date": t0.strftime("%Y-%m-%d"),
            "current_date": t1.strftime("%Y-%m-%d"),
        },
        "deltas": {
            "delta_length_mm": round(delta_l, 2),
            "percent_length_change": round(pct_l, 2),
            "delta_mean_width_mm": round(delta_w_mean, 3),
            "percent_mean_width_change": round(pct_w_mean, 2),
            "delta_max_width_mm": round(delta_w_max, 3),
            "percent_max_width_change": round(pct_w_max, 2),
            "delta_area_mm2": round(delta_a, 2),
            "percent_area_change": round(pct_a, 2),
            "delta_volume_mm3": round(delta_v, 2) if delta_v is not None else None,
            "percent_volume_change": round(pct_v, 2) if pct_v is not None else None,
        },
        "annualized_growth_rates": {
            "length_rate_mm_per_year": round(rate_l_mm_yr, 2),
            "mean_width_rate_mm_per_year": round(rate_w_mm_yr, 3),
            "area_rate_mm2_per_year": round(rate_a_mm2_yr, 2),
            "volume_rate_mm3_per_year": round(rate_v_mm3_yr, 2) if rate_v_mm3_yr is not None else None,
        },
        "uncertainty_evaluation": {
            "confidence_coverage_k": confidence_k,
            "sigma_delta_length_mm": round(sigma_delta_l, 3),
            "noise_threshold_length_mm": round(thresh_l, 3),
            "is_length_statistically_significant": length_growth_significant,
            "sigma_delta_width_mm": round(sigma_delta_w, 3),
            "noise_threshold_width_mm": round(thresh_w, 3),
            "is_width_statistically_significant": width_growth_significant,
            "sigma_delta_area_mm2": round(sigma_delta_a, 3),
            "noise_threshold_area_mm2": round(thresh_a, 3),
            "is_area_statistically_significant": area_growth_significant,
        },
        "spatial_summary": {
            "stable_pixels": spatial_res["stable_pixels"] if spatial_res else None,
            "propagated_pixels": spatial_res["propagated_pixels"] if spatial_res else None,
            "widened_pixels": spatial_res["widened_pixels"] if spatial_res else None,
            "growth_area_ratio": spatial_res["growth_area_ratio"] if spatial_res else None,
        } if spatial_res else None,
    }


def evaluate_temporal_progression_accuracy(
    predicted_progressions: List[dict],
    ground_truth_progressions: List[dict],
) -> dict:
    """
    Research benchmark evaluation for temporal crack growth estimation.

    Calculates:
    - Growth Rate MAE (mm/year)
    - False Growth Alert Rate (%)
    - Genuine Growth Detection Recall (%)
    """
    if len(predicted_progressions) != len(ground_truth_progressions) or len(predicted_progressions) == 0:
        raise ValueError("Predicted and ground truth progression lists must have matching non-zero lengths.")

    rate_diffs = []
    false_alerts = 0
    stable_count = 0
    true_positives = 0
    actual_positives = 0

    for pred, gt in zip(predicted_progressions, ground_truth_progressions):
        p_rate = pred["annualized_growth_rates"]["length_rate_mm_per_year"]
        g_rate = gt["length_rate_mm_per_year"]
        rate_diffs.append(abs(p_rate - g_rate))

        is_gt_growth = gt.get("is_genuine_growth", False)
        is_pred_growth = pred.get("is_genuine_growth", False)

        if is_gt_growth:
            actual_positives += 1
            if is_pred_growth:
                true_positives += 1
        else:
            stable_count += 1
            if is_pred_growth:
                false_alerts += 1

    rate_mae = float(np.mean(rate_diffs))
    false_alert_rate = float((false_alerts / stable_count) * 100.0) if stable_count > 0 else 0.0
    recall = float((true_positives / actual_positives) * 100.0) if actual_positives > 0 else 100.0

    return {
        "growth_rate_mae_mm_per_year": round(rate_mae, 3),
        "false_growth_alert_rate_percent": round(false_alert_rate, 2),
        "genuine_growth_recall_percent": round(recall, 2),
        "total_evaluations": len(predicted_progressions),
        "stable_specimens_evaluated": stable_count,
        "growing_specimens_evaluated": actual_positives,
    }
