"""
test_advanced_features.py
=========================

Unit test suite verifying all 4 advanced candidate research modules in crack_quantification:
- Candidate A: Crack-Width Estimation (Normal rays, EDT, boundary-reliability under restoration, ACI 224R classification, accuracy benchmarks)
- Candidate B: Crack-Volume Estimation (Physical surface recess integration, depth uncertainty bounds, volume benchmarks)
- Candidate C: Crack-Area Quantification (Perspective-corrected area, boundary uncertainty bounds ±Δ mm², accuracy benchmarks)
- Candidate D: Temporal Crack Progression (Uncertainty-aware change detection, growth velocity mm/year, spatial progression maps, alert levels)
"""

import numpy as np
from datetime import datetime

from crack_quantification.skeleton import skeletonize_mask
from crack_quantification.width import (
    compute_width_profile,
    compute_normal_width_profile,
    compute_advanced_width_profile,
    assess_width_reliability,
    classify_crack_width,
    evaluate_width_accuracy,
)
from crack_quantification.area import (
    compute_area,
    compute_area_uncertainty,
    compute_perspective_corrected_area,
    evaluate_area_accuracy,
)
from crack_quantification.volume import (
    compute_crack_volume,
    compute_volume_uncertainty,
    evaluate_volume_accuracy,
    estimate_surface_baseline,
)
from crack_quantification.temporal import (
    analyze_temporal_progression,
    compute_spatial_progression_map,
    evaluate_temporal_progression_accuracy,
)
from crack_quantification.quantify import quantify_crack, format_report


def create_synthetic_crack(size=120, thickness=6):
    """Build a synthetic diagonal crack mask with a known physical thickness."""
    mask = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    band = np.abs(xx - yy) < thickness
    mask[band] = 1
    return mask


def test_candidate_a_width():
    print("[TEST 1/5] Candidate A: Crack-Width Estimation...")
    mask = create_synthetic_crack(size=100, thickness=8)
    skel = skeletonize_mask(mask, method="guo_hall")

    # 1. Normal-ray profile
    norm_res = compute_normal_width_profile(mask, skel)
    assert norm_res["normal_stats"]["mean_width_px"] > 0, "Normal width should be positive"
    assert 5.0 <= norm_res["normal_stats"]["mean_width_px"] <= 11.0, f"Expected width around 8px, got {norm_res['normal_stats']['mean_width_px']}"

    # 2. Advanced width with calibration
    adv_res = compute_advanced_width_profile(mask, skel, mm_per_px=0.2)
    assert adv_res["consensus_stats_px"]["mean_width_px"] > 0
    assert adv_res["calibrated_stats_mm"]["mean_width_mm"] > 0
    assert adv_res["classification"] is not None

    # 3. Boundary reliability assessment
    rel_res = assess_width_reliability(mask, skel, mm_per_px=0.2)
    assert 0.0 <= rel_res["reliability_score"] <= 1.0
    assert rel_res["confidence_interval_95_mm"][0] <= rel_res["confidence_interval_95_mm"][1]

    # 4. ACI Classification
    aci_hairline = classify_crack_width(0.05)
    assert aci_hairline["category"] == "Hairline"
    aci_severe = classify_crack_width(0.45)
    assert aci_severe["category"] == "Severe"

    # 5. Benchmark accuracy evaluation
    est = [0.15, 0.28, 0.55]
    gt = [0.14, 0.30, 0.50]
    acc = evaluate_width_accuracy(est, gt)
    assert acc["mae_mm"] >= 0.0
    assert acc["rmse_mm"] >= 0.0
    assert acc["mape_percent"] >= 0.0
    print("   [OK] Candidate A passed successfully.")


def test_candidate_b_volume():
    print("[TEST 2/5] Candidate B: Crack-Volume Estimation...")
    mask = create_synthetic_crack(size=100, thickness=6)
    # Synthetic depth: surface is at 1000mm, crack recess is at 1004mm (4mm deep)
    depth_map = np.full((100, 100), 1000.0, dtype=np.float32)
    depth_map[mask > 0] = 1004.0  # 4mm crevice depth

    vol_res = compute_crack_volume(mask, depth_map, mm_per_px=0.25)
    assert vol_res["volume_mm3"] > 0, "Volume should be positive"
    assert vol_res["volume_cm3"] > 0
    assert 3.5 <= vol_res["mean_recess_depth_mm"] <= 4.5

    # Volume uncertainty
    unc_res = compute_volume_uncertainty(mask, depth_map, mm_per_px=0.25)
    assert unc_res["nominal_volume_mm3"] > 0
    assert unc_res["volume_confidence_interval_95_mm3"][0] <= unc_res["volume_confidence_interval_95_mm3"][1]

    # Volume accuracy benchmark
    vol_acc = evaluate_volume_accuracy([120.0, 450.0], [125.0, 440.0])
    assert vol_acc["volume_mae_mm3"] >= 0.0
    print("   [OK] Candidate B passed successfully.")


def test_candidate_c_area():
    print("[TEST 3/5] Candidate C: Crack-Area Quantification...")
    mask = create_synthetic_crack(size=100, thickness=6)

    # 1. Base area
    area_res = compute_area(mask)
    assert area_res["area_px"] > 0

    # 2. Boundary uncertainty
    unc_res = compute_area_uncertainty(mask, mm_per_px=0.2)
    assert unc_res["nominal_area_mm2"] > 0
    assert unc_res["min_bound_area_px"] < unc_res["nominal_area_px"] < unc_res["max_bound_area_px"]
    assert unc_res["uncertainty_delta_mm2"] > 0

    # 3. Perspective correction test with identity homography
    H_eye = np.eye(3, dtype=np.float32)
    persp_res = compute_perspective_corrected_area(
        mask, H_px=H_eye, rectified_width_mm=50.0, rectified_height_mm=50.0
    )
    assert persp_res["perspective_corrected"] is True
    assert persp_res["area_mm2"] > 0

    # 4. Area accuracy evaluation
    acc = evaluate_area_accuracy([50.0, 100.0], [52.0, 98.0])
    assert acc["mae_mm2"] >= 0.0
    print("   [OK] Candidate C passed successfully.")


def test_candidate_d_temporal():
    print("[TEST 4/5] Candidate D: Temporal Crack Progression...")
    mask_t0 = create_synthetic_crack(size=100, thickness=4)
    mask_t1 = create_synthetic_crack(size=100, thickness=8)  # Widened crack

    # Baseline inspection
    t0_data = {
        "timestamp": "2025-01-01",
        "length_mm": 50.0,
        "mean_width_mm": 0.18,
        "max_width_mm": 0.22,
        "area_mm2": 80.0,
        "volume_mm3": 120.0,
        "mask": mask_t0,
    }

    # Follow-up inspection 1 year later with growth
    t1_data = {
        "timestamp": "2026-01-01",
        "length_mm": 62.0,       # +12 mm growth
        "mean_width_mm": 0.35,   # +0.17 mm growth (crosses severe threshold)
        "max_width_mm": 0.45,
        "area_mm2": 150.0,
        "volume_mm3": 250.0,
        "mask": mask_t1,
    }

    prog = analyze_temporal_progression(t0_data, t1_data)
    assert prog["is_genuine_growth"] is True, "Should flag genuine growth exceeding noise"
    assert prog["alert_level"] in ("ACTIVE_GROWTH", "CRITICAL_ACCELERATION")
    assert prog["deltas"]["delta_length_mm"] == 12.0
    assert prog["annualized_growth_rates"]["length_rate_mm_per_year"] > 10.0
    assert prog["spatial_summary"] is not None

    # Test noise case (delta within 0.1mm measurement noise)
    t1_noise = dict(t0_data)
    t1_noise["timestamp"] = "2025-06-01"
    t1_noise["length_mm"] = 50.2  # +0.2mm within noise
    t1_noise["mean_width_mm"] = 0.182
    prog_noise = analyze_temporal_progression(t0_data, t1_noise)
    assert prog_noise["is_genuine_growth"] is False, "Small delta within noise should NOT trigger genuine growth alert"
    assert prog_noise["alert_level"] == "STABLE"

    # Benchmark accuracy
    eval_res = evaluate_temporal_progression_accuracy(
        [prog, prog_noise],
        [{"length_rate_mm_per_year": 12.0, "is_genuine_growth": True},
         {"length_rate_mm_per_year": 0.0, "is_genuine_growth": False}]
    )
    assert eval_res["false_growth_alert_rate_percent"] == 0.0
    print("   [OK] Candidate D passed successfully.")


def test_full_pipeline_orchestration():
    print("[TEST 5/5] Full Pipeline Orchestration (quantify_crack)...")
    mask = create_synthetic_crack(size=100, thickness=6)
    depth_map = np.full((100, 100), 1000.0, dtype=np.float32)
    depth_map[mask > 0] = 1003.5

    report = quantify_crack(mask, mm_per_px=0.25, depth_map_mm=depth_map)
    assert report["length_mm"] > 0
    assert report["mean_width_mm"] > 0
    assert report["area_mm2"] > 0
    assert report["volume"] is not None
    assert report["volume"]["volume_mm3"] > 0
    assert report["area_uncertainty"] is not None
    assert report["aci_classification"] is not None

    formatted = format_report(report, image_id="Orchestration Test Specimen")
    assert "Length" in formatted
    assert "Area Bounds" in formatted
    assert "Recess Volume" in formatted
    print("\n--- Unified Engineering Report Preview ---")
    print(formatted)
    print("   [OK] Full orchestration passed successfully.")


if __name__ == "__main__":
    print("==========================================================")
    print("CRACK QUANTIFICATION ADVANCED RESEARCH MODULES TEST SUITE")
    print("==========================================================")
    test_candidate_a_width()
    test_candidate_b_volume()
    test_candidate_c_area()
    test_candidate_d_temporal()
    test_full_pipeline_orchestration()
    print("\n[SUCCESS] ALL ADVANCED RESEARCH MODULE TESTS PASSED (5/5)!")
