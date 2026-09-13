"""
test_quantification.py
======================

Quick self-test of the crack_quantification pipeline using a synthetic crack
mask, so no real model or image is required.

Run:
    python crack_quantification/test_quantification.py
"""

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .quantify import quantify_crack, format_report


def make_synthetic_mask(size=100, thickness=6):
    """
    Build a diagonal crack mask for testing.

    The crack is a slightly thick diagonal band of width `thickness` pixels,
    roughly 45 degrees.
    """
    mask = np.zeros((size, size), dtype=np.uint8)
    yy, xx = np.mgrid[0:size, 0:size]
    # Equation of a diagonal strip: |x - y| < thickness
    band = np.abs(xx - yy) < thickness
    mask[band] = 1
    return mask


def main():
    print("=== Crack Quantification Self-Test ===")
    mask = make_synthetic_mask()
    print(f"Mask shape: {mask.shape}, foreground px: {int(mask.sum())}")

    # Calibration example: 0.2 mm per pixel
    mm_per_px = 0.2

    report = quantify_crack(mask, mm_per_px=mm_per_px)

    print("\n--- Report (with calibration) ---")
    print(format_report(report, image_id="Synthetic Test 001"))

    # Also run without calibration to show pixel-only output
    report_px = quantify_crack(mask, mm_per_px=None)
    print("\n--- Report (pixel units only) ---")
    print(format_report(report_px, image_id="Synthetic Test 002"))

# A few sanity assertions
    stats = report["width_stats_px"]
    assert report["length_px"] > 0, "Length should be positive"
    assert report["area_px"] > 0, "Area should be positive"
    assert stats["max_width_px"] >= stats["min_width_px"], "Width ordering"
    assert report["mean_width_mm"] > 0, "Mean width should be positive"
    assert report["severity"]["severity"] in ("Minor", "Moderate", "Severe")

    print("\n[SUCCESS] All sanity checks passed.")

    # Optional: save visualizations
    if CV2_AVAILABLE:
        from .skeleton import skeletonize_mask
        skel = skeletonize_mask(mask)
        vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        vis[mask > 0] = (0, 0, 255)          # crack in red
        vis[skel > 0] = (0, 255, 0)          # skeleton in green
        out_path = "crack_quantification/synthetic_overlay.png"
        cv2.imwrite(out_path, vis)
        print(f"Saved overlay visualization to {out_path}")


if __name__ == "__main__":
    main()
