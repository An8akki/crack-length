"""
run_quantification.py
=====================

CLI / demo runner for the crack quantification package.

Accepts a binary crack MASK image (png/jpg) and an optional calibration
(mm per pixel). It runs the full pipeline and prints the final report.

Usage:
    python crack_quantification/run_quantification.py --mask <path> [--mm-per-px 0.277] [--image-id "Image 001"]

If --mm-per-px is omitted, results are reported in pixel units and severity is
marked N/A.

Mask images may be saved as 0/1 or 0/255 (any nonzero = crack).
"""

import argparse
import os

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from .quantify import quantify_crack, format_report


def load_mask(path: str) -> np.ndarray:
    """Load a binary mask image and return a uint8 single-channel array."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mask file not found: {path}")

    if CV2_AVAILABLE:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Failed to read mask image: {path}")
        return (img > 0).astype(np.uint8)

    # Fallback using PIL
    from PIL import Image
    pil = Image.open(path).convert("L")
    arr = np.array(pil)
    return (arr > 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Crack quantification from a binary mask")
    parser.add_argument("--mask", required=True, help="Path to binary mask image")
    parser.add_argument("--mm-per-px", type=float, default=None,
                        help="Calibration scale (mm per pixel). Optional.")
    parser.add_argument("--image-id", type=str, default="Image 001",
                        help="Label for the image in the report.")
    args = parser.parse_args()

    mask = load_mask(args.mask)
    print(f"Loaded mask: {mask.shape}, foreground pixels: {int(mask.sum())}")

    report = quantify_crack(mask, mm_per_px=args.mm_per_px)
    print()
    print(format_report(report, image_id=args.image_id))
    print()


if __name__ == "__main__":
    main()
