"""
integrate_pipeline.py
=====================

END-TO-END INTEGRATION
----------------------

This script connects the EXISTING segmentation detector to the crack
quantification package, so you can go from a raw image to a full physical
crack report in one run.

    Raw Image
        |
        v
   [segmentation model]   (YOLO seg, once only -> binary mask)
        |
        v
   [scale recovery]       (mm/pixel, automatic or given)
        |
        v
   [quantify_crack]       (skeleton -> length / width / area / severity)
        |
        v
   Final Report (+ XAI explanation of the px->mm conversion)

Usage:
    python -m crack_quantification.integrate_pipeline --image <path>
        [--mm-per-px 0.277] [--image-id "Image 001"] [--conf 0.3]

If --mm-per-px is omitted, the script tries automatic scale recovery from
camera intrinsics (pass --focal-mm, --sensor-mm, --distance-mm). Otherwise it
falls back to a default scale and marks it as assumed.
"""

import argparse
import os

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# --------------------------------------------------------------------------- #
# 1. Get the binary crack mask from the existing segmentation model
# --------------------------------------------------------------------------- #
def get_mask_from_segmentation(image_rgb: np.ndarray, conf: float, model=None):
    """
    Run the YOLO segmentation model once and return a binary crack mask.

    This is the ONLY place the detector is touched. The returned mask (0/1)
    drives everything downstream.
    """
    from ultralytics import YOLO

    if model is None:
        model = YOLO("./segmentation_model/weights/best.pt")

    results = model.predict(source=image_rgb, conf=conf, save=False)
    if not results or len(results) == 0 or results[0].masks is None:
        raise RuntimeError("No segmentation masks produced by the model.")

    # results[0].masks.data : (N, H, W) float 0/1 masks at model resolution
    masks = results[0].masks.data.cpu().numpy()

    h, w = image_rgb.shape[:2]
    # Combine all masks into a single binary crack mask, resized to original.
    combined = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        resized = cv2.resize(
            m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
        )
        combined[resized > 0] = 1

    return (combined > 0).astype(np.uint8)


# --------------------------------------------------------------------------- #
# 2. XAI: explain how mm per pixel is derived
# --------------------------------------------------------------------------- #
def explain_scale(method: str, inputs: dict) -> str:
    """
    Produce a short, human-readable ("explainable AI") explanation of how the
    px->mm scale was computed, so a civil engineer can trust the number.
    """
    if method in ("given", "assumed"):
        return (
            f"Scale: {inputs['mm_per_px']:.4f} mm/px "
            f"({method} by user / ruler). 1 pixel = that many mm on the surface."
        )

    if method == "aruco":
        return (
            f"Scale: {inputs['mm_per_px']:.4f} mm/px. A physical ArUco marker "
            f"of known size {inputs['marker_mm']:.0f} mm spans "
            f"{inputs['marker_px']:.1f} px, so mm/px = marker_mm / marker_px "
            f"= {inputs['marker_mm']:.1f} / {inputs['marker_px']:.1f}."
        )

    if method == "intrinsics":
        # mm_per_px = (sensor_width_mm / image_width_px) * (distance / focal)
        pixel_size = inputs["sensor_width_mm"] / inputs["image_width_px"]
        mag = inputs["distance_mm"] / inputs["focal_mm"]
        return (
            f"Scale: {inputs['mm_per_px']:.4f} mm/px from camera geometry.\n"
            "  - sensor pixel size = sensor_width / image_width = "
            f"{inputs['sensor_width_mm']:.2f}mm / {inputs['image_width_px']}px "
            f"= {pixel_size:.5f} mm/px\n"
            "  - magnification = distance / focal_length = "
            f"{inputs['distance_mm']:.0f}mm / {inputs['focal_mm']:.1f}mm = "
            f"{mag:.2f}x\n"
            "  - mm/px = pixel_size x magnification = "
            f"{pixel_size:.5f} x {mag:.2f}"
        )

    return f"Scale: {inputs.get('mm_per_px', 0):.4f} mm/px (assumed default)."


def scales_to_mm(mm_per_px, pix_length, pix_width, pix_area):
    """Convert pixel quantities to mm using the recovered scale."""
    return {
        "length_mm": pix_length * mm_per_px,
        "width_mm": pix_width * mm_per_px,
        "area_mm2": pix_area * (mm_per_px ** 2),
    }


# --------------------------------------------------------------------------- #
# 3. Main pipeline
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Integration: segmentation -> scale -> quantification"
    )
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--image-id", default="Image 001")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--mm-per-px", type=float, default=None)

    # Scale-recovery options (camera intrinsics method)
    parser.add_argument("--focal-mm", type=float, default=None)
    parser.add_argument("--sensor-mm", type=float, default=None)
    parser.add_argument("--distance-mm", type=float, default=None)

    # 3D Depth estimation option
    parser.add_argument("--depth", default=None, help="Optional path to 8-bit depth map for relative recess estimation")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    # ---- A. Load image (BGR internally for OpenCV) ----
    if CV2_AVAILABLE:
        image_bgr = cv2.imread(args.image)
        if image_bgr is None:
            raise ValueError(f"Could not read image: {args.image}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    else:
        from PIL import Image
        image_rgb = np.array(Image.open(args.image).convert("RGB"))

    # ---- B. Segmentation model -> binary mask (detector touched once) ----
    print("Running segmentation model to obtain binary crack mask...")
    mask = get_mask_from_segmentation(image_rgb, args.conf)
    print(f"  Mask shape {mask.shape}, foreground pixels = {int(mask.sum())}")
    if int(mask.sum()) == 0:
        print("  No crack pixels detected. Nothing to quantify.")
        return

    # ---- C. Scale recovery (mm per pixel) ----
    # If the user supplies --mm-per-px, use it directly.
    # Otherwise, AUTO-CALIBRATE: read EXIF -> camera DB -> estimate distance.
    mm_per_px = args.mm_per_px
    if mm_per_px is not None:
        method = "given"
        xai_inputs = {"mm_per_px": mm_per_px}
    else:
        from .autocalibrate import auto_calibrate

        # If geometry flags are given, pass them as an explicit distance override
        dist_override = args.distance_mm
        if args.sensor_mm and args.focal_mm and args.distance_mm:
            # We have full intrinsics -> compute directly for exactness
            img_w = image_rgb.shape[1]
            pixel_size = args.sensor_mm / img_w
            mm_per_px = pixel_size * (args.distance_mm / args.focal_mm)
            method = "intrinsics"
            xai_inputs = {
                "mm_per_px": mm_per_px,
                "sensor_width_mm": args.sensor_mm,
                "image_width_px": img_w,
                "focal_mm": args.focal_mm,
                "distance_mm": args.distance_mm,
            }
        else:
            # Automatic: EXIF + camera DB + estimated distance
            ac = auto_calibrate(
                args.image,
                image_rgb=image_rgb,
                distance_mm=dist_override,
            )
            mm_per_px = ac["mm_per_px"]
            method = ac["method"]
            xai_inputs = {"mm_per_px": mm_per_px}
            xai_inputs["_auto"] = ac  # store full auto result

    # ---- D. Quantify the mask (length / width / area / severity) ----
    from .quantify import quantify_crack, format_report

    report = quantify_crack(mask, mm_per_px=mm_per_px, image_rgb=image_rgb)

    # ---- E. Report + XAI line ----
    print("\n" + "=" * 60)
    print(format_report(report, image_id=args.image_id))
    print("=" * 60)

    # Quick manual check: px -> mm using the recovered scale
    mean_w_px = report["width_stats_px"]["mean_width_px"]
    converted = scales_to_mm(
        mm_per_px,
        report["length_px"],
        mean_w_px,
        report["area_px"],
    )
    print("\n[px -> mm check (using recovered mm/px)]")
    print(
        f"  length: {report['length_px']:.1f} px * {mm_per_px:.4f} "
        f"= {converted['length_mm']:.1f} mm"
    )
    print(
        f"  width : {mean_w_px:.2f} px * {mm_per_px:.4f} "
        f"= {converted['width_mm']:.2f} mm"
    )
    print(
        f"  area  : {report['area_px']} px * {mm_per_px ** 2:.5f} "
        f"= {converted['area_mm2']:.1f} mm^2"
    )

    print("\n[Explainable AI - how mm/px was obtained]")
    print(explain_scale(method, xai_inputs))

    # ---- F. Optional 3D Relative Recess Estimation ----
    if args.depth:
        print("\n" + "=" * 60)
        print("          3D RELATIVE RECESS & DEPTH ESTIMATION")
        print("=" * 60)
        from .recess_depth import estimate_crack_recess
        try:
            depth_res = estimate_crack_recess(image_rgb=image_rgb, mask=mask, depth_map=args.depth)
            print(f"Predicted Relative Depth: {depth_res.predicted_relative_depth:.2f} sensor-depth units")
            print(f"Prediction Interval 90%:  [{depth_res.interval_90[0]:.2f}, {depth_res.interval_90[1]:.2f}] sensor units")
            print(f"Prediction Interval 80%:  [{depth_res.interval_80[0]:.2f}, {depth_res.interval_80[1]:.2f}] sensor units")
            print(f"Reliability Tier:         {depth_res.reliability}")
            print("Reliability Assessment:")
            for reason in depth_res.reliability_reasons:
                print(f"  - {reason}")
            print("-" * 60)
            print("Physical Crack Depth:     NOT AVAILABLE (Software-Only Scope)")
            print(f"System Calibration State: {depth_res.calibration_state}")
        except Exception as e:
            print(f"Depth estimation failed: {e}")
        print("=" * 60)


if __name__ == "__main__":
    main()
