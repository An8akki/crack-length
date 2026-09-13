"""
autocalibrate.py
================

FULLY AUTOMATIC SCALE RECOVERY
------------------------------

The user supplies ONLY an image. This module derives `mm_per_px` without asking
for manual focal length / sensor / distance.

Stages (progressive, best-effort):
    Stage 1  read EXIF   -> focal_length_mm, image_width_px, camera model
    Stage 2  camera DB   -> sensor_width_mm  (lookup by camera model)
    Stage 3  depth model -> camera distance   (optional, if a depth map provider
                             is registered; otherwise a sensible default with a
                             confidence flag)

The result is a `Calibration` plus a human-readable (XAI) explanation of how
the scale was obtained and how confident we are.

If neither EXIF nor a camera database match is available, the module reports a
low-confidence "assumed" scale so the pipeline still runs, but clearly flags it.
"""

import os

import numpy as np

try:
    from PIL import Image, ExifTags
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover
    PIL_AVAILABLE = False

from .calibration import Calibration
from .scale_recovery import mm_per_px_from_geometry, intrinsic_from_exif


# --------------------------------------------------------------------------- #
# Camera sensor database (Option 4). Add models as needed.
# {model_substring: {'sensor_width_mm': float, 'default_focal_mm': float}}
# --------------------------------------------------------------------------- #
CAMERA_DB = {
    "iPhone": {"sensor_width_mm": 6.17, "default_focal_mm": 6.86},
    "iPhone 15 Pro": {"sensor_width_mm": 6.17, "default_focal_mm": 6.86},
    "iPhone 14 Pro": {"sensor_width_mm": 6.17, "default_focal_mm": 6.86},
    "iPhone 13 Pro": {"sensor_width_mm": 6.17, "default_focal_mm": 6.86},
    "iPhone 12 Pro": {"sensor_width_mm": 6.17, "default_focal_mm": 6.86},
    # Samsung Galaxy S series (main wide lens, ~1/1.3" sensor)
    "Samsung S24 Ultra": {"sensor_width_mm": 6.3, "default_focal_mm": 6.4},
    "Samsung S23 Ultra": {"sensor_width_mm": 6.3, "default_focal_mm": 6.4},
    "Samsung S22 Ultra": {"sensor_width_mm": 6.3, "default_focal_mm": 6.4},
    "SM-S9": {"sensor_width_mm": 6.3, "default_focal_mm": 6.4},
    # Google Pixel (main wide, ~1/1.3")
    "Pixel 8": {"sensor_width_mm": 6.17, "default_focal_mm": 6.9},
    "Pixel 7": {"sensor_width_mm": 6.17, "default_focal_mm": 6.9},
    "Pixel 6": {"sensor_width_mm": 6.17, "default_focal_mm": 6.9},
    # Canon/Mirrorless crop (APSC)
    "Canon": {"sensor_width_mm": 22.3, "default_focal_mm": 24.0},
    "NIKON": {"sensor_width_mm": 23.5, "default_focal_mm": 24.0},
    "SONY": {"sensor_width_mm": 23.5, "default_focal_mm": 24.0},
    # Full-frame
    "ILCE": {"sensor_width_mm": 36.0, "default_focal_mm": 35.0},
    "EOS R": {"sensor_width_mm": 36.0, "default_focal_mm": 35.0},
}


def _lookup_camera(model: str, focal_length_mm: float):
    """
    Best-effort camera lookup. Returns sensor_width_mm and focal_length_mm.

    If the model is matched, sensor width is taken from the DB. Focal length
    from the DB is used only if EXIF did not provide one.
    """
    if not model:
        return None, focal_length_mm
    model_lower = model.lower()
    for key, info in CAMERA_DB.items():
        if key.lower() in model_lower:
            sensor = info["sensor_width_mm"]
            focal = focal_length_mm or info["default_focal_mm"]
            return sensor, focal
    return None, focal_length_mm


# --------------------------------------------------------------------------- #
# EXIF extraction
# --------------------------------------------------------------------------- #
def extract_exif(image_path: str) -> dict:
    """
    Extract EXIF metadata from an image file using Pillow.

    Returns a dict with keys:
        camera_model, focal_length_mm, focal_35mm, image_width_px,
        image_height_px, sensor_width_mm (if inferable), and the raw exif dict.
    """
    if not PIL_AVAILABLE:
        return {}

    result = {
        "camera_model": None,
        "focal_length_mm": None,
        "focal_35mm": None,
        "image_width_px": None,
        "image_height_px": None,
        "sensor_width_mm": None,
        "raw_exif": {},
    }

    try:
        img = Image.open(image_path)
        result["image_width_px"] = img.width
        result["image_height_px"] = img.height

        exif_raw = img._getexif() or {}
        mapped = {}
        for k, v in exif_raw.items():
            tag = ExifTags.TAGS.get(k, k)
            mapped[tag] = v
        result["raw_exif"] = mapped

        # Camera model
        model = mapped.get("Model") or mapped.get("Make")
        if model:
            make = mapped.get("Make")
            result["camera_model"] = f"{make} {model}".strip()

        # Focal length (rational)
        fl = mapped.get("FocalLength")
        if isinstance(fl, tuple) and len(fl) == 2 and fl[1]:
            result["focal_length_mm"] = fl[0] / fl[1]

        # 35mm equivalent
        fl35 = (
            mapped.get("FocalLengthIn35mmFilm")
            or mapped.get("FocalLengthIn35mmFilmEquivalent")
        )
        if fl35:
            result["focal_35mm"] = float(fl35)

        # Infer sensor width from 35mm equivalent if focal present
        fl_mm = result["focal_length_mm"]
        if fl35 and fl_mm:
            crop = float(fl35) / fl_mm
            if 1.0 <= crop <= 8.0:
                result["sensor_width_mm"] = 36.0 / crop

    except Exception as e:  # pragma: no cover
        print(f"[autocalibrate] EXIF read failed: {e}")

    return result


# --------------------------------------------------------------------------- #
# Depth-based distance estimation hook
# --------------------------------------------------------------------------- #
# Users can register a depth map provider:
#   register_depth_provider(fn)  where fn(image_rgb) -> metric depth map (mm)
_DEPTH_PROVIDER = None


def register_depth_provider(fn):
    """
    Register a callable that returns a metric depth map (mm) for an RGB image.

    Use with a metric depth model (Metric3D, UniDepth, Depth Anything V2 metric)
    to get an automatic camera distance. Until registered, distance is treated
    as unknown and a default is used with low confidence.
    """
    global _DEPTH_PROVIDER
    _DEPTH_PROVIDER = fn


def estimate_distance_mm(image_rgb: np.ndarray):
    """
    Estimate camera-to-surface distance in mm.

    Tries the registered metric-depth provider first. Falls back to a default
    (1000 mm) with a low-confidence flag.
    """
    if _DEPTH_PROVIDER is not None:
        try:
            depth = _DEPTH_PROVIDER(image_rgb)
            depth = np.asarray(depth, dtype=np.float64)
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            # Median of central region = typical wall distance
            h, w = depth.shape
            y0, y1 = int(h * 0.3), int(h * 0.7)
            x0, x1 = int(w * 0.3), int(w * 0.7)
            region = depth[y0:y1, x0:x1]
            valid = region[np.isfinite(region) & (region > 0)]
            if valid.size:
                return float(np.median(valid)), 0.05, "depth_model"
        except Exception as e:  # pragma: no cover
            print(f"[autocalibrate] depth provider failed: {e}")

    # Default fallback (assumed distance). Low confidence.
    return 1000.0, 0.4, "assumed_distance"


# --------------------------------------------------------------------------- #
# Main automatic calibration
# --------------------------------------------------------------------------- #
def auto_calibrate(
    image_path: str,
    image_rgb: np.ndarray = None,
    distance_mm: float = None,
    *,
    default_distance_mm: float = 1000.0,
) -> dict:
    """
    Automatically recover the mm-per-pixel scale for an image.

    Parameters
    ----------
    image_path : str
        Path to the image (used for EXIF).
    image_rgb : np.ndarray, optional
        RGB image (used for depth-based distance estimation).
    distance_mm : float, optional
        If provided, overrides automatic distance estimation.
    default_distance_mm : float
        Fallback distance when none can be estimated.

    Returns
    -------
    dict
        {
            'calibration': Calibration,
            'mm_per_px': float,
            'method': str,
            'confidence': float (0..1),
            'explanation': str (XAI),
            'exif': dict,
        }
    """
    exif = extract_exif(image_path) if image_path else {}

    focal = exif.get("focal_length_mm")
    sensor = exif.get("sensor_width_mm")
    img_w = exif.get("image_width_px")
    camera_model = exif.get("camera_model")

    # ---- Stage 2: camera database lookup for sensor width ----
    db_sensor, db_focal = _lookup_camera(camera_model, focal)
    if sensor is None:
        sensor = db_sensor
    if focal is None:
        focal = db_focal

    # Image width fallback
    if img_w is None and image_rgb is not None:
        img_w = image_rgb.shape[1]

    # ---- Stage 3: distance ----
    if distance_mm is None:
        distance_mm, dist_rel_err, dist_method = estimate_distance_mm(image_rgb)
    else:
        dist_rel_err, dist_method = 0.05, "provided"

    # ---- Build explanation + confidence ----
    if focal and sensor and img_w:
        # Full intrinsics available
        mmpp = mm_per_px_from_geometry(focal, sensor, img_w, distance_mm)
        method = "exif_camera_geometry"

        # Confidence: high if EXIF focal + camera DB sensor; lower if distance
        # was assumed; lower still if sensor was DB-guessed.
        conf = 0.9
        if dist_method == "assumed_distance":
            conf -= 0.3
        if sensor != exif.get("sensor_width_mm") and not exif.get("sensor_width_mm"):
            # sensor came from DB guess, slight reduction
            conf -= 0.1
        conf = max(0.2, min(1.0, conf))

        explanation = (
            f"EXIF focal={focal:.1f}mm, image width={img_w}px, "
            f"sensor={sensor:.1f}mm (from {'EXIF' if exif.get('sensor_width_mm') else 'camera DB'}), "
            f"distance={distance_mm:.0f}mm ({dist_method}). "
            f"mm/px = {sensor:.1f}/{img_w} * {distance_mm:.0f}/{focal:.1f} = {mmpp:.4f}"
        )
    elif img_w and (focal or db_focal):
        # Focal present but no sensor -> use 35mm-equivalent assumption (36mm FF)
        f_use = focal or db_focal
        sensor_ff = 36.0
        mmpp = mm_per_px_from_geometry(f_use, sensor_ff, img_w, distance_mm)
        method = "ff_assumption"
        conf = 0.6 if dist_method != "assumed_distance" else 0.4
        explanation = (
            f"EXIF focal={f_use:.1f}mm; sensor assumed 36mm (full-frame); "
            f"distance={distance_mm:.0f}mm ({dist_method}). "
            f"mm/px = 36/{img_w} * {distance_mm:.0f}/{f_use:.1f} = {mmpp:.4f}"
        )
    else:
        # Nothing usable -> assumed default scale
        mmpp = 0.277
        method = "assumed"
        conf = 0.2
        explanation = (
            f"No EXIF / camera metadata found. Using assumed default "
            f"mm/px = {mmpp:.4f} (LOW confidence). Provide a known scale or "
            f"an image with EXIF for accurate results."
        )

    return {
        "calibration": Calibration(mmpp),
        "mm_per_px": float(mmpp),
        "method": method,
        "confidence": float(conf),
        "explanation": explanation,
        "exif": exif,
    }
