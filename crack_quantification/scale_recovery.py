"""
scale_recovery.py
=================

Automatic recovery of the real-world measurement scale (mm per pixel) from a
single RGB image, without requiring calibration markers.

Four approaches are provided:

1. Metadata / camera intrinsics  (`scale_from_metadata`)
   - Uses EXIF focal length, sensor size, image width and (if available) the
     camera-to-surface distance.

2. Camera distance estimation   (`scale_from_camera_distance`)
   - A camera-to-wall distance d (e.g. from a monocular distance regressor or
     vanishing-point geometry) combined with intrinsics:
         mm_per_px = (sensor_width_mm / image_width_px) * (distance / focal_length)

3. Monocular depth estimation   (`scale_from_depth_map`)
   - A depth map (metric or relative) covering the wall is fitted to a plane,
     then the local scale is derived via the pinhole model per pixel.

4. Fusion                       (`fuse_scales`)
   - Probabilistic (inverse-variance weighted) combination of whichever of the
     above estimates are available, producing a more robust single scale.

All methods return a `Calibration` (mm_per_px) so the rest of the pipeline
(length / width / area / severity) is untouched.
"""

import numpy as np

from .calibration import Calibration


# --------------------------------------------------------------------------- #
# Shared pinhole helpers
# --------------------------------------------------------------------------- #
def mm_per_px_from_geometry(
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
    distance_mm: float,
) -> float:
    """Pinhole model: mm per pixel at a given perpendicular distance."""
    if (focal_length_mm <= 0 or sensor_width_mm <= 0
            or image_width_px <= 0 or distance_mm <= 0):
        raise ValueError("All pinhole inputs must be positive")
    pixel_size_mm = sensor_width_mm / float(image_width_px)
    return float(pixel_size_mm) * (distance_mm / focal_length_mm)


def intrinsic_from_exif(exif: dict) -> dict:
    """
    Best-effort extraction of focal length and sensor size from EXIF data.

    EXIF keys are handled loosely (Piexif / PIL / dict). Returns a dict with
    'focal_length_mm', 'sensor_width_mm', 'image_width_px' or None for unknown.
    """
    result = {
        "focal_length_mm": None,
        "sensor_width_mm": None,
        "image_width_px": None,
    }

    # Focal length
    fl = (
        exif.get("FocalLength")
        or exif.get("focal_length")
        or exif.get("FocalLengthIn35mmFilm")
    )
    if fl is not None:
        try:
            if isinstance(fl, tuple):
                fl = fl[0] / fl[1] if fl[1] else None
            result["focal_length_mm"] = float(fl)
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # Image width
    iw = exif.get("PixelXDimension") or exif.get("ImageWidth") or exif.get("width")
    if iw is not None:
        try:
            result["image_width_px"] = int(iw)
        except (TypeError, ValueError):
            pass

    # Sensor width: often not in EXIF; infer from 35mm-equivalent if present.
    sw = exif.get("SensorWidth") or exif.get("sensor_width_mm")
    if sw is not None:
        try:
            result["sensor_width_mm"] = float(sw)
        except (TypeError, ValueError):
            pass
    else:
        fl35 = exif.get("FocalLengthIn35mmFilm")
        fl_mm = result["focal_length_mm"]
        if fl35 and fl_mm:
            crop = float(fl35) / fl_mm
            result["sensor_width_mm"] = 36.0 / crop

    return result


# --------------------------------------------------------------------------- #
# Method 1: metadata / intrinsics
# --------------------------------------------------------------------------- #
def scale_from_metadata(
    exif: dict = None,
    *,
    focal_length_mm: float = None,
    sensor_width_mm: float = None,
    image_width_px: int = None,
    distance_mm: float = None,
) -> Calibration:
    """
    Recover scale from camera metadata (EXIF) and an optional known distance.

    If `exif` is provided, it is parsed for focal length / sensor / width.
    Missing values can be supplied directly as keyword arguments.
    `distance_mm` is required to produce a metric scale; if omitted, an error
    is raised (metadata alone cannot give absolute scale).
    """
    if exif:
        parsed = intrinsic_from_exif(exif)
        focal_length_mm = focal_length_mm or parsed["focal_length_mm"]
        sensor_width_mm = sensor_width_mm or parsed["sensor_width_mm"]
        image_width_px = image_width_px or parsed["image_width_px"]

    if distance_mm is None:
        raise ValueError(
            "scale_from_metadata requires a camera distance (mm). "
            "Metadata alone cannot recover absolute scale."
        )
    if (focal_length_mm is None or sensor_width_mm is None
            or image_width_px is None):
        raise ValueError(
            "Missing required intrinsics (focal_length_mm, sensor_width_mm, "
            "image_width_px)."
        )

    mmpp = mm_per_px_from_geometry(
        focal_length_mm, sensor_width_mm, image_width_px, distance_mm
    )
    return Calibration(mmpp)


# --------------------------------------------------------------------------- #
# Method 2: camera distance estimation
# --------------------------------------------------------------------------- #
def scale_from_camera_distance(
    distance_mm: float,
    *,
    focal_length_mm: float = None,
    sensor_width_mm: float = None,
    image_width_px: int = None,
    exif: dict = None,
) -> Calibration:
    """
    Recover scale from an estimated camera-to-wall distance plus intrinsics.

    `distance_mm` may come from any distance estimator (monocular geometry,
    vanishing points, a regression network, or a depth-derived plane distance).
    """
    if exif:
        parsed = intrinsic_from_exif(exif)
        focal_length_mm = focal_length_mm or parsed["focal_length_mm"]
        sensor_width_mm = sensor_width_mm or parsed["sensor_width_mm"]
        image_width_px = image_width_px or parsed["image_width_px"]

    if (focal_length_mm is None or sensor_width_mm is None
            or image_width_px is None):
        raise ValueError(
            "scale_from_camera_distance requires focal_length_mm, "
            "sensor_width_mm, image_width_px."
        )
    if distance_mm <= 0:
        raise ValueError("distance_mm must be positive")

    mmpp = mm_per_px_from_geometry(
        focal_length_mm, sensor_width_mm, image_width_px, distance_mm
    )
    return Calibration(mmpp)


# --------------------------------------------------------------------------- #
# Method 3: monocular depth estimation
# --------------------------------------------------------------------------- #
def scale_from_depth_map(
    depth_map: np.ndarray,
    *,
    focal_length_px: float = None,
    sensor_width_mm: float = None,
    image_width_px: int = None,
    metric: bool = True,
    reference_distance_mm: float = None,
) -> Calibration:
    """
    Recover scale from a monocular depth map.

    The wall plane is fitted to the depth map (least-squares plane fit over
    the central region). The perpendicular distance to the plane is then used
    with the pinhole model to compute mm_per_px.

    Parameters
    ----------
    depth_map : np.ndarray
        Depth map. If `metric=True`, values are in millimetres. Otherwise it is
        treated as relative depth and must be paired with `reference_distance_mm`
        (known metric distance at some pixel) to rescale.
    focal_length_px : float
        Focal length in pixels (fx = focal_mm * image_width / sensor_width_mm).
    """
    if depth_map is None:
        raise ValueError("depth_map cannot be None")
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim == 3:
        depth = depth[:, :, 0]

    if focal_length_px is None:
        raise ValueError("scale_from_depth_map requires focal_length_px")

    if not metric:
        if reference_distance_mm is None:
            raise ValueError(
                "For a relative depth map, reference_distance_mm is required "
                "to rescale to metric units."
            )
        ref_val = float(np.median(depth))
        if ref_val <= 0:
            ref_val = float(np.max(depth))
        depth = depth / ref_val * reference_distance_mm

    # Fit a plane z = a*x + b*y + c to the central region of the depth map.
    h, w = depth.shape
    # Use central 60% region to avoid edges
    y0, y1 = int(h * 0.2), int(h * 0.8)
    x0, x1 = int(w * 0.2), int(w * 0.8)
    region = depth[y0:y1, x0:x1]
    ys, xs = np.mgrid[y0:y1, x0:x1]
    xf = xs.flatten().astype(np.float64)
    yf = ys.flatten().astype(np.float64)
    zf = region.flatten()
    valid = np.isfinite(zf) & (zf > 0)
    if valid.sum() < 10:
        raise ValueError("Not enough valid depth values to fit a wall plane.")

    A = np.column_stack([xf[valid], yf[valid], np.ones(valid.sum())])
    coef, *_ = np.linalg.lstsq(A, zf[valid], rcond=None)
    # Plane perpendicular distance at image centre (wc/2, hc/2)
    cx, cy = w / 2.0, h / 2.0
    ax, by, c = coef
    dist_mm = abs(ax * cx + by * cy + c) / np.sqrt(ax * ax + by * by + 1.0)

    # Pinhole with focal length in pixels:
    #   mm_per_px = distance_mm / focal_length_px
    # (focal_length_px = focal_mm * image_width_px / sensor_width_mm, so the
    #  sensor width cancels out.)
    if focal_length_px <= 0:
        raise ValueError("focal_length_px must be positive")
    mmpp = dist_mm / focal_length_px
    return Calibration(float(mmpp))


# --------------------------------------------------------------------------- #
# Method 4: Fusion
# --------------------------------------------------------------------------- #
def fuse_scales(estimates: list) -> Calibration:
    """
    Probabilistic fusion of multiple mm_per_px estimates using
    inverse-variance weighting (best linear unbiased estimator for
    independent measurements).

    Parameters
    ----------
    estimates : list of dict
        Each dict has 'mm_per_px' and optional 'std_mm_per_px' (uncertainty).
        If 'std' is missing, a default is used.

    Returns
    -------
    Calibration
        Fused scale.
    """
    if not estimates:
        raise ValueError("No scale estimates provided for fusion")

    weights = []
    vals = []
    default_std = 0.05 * max([e["mm_per_px"] for e in estimates], default=1.0)

    for e in estimates:
        val = e["mm_per_px"]
        if val is None or val <= 0:
            continue
        std = e.get("std_mm_per_px") or e.get("std") or default_std
        vals.append(val)
        weights.append(1.0 / (std * std))

    if not vals:
        raise ValueError("No valid scale estimates for fusion")

    weights = np.array(weights)
    vals = np.array(vals)
    fused = float(np.sum(weights * vals) / np.sum(weights))
    return Calibration(fused)
