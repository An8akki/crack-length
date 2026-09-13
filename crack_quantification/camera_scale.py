"""Camera-scale recovery kept separate from crack-length estimation.

This module distinguishes camera calibration (intrinsics), scene metric scale
(a known planar reference or metric depth), and crack measurement. The GLB
heightmap in ``image_3d_heightmap.py`` is deliberately not accepted here: its
Z coordinate is normalized grayscale, not metric depth.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .calibration import Calibration
from .depth_3d import crack_length_from_depth
from .reference_scale import homography_from_reference
from .scale_recovery import mm_per_px_from_geometry


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics in pixel units."""

    fx: float
    fy: float
    cx: float
    cy: float
    distortion: tuple = ()

    def matrix(self):
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)


def intrinsics_from_sensor(focal_length_mm, sensor_width_mm, image_width_px,
                           image_height_px, sensor_height_mm=None):
    """Convert physical camera specifications to pixel focal lengths."""
    if sensor_height_mm is None:
        sensor_height_mm = sensor_width_mm * image_height_px / image_width_px
    fx = focal_length_mm * image_width_px / sensor_width_mm
    fy = focal_length_mm * image_height_px / sensor_height_mm
    return CameraIntrinsics(float(fx), float(fy), image_width_px / 2.0,
                            image_height_px / 2.0)


def calibrate_checkerboard(image_paths, pattern_size, square_size_mm):
    """Estimate camera intrinsics from checkerboard images.

    ``pattern_size`` is the number of inner corners as ``(columns, rows)``.
    At least several views with varied board orientation are required for a
    useful calibration; this function does not infer metric scale from a board
    image unless ``square_size_mm`` is supplied.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for checkerboard calibration") from exc
    if square_size_mm <= 0:
        raise ValueError("square_size_mm must be positive")
    columns, rows = pattern_size
    object_points = np.zeros((columns * rows, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_points *= float(square_size_mm)
    object_list, image_list = [], []
    image_size = None
    for image_path in image_paths:
        image = cv2.imread(str(Path(image_path)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(image_path)
        image_size = (image.shape[1], image.shape[0])
        found, corners = cv2.findChessboardCorners(image, (columns, rows), None)
        if found:
            refined = cv2.cornerSubPix(
                image, corners, (11, 11), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            )
            object_list.append(object_points.copy())
            image_list.append(refined)
    if len(object_list) < 3:
        raise ValueError("At least three valid checkerboard views are required")
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_list, image_list, image_size, None, None
    )
    return {"intrinsics": CameraIntrinsics(
        float(matrix[0, 0]), float(matrix[1, 1]),
        float(matrix[0, 2]), float(matrix[1, 2]),
        tuple(float(value) for value in distortion.ravel())
    ), "rms_reprojection_error_px": float(rms),
            "views_used": len(object_list)}


def scale_from_known_distance(intrinsics, distance_mm):
    """Return centre-image scale for a fronto-parallel surface."""
    if distance_mm <= 0:
        raise ValueError("distance_mm must be positive")
    return Calibration(float(distance_mm / ((intrinsics.fx + intrinsics.fy) / 2.0)))


def planar_reference_scale(corners_px, width_mm, height_mm, crack_path_yx=None):
    """Recover a perspective-correct planar scale from a known rectangle."""
    homography = homography_from_reference(corners_px, width_mm, height_mm)
    result = {"homography_image_to_surface": homography}
    if crack_path_yx is not None:
        from .reference_scale import polyline_length_on_reference_plane
        result["crack_length_mm"] = polyline_length_on_reference_plane(
            crack_path_yx, homography
        )
    return result


def scale_from_metric_depth(depth_map, intrinsics, crack_path_yx=None):
    """Recover local metric scale or a 3D crack length from metric depth."""
    depth = np.asarray(depth_map, dtype=np.float64)
    if depth.ndim != 2:
        raise ValueError("depth_map must be a 2D metric depth array")
    valid = depth[np.isfinite(depth) & (depth > 0)]
    if valid.size == 0:
        raise ValueError("depth_map has no positive metric values")
    result = {
        "median_depth_mm": float(np.median(valid)),
        "centre_mm_per_px": float(np.median(valid) /
                                   ((intrinsics.fx + intrinsics.fy) / 2.0)),
    }
    if crack_path_yx is not None:
        result.update(crack_length_from_depth(
            crack_path_yx, depth, intrinsics.fx, intrinsics.fy,
            intrinsics.cx, intrinsics.cy
        ))
    return result


def recover_camera_scale(*, intrinsics=None, focal_length_mm=None,
                         sensor_width_mm=None, image_width_px=None,
                         image_height_px=None, distance_mm=None,
                         reference_corners_px=None, reference_width_mm=None,
                         reference_height_mm=None, depth_map=None,
                         crack_path_yx=None):
    """Choose a metric scale source without mixing it with crack tracing.

    Priority is perspective-correct reference, metric depth, then explicit
    fronto-parallel geometry. The return value includes a method label and
    confidence rationale so an experiment can compare methods honestly.
    """
    if reference_corners_px is not None:
        if reference_width_mm is None or reference_height_mm is None:
            raise ValueError("Reference width and height are required")
        result = planar_reference_scale(
            reference_corners_px, reference_width_mm, reference_height_mm,
            crack_path_yx
        )
        result.update({"method": "reference_homography", "confidence": "high"})
        return result

    if depth_map is not None:
        if intrinsics is None:
            raise ValueError("Metric depth recovery requires pixel intrinsics")
        result = scale_from_metric_depth(depth_map, intrinsics, crack_path_yx)
        result.update({"method": "metric_depth_3d", "confidence": "depends_on_depth"})
        return result

    if intrinsics is not None and distance_mm is not None:
        result = {"calibration": scale_from_known_distance(intrinsics, distance_mm),
                  "method": "intrinsics_distance", "confidence": "depends_on_distance"}
        if crack_path_yx is not None:
            result["crack_length_mm"] = float(
                len(crack_path_yx) - 1
            ) * result["calibration"].mm_per_px
        return result

    required = (focal_length_mm, sensor_width_mm, image_width_px,
                distance_mm)
    if all(value is not None for value in required):
        return {"calibration": Calibration(mm_per_px_from_geometry(*required)),
                "method": "pinhole_geometry", "confidence": "depends_on_inputs"}
    raise ValueError("Provide a planar reference, metric depth, or complete camera geometry")