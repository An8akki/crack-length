"""
calibration.py
==============

Convert pixel measurements to real-world units (millimetres).

Three methods are provided:

1. Known mm/pixel scale  (e.g. a ruler in the image, or a fixed sensor setup)
2. ArUco marker         (place a known-size marker beside the crack)
3. Camera intrinsics    (focal length, sensor size, working distance)

Each method returns a `Calibration` object which knows how to convert pixel
lengths, widths and areas to their real-world equivalents.
"""

import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV2_AVAILABLE = False


class Calibration:
    """
    Holds the pixel->physical scale and provides conversion helpers.

    Attributes
    ----------
    mm_per_px : float
        Millimetres spanned by a single image pixel.
    """

    def __init__(self, mm_per_px: float):
        if mm_per_px is None or mm_per_px <= 0:
            raise ValueError("mm_per_px must be a positive value")
        self.mm_per_px = float(mm_per_px)

    # ---- length ----
    def px_to_mm(self, length_px: float) -> float:
        """Convert a pixel length to millimetres."""
        return float(length_px) * self.mm_per_px

    def mm_to_px(self, length_mm: float) -> float:
        """Convert a length in millimetres to pixels."""
        return float(length_mm) / self.mm_per_px

    # ---- area ----
    def px2_to_mm2(self, area_px: float) -> float:
        """Convert a pixel area (px^2) to square millimetres."""
        return float(area_px) * (self.mm_per_px ** 2)

    def mm2_to_px2(self, area_mm2: float) -> float:
        """Convert an area in square millimetres to pixels."""
        return float(area_mm2) / (self.mm_per_px ** 2)

    # ---- width ----
    def width_px_to_mm(self, width_px: float) -> float:
        """Convert a pixel width to millimetres (same as px_to_mm)."""
        return self.px_to_mm(width_px)

    def __repr__(self):
        return f"Calibration(mm_per_px={self.mm_per_px:.6f})"


def scale_from_mm_per_pixel(mm_per_px: float) -> Calibration:
    """
    Method 1: Directly specify the scale (mm per pixel).

    Example: a 50 mm marker spans 180 px -> scale 50/180 = 0.277 mm/px.
    """
    return Calibration(mm_per_px)


def _detect_aruco_scale(image: np.ndarray, marker_size_mm: float) -> float:
    """
    Detect an ArUco marker in the image and return mm_per_px.

    Returns the scale (mm per pixel) derived from the marker's known physical
    size measured in pixels.
    """
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV (cv2) is required for ArUco calibration")

    # Try a range of common dictionaries; fall back gracefully.
    dict_list = [
        cv2.aruco.DICT_4X4_50,
        cv2.aruco.DICT_5X5_50,
        cv2.aruco.DICT_6X6_50,
        cv2.aruco.DICT_ARUCO_ORIGINAL,
    ]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for dict_id in dict_list:
        try:
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
        except AttributeError:
            # Older OpenCV API
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(
                gray, dictionary, parameters=parameters
            )
        except Exception:
            continue

        if ids is not None and len(ids) > 0:
            # Use the first detected marker; measure its side length in px.
            c = corners[0][0]  # 4 corner points
            side_px = np.mean(
                [np.linalg.norm(c[i] - c[(i + 1) % 4]) for i in range(4)]
            )
            return marker_size_mm / side_px

    raise RuntimeError("No ArUco marker detected in the image")


def scale_from_aruco_marker(image: np.ndarray, marker_size_mm: float) -> Calibration:
    """
    Method 2: Derive scale from a known-size ArUco marker in the image.

    Parameters
    ----------
    image : np.ndarray
        BGR image containing the crack and the marker.
    marker_size_mm : float
        Physical edge length of the marker, in millimetres.

    Returns
    -------
    Calibration
    """
    mm_per_px = _detect_aruco_scale(image, marker_size_mm)
    return Calibration(mm_per_px)


def scale_from_camera_intrinsics(
    focal_length_mm: float,
    sensor_width_mm: float,
    image_width_px: int,
    distance_mm: float,
) -> Calibration:
    """
    Method 3: Derive scale from camera intrinsics and working distance.

    Using the pinhole model:

        mm_per_px = (sensor_width_mm / image_width_px) * (distance_mm / focal_length_mm)

    Parameters
    ----------
    focal_length_mm : float
        Camera focal length (physical, e.g. 4.5 mm).
    sensor_width_mm : float
        Camera sensor width (e.g. 6.17 mm for 1/2.3" sensor).
    image_width_px : int
        Image width in pixels.
    distance_mm : float
        Perpendicular distance from camera to the surface, in millimetres.

    Returns
    -------
    Calibration
    """
    if focal_length_mm <= 0:
        raise ValueError("focal_length_mm must be positive")
    if sensor_width_mm <= 0:
        raise ValueError("sensor_width_mm must be positive")
    if image_width_px <= 0:
        raise ValueError("image_width_px must be positive")
    if distance_mm <= 0:
        raise ValueError("distance_mm must be positive")

    pixel_size_mm = sensor_width_mm / float(image_width_px)
    magnification = distance_mm / focal_length_mm
    mm_per_px = pixel_size_mm * magnification
    return Calibration(mm_per_px)
