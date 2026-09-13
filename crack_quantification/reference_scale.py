"""Reference-object scale recovery, including planar perspective correction."""

import numpy as np

from .calibration import Calibration


def scale_from_reference_segment(known_length_mm, endpoint_a, endpoint_b):
    """Estimate a baseline mm/px scale from two endpoints on the surface."""
    if known_length_mm <= 0:
        raise ValueError("known_length_mm must be positive")
    pixel_length = float(np.linalg.norm(np.subtract(endpoint_a, endpoint_b)))
    if pixel_length <= 0:
        raise ValueError("Reference endpoints must be distinct")
    return Calibration(known_length_mm / pixel_length)


def homography_from_reference(corners_px, width_mm, height_mm):
    """Return image-to-surface homography for ordered quadrilateral corners.

    Corners must be ordered around the marker and correspond to
    ``(0,0), (width,0), (width,height), (0,height)`` in millimetres.
    """
    corners = np.asarray(corners_px, dtype=np.float64)
    if corners.shape != (4, 2):
        raise ValueError("corners_px must have shape (4, 2)")
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("Reference dimensions must be positive")
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("OpenCV is required for planar reference calibration") from exc
    target = np.array([[0, 0], [width_mm, 0], [width_mm, height_mm],
                       [0, height_mm]], dtype=np.float64)
    return cv2.getPerspectiveTransform(corners.astype(np.float32),
                                       target.astype(np.float32))


def polyline_length_on_reference_plane(points_px, homography):
    """Map a crack centreline to the reference plane and sum 2D lengths."""
    points = np.asarray(points_px, dtype=np.float64)
    if len(points) < 2:
        return 0.0
    homogeneous = np.column_stack([points[:, 1], points[:, 0],
                                   np.ones(len(points))])
    mapped = (np.asarray(homography) @ homogeneous.T).T
    mapped = mapped[:, :2] / mapped[:, 2:3]
    return float(np.linalg.norm(np.diff(mapped, axis=0), axis=1).sum())


def detect_aruco_reference(image, marker_size_mm, dictionary_name="DICT_4X4_50"):
    """Detect an ArUco marker and return its corners plus a baseline scale.

    Detection is optional because OpenCV builds differ in ArUco availability.
    The returned corner order is the order supplied by OpenCV.
    """
    try:
        import cv2
        aruco = cv2.aruco
    except (ImportError, AttributeError) as exc:
        raise ImportError("OpenCV ArUco support is required") from exc
    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))
    detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)
    if ids is None or len(corners) == 0:
        raise ValueError("No ArUco reference marker detected")
    marker = corners[0].reshape(4, 2)
    side_px = np.mean([
        np.linalg.norm(marker[i] - marker[(i + 1) % 4]) for i in range(4)
    ])
    return {"corners_px": marker, "marker_id": int(ids[0, 0]),
            "calibration": Calibration(marker_size_mm / side_px),
            "marker_px": float(side_px)}