"""Automatic reference marker detection (ArUco / AprilTag) for planar calibration."""

from typing import Dict, List, Optional, Tuple
import numpy as np


DICT_NAMES = [
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_6X6_50",
    "DICT_6X6_250",
    "DICT_APRILTAG_36h11",
    "DICT_ARUCO_ORIGINAL",
]


def detect_aruco_marker(
    image: np.ndarray,
    marker_size_mm: float = 100.0,
    preferred_dict: Optional[str] = None
) -> Dict:
    """Detect ArUco or AprilTag reference markers in an RGB or grayscale image.

    Args:
        image: RGB or grayscale numpy image
        marker_size_mm: Physical side length of the marker in mm (default 100mm)
        preferred_dict: Optional dictionary name to test first

    Returns:
        dict with:
            'detected': bool
            'marker_id': int or None
            'dictionary': str or None
            'corners_px': (4, 2) ndarray in clockwise order (TL, TR, BR, BL)
            'marker_width_mm': float
            'marker_height_mm': float
            'message': str
    """
    try:
        import cv2
        if not hasattr(cv2, "aruco"):
            return {
                "detected": False,
                "message": "OpenCV ArUco module is not available in this OpenCV build.",
                "corners_px": None,
            }
        aruco = cv2.aruco
    except ImportError as e:
        return {
            "detected": False,
            "message": f"OpenCV import failed: {e}",
            "corners_px": None,
        }

    # Convert to grayscale if needed
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    dicts_to_try = [preferred_dict] if preferred_dict and hasattr(aruco, preferred_dict) else []
    dicts_to_try += [d for d in DICT_NAMES if hasattr(aruco, d) and d != preferred_dict]

    best_result = None

    for d_name in dicts_to_try:
        try:
            d_val = getattr(aruco, d_name)
            dictionary = aruco.getPredefinedDictionary(d_val)

            # Support both new OpenCV 4.7+ ArucoDetector and legacy detectMarkers
            if hasattr(aruco, "ArucoDetector"):
                params = aruco.DetectorParameters()
                detector = aruco.ArucoDetector(dictionary, params)
                corners, ids, _ = detector.detectMarkers(gray)
            else:
                params = aruco.DetectorParameters_create()
                corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)

            if ids is not None and len(corners) > 0:
                # Select the largest marker if multiple are detected
                max_area = -1.0
                best_corner = None
                best_id = None
                for idx, c in enumerate(corners):
                    pts = c.reshape(4, 2)
                    area = cv2.contourArea(pts.astype(np.float32))
                    if area > max_area:
                        max_area = area
                        best_corner = pts
                        best_id = int(ids[idx][0])

                if best_corner is not None:
                    best_result = {
                        "detected": True,
                        "marker_id": best_id,
                        "dictionary": d_name,
                        "corners_px": best_corner.astype(np.float32),
                        "marker_width_mm": float(marker_size_mm),
                        "marker_height_mm": float(marker_size_mm),
                        "message": f"Detected marker ID {best_id} using {d_name}",
                    }
                    break
        except Exception:
            continue

    if best_result is not None:
        return best_result

    return {
        "detected": False,
        "marker_id": None,
        "dictionary": None,
        "corners_px": None,
        "marker_width_mm": float(marker_size_mm),
        "marker_height_mm": float(marker_size_mm),
        "message": "No fiducial marker (ArUco / AprilTag) detected in image.",
    }


def detect_planar_marker(
    image: np.ndarray,
    marker_size_mm: float = 100.0,
) -> Dict:
    """Convenience alias for detect_aruco_marker."""
    return detect_aruco_marker(image, marker_size_mm)
