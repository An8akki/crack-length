"""Planar crack measurement executing identical skeletonization and graph network analysis."""

from typing import Dict, Optional, Tuple
import numpy as np

from crack_quantification.skeleton import skeletonize_mask
from crack_quantification.graph_analysis import analyze_skeleton
from crack_quantification.length import compute_length, _connected_components

from .metric_transform import MetricCoordinateTransform, compute_polyline_metric_length


def quantify_planar_crack(
    rectified_mask: np.ndarray,
    ref_width_mm: float,
    ref_height_mm: float,
    H_metric: Optional[np.ndarray] = None
) -> Dict:
    """Compute physical crack measurements from a perspective-rectified binary mask.

    Uses the exact same skeleton algorithms (Guo-Hall, Zhang-Suen, Morphological)
    and MST graph topology as the 2D pipeline, ensuring parity of algorithm.

    Args:
        rectified_mask: Binary mask in rectified coordinate system (H_rect, W_rect)
        ref_width_mm: Known physical width of the reference plane in mm
        ref_height_mm: Known physical height of the reference plane in mm
        H_metric: Optional 3x3 homography mapping image px -> metric mm

    Returns:
        dict of planar measurements and per-skeleton comparison
    """
    rh, rw = rectified_mask.shape[:2]
    scale_x = float(ref_width_mm / max(1, rw))
    scale_y = float(ref_height_mm / max(1, rh))
    mean_scale = float(0.5 * (scale_x + scale_y))

    mask = (rectified_mask > 0).astype(np.uint8)
    if not mask.any():
        return {
            "success": False,
            "message": "Rectified crack mask is empty.",
            "network_length_mm": 0.0,
            "mst_length_mm": 0.0,
            "main_branch_mm": 0.0,
            "mean_width_mm": 0.0,
            "max_width_mm": 0.0,
            "area_mm2": 0.0,
            "mean_scale_mm_per_px": mean_scale,
            "per_method": {},
        }

    # 1. Compute 3 skeletonizations on rectified mask
    skel_gh = skeletonize_mask(mask, method="guo_hall")
    skel_zs = skeletonize_mask(mask, method="zhang_suen")
    skel_mo = skeletonize_mask(mask, method="skeletonize")

    primary_skel = skel_gh if skel_gh.any() else (skel_zs if skel_zs.any() else skel_mo)

    # 2. Graph analysis on primary skeleton
    graph = analyze_skeleton(primary_skel)
    raw_2d = compute_length(primary_skel)

    # 3. Compute accurate physical length for MST and main path
    # Convert graph edges directly in metric millimeters
    mst_len_mm = 0.0
    if "mst_edges" in graph and graph["mst_edges"]:
        for edge in graph["mst_edges"]:
            pt1 = edge["start"]
            pt2 = edge["end"]
            dx = (pt2[1] - pt1[1]) * scale_x
            dy = (pt2[0] - pt1[0]) * scale_y
            mst_len_mm += float(np.sqrt(dx ** 2 + dy ** 2))
    else:
        mst_len_mm = float(graph["mst_length_px"]) * mean_scale

    # Main path in metric mm
    main_path_mm = 0.0
    if "main_path" in graph and len(graph["main_path"]) >= 2:
        pts = np.array(graph["main_path"])  # (N, 2) in (y, x)
        dx = np.diff(pts[:, 1]) * scale_x
        dy = np.diff(pts[:, 0]) * scale_y
        main_path_mm = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
    else:
        main_path_mm = float(graph["main_path_px"]) * mean_scale

    # Network length
    network_len_mm = float(graph["network_length_px"]) * mean_scale

    # 4. Crack surface area in physical mm²
    area_px = int(np.count_nonzero(mask))
    area_mm2 = float(area_px * scale_x * scale_y)

    # 5. Crack width in physical mm
    try:
        import cv2
        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        core_widths = dist_transform[primary_skel > 0] * 2.0
        if len(core_widths) > 0:
            mean_width_px = float(np.mean(core_widths))
            max_width_px = float(np.max(core_widths))
        else:
            mean_width_px = 1.0
            max_width_px = 1.0
    except Exception:
        mean_width_px = 1.0
        max_width_px = 1.0

    mean_width_mm = float(mean_width_px * mean_scale)
    max_width_mm = float(max_width_px * mean_scale)

    # 6. Per-method evaluation under planar homography
    per_method = {}
    for name, sk in [("guo_hall", skel_gh), ("zhang_suen", skel_zs), ("morphological", skel_mo)]:
        if sk.any():
            g_m = analyze_skeleton(sk)
            per_method[name] = {
                "network_px": float(g_m["network_length_px"]),
                "network_mm": round(float(g_m["network_length_px"]) * mean_scale, 2),
                "main_path_px": float(g_m["main_path_px"]),
                "main_path_mm": round(float(g_m["main_path_px"]) * mean_scale, 2),
                "mst_px": float(g_m["mst_length_px"]),
                "mst_mm": round(float(g_m["mst_length_px"]) * mean_scale, 2),
                "junctions": g_m["junction_count"],
            }
        else:
            per_method[name] = {"network_px": 0, "network_mm": 0, "main_path_px": 0, "main_path_mm": 0, "mst_px": 0, "mst_mm": 0, "junctions": 0}

    return {
        "success": True,
        "network_length_mm": round(network_len_mm, 2),
        "mst_length_mm": round(mst_len_mm, 2),
        "main_branch_mm": round(main_path_mm, 2),
        "network_length_px": round(float(graph["network_length_px"]), 1),
        "mst_length_px": round(float(graph["mst_length_px"]), 1),
        "main_branch_px": round(float(graph["main_path_px"]), 1),
        "mean_width_mm": round(mean_width_mm, 3),
        "max_width_mm": round(max_width_mm, 3),
        "area_mm2": round(area_mm2, 2),
        "area_px": area_px,
        "rectified_width_px": rw,
        "rectified_height_px": rh,
        "scale_x_mm_per_px": round(scale_x, 5),
        "scale_y_mm_per_px": round(scale_y, 5),
        "mean_scale_mm_per_px": round(mean_scale, 5),
        "junctions": graph["junction_count"],
        "endpoints": graph["endpoint_count"],
        "primary_skeleton": primary_skel,
        "skeletons": {
            "guo_hall": skel_gh,
            "zhang_suen": skel_zs,
            "morphological": skel_mo,
        },
        "per_method": per_method,
    }
