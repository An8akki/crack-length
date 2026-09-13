"""Reusable factorial evaluation for crack-path and scale methods.

The runner accepts pre-segmented masks and ground-truth lengths, so detector
quality can be held constant while path and scale hypotheses are compared.
Input rows use ``mask``, ``true_length_mm`` and one of ``mm_per_px``,
``reference`` or ``depth_context``. Results are JSON/CSV-friendly dictionaries.
"""

import time

import numpy as np

from .graph_analysis import analyze_skeleton
from .skeleton import skeletonize_mask
from .scale_recovery import scale_from_camera_distance
from .reference_scale import scale_from_reference_segment


PATH_METHODS = ("zhang_suen", "guo_hall", "morphological")
PATH_MEASURES = ("network", "main_path", "mst")


def _scale_for_row(row, method):
    if method == "reference":
        reference = row["reference"]
        return scale_from_reference_segment(
            reference["length_mm"], reference["endpoint_a"], reference["endpoint_b"]
        ).mm_per_px
    if method == "camera_geometry":
        geometry = row["camera_geometry"]
        return scale_from_camera_distance(**geometry).mm_per_px
    if method == "depth_3d":
        # A depth row may provide a local metric scale directly when testing
        # monocular relative-depth models, or a full 3D path context.
        if "mm_per_px" in row.get("depth_context", {}):
            return float(row["depth_context"]["mm_per_px"])
        raise ValueError("depth_context requires mm_per_px for 2D scale evaluation")
    if method == "known":
        return float(row["mm_per_px"])
    raise ValueError("Unknown scale method: " + method)


def evaluate_rows(rows, scale_methods=("reference", "camera_geometry", "depth_3d")):
    """Evaluate all meaningful path/scale combinations and return raw rows."""
    results = []
    for index, row in enumerate(rows):
        for path_method in PATH_METHODS:
            started = time.perf_counter()
            skeleton_method = "skeletonize" if path_method == "morphological" else path_method
            skeleton = skeletonize_mask(row["mask"], method=skeleton_method)
            graph = analyze_skeleton(skeleton)
            elapsed = time.perf_counter() - started
            for measure in PATH_MEASURES:
                graph_key = {
                    "network": "network_length_px",
                    "main_path": "main_path_px",
                    "mst": "mst_length_px",
                }[measure]
                pixel_length = graph[graph_key]
                for scale_method in scale_methods:
                    try:
                        mm_per_px = _scale_for_row(row, scale_method)
                    except (KeyError, ValueError):
                        continue
                    estimate = pixel_length * mm_per_px
                    error = estimate - float(row["true_length_mm"])
                    results.append({
                        "sample_id": row.get("sample_id", index),
                        "path_method": path_method,
                        "measure": measure,
                        "scale_method": scale_method,
                        "pixel_length": float(pixel_length),
                        "estimate_mm": float(estimate),
                        "true_length_mm": float(row["true_length_mm"]),
                        "absolute_error_mm": float(abs(error)),
                        "percentage_error": float(100 * error / row["true_length_mm"]),
                        "processing_time_s": float(elapsed),
                        "components": len(graph["components"]),
                        "branches": len(graph["branch_lengths_px"]),
                    })
    return results


def summarize_results(results):
    """Aggregate MAE, RMSE, MAPE, signed-error spread, and runtime."""
    grouped = {}
    for result in results:
        key = (result["path_method"], result["measure"], result["scale_method"])
        grouped.setdefault(key, []).append(result)
    summary = []
    for key, values in grouped.items():
        errors = np.array([v["estimate_mm"] - v["true_length_mm"] for v in values])
        absolute = np.abs(errors)
        summary.append({
            "path_method": key[0], "measure": key[1], "scale_method": key[2],
            "n": len(values), "mae_mm": float(np.mean(absolute)),
            "rmse_mm": float(np.sqrt(np.mean(errors ** 2))),
            "mape_percent": float(np.mean(100 * absolute /
                                           np.array([v["true_length_mm"] for v in values]))),
            "error_std_mm": float(np.std(errors)),
            "mean_processing_time_s": float(np.mean([v["processing_time_s"] for v in values])),
        })
    return sorted(summary, key=lambda item: item["mae_mm"])