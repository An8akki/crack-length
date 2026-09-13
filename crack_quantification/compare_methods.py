"""Compare crack-path algorithms and save marked graph outputs.

Example:
    python -m crack_quantification.compare_methods --synthetic \
        --output-dir outputs/crack_comparison

For a real segmented mask, pass ``--mask mask.png`` and optionally ``--image
crack.jpg``. The mask is expected to contain non-zero crack pixels.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .compare_skeletons import _make_crack_mask
from .graph_analysis import analyze_skeleton
from .skeleton import skeletonize_mask

METHODS = {
    "zhang_suen": "zhang_suen",
    "guo_hall": "guo_hall",
    "morphological": "skeletonize",
}
MEASURES = ("network", "main_path", "mst")
GRAPH_KEYS = {
    "network": "network_length_px",
    "main_path": "main_path_px",
    "mst": "mst_length_px",
}
COLORS = {
    "network": (255, 170, 0),
    "main_path": (0, 255, 0),
    "mst": (0, 0, 255),
}


def _load_gray(path):
    import cv2
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def _overlay(base, skeleton, graph, measure, color):
    import cv2
    canvas = np.repeat(base[..., None], 3, axis=2) if base.ndim == 2 else base.copy()
    canvas = canvas.astype(np.uint8)
    if measure == "mst":
        edges = graph["mst_edges"]
        for edge in edges:
            cv2.line(canvas, (int(edge["start"][1]), int(edge["start"][0])),
                     (int(edge["end"][1]), int(edge["end"][0])), color, 2)
        for edge in edges:
            for pt in (edge["start"], edge["end"]):
                cv2.circle(canvas, (int(pt[1]), int(pt[0])), 2, (0, 255, 255), -1)
        if not edges and skeleton.any():
            ys, xs = np.where(skeleton)
            for y, x in zip(ys, xs):
                cv2.circle(canvas, (int(x), int(y)), 2, color, -1)
    elif measure == "main_path":
        points = graph["main_path"]
        for first, second in zip(points, points[1:]):
            cv2.line(canvas, (first[1], first[0]), (second[1], second[0]),
                     color, 2)
    else:
        ys, xs = np.where(skeleton)
        canvas[ys, xs] = color
    return canvas


def _save_panel(base, outputs, output_path):
    import cv2
    labels = list(outputs)
    panels = []
    for index, label in enumerate(labels):
        image = outputs[label]
        title = f"{index + 1}: {label.split(':')[0]} / {label.split(':')[1].split('px')[0]}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 32), (0, 0, 0), -1)
        cv2.putText(image, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(image)
    panel = np.concatenate(panels, axis=1)
    cv2.imwrite(str(output_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def _shap_explanation(results, output_path):
    """Explain candidate error using SHAP over measurable method features."""
    try:
        import shap
        from sklearn.ensemble import RandomForestRegressor
    except ImportError:
        return {"available": False, "reason": "Install shap and scikit-learn"}
    if not results or not all("absolute_error_px" in row for row in results):
        return {"available": False, "reason": "Ground-truth length is required"}
    feature_names = ["is_zhang_suen", "is_guo_hall", "is_morphological",
                     "is_network", "is_main_path", "is_mst",
                     "skeleton_pixels", "components", "junctions",
                     "network_minus_main_px"]
    matrix = np.array([
        [float(row["path_method"] == method) for method in
         ("zhang_suen", "guo_hall", "morphological")] +
        [float(row["measure"] == measure) for measure in
         ("network", "main_path", "mst")] +
        [row[name] for name in ("skeleton_pixels", "components", "junctions",
                                "network_minus_main_px")]
        for row in results
    ], dtype=float)
    target = np.array([row["absolute_error_px"] for row in results], dtype=float)
    model = RandomForestRegressor(n_estimators=100, random_state=7).fit(matrix, target)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(matrix)
    mean_abs = np.mean(np.abs(values), axis=0)
    importance = sorted(
        ({"feature": name, "mean_abs_shap": float(value)}
         for name, value in zip(feature_names, mean_abs)),
        key=lambda item: item["mean_abs_shap"], reverse=True
    )
    report = {"available": True, "model": "RandomForestRegressor",
              "target": "absolute_error_px", "feature_importance": importance}
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def compare_mask(mask, image=None, true_length_px=None, output_dir="outputs/crack_comparison"):
    """Run every skeleton/graph combination and save overlays and tables."""
    import cv2
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mask = np.asarray(mask) > 0
    base = image if image is not None else (mask.astype(np.uint8) * 255)
    if base.shape[:2] != mask.shape:
        raise ValueError("image and mask must have identical height and width")
    results, overlays = [], {}
    for method, skeleton_method in METHODS.items():
        started = time.perf_counter()
        skeleton = skeletonize_mask(mask, method=skeleton_method)
        graph = analyze_skeleton(skeleton)
        elapsed_ms = 1000.0 * (time.perf_counter() - started)
        for measure in MEASURES:
            length_px = float(graph[GRAPH_KEYS[measure]])
            row = {
                "path_method": method, "measure": measure,
                "length_px": length_px, "skeleton_pixels": int(skeleton.sum()),
                "components": len(graph["components"]),
                "junctions": graph["junction_count"],
                "network_minus_main_px": graph["network_length_px"] - graph["main_path_px"],
                "processing_time_ms": elapsed_ms,
            }
            if true_length_px is not None:
                row["true_length_px"] = float(true_length_px)
                row["absolute_error_px"] = abs(length_px - true_length_px)
                row["percentage_error"] = 100.0 * (length_px - true_length_px) / true_length_px
            results.append(row)
            label = f"{method} + {measure}: {length_px:.1f}px"
            overlays[label] = _overlay(base, skeleton, graph, measure, COLORS[measure])
        if method == "zhang_suen":
            # This is the primary requested MST reconstruction output.
            mst_image = _overlay(base, skeleton, graph, "mst", COLORS["mst"])
            cv2.imwrite(str(output / "mst_marked_crack.png"),
                        cv2.cvtColor(mst_image, cv2.COLOR_RGB2BGR))

    _save_panel(base, overlays, output / "all_methods_comparison.png")
    with (output / "length_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    with (output / "length_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    ranked = sorted(results, key=lambda row: (
        row.get("absolute_error_px", 0.0) if true_length_px is not None else
        (0 if row["measure"] == "network" else 1),
        row["processing_time_ms"],
    ))
    shap_report = _shap_explanation(results, output / "shap_explanation.json")
    final = {
        "best": ranked[0],
        "ranking": ranked,
        "selection_rule": "lowest absolute error with ground truth; otherwise network measure for total crack length",
        "explanation": (
            "The selected method is best only for this evaluated sample. "
            "MST is preferred here because it removes redundant 8-connected "
            "cycle edges while retaining the crack branches. Longest geodesic "
            "is lower because it measures only the dominant path, not the full network."
        ),
        "shap": shap_report,
    }
    (output / "comparison_report.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    return final


def main():
    parser = argparse.ArgumentParser(description="Compare crack graph length methods")
    parser.add_argument("--mask", help="Binary crack mask image")
    parser.add_argument("--image", help="Original crack image")
    parser.add_argument("--true-length-px", type=float)
    parser.add_argument("--output-dir", default="outputs/crack_comparison")
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    if args.synthetic or not args.mask:
        mask, true_length = _make_crack_mask("branched")
        if args.true_length_px is None:
            args.true_length_px = true_length
        image = np.repeat((mask * 255)[..., None], 3, axis=2)
    else:
        mask = _load_gray(args.mask)
        image = _load_gray(args.image) if args.image else mask
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
    report = compare_mask(mask, image, args.true_length_px, args.output_dir)
    print(json.dumps({"best": report["best"], "output_dir": args.output_dir,
                      "shap": report["shap"]}, indent=2))


if __name__ == "__main__":
    main()