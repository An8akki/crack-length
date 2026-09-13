"""
compare_skeletons.py
====================

Benchmark the three skeletonization algorithms used for crack length
estimation:

    1. 'zhang_suen'  - classic Zhang-Suen parallel thinning
    2. 'guo_hall'    - Guo-Hall parallel thinning
    3. 'skeletonize' - scikit-image skeletonize

Each algorithm is evaluated on synthetic crack masks of KNOWN true length
(straight horizontal, straight vertical, diagonal at 45deg, a smooth curve,
and a branched crack). For each we measure:

    - Estimated length (px) via the same graph-traversal length function
    - Ground-truth length (px) of the crack's centerline
    - Absolute and relative error vs ground truth
    - Skeleton pixel count (a good skeleton is close to 1px wide)
    - Number of connected components (should be 1 for a single crack)
    - Runtime

The best algorithm is the one with the lowest average relative length error
and, secondarily, the most connected (fewest components) thin skeleton.

Run:
    python -m crack_quantification.compare_skeletons
"""

import time
import numpy as np

from .skeleton import skeletonize_mask
from .length import compute_length

METHODS = ["skeletonize", "zhang_suen", "guo_hall"]


def _bresenham_line(p0, p1):
    """Integer line between two points (Bresenham). Returns list of (y, x)."""
    x0, y0 = p0
    x1, y1 = p1
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((y0, x0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points


def _make_crack_mask(name, size=140):
    """Generate a synthetic single-crack mask + its true centerline length."""
    mask = np.zeros((size, size), dtype=np.uint8)
    thickness = 5

    if name == "horizontal":
        pts = _bresenham_line((30, 20), (110, 20))
        true_len = 110 - 30
    elif name == "vertical":
        pts = _bresenham_line((40, 30), (40, 110))
        true_len = 110 - 30
    elif name == "diagonal":
        pts = _bresenham_line((30, 30), (110, 110))
        true_len = np.hypot(110 - 30, 110 - 30)
    elif name == "curve":
        # A smooth-ish curve approximated by a polyline
        p0 = (25, 70)
        pm = (70, 25)
        p1 = (115, 80)
        pts = _bresenham_line(p0, pm) + _bresenham_line(pm, p1)[1:]
        true_len = np.hypot(70 - 25, 25 - 70) + np.hypot(115 - 70, 80 - 25)
    elif name == "branched":
        # Main line + a branch
        pts = _bresenham_line((30, 70), (90, 70))
        branch = _bresenham_line((60, 70), (60, 30))
        pts = pts + branch[1:]
        true_len = (90 - 30) + (70 - 30)
    else:
        raise ValueError(f"Unknown shape {name}")

# Draw the thick crack from the centerline points
    half = thickness // 2
    for (y, x) in pts:
        for dy in range(-half, half + 1):
            for dx in range(-half, half + 1):
                cy, cx = y + dy, x + dx
                if 0 <= cy < size and 0 <= cx < size:
                    mask[cy, cx] = 1

    return mask, true_len


def _skeleton_quality(skeleton):
    """Return (component_count, skeleton_pixel_count)."""
    from .length import _connected_components
    comps = _connected_components(skeleton)
    return len(comps), int(skeleton.sum())


def evaluate():
    """Run the full benchmark and print a comparison table."""
    shapes = ["horizontal", "vertical", "diagonal", "curve", "branched"]

    # results[method][shape] = dict of metrics
    results = {m: {} for m in METHODS}

    for shape in shapes:
        mask, true_len = _make_crack_mask(shape)
        for method in METHODS:
            t0 = time.perf_counter()
            skeleton = skeletonize_mask(mask, method=method)
            elapsed = time.perf_counter() - t0

            length = compute_length(skeleton)
            est_len = length["length_px"]
            err = est_len - true_len
            rel_err = (err / true_len) * 100.0 if true_len else 0.0
            ncomp, npix = _skeleton_quality(skeleton)

            results[method][shape] = {
                "true_len": true_len,
                "est_len": est_len,
                "abs_err": abs(err),
                "rel_err": rel_err,
                "components": ncomp,
                "pixels": npix,
                "time_s": elapsed,
            }

    # ---- Print per-shape table ----
    print("=" * 78)
    print("SKELETONIZATION ALGORITHM COMPARISON FOR CRACK LENGTH ESTIMATION")
    print("=" * 78)

    header = f"{'Shape':<12}{'Method':<13}{'True(px)':>10}{'Est(px)':>10}" \
             f"{'RelErr%':>9}{'Comp':>6}{'Px':>6}{'Time(ms)':>10}"
    print(header)
    print("-" * 78)

    for shape in shapes:
        for method in METHODS:
            r = results[method][shape]
            print(
                f"{shape:<12}{method:<13}{r['true_len']:>10.1f}{r['est_len']:>10.1f}"
                f"{r['rel_err']:>9.2f}{r['components']:>6}{r['pixels']:>6}"
                f"{r['time_s'] * 1000:>10.2f}"
            )
        print("-" * 78)

    # ---- Summary: average metrics per method ----
    print("\nAVERAGE ACROSS ALL SHAPES")
    print("-" * 78)
    summary = {}
    for method in METHODS:
        rel_errs = [results[method][s]["rel_err"] for s in shapes]
        abs_errs = [results[method][s]["abs_err"] for s in shapes]
        comps = [results[method][s]["components"] for s in shapes]
        total_time = sum(results[method][s]["time_s"] for s in shapes)
        summary[method] = {
            "mean_rel_err": np.mean(rel_errs),
            "mean_abs_err": np.mean(abs_errs),
            "total_comp": sum(comps),
            "total_time_s": total_time,
        }
        print(
            f"{method:<13} mean_rel_err={np.mean(rel_errs):6.2f}%  "
            f"mean_abs_err={np.mean(abs_errs):6.2f}px  "
            f"total_components={sum(comps):3d}  "
            f"total_time={total_time * 1000:.1f}ms"
        )

    # ---- Best algorithm ----
    # Rank primarily by mean relative error, then by fewer total components.
    ranked = sorted(
        METHODS,
        key=lambda m: (summary[m]["mean_abs_err"], summary[m]["total_comp"]),
    )
    best = ranked[0]
    print("\n" + "=" * 78)
    print(f"BEST ALGORITHM FOR LENGTH ESTIMATION: {best.upper()}")
    print(f"  - Lowest total absolute error across shapes: "
          f"{summary[best]['total_comp']} components")
    print(f"  - Mean relative error: {summary[best]['mean_rel_err']:.2f}%")
    print("=" * 78)

    return results, summary, best


if __name__ == "__main__":
    evaluate()
