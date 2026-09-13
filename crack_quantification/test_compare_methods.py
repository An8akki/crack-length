"""Tests for MST reconstruction and comparison artifacts."""

import json

from .compare_methods import compare_mask
from .compare_skeletons import _make_crack_mask
from .graph_analysis import analyze_skeleton
from .skeleton import skeletonize_mask


def test_mst_edges_form_a_reconstructable_tree():
    mask, _ = _make_crack_mask("branched")
    graph = analyze_skeleton(skeletonize_mask(mask, method="guo_hall"))
    assert graph["mst_edges"]
    assert abs(sum(edge["length_px"] for edge in graph["mst_edges"])
               - graph["mst_length_px"]) < 1e-9


def test_comparison_writes_mst_and_method_outputs(tmp_path):
    mask, true_length = _make_crack_mask("branched")
    report = compare_mask(mask, true_length_px=true_length, output_dir=tmp_path)
    assert report["best"]["absolute_error_px"] >= 0
    assert (tmp_path / "mst_marked_crack.png").exists()
    assert (tmp_path / "all_methods_comparison.png").exists()
    assert (tmp_path / "length_results.csv").exists()
    saved = json.loads((tmp_path / "comparison_report.json").read_text())
    assert saved["best"]["path_method"] in ("zhang_suen", "guo_hall", "morphological")