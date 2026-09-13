"""Regression tests for topology and metric scale research primitives."""

import numpy as np

from .depth_3d import crack_length_from_depth
from .graph_analysis import analyze_skeleton
from .reference_scale import scale_from_reference_segment
from .research_experiment import evaluate_rows


def test_branched_network_and_geodesic_are_distinct():
    skeleton = np.zeros((31, 31), dtype=bool)
    skeleton[15, 5:27] = True
    skeleton[5:16, 15] = True
    report = analyze_skeleton(skeleton)
    assert report["network_length_px"] > report["main_path_px"]
    assert report["mst_length_px"] > 0
    assert report["endpoint_count"] >= 3


def test_reference_segment_recovers_known_scale():
    calibration = scale_from_reference_segment(100.0, (20, 10), (20, 260))
    assert calibration.mm_per_px == 0.4


def test_metric_depth_backprojection_preserves_flat_surface_length():
    depth = np.full((20, 100), 1000.0)
    result = crack_length_from_depth([(10, 10), (10, 60)], depth, 50, 50)
    assert np.isclose(result["length"], 1000.0)


def test_factorial_runner_emits_all_combinations():
    mask = np.zeros((30, 50), dtype=np.uint8)
    mask[15, 5:45] = 1
    row = {
        "mask": mask,
        "true_length_mm": 40.0,
        "reference": {"length_mm": 100.0, "endpoint_a": (2, 2),
                       "endpoint_b": (2, 102)},
        "camera_geometry": {"distance_mm": 1000.0, "focal_length_mm": 50.0,
                             "sensor_width_mm": 36.0, "image_width_px": 100},
        "depth_context": {"mm_per_px": 1.0},
    }
    assert len(evaluate_rows([row])) == 27