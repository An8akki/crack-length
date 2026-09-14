"""Comprehensive unit tests for the Scientific Quantitative Crack Validation Platform."""

import unittest
import numpy as np

from synthetic_specimens.crack_generator import (
    generate_parametric_crack,
    compute_analytical_ground_truth,
)
from synthetic_specimens.surface_models import (
    get_surface_model,
    PlanarWall,
    CylindricalSurface,
    IrregularSurface,
)
from synthetic_specimens.camera_simulation import (
    CameraParameters,
    project_surface_and_crack_to_image,
)
from synthetic_specimens.reference_generator import generate_reference_target
from validation_platform.real_specimen_schema import create_specimen_record
from validation_platform.experiment_runner import (
    run_specimen_trial,
    aggregate_experiment_statistics,
)
from validation_platform.reliability_model import MeasurementReliabilityModel
from validation_platform.adaptive_selector import recommend_measurement_method


class TestScientificValidationPlatform(unittest.TestCase):

    def test_parametric_crack_analytical_ground_truth(self):
        # Generate straight crack with target 400 mm
        crack = generate_parametric_crack(
            crack_type="straight", target_length_mm=400.0,
            surface_width_mm=1000.0, surface_height_mm=1000.0, seed=42
        )
        gt = compute_analytical_ground_truth(crack)

        # Assert length is close to 400 mm (slight Brownian roughness)
        self.assertAlmostEqual(gt["total_centerline_length_mm"], 400.0, delta=25.0)
        self.assertGreater(gt["total_area_mm2"], 0.0)
        self.assertGreater(gt["mean_width_mm"], 0.0)
        self.assertEqual(gt["branch_count"], 1)
        self.assertEqual(gt["endpoint_count"], 2)

    def test_branching_crack_topology(self):
        crack = generate_parametric_crack(
            crack_type="branching", target_length_mm=500.0, seed=123
        )
        gt = compute_analytical_ground_truth(crack)

        self.assertGreaterEqual(gt["branch_count"], 2)
        self.assertGreaterEqual(gt["junction_count"], 1)
        # In trees, endpoints = junctions + 2
        self.assertEqual(gt["endpoint_count"], gt["junction_count"] + 2)
        # Total network length must be strictly greater than dominant branch
        self.assertGreater(gt["total_network_length_mm"], gt["dominant_branch_length_mm"])

    def test_surface_manifold_models(self):
        flat = get_surface_model("planar_wall", 1000.0, 1000.0)
        self.assertTrue(flat.is_planar())
        pts = np.array([[100.0, 100.0], [300.0, 100.0]])
        self.assertAlmostEqual(flat.compute_3d_arc_length(pts), 200.0, places=2)

        curved = get_surface_model("cylindrical", 1000.0, 1000.0, radius_mm=600.0)
        self.assertFalse(curved.is_planar())
        # Arc length across curved flank should be strictly longer in 3D than flat 2D
        x_pts = np.linspace(300.0, 700.0, 50)
        y_pts = np.full_like(x_pts, 500.0)
        pts_flank = np.column_stack([x_pts, y_pts])
        arc_3d = curved.compute_3d_arc_length(pts_flank)
        self.assertGreater(arc_3d, 400.0)

    def test_camera_simulation_projection(self):
        crack = generate_parametric_crack("curved", target_length_mm=450.0, seed=7)
        surface = get_surface_model("planar_wall", 1000.0, 1000.0)
        cam = CameraParameters(distance_mm=1000.0, yaw_deg=20.0, pitch_deg=10.0)
        ref = generate_reference_target(surface, 500.0, 500.0)

        rendered = project_surface_and_crack_to_image(
            crack, surface, cam, reference_corners_3d=ref["corners_3d"], seed=7
        )

        self.assertEqual(rendered["image_rgb"].shape, (720, 1280, 3))
        self.assertEqual(rendered["crack_mask"].shape, (720, 1280))
        self.assertTrue(rendered["crack_mask"].any())
        self.assertIsNotNone(rendered["projected_reference_px"])
        self.assertEqual(rendered["projected_reference_px"].shape, (4, 2))

    def test_specimen_trial_and_error_quantification(self):
        crack = generate_parametric_crack("straight", target_length_mm=350.0, seed=99)
        surface = get_surface_model("planar_wall", 1000.0, 1000.0)
        cam = CameraParameters(distance_mm=1000.0, yaw_deg=0.0, pitch_deg=0.0)
        ref = generate_reference_target(surface, 500.0, 500.0)

        rendered = project_surface_and_crack_to_image(
            crack, surface, cam, reference_corners_3d=ref["corners_3d"], seed=99
        )

        rec = run_specimen_trial(
            image_rgb=rendered["image_rgb"],
            ground_truth=rendered["ground_truth"],
            specimen_id="TEST-001",
            surface_type="planar_wall",
            is_planar=True,
            camera_distance_mm=1000.0,
            reference_corners_px=rendered["projected_reference_px"],
            reference_dims_mm=(500.0, 500.0),
            crack_mask=rendered["crack_mask"]
        )

        self.assertIsNotNone(rec.estimate_2d_mm)
        self.assertIsNotNone(rec.estimate_planar_mm)
        self.assertIsNotNone(rec.error_planar_pct)
        # Planar error on frontal view should be low (< 10%)
        self.assertLess(rec.error_planar_pct, 10.0)

    def test_adaptive_method_selector(self):
        # Case 1: High planarity + valid reference -> recommend PLANAR
        diag_planar = {
            "planarity_score": 92.0,
            "planarity_label": "HIGH",
            "is_planar": True,
            "reprojection_error_px": 0.45,
            "homography_status": "VALID",
            "camera_yaw_deg": 15.0,
            "length_planar_mm": 500.0,
            "length_3d_mm": 515.0,
        }
        rec1 = recommend_measurement_method(diag_planar)
        self.assertEqual(rec1.recommended_method, "PLANAR")
        self.assertEqual(rec1.reliability, "HIGH")

        # Case 2: Low planarity (curved cylinder) -> recommend 3D
        diag_curved = {
            "planarity_score": 45.0,
            "planarity_label": "LOW",
            "is_planar": False,
            "length_3d_mm": 530.0,
        }
        rec2 = recommend_measurement_method(diag_curved)
        self.assertEqual(rec2.recommended_method, "3D")

        # Case 3: Extreme angle without reference -> UNRELIABLE
        diag_unreliable = {
            "is_physical_scale_calibrated": False,
            "homography_status": "NOT_CONFIGURED",
            "camera_yaw_deg": 50.0,
        }
        rec3 = recommend_measurement_method(diag_unreliable)
        self.assertEqual(rec3.recommended_method, "UNRELIABLE")
        self.assertFalse(rec3.is_measurement_acceptable)


if __name__ == "__main__":
    unittest.main()
