"""Unit and integration tests for planar_calibration module."""

import unittest
import numpy as np

from planar_calibration.homography import (
    order_quadrilateral_corners,
    validate_quadrilateral,
    compute_reprojection_error,
    assess_perspective_distortion,
    estimate_homography,
)
from planar_calibration.rectification import (
    rectify_image_and_mask,
    create_calibration_overlay,
    create_rectified_crack_overlay,
)
from planar_calibration.metric_transform import (
    MetricCoordinateTransform,
    compute_polyline_metric_length,
)
from planar_calibration.planar_measurement import quantify_planar_crack
from planar_calibration.planar_validation import (
    assess_planarity_heuristic,
    run_measurement_sanity_checks,
    compute_research_comparison,
)
from planar_calibration.reference_detection import resolve_planar_reference


class TestPlanarCalibration(unittest.TestCase):

    def setUp(self):
        # Synthetic known rectangle in image: (100, 100) to (500, 500)
        # Dimensions: 500 mm x 500 mm (1.25 mm/px)
        self.corners = np.array([
            [100.0, 100.0],  # TL
            [500.0, 100.0],  # TR
            [500.0, 500.0],  # BR
            [100.0, 500.0],  # BL
        ], dtype=np.float32)

        self.ref_w_mm = 500.0
        self.ref_h_mm = 500.0

    def test_corner_ordering(self):
        # Permute corners
        shuffled = np.array([self.corners[2], self.corners[0], self.corners[3], self.corners[1]])
        ordered = order_quadrilateral_corners(shuffled)
        np.testing.assert_allclose(ordered, self.corners, atol=1e-3)

    def test_quadrilateral_validation(self):
        is_valid, msg = validate_quadrilateral(self.corners)
        self.assertTrue(is_valid, f"Expected valid, got: {msg}")

        # Test collinear points
        collinear = np.array([[100, 100], [200, 100], [300, 100], [400, 100]], dtype=np.float32)
        is_valid_c, _ = validate_quadrilateral(collinear)
        self.assertFalse(is_valid_c)

        # Test self-intersecting hourglass
        hourglass = np.array([[100, 100], [500, 500], [500, 100], [100, 500]], dtype=np.float32)
        is_valid_h, _ = validate_quadrilateral(hourglass)
        self.assertFalse(is_valid_h)

    def test_homography_estimation_perfect_square(self):
        res = estimate_homography(self.corners, self.ref_w_mm, self.ref_h_mm)
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["status"], "VALID")
        self.assertLess(res["reprojection_error_px"], 0.1)
        self.assertEqual(res["distortion_rating"], "LOW")
        # 500mm / 400px = 1.25 mm/px
        self.assertAlmostEqual(res["mean_mm_per_px"], 1.25, places=2)

    def test_metric_coordinate_transform(self):
        res = estimate_homography(self.corners, self.ref_w_mm, self.ref_h_mm)
        transformer = MetricCoordinateTransform(
            res["H_metric"], res["rectified_width_px"], res["rectified_height_px"],
            self.ref_w_mm, self.ref_h_mm
        )

        # Test image to metric mapping for corners
        corners_metric = transformer.image_to_physical_mm(self.corners)
        expected_metric = np.array([
            [0.0, 0.0],
            [500.0, 0.0],
            [500.0, 500.0],
            [0.0, 500.0]
        ])
        np.testing.assert_allclose(corners_metric, expected_metric, atol=0.5)

    def test_rectification_and_quantification(self):
        # Create a synthetic image with a straight line crack from (200, 250) to (400, 250)
        # Length in image = 200 px. At 1.25 mm/px, physical length should be 250 mm.
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        mask = np.zeros((600, 600), dtype=np.uint8)
        mask[249:252, 200:401] = 1

        res = estimate_homography(self.corners, self.ref_w_mm, self.ref_h_mm)
        rect_img, rect_mask = rectify_image_and_mask(
            img, mask, res["H_px"], res["rectified_width_px"], res["rectified_height_px"]
        )

        self.assertIsNotNone(rect_mask)
        self.assertTrue(rect_mask.any())

        crack_metrics = quantify_planar_crack(rect_mask, self.ref_w_mm, self.ref_h_mm)
        self.assertTrue(crack_metrics["success"])
        # Expected: ~250 mm
        self.assertAlmostEqual(crack_metrics["main_branch_mm"], 250.0, delta=10.0)

    def test_sanity_checks_and_research_comparison(self):
        # Discrepancy between 3D (2380mm) and Planar (405mm)
        warnings = run_measurement_sanity_checks(
            length_2d_mm=317.0, length_planar_mm=405.0, length_3d_mm=2380.0
        )
        self.assertTrue(any("Discrepancy" in w["title"] for w in warnings))

        # Ground truth comparison
        comp = compute_research_comparison(
            {"2d": 317.0, "planar": 395.0, "3d": 420.0},
            ground_truth_mm=400.0
        )
        self.assertTrue(comp["has_ground_truth"])
        self.assertEqual(comp["best_performing_method"], "PLANAR")
        self.assertAlmostEqual(comp["best_error_pct"], 1.25, delta=0.5)


if __name__ == "__main__":
    unittest.main()
