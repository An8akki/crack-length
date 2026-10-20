"""
Local Surface Estimator for Step 4C Pipeline
Pinnacle Concrete Crack-Depth Project

Estimates the local concrete datum plane surrounding a crack:
1. Ring Band Generator (Multi-scale morphological dilations: 10px, 20px, 40px)
2. Local Ring Median
3. Robust RANSAC Linear Plane (Z = Ax + By + C)
4. Bivariate Quadratic Polynomial Surface (Z = ax^2 + by^2 + cxy + dx + ey + f)
"""

import numpy as np
import cv2
from sklearn.linear_model import RANSACRegressor, LinearRegression, HuberRegressor


class LocalSurfaceEstimator:
    def __init__(self):
        pass

    @staticmethod
    def generate_ring_mask(binary_mask, inner_dilation=5, outer_dilation=25, invalid_mask=None):
        r"""
        Generates a surrounding ring mask around the crack:
        Ring = Dilate(mask, outer) \ Dilate(mask, inner) \ InvalidPixels
        """
        k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_dilation * 2 + 1, inner_dilation * 2 + 1))
        k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_dilation * 2 + 1, outer_dilation * 2 + 1))

        dilated_inner = cv2.dilate(binary_mask, k_inner)
        dilated_outer = cv2.dilate(binary_mask, k_outer)

        ring = (dilated_outer > 0) & (dilated_inner == 0)
        if invalid_mask is not None:
            ring = ring & (~invalid_mask)
        return ring.astype(np.uint8)

    @staticmethod
    def fit_local_median(depth_map, ring_mask):
        """Estimates surface as scalar median of valid pixels in the surrounding ring."""
        ring_pixels = depth_map[ring_mask > 0]
        if len(ring_pixels) == 0:
            return float(np.median(depth_map[depth_map > 0])) if np.any(depth_map > 0) else 128.0, 0.0
        val = float(np.median(ring_pixels))
        residuals = np.abs(ring_pixels - val)
        std_residual = float(np.std(residuals))
        return val, std_residual

    @staticmethod
    def fit_ransac_plane(depth_map, ring_mask, max_samples=10000):
        """
        Fits robust linear plane Z = Ax + By + C on ring pixels using RANSAC.
        Returns:
            plane_fn: callable (x, y) -> estimated Z
            std_residual: float standard deviation of fit residuals
            params: (A, B, C)
        """
        y_coords, x_coords = np.where(ring_mask > 0)
        z_vals = depth_map[y_coords, x_coords]

        if len(z_vals) < 30:
            med_val, std_res = LocalSurfaceEstimator.fit_local_median(depth_map, ring_mask)
            return (lambda x, y: np.full_like(x, med_val, dtype=np.float64)), std_res, (0.0, 0.0, med_val)

        if len(z_vals) > max_samples:
            sub = np.random.choice(len(z_vals), max_samples, replace=False)
            x_fit, y_fit, z_fit = x_coords[sub], y_coords[sub], z_vals[sub]
        else:
            x_fit, y_fit, z_fit = x_coords, y_coords, z_vals

        X = np.column_stack([x_fit, y_fit])
        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=30,
            residual_threshold=4.0,
            random_state=42
        )
        ransac.fit(X, z_fit)
        A, B = ransac.estimator_.coef_
        C = ransac.estimator_.intercept_

        # Compute residual standard deviation on inliers
        inlier_mask = ransac.inlier_mask_
        residuals = z_fit[inlier_mask] - (A * x_fit[inlier_mask] + B * y_fit[inlier_mask] + C)
        std_res = float(np.std(residuals)) if len(residuals) > 0 else 0.0

        plane_fn = lambda x, y: A * x + B * y + C
        return plane_fn, round(std_res, 3), (round(A, 6), round(B, 6), round(C, 3))

    @staticmethod
    def fit_quadratic_surface(depth_map, ring_mask, max_samples=10000):
        """
        Fits 2nd-order bivariate polynomial: Z = ax^2 + by^2 + cxy + dx + ey + f
        using Huber robust regression to handle macro-scale lens/illumination field curvature.
        """
        y_coords, x_coords = np.where(ring_mask > 0)
        z_vals = depth_map[y_coords, x_coords]

        if len(z_vals) < 50:
            return LocalSurfaceEstimator.fit_ransac_plane(depth_map, ring_mask)

        if len(z_vals) > max_samples:
            sub = np.random.choice(len(z_vals), max_samples, replace=False)
            x_fit, y_fit, z_fit = x_coords[sub], y_coords[sub], z_vals[sub]
        else:
            x_fit, y_fit, z_fit = x_coords, y_coords, z_vals

        # Design matrix: [x^2, y^2, x*y, x, y]
        # Normalizing coordinates to [-1, 1] for numerical stability
        w, h = 1920.0, 1080.0
        xn = (x_fit - w / 2.0) / (w / 2.0)
        yn = (y_fit - h / 2.0) / (h / 2.0)
        X_poly = np.column_stack([xn**2, yn**2, xn * yn, xn, yn])

        huber = HuberRegressor(max_iter=100, alpha=1.0)
        huber.fit(X_poly, z_fit)
        coeffs = huber.coef_
        intercept = huber.intercept_

        # Prediction function
        def poly_fn(x, y):
            x_norm = (x - w / 2.0) / (w / 2.0)
            y_norm = (y - h / 2.0) / (h / 2.0)
            return (coeffs[0] * x_norm**2 +
                    coeffs[1] * y_norm**2 +
                    coeffs[2] * x_norm * y_norm +
                    coeffs[3] * x_norm +
                    coeffs[4] * y_norm +
                    intercept)

        pred_z = poly_fn(x_fit, y_fit)
        residuals = z_fit - pred_z
        std_res = float(np.std(residuals))

        return poly_fn, round(std_res, 3), tuple(coeffs) + (intercept,)
