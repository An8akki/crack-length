"""
Machine Learning Feasibility Evaluator for Step 4C Pipeline
Pinnacle Concrete Crack-Depth Project

Investigates whether independent 2D features (RGB luminance/contrast + 2D crack morphology)
can predict relative depth recess without depth sensor leakage.

Features:
- RGB mean & std inside crack (R, G, B)
- RGB mean & std outside crack in ring (R, G, B)
- Luminance contrast ratio (Y_ring / Y_crack)
- RGB edge gradient magnitude along crack lip
- Crack morphology: area, length, median width, max width, tortuosity, branch points

Target: Deterministic relative depth recess (depth units)
Models: Ridge Regression and Random Forest Regressor
Split: 70% Train / 15% Val / 15% Test
"""

import numpy as np
import cv2
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class MLEstimator:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.feature_names = [
            "mean_luminance_contrast",
            "rgb_crack_mean_intensity",
            "rgb_ring_mean_intensity",
            "rgb_edge_gradient",
            "median_width_px",
            "max_width_px",
            "crack_length_px",
            "crack_area_px",
            "tortuosity",
            "branch_points"
        ]

    @staticmethod
    def extract_rgb_features(rgb_image, binary_mask, ring_mask):
        """
        Extracts purely independent optical 2D features from the RGB frame.
        Zero depth information is consumed.
        """
        # Convert to grayscale luminance Y
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY).astype(np.float64)

        crack_idx = (binary_mask > 0)
        ring_idx = (ring_mask > 0)

        crack_y = gray[crack_idx]
        ring_y = gray[ring_idx]

        mean_y_crack = float(np.mean(crack_y)) if len(crack_y) > 0 else 128.0
        mean_y_ring = float(np.mean(ring_y)) if len(ring_y) > 0 else 128.0
        luminance_contrast = float(mean_y_ring / max(1.0, mean_y_crack))

        # Edge gradient on RGB
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        g_mag = np.sqrt(gx**2 + gy**2)

        # Boundary ring
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        boundary = (cv2.dilate(binary_mask, kernel) > 0) ^ crack_idx
        edge_grad = float(np.mean(g_mag[boundary])) if np.any(boundary) else 0.0

        return {
            "mean_luminance_contrast": round(luminance_contrast, 3),
            "rgb_crack_mean_intensity": round(mean_y_crack, 2),
            "rgb_ring_mean_intensity": round(mean_y_ring, 2),
            "rgb_edge_gradient": round(edge_grad, 2)
        }

    def train_and_evaluate(self, X, y, train_idx, val_idx, test_idx):
        """
        Trains Ridge and Random Forest models and evaluates on blind test set.
        """
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # 1. Ridge Baseline
        ridge = Ridge(alpha=10.0, random_state=self.random_state)
        ridge.fit(X_train, y_train)
        pred_ridge_test = ridge.predict(X_test)

        mae_ridge = mean_absolute_error(y_test, pred_ridge_test)
        rmse_ridge = np.sqrt(mean_squared_error(y_test, pred_ridge_test))
        r2_ridge = r2_score(y_test, pred_ridge_test)

        # 2. Random Forest Regressor
        rf = RandomForestRegressor(n_estimators=100, max_depth=6, min_samples_leaf=3, random_state=self.random_state)
        rf.fit(X_train, y_train)
        pred_rf_test = rf.predict(X_test)

        mae_rf = mean_absolute_error(y_test, pred_rf_test)
        rmse_rf = np.sqrt(mean_squared_error(y_test, pred_rf_test))
        r2_rf = r2_score(y_test, pred_rf_test)

        # Feature importances
        importances = dict(zip(self.feature_names, [round(float(v), 4) for v in rf.feature_importances_]))

        results = {
            "dataset_split": {
                "train_count": len(train_idx),
                "val_count": len(val_idx),
                "test_count": len(test_idx)
            },
            "ridge_regression": {
                "test_mae": round(float(mae_ridge), 3),
                "test_rmse": round(float(rmse_ridge), 3),
                "test_r2": round(float(r2_ridge), 3)
            },
            "random_forest": {
                "test_mae": round(float(mae_rf), 3),
                "test_rmse": round(float(rmse_rf), 3),
                "test_r2": round(float(r2_rf), 3),
                "feature_importances": importances
            },
            "test_actual": [round(float(v), 2) for v in y_test],
            "test_pred_rf": [round(float(v), 2) for v in pred_rf_test]
        }
        return results, rf
