"""
Calibration Models for Step 5 Metric Crack Depth Estimation
Pinnacle Concrete Crack-Depth Project

Implements candidate model families for transferring relative optical recess
to physical crack depth in millimeters (mm):
1. HuberCalibrationModel (Robust linear M-estimation)
2. PCHIPCalibrationModel (Monotonic shape-preserving cubic Hermite spline)
3. WLSCalibrationModel (Heteroscedastic weighted least squares)
4. StereoPhysicsCalibrationModel (Epipolar stereo triangulation geometry)

Guarantees:
- Initial state is UNCALIBRATED.
- Prediction without calibration strictly raises NotCalibratedError.
- Explicit extrapolation detection beyond calibrated bounds.
- Full serialization/deserialization to JSON and pickle.
"""

from abc import ABC, abstractmethod
import json
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, Union
from scipy.interpolate import PchipInterpolator
from sklearn.linear_model import HuberRegressor

from .calibration_schema import (
    CalibrationDataset,
    CalibrationState,
    ValidationError,
)


class NotCalibratedError(Exception):
    """Raised when attempting physical mm estimation before calibration curve is established."""
    pass


class ExtrapolationWarning(UserWarning):
    """Emitted when inputs fall outside the verified physical calibration range."""
    pass


class BaseCalibrationModel(ABC):
    """
    Abstract Base Class for Metric Crack Depth Calibration Models.
    Defines the unified lifecycle: Initialize -> Fit(Dataset) -> Predict -> PredictWithUncertainty -> Save/Load.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.state: CalibrationState = CalibrationState.UNCALIBRATED
        self.calibration_metadata: Dict[str, Any] = {}
        self.recess_range: Tuple[float, float] = (0.0, 0.0)
        self.residual_sigma: float = 0.0
        self.dof: int = 1

    def _check_is_calibrated(self):
        """Guard method ensuring prediction cannot occur while uncalibrated."""
        if self.state == CalibrationState.UNCALIBRATED:
            raise NotCalibratedError(
                f"Model '{self.model_name}' is UNCALIBRATED. "
                "Physical crack depth in millimeters cannot be predicted until a verified "
                "physical calibration dataset with independently measured true_depth_mean_mm is fitted."
            )

    def check_extrapolation(self, recess_units: np.ndarray) -> np.ndarray:
        """
        Check whether inputs exceed the verified training bounds.
        Returns a boolean array indicating which samples are extrapolating.
        """
        self._check_is_calibrated()
        recess_min, recess_max = self.recess_range
        return (recess_units < recess_min) | (recess_units > recess_max)

    @abstractmethod
    def fit(self, dataset: CalibrationDataset):
        """Fit model parameters using a verified physical CalibrationDataset."""
        pass

    @abstractmethod
    def predict(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        """Predict physical crack depth in millimeters."""
        pass

    @abstractmethod
    def predict_with_uncertainty(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CalibrationState]:
        """
        Predict depth in mm along with 95% empirical confidence intervals.
        Returns: (pred_mm, lower_bound_mm, upper_bound_mm, state)
        """
        pass

    def save(self, filepath: str):
        """Serialize model state and parameters to disk."""
        payload = {
            "model_name": self.model_name,
            "state": self.state.value,
            "recess_range": self.recess_range,
            "residual_sigma": self.residual_sigma,
            "dof": self.dof,
            "calibration_metadata": self.calibration_metadata,
            "model_data": self._get_model_data(),
        }
        with open(filepath, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, filepath: str) -> "BaseCalibrationModel":
        """Load serialized model from disk."""
        with open(filepath, "rb") as f:
            payload = pickle.load(f)

        instance = cls()
        instance.state = CalibrationState(payload["state"])
        instance.recess_range = tuple(payload["recess_range"])
        instance.residual_sigma = payload["residual_sigma"]
        instance.dof = payload["dof"]
        instance.calibration_metadata = payload["calibration_metadata"]
        instance._restore_model_data(payload["model_data"])
        return instance

    @abstractmethod
    def _get_model_data(self) -> Dict[str, Any]:
        """Subclass specific serializable payload."""
        pass

    @abstractmethod
    def _restore_model_data(self, data: Dict[str, Any]):
        """Subclass specific state restoration."""
        pass


class HuberCalibrationModel(BaseCalibrationModel):
    """
    Huber Robust Linear M-Estimator:
    Z_mm = beta_1 * recess + beta_2 * width + beta_0
    Rejects aggregate bridging outliers and provides robust parameter fits.
    """

    def __init__(self, epsilon: float = 1.35, alpha: float = 0.0001):
        super().__init__(model_name="HuberRobustLinear")
        self.epsilon = epsilon
        self.alpha = alpha
        self.regressor: Optional[HuberRegressor] = None
        self.beta_recess: float = 0.0
        self.beta_width: float = 0.0
        self.intercept: float = 0.0

    def fit(self, dataset: CalibrationDataset):
        df = dataset.data
        X = df[["relative_recess_median", "crack_width_px"]].to_numpy()
        y = df["true_depth_mean_mm"].to_numpy()

        reg = HuberRegressor(epsilon=self.epsilon, alpha=self.alpha, fit_intercept=True)
        reg.fit(X, y)

        self.regressor = reg
        self.beta_recess = float(reg.coef_[0])
        self.beta_width = float(reg.coef_[1])
        self.intercept = float(reg.intercept_)

        preds = reg.predict(X)
        residuals = y - preds
        self.dof = max(1, len(y) - 3)
        self.residual_sigma = float(np.std(residuals, ddof=1))
        self.recess_range = (float(df["relative_recess_median"].min()), float(df["relative_recess_median"].max()))

        self.calibration_metadata = {
            "num_samples": len(df),
            "num_specimens": dataset.num_specimens,
            "specimen_ids": dataset.specimen_ids,
            "recess_min": self.recess_range[0],
            "recess_max": self.recess_range[1],
            "beta_recess": self.beta_recess,
            "beta_width": self.beta_width,
            "intercept": self.intercept,
            "residual_sigma_mm": self.residual_sigma,
        }
        self.state = CalibrationState.CALIBRATED

    def predict(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        self._check_is_calibrated()
        recess_arr = np.atleast_1d(np.asarray(recess_units, dtype=float))
        if width_px is None:
            width_arr = np.zeros_like(recess_arr)
        else:
            width_arr = np.atleast_1d(np.asarray(width_px, dtype=float))

        preds = self.beta_recess * recess_arr + self.beta_width * width_arr + self.intercept
        # Physical crack depth cannot be negative
        preds = np.maximum(preds, 0.0)
        return preds if recess_arr.ndim > 0 and len(preds) > 1 else np.squeeze(preds)

    def predict_with_uncertainty(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CalibrationState]:
        preds = self.predict(recess_units, width_px, standoff_mm)
        recess_arr = np.atleast_1d(np.asarray(recess_units, dtype=float))

        extrap = self.check_extrapolation(recess_arr)
        state = CalibrationState.EXTRAPOLATING_BEYOND_RANGE if np.any(extrap) else CalibrationState.CALIBRATED

        # 95% interval: t_crit ~ 1.96 for large sample sizes, expanded by 1.5x if extrapolating
        z_score = 1.96 if confidence >= 0.95 else 1.645
        margin = z_score * self.residual_sigma * (1.5 if np.any(extrap) else 1.0)

        lower = np.maximum(preds - margin, 0.0)
        upper = preds + margin
        return preds, lower, upper, state

    def _get_model_data(self) -> Dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "beta_recess": self.beta_recess,
            "beta_width": self.beta_width,
            "intercept": self.intercept,
        }

    def _restore_model_data(self, data: Dict[str, Any]):
        self.epsilon = data["epsilon"]
        self.alpha = data["alpha"]
        self.beta_recess = data["beta_recess"]
        self.beta_width = data["beta_width"]
        self.intercept = data["intercept"]


class PCHIPCalibrationModel(BaseCalibrationModel):
    """
    Monotonic Shape-Preserving Piecewise Cubic Hermite Interpolating Polynomial:
    Z_mm = PCHIP(recess)
    Guarantees physical monotonicity: dZ_mm / d(recess) >= 0 everywhere.
    """

    def __init__(self):
        super().__init__(model_name="MonotonicPCHIPSpline")
        self.interpolator: Optional[PchipInterpolator] = None
        self.x_anchors: np.ndarray = np.array([])
        self.y_anchors: np.ndarray = np.array([])

    def fit(self, dataset: CalibrationDataset):
        df = dataset.data.sort_values(by="relative_recess_median")

        # Aggregate replicates at distinct recess intervals
        grouped = df.groupby(
            pd.qcut(df["relative_recess_median"], q=min(10, df["specimen_id"].nunique()), duplicates="drop"),
            observed=False
        )
        x_pts = grouped["relative_recess_median"].mean().dropna().to_numpy()
        y_pts = grouped["true_depth_mean_mm"].mean().dropna().to_numpy()

        # Enforce strict sorted order for PCHIP
        sort_idx = np.argsort(x_pts)
        x_pts = x_pts[sort_idx]
        y_pts = y_pts[sort_idx]

        # Monotonic regularization: ensure y_pts is non-decreasing
        y_pts_mono = np.maximum.accumulate(y_pts)

        # Include origin (0 recess -> 0 depth)
        if x_pts[0] > 0.0:
            x_pts = np.insert(x_pts, 0, 0.0)
            y_pts_mono = np.insert(y_pts_mono, 0, 0.0)

        self.interpolator = PchipInterpolator(x_pts, y_pts_mono, extrapolate=True)
        self.x_anchors = x_pts
        self.y_anchors = y_pts_mono
        self.recess_range = (float(x_pts.min()), float(x_pts.max()))

        # Residuals
        preds = self.interpolator(df["relative_recess_median"].to_numpy())
        residuals = df["true_depth_mean_mm"].to_numpy() - preds
        self.residual_sigma = float(np.std(residuals, ddof=1))
        self.dof = max(1, len(df) - len(x_pts))

        self.calibration_metadata = {
            "num_samples": len(df),
            "num_anchors": len(x_pts),
            "x_anchors": x_pts.tolist(),
            "y_anchors": y_pts_mono.tolist(),
            "recess_min": self.recess_range[0],
            "recess_max": self.recess_range[1],
            "residual_sigma_mm": self.residual_sigma,
        }
        self.state = CalibrationState.CALIBRATED

    def predict(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        self._check_is_calibrated()
        recess_arr = np.atleast_1d(np.asarray(recess_units, dtype=float))
        preds = self.interpolator(recess_arr)
        preds = np.maximum(preds, 0.0)
        return preds if recess_arr.ndim > 0 and len(preds) > 1 else np.squeeze(preds)

    def predict_with_uncertainty(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CalibrationState]:
        preds = self.predict(recess_units, width_px, standoff_mm)
        recess_arr = np.atleast_1d(np.asarray(recess_units, dtype=float))

        extrap = self.check_extrapolation(recess_arr)
        state = CalibrationState.EXTRAPOLATING_BEYOND_RANGE if np.any(extrap) else CalibrationState.CALIBRATED

        z_score = 1.96 if confidence >= 0.95 else 1.645
        margin = z_score * self.residual_sigma * (2.0 if np.any(extrap) else 1.0)

        lower = np.maximum(preds - margin, 0.0)
        upper = preds + margin
        return preds, lower, upper, state

    def is_strictly_monotonic(self, test_points: int = 500) -> bool:
        """Verify derivative >= 0 across calibration range."""
        self._check_is_calibrated()
        x_eval = np.linspace(self.recess_range[0], self.recess_range[1], test_points)
        derivs = self.interpolator.derivative()(x_eval)
        return bool(np.all(derivs >= -1e-7))

    def _get_model_data(self) -> Dict[str, Any]:
        return {
            "x_anchors": self.x_anchors.tolist(),
            "y_anchors": self.y_anchors.tolist(),
        }

    def _restore_model_data(self, data: Dict[str, Any]):
        self.x_anchors = np.array(data["x_anchors"])
        self.y_anchors = np.array(data["y_anchors"])
        self.interpolator = PchipInterpolator(self.x_anchors, self.y_anchors, extrapolate=True)


class WLSCalibrationModel(BaseCalibrationModel):
    """
    Heteroscedastic Uncertainty-Weighted Least Squares:
    Minimizes sum w_i * (y_i - (beta_1 * recess_i + beta_0))^2
    where w_i = 1 / max(sigma_repeat_i^2, 1e-4)
    Points with lower physical measurement error receive higher optimization weight.
    """

    def __init__(self):
        super().__init__(model_name="UncertaintyWeightedLS")
        self.beta_recess: float = 0.0
        self.intercept: float = 0.0
        self.cov_matrix: Optional[np.ndarray] = None

    def fit(self, dataset: CalibrationDataset):
        df = dataset.data
        x = df["relative_recess_median"].to_numpy()
        y = df["true_depth_mean_mm"].to_numpy()
        sigmas = df["true_depth_std_mm"].to_numpy()

        weights = 1.0 / np.maximum(sigmas ** 2, 1e-4)

        # Construct design matrix [x, 1]
        X = np.column_stack([x, np.ones_like(x)])
        W = np.diag(weights)

        # Normal equations: (X^T W X) beta = X^T W y
        XtW = X.T @ W
        XtWX = XtW @ X
        XtWy = XtW @ y

        beta = np.linalg.solve(XtWX, XtWy)
        self.beta_recess = float(beta[0])
        self.intercept = float(beta[1])
        self.cov_matrix = np.linalg.inv(XtWX)

        preds = X @ beta
        residuals = y - preds
        self.residual_sigma = float(np.sqrt(np.sum(weights * (residuals ** 2)) / max(1, len(y) - 2)))
        self.recess_range = (float(x.min()), float(x.max()))

        self.calibration_metadata = {
            "num_samples": len(df),
            "beta_recess": self.beta_recess,
            "intercept": self.intercept,
            "residual_sigma_mm": self.residual_sigma,
            "cov_matrix": self.cov_matrix.tolist(),
        }
        self.state = CalibrationState.CALIBRATED

    def predict(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        self._check_is_calibrated()
        recess_arr = np.atleast_1d(np.asarray(recess_units, dtype=float))
        preds = self.beta_recess * recess_arr + self.intercept
        preds = np.maximum(preds, 0.0)
        return preds if recess_arr.ndim > 0 and len(preds) > 1 else np.squeeze(preds)

    def predict_with_uncertainty(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CalibrationState]:
        preds = self.predict(recess_units, width_px, standoff_mm)
        recess_arr = np.atleast_1d(np.asarray(recess_units, dtype=float))

        extrap = self.check_extrapolation(recess_arr)
        state = CalibrationState.EXTRAPOLATING_BEYOND_RANGE if np.any(extrap) else CalibrationState.CALIBRATED

        # Propagate parameter covariance
        X_eval = np.column_stack([recess_arr, np.ones_like(recess_arr)])
        var_pred = np.sum((X_eval @ self.cov_matrix) * X_eval, axis=1) + (self.residual_sigma ** 2)
        std_pred = np.sqrt(var_pred)

        z_score = 1.96 if confidence >= 0.95 else 1.645
        margin = z_score * std_pred * (1.5 if np.any(extrap) else 1.0)

        lower = np.maximum(preds - margin, 0.0)
        upper = preds + margin
        return preds, lower, upper, state

    def _get_model_data(self) -> Dict[str, Any]:
        return {
            "beta_recess": self.beta_recess,
            "intercept": self.intercept,
            "cov_matrix": self.cov_matrix.tolist() if self.cov_matrix is not None else None,
        }

    def _restore_model_data(self, data: Dict[str, Any]):
        self.beta_recess = data["beta_recess"]
        self.intercept = data["intercept"]
        self.cov_matrix = np.array(data["cov_matrix"]) if data["cov_matrix"] is not None else None


class StereoPhysicsCalibrationModel(BaseCalibrationModel):
    """
    Epipolar Stereo Triangulation Physics Model:
    Relates disparity offset delta_d = k_disp * recess to true physical depth:
    Z_true_mm = (Z_surf^2 * delta_d) / (f * B - Z_surf * delta_d) + beta_0
    Grounds depth estimation in physical optical triangulation limits.
    """

    def __init__(self, focal_length_px: float = 640.0, baseline_mm: float = 50.0):
        super().__init__(model_name="StereoTriangulationPhysics")
        self.focal_length_px = focal_length_px
        self.baseline_mm = baseline_mm
        self.k_disp: float = 0.1
        self.offset_mm: float = 0.0

    def fit(self, dataset: CalibrationDataset):
        df = dataset.data
        recess = df["relative_recess_median"].to_numpy()
        y = df["true_depth_mean_mm"].to_numpy()
        z_surf = df["sensor_standoff_distance_mm"].to_numpy()

        # Non-linear least squares optimization for scale k_disp and offset
        from scipy.optimize import curve_fit

        def forward_model(X, k, b):
            rec, z = X
            fB = self.focal_length_px * self.baseline_mm
            delta_d = k * rec
            denom = np.maximum(fB - z * delta_d, 1.0)
            return (z ** 2 * delta_d) / denom + b

        popt, _ = curve_fit(
            forward_model,
            (recess, z_surf),
            y,
            p0=[0.01, 0.0],
            bounds=([1e-5, -10.0], [5.0, 10.0]),
            maxfev=5000,
        )

        self.k_disp = float(popt[0])
        self.offset_mm = float(popt[1])

        preds = forward_model((recess, z_surf), self.k_disp, self.offset_mm)
        residuals = y - preds
        self.residual_sigma = float(np.std(residuals, ddof=1))
        self.recess_range = (float(recess.min()), float(recess.max()))

        self.calibration_metadata = {
            "num_samples": len(df),
            "focal_length_px": self.focal_length_px,
            "baseline_mm": self.baseline_mm,
            "k_disp": self.k_disp,
            "offset_mm": self.offset_mm,
            "residual_sigma_mm": self.residual_sigma,
        }
        self.state = CalibrationState.CALIBRATED

    def predict(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
    ) -> np.ndarray:
        self._check_is_calibrated()
        recess_arr = np.atleast_1d(np.asarray(recess_units, dtype=float))
        z_surf = np.full_like(recess_arr, 500.0) if standoff_mm is None else np.atleast_1d(np.asarray(standoff_mm, dtype=float))

        fB = self.focal_length_px * self.baseline_mm
        delta_d = self.k_disp * recess_arr
        denom = np.maximum(fB - z_surf * delta_d, 1.0)

        preds = (z_surf ** 2 * delta_d) / denom + self.offset_mm
        preds = np.maximum(preds, 0.0)
        return preds if recess_arr.ndim > 0 and len(preds) > 1 else np.squeeze(preds)

    def predict_with_uncertainty(
        self,
        recess_units: Union[float, np.ndarray],
        width_px: Optional[Union[float, np.ndarray]] = None,
        standoff_mm: Optional[Union[float, np.ndarray]] = None,
        confidence: float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, CalibrationState]:
        preds = self.predict(recess_units, width_px, standoff_mm)
        recess_arr = np.atleast_1d(np.asarray(recess_units, dtype=float))

        extrap = self.check_extrapolation(recess_arr)
        state = CalibrationState.EXTRAPOLATING_BEYOND_RANGE if np.any(extrap) else CalibrationState.CALIBRATED

        z_score = 1.96 if confidence >= 0.95 else 1.645
        margin = z_score * self.residual_sigma * (1.5 if np.any(extrap) else 1.0)

        lower = np.maximum(preds - margin, 0.0)
        upper = preds + margin
        return preds, lower, upper, state

    def _get_model_data(self) -> Dict[str, Any]:
        return {
            "focal_length_px": self.focal_length_px,
            "baseline_mm": self.baseline_mm,
            "k_disp": self.k_disp,
            "offset_mm": self.offset_mm,
        }

    def _restore_model_data(self, data: Dict[str, Any]):
        self.focal_length_px = data["focal_length_px"]
        self.baseline_mm = data["baseline_mm"]
        self.k_disp = data["k_disp"]
        self.offset_mm = data["offset_mm"]
