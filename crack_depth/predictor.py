"""
Production Predictor Engine for Relative Concrete Crack Recess Estimation
Integrated into crack-length system.

Estimates relative crevice recess (ΔZ) in dimensionless sensor units using
the frozen ExtraTreesRegressor and Top-25 independent features.
Provides split conformal prediction intervals and explainable 5-tier reliability.
"""

import os
import sys
import json
from dataclasses import dataclass, asdict
from typing import Union, Optional, Tuple, Dict, Any, List
import numpy as np
import cv2
import joblib

from .core.preprocessor import DepthPreprocessor
from .core.morphology import CrackMorphologyExtractor
from .core.surface_estimator import LocalSurfaceEstimator
from .core.ml_estimator import MLEstimator
from .core.calibration_schema import CalibrationState
from .core.calibration_models import NotCalibratedError


@dataclass
class FinalPredictionResult:
    """Structured result container for crack depth estimation."""
    predicted_relative_depth: float
    unit: str
    interval_80: Tuple[float, float]
    interval_90: Tuple[float, float]
    interval_95: Tuple[float, float]
    reliability: str
    reliability_reasons: List[str]
    median_crack_width_px: float
    crack_area_px: int
    valid_depth_ratio: float
    surface_noise_sigma: float
    plane_inlier_ratio: float
    calibration_state: str
    features: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FinalCrackPredictor:
    """
    Production Predictor for Relative Concrete Crack Recess Estimation.
    Integrated into the crack-length quantification platform.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
    ):
        pkg_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(pkg_dir, "models")
        self.config_path = config_path or os.path.join(models_dir, "final_config.json")
        self.model_path = model_path or os.path.join(models_dir, "final_extra_trees.joblib")
        self.scaler_path = scaler_path or os.path.join(models_dir, "final_scaler.joblib")

        # Load frozen configuration
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        # Load frozen model and scaler
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
        if not os.path.exists(self.scaler_path):
            raise FileNotFoundError(f"Scaler file not found at: {self.scaler_path}")

        self.model = joblib.load(self.model_path)
        self.scaler = joblib.load(self.scaler_path)

        self.feature_names = self.config["features"]["feature_list"]
        self.q80 = float(self.config["conformal_uncertainty"]["intervals"]["80_percent"]["half_width_units"])
        self.q90 = float(self.config["conformal_uncertainty"]["intervals"]["90_percent"]["half_width_units"])
        self.q95 = float(self.config["conformal_uncertainty"]["intervals"]["95_percent"]["half_width_units"])

        # Core preprocessing & extraction components
        self.preprocessor = DepthPreprocessor()
        self.morph_extractor = CrackMorphologyExtractor()
        self.surface_estimator = LocalSurfaceEstimator()
        self.ml_evaluator = MLEstimator()

    @property
    def calibration_state(self) -> CalibrationState:
        return CalibrationState.UNCALIBRATED

    def predict_depth_mm(self, *args, **kwargs):
        """
        Permanent safety gate: Physical millimeter depth estimation is strictly blocked.
        """
        raise NotCalibratedError(
            "predict_depth_mm() is permanently disabled. The system operates strictly under "
            "a software-only scope without physical millimeter depth calibration."
        )

    def _load_image_if_needed(self, img_input: Union[str, np.ndarray], is_color: bool = False) -> np.ndarray:
        """Loads image from path or validates existing numpy array."""
        if isinstance(img_input, str):
            if not os.path.exists(img_input):
                raise FileNotFoundError(f"Input file does not exist: {img_input}")
            flag = cv2.IMREAD_COLOR if is_color else cv2.IMREAD_UNCHANGED
            img = cv2.imread(img_input, flag)
            if img is None:
                raise ValueError(f"Failed to read image from path: {img_input}")
            return img
        elif isinstance(img_input, np.ndarray):
            return img_input.copy()
        else:
            raise TypeError(f"Expected file path str or numpy.ndarray, got {type(img_input)}")

    def validate_inputs(
        self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validates shapes, dtypes, and value ranges for RGB-D-mask triplet."""
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"RGB image must have 3 channels (H, W, 3), got shape {rgb.shape}")
        if rgb.dtype != np.uint8:
            raise TypeError(f"RGB image must be uint8, got {rgb.dtype}")

        if depth.ndim == 3:
            if depth.shape[2] == 1:
                depth = depth.squeeze(axis=2)
            else:
                depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        if depth.ndim != 2:
            raise ValueError(f"Depth map must be 2D (H, W), got shape {depth.shape}")
        if depth.dtype != np.uint8:
            raise TypeError(f"Depth map must be 8-bit uint8 [0, 255], got {depth.dtype}")

        if mask.ndim == 3:
            if mask.shape[2] == 1:
                mask = mask.squeeze(axis=2)
            else:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.ndim != 2:
            raise ValueError(f"Crack mask must be 2D (H, W), got shape {mask.shape}")
        mask_binary = (mask > 0).astype(np.uint8)

        if rgb.shape[:2] != depth.shape[:2] or rgb.shape[:2] != mask_binary.shape[:2]:
            raise ValueError(
                f"Dimension mismatch between inputs: RGB {rgb.shape[:2]}, "
                f"Depth {depth.shape[:2]}, Mask {mask_binary.shape[:2]}"
            )

        return rgb, depth, mask_binary

    def extract_features(
        self, rgb: np.ndarray, depth: np.ndarray, mask: np.ndarray
    ) -> Tuple[Dict[str, float], float, float, float]:
        """
        Extracts the exact Top-25 independent features without target depth leakage.
        Returns: (feature_dict, valid_depth_ratio, surface_noise_sigma, plane_inlier_ratio)
        """
        crack_pixels = int(np.sum(mask > 0))
        if crack_pixels == 0:
            raise ValueError("Input crack mask is completely empty (0 crack pixels).")

        invalid_depth = (depth == 0)
        valid_crack_pixels = np.sum((mask > 0) & (~invalid_depth))
        valid_depth_ratio = float(valid_crack_pixels / max(1, crack_pixels))

        # 1. 2D Morphology & Skeleton Analysis
        m_features = self.morph_extractor.analyze_crack(mask)

        y_indices, x_indices = np.where(mask > 0)
        dx = float(np.max(x_indices) - np.min(x_indices) + 1)
        dy = float(np.max(y_indices) - np.min(y_indices) + 1)
        bbox_aspect_ratio = float(max(dx, dy) / max(1.0, min(dx, dy)))

        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        skel_idx = (m_features["skeleton"] > 0)
        if np.any(skel_idx):
            skel_widths = dist_transform[skel_idx] * 2.0
            p75_w = float(np.percentile(skel_widths, 75))
            p25_w = float(np.percentile(skel_widths, 25))
        else:
            p75_w = p25_w = float(m_features["median_width_px"])

        # 2. Depth Bilateral Filtering & Intact Ring Surface Estimation
        depth_filtered = self.preprocessor.filter_bilateral(
            depth, d=7, sigma_color=15.0, sigma_space=7.0
        )
        ring_20 = self.surface_estimator.generate_ring_mask(
            mask, inner_dilation=5, outer_dilation=20, invalid_mask=invalid_depth
        )

        np.random.seed(42)
        plane_fn, surface_noise, plane_params = self.surface_estimator.fit_ransac_plane(
            depth_filtered, ring_20
        )
        A, B, C = plane_params

        ring_pixels = depth_filtered[ring_20 > 0]
        local_surface_median = float(np.median(ring_pixels)) if len(ring_pixels) > 0 else 128.0

        if len(ring_pixels) >= 30:
            y_r, x_r = np.where(ring_20 > 0)
            plane_preds = A * x_r + B * y_r + C
            res_r = np.abs(ring_pixels - plane_preds)
            plane_inliers = np.sum(res_r <= 4.0)
            plane_inlier_ratio = float(plane_inliers / len(ring_pixels))
        else:
            plane_inlier_ratio = 1.0

        # 3. 2D RGB Photometric Features
        rgb_feat = self.ml_evaluator.extract_rgb_features(rgb, mask, ring_20)

        # 4. Assembling and Engineering Domain Features
        c_area = float(m_features["area_px"])
        c_len = float(m_features["length_px"])
        m_width = float(m_features["median_width_px"])
        b_points = float(m_features["branch_points"])

        hydraulic_radius_proxy = c_area / (2.0 * c_len + 2.0 * m_width + 1e-4)
        log_median_width = float(np.log1p(m_width))
        aperture_roughness_ratio = m_width / (surface_noise + 1e-4)
        compactness = c_area / (c_len ** 2 + 1e-4)
        aspect_width_ratio = bbox_aspect_ratio / (m_width + 1e-4)
        contrast_roughness_product = float(rgb_feat["mean_luminance_contrast"] * surface_noise)
        rgb_contrast_ratio = float(
            rgb_feat["rgb_crack_mean_intensity"] / (rgb_feat["rgb_ring_mean_intensity"] + 1e-4)
        )
        branch_density_per_1kpx = float((b_points * 1000.0) / (c_len + 1e-4))

        feat_dict: Dict[str, float] = {
            "hydraulic_radius_proxy": float(hydraulic_radius_proxy),
            "median_width_px": float(m_width),
            "orientation_deg": float(m_features["orientation_deg"]),
            "log_median_width": float(log_median_width),
            "mean_width_px": float(m_features["mean_width_px"]),
            "aperture_roughness_ratio": float(aperture_roughness_ratio),
            "max_width_px": float(m_features["max_width_px"]),
            "crack_area_px": float(c_area),
            "compactness": float(compactness),
            "p75_width_px": float(p75_w),
            "p25_width_px": float(p25_w),
            "plane_inlier_ratio": float(plane_inlier_ratio),
            "tortuosity": float(m_features["tortuosity"]),
            "bbox_width_px": float(dx),
            "aspect_width_ratio": float(aspect_width_ratio),
            "mean_luminance_contrast": float(rgb_feat["mean_luminance_contrast"]),
            "local_surface_median": float(local_surface_median),
            "rgb_ring_mean_intensity": float(rgb_feat["rgb_ring_mean_intensity"]),
            "contrast_roughness_product": float(contrast_roughness_product),
            "branch_points": float(b_points),
            "rgb_edge_gradient": float(rgb_feat["rgb_edge_gradient"]),
            "rgb_contrast_ratio": float(rgb_contrast_ratio),
            "skeleton_length_px": float(c_len),
            "branch_density_per_1kpx": float(branch_density_per_1kpx),
            "main_component_ratio": float(m_features["main_component_ratio"]),
        }

        return feat_dict, valid_depth_ratio, float(surface_noise), float(plane_inlier_ratio)

    def classify_reliability(
        self,
        median_width: float,
        valid_depth_ratio: float,
        surface_noise: float,
        plane_inlier_ratio: float,
        crack_pixels: int,
    ) -> Tuple[str, List[str]]:
        """Transparent, explainable 5-tier reliability classification."""
        reasons = []

        if crack_pixels == 0:
            return "INVALID", ["Zero crack pixels in mask."]
        if valid_depth_ratio < 0.10:
            return "INVALID", [f"Valid depth ratio ({valid_depth_ratio*100:.1f}%) < 10% dropout limit."]

        if median_width >= 20.0 and valid_depth_ratio >= 0.95 and surface_noise <= 2.30 and plane_inlier_ratio >= 0.85:
            reasons.append("Aperture >= 20 px enables optical depth penetration.")
            reasons.append("High valid depth ratio (>= 95%).")
            reasons.append("Low flanking surface roughness.")
            return "HIGH", reasons

        if median_width >= 15.0 and valid_depth_ratio >= 0.90 and plane_inlier_ratio >= 0.70:
            reasons.append("Aperture >= 15 px provides detectable optical signal.")
            reasons.append("Acceptable valid depth ratio (>= 90%).")
            return "RELIABLE", reasons

        if median_width >= 10.0 and valid_depth_ratio >= 0.80:
            reasons.append("Narrow aperture (10-15 px); optical attenuation present.")
            reasons.append("Valid depth ratio >= 80%.")
            return "BORDERLINE", reasons

        reasons.append("Hairline crack (< 10 px) or low valid depth ratio (< 80%).")
        reasons.append("Signal near or below passive stereo disparity noise floor.")
        return "UNRELIABLE", reasons

    def predict(
        self,
        rgb_input: Union[str, np.ndarray],
        depth_input: Union[str, np.ndarray],
        mask_input: Union[str, np.ndarray],
    ) -> FinalPredictionResult:
        """
        Executes end-to-end inference on an RGB-D-mask sample.
        """
        rgb = self._load_image_if_needed(rgb_input, is_color=True)
        depth = self._load_image_if_needed(depth_input, is_color=False)
        mask = self._load_image_if_needed(mask_input, is_color=False)

        rgb_val, depth_val, mask_val = self.validate_inputs(rgb, depth, mask)

        feat_dict, valid_ratio, surf_noise, inlier_ratio = self.extract_features(
            rgb_val, depth_val, mask_val
        )

        feat_vector = np.array([[feat_dict[fname] for fname in self.feature_names]], dtype=float)
        feat_scaled = self.scaler.transform(feat_vector)
        raw_pred = float(self.model.predict(feat_scaled)[0])

        int_80 = (round(raw_pred - self.q80, 2), round(raw_pred + self.q80, 2))
        int_90 = (round(raw_pred - self.q90, 2), round(raw_pred + self.q90, 2))
        int_95 = (round(raw_pred - self.q95, 2), round(raw_pred + self.q95, 2))

        rel_tier, rel_reasons = self.classify_reliability(
            median_width=feat_dict["median_width_px"],
            valid_depth_ratio=valid_ratio,
            surface_noise=surf_noise,
            plane_inlier_ratio=inlier_ratio,
            crack_pixels=int(feat_dict["crack_area_px"]),
        )

        return FinalPredictionResult(
            predicted_relative_depth=round(raw_pred, 2),
            unit="mm",
            interval_80=int_80,
            interval_90=int_90,
            interval_95=int_95,
            reliability=rel_tier,
            reliability_reasons=rel_reasons,
            median_crack_width_px=round(feat_dict["median_width_px"], 2),
            crack_area_px=int(feat_dict["crack_area_px"]),
            valid_depth_ratio=round(valid_ratio, 4),
            surface_noise_sigma=round(surf_noise, 3),
            plane_inlier_ratio=round(inlier_ratio, 3),
            calibration_state="UNCALIBRATED",
            features={k: round(v, 4) for k, v in feat_dict.items()},
        )
