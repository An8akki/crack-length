"""Empirical machine learning measurement error predictor and confidence interval model."""

from typing import Dict, List, Optional, Tuple
import os
import joblib
import numpy as np

FEATURE_NAMES = [
    "camera_distance_mm",
    "camera_angle_deg",
    "planarity_score",
    "reprojection_error_px",
    "is_planar_flag",
    "method_disagreement_ratio",
    "lighting_penalty",
]


class MeasurementReliabilityModel:
    """Predicts expected physical measurement error and reliability from observable diagnostics."""

    def __init__(self, model=None):
        self.model = model
        self.is_trained = model is not None

    def _extract_features(self, diagnostics: Dict) -> np.ndarray:
        """Extract standardized numeric feature vector from observable diagnostics."""
        dist = float(diagnostics.get("camera_distance_mm", 1000.0))
        angle = float(diagnostics.get("camera_angle_deg") or diagnostics.get("camera_yaw_deg", 0.0))
        planarity = float(diagnostics.get("planarity_score", 90.0))
        reproj = float(diagnostics.get("reprojection_error_px") or 0.0)
        is_planar = 1.0 if diagnostics.get("is_planar", True) else 0.0

        # Method disagreement
        len_2d = diagnostics.get("length_2d_mm")
        len_planar = diagnostics.get("length_planar_mm")
        len_3d = diagnostics.get("length_3d_mm")

        disagree = 1.0
        if len_planar and len_2d and len_2d > 0:
            disagree = max(disagree, abs(len_planar - len_2d) / len_2d)
        if len_3d and len_planar and len_planar > 0:
            disagree = max(disagree, abs(len_3d - len_planar) / len_planar)

        light_pen = 0.0
        cond = str(diagnostics.get("lighting_condition", "normal")).lower()
        if cond in ("shadow", "moisture"): light_pen = 0.5
        elif cond in ("glare", "low_light"): light_pen = 1.0

        feats = np.array([dist, angle, planarity, reproj, is_planar, disagree, light_pen], dtype=np.float32)
        return feats.reshape(1, -1)

    def train(self, records: List) -> Dict:
        """Train the error predictor on experimental trial records."""
        try:
            from sklearn.ensemble import RandomForestRegressor
        except ImportError:
            return {"trained": False, "reason": "scikit-learn is required"}

        X_list = []
        y_list = []

        for r in records:
            diag = {
                "camera_distance_mm": r.camera_distance_mm,
                "camera_angle_deg": max(abs(r.camera_yaw_deg), abs(r.camera_pitch_deg)),
                "planarity_score": r.planarity_score,
                "reprojection_error_px": r.reprojection_error_px or 0.0,
                "is_planar": r.is_planar,
                "length_2d_mm": r.estimate_2d_mm,
                "length_planar_mm": r.estimate_planar_mm,
                "length_3d_mm": r.estimate_3d_mm,
                "lighting_condition": r.lighting_condition,
            }
            # Target is the lowest achieved percentage error among active methods
            errs = [e for e in (r.error_planar_pct, r.error_3d_pct, r.error_2d_pct) if e is not None]
            if errs:
                X_list.append(self._extract_features(diag)[0])
                y_list.append(min(errs))

        if len(X_list) < 5:
            return {"trained": False, "reason": "Insufficient samples (minimum 5 required)"}

        X = np.array(X_list)
        y = np.array(y_list)

        reg = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=42)
        reg.fit(X, y)
        self.model = reg
        self.is_trained = True

        importances = dict(zip(FEATURE_NAMES, [round(float(v), 3) for v in reg.feature_importances_]))
        return {
            "trained": True,
            "samples_used": len(X),
            "feature_importance": importances,
        }

    def predict_error(self, measured_length_mm: float, diagnostics: Dict) -> Dict:
        """Predict expected measurement error and confidence interval for a crack observation."""
        if not self.is_trained or self.model is None:
            # Analytical heuristic estimate when ML model has not yet been fitted
            angle = float(diagnostics.get("camera_angle_deg") or diagnostics.get("camera_yaw_deg", 0.0))
            reproj = float(diagnostics.get("reprojection_error_px") or 0.5)
            planarity = float(diagnostics.get("planarity_score", 90.0))

            base_pct = 2.0 + (angle / 15.0) * 1.5 + reproj * 0.8 + max(0.0, (85.0 - planarity) * 0.25)
            pred_pct = min(40.0, max(0.8, base_pct))
            is_ml = False
        else:
            feats = self._extract_features(diagnostics)
            pred_pct = float(np.clip(self.model.predict(feats)[0], 0.4, 60.0))
            is_ml = True

        delta_mm = measured_length_mm * (pred_pct / 100.0)

        # Classify Reliability
        if pred_pct < 4.0:
            rel_label = "HIGH"
        elif pred_pct < 12.0:
            rel_label = "MEDIUM"
        elif pred_pct < 25.0:
            rel_label = "LOW"
        else:
            rel_label = "UNRELIABLE"

        ci_lower = max(0.0, measured_length_mm - 1.96 * delta_mm)
        ci_upper = measured_length_mm + 1.96 * delta_mm

        return {
            "measured_length_mm": round(measured_length_mm, 2),
            "expected_error_mm": round(delta_mm, 2),
            "expected_relative_error_pct": round(pred_pct, 2),
            "confidence_interval_95": [round(ci_lower, 2), round(ci_upper, 2)],
            "measurement_reliability": rel_label,
            "is_empirically_trained": is_ml,
            "confidence_interval_formatted": f"{measured_length_mm:.1f} mm ± {delta_mm:.1f} mm",
        }


def train_reliability_model(records: List, save_path: str = "outputs/reliability_model.joblib") -> Tuple[MeasurementReliabilityModel, Dict]:
    """Train and persist a MeasurementReliabilityModel."""
    model = MeasurementReliabilityModel()
    info = model.train(records)
    if info.get("trained"):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model.model, save_path)
    return model, info


def predict_measurement_reliability(
    diagnostics_or_length,
    diagnostics: Optional[Dict] = None,
    model_path: str = "outputs/reliability_model.joblib"
) -> Dict:
    """Convenience helper to predict measurement reliability using trained model or heuristic fallback."""
    if isinstance(diagnostics_or_length, dict):
        diag = diagnostics_or_length
        measured_length_mm = float(
            diag.get("length_planar_mm") or
            diag.get("length_3d_mm") or
            diag.get("length_2d_mm") or
            diag.get("measured_length_mm") or
            100.0
        )
    else:
        measured_length_mm = float(diagnostics_or_length)
        diag = diagnostics or {}

    loaded_model = None
    if os.path.exists(model_path):
        try:
            loaded_model = joblib.load(model_path)
        except Exception:
            loaded_model = None
    rm = MeasurementReliabilityModel(loaded_model)
    return rm.predict_error(measured_length_mm, diag)
