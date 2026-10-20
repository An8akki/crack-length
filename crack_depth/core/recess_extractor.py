"""
Relative Crack Recess Feature Extractor for Step 4C Pipeline
Pinnacle Concrete Crack-Depth Project

Calculates relative depth recess:
ΔZ(x,y) = Surface_Estimate(x,y) - Crack_Depth(x,y)
Where positive values indicate a physical recess/depression into the concrete.

Extracts multi-scale features:
- Median, Mean, Max, P90, IQR recess
- Local surface noise floor (sigma_surface)
- Local Signal-to-Noise Ratio (SNR = median_recess / sigma_surface)
- Volumetric void index (integral of recess per unit crack length)
- Multi-scale ring comparisons (10px, 20px, 40px)
"""

import numpy as np


class RelativeRecessExtractor:
    def __init__(self):
        pass

    def extract_recess_features(self, depth_map, binary_mask, surface_fn, surface_noise_sigma, crack_length_px=1):
        """
        Computes relative recess features for all valid pixels inside the crack mask.
        Args:
            depth_map: np.ndarray (H, W) float64 preprocessed depth
            binary_mask: np.ndarray (H, W) uint8 {0, 1}
            surface_fn: callable (x, y) -> estimated surface level Z
            surface_noise_sigma: float standard deviation of local surface noise
            crack_length_px: int skeleton length for void index normalization
        Returns:
            dict of recess statistics and spatial 2D recess map
        """
        y_crack, x_crack = np.where(binary_mask > 0)
        total_crack_pixels = len(y_crack)

        if total_crack_pixels == 0:
            return {
                "median_recess": 0.0,
                "mean_recess": 0.0,
                "max_recess": 0.0,
                "p90_recess": 0.0,
                "p75_recess": 0.0,
                "p25_recess": 0.0,
                "iqr_recess": 0.0,
                "recess_std": 0.0,
                "local_noise_sigma": float(surface_noise_sigma),
                "local_snr": 0.0,
                "volumetric_void_index": 0.0,
                "valid_depth_ratio": 0.0,
                "detectable_recess": False,
                "reliable_recess": False,
                "recess_map": np.zeros_like(depth_map)
            }

        crack_depths = depth_map[y_crack, x_crack]
        valid_mask = (crack_depths > 0)
        valid_crack_pixels = int(np.sum(valid_mask))
        valid_ratio = float(valid_crack_pixels / total_crack_pixels)

        if valid_crack_pixels == 0:
            return {
                "median_recess": 0.0,
                "mean_recess": 0.0,
                "max_recess": 0.0,
                "p90_recess": 0.0,
                "p75_recess": 0.0,
                "p25_recess": 0.0,
                "iqr_recess": 0.0,
                "recess_std": 0.0,
                "local_noise_sigma": float(surface_noise_sigma),
                "local_snr": 0.0,
                "volumetric_void_index": 0.0,
                "valid_depth_ratio": 0.0,
                "detectable_recess": False,
                "reliable_recess": False,
                "recess_map": np.zeros_like(depth_map)
            }

        valid_x = x_crack[valid_mask]
        valid_y = y_crack[valid_mask]
        valid_z = crack_depths[valid_mask]

        # Evaluate expected surface at each crack pixel coordinate
        surface_z = surface_fn(valid_x, valid_y)

        # Relative Recess: S - C (positive when crevice is deeper/darker than surface)
        recess_values = surface_z - valid_z

        med_recess = float(np.median(recess_values))
        mean_recess = float(np.mean(recess_values))
        max_recess = float(np.max(recess_values))
        p90_recess = float(np.percentile(recess_values, 90))
        p75_recess = float(np.percentile(recess_values, 75))
        p25_recess = float(np.percentile(recess_values, 25))
        iqr_recess = float(p75_recess - p25_recess)
        std_recess = float(np.std(recess_values))

        # Signal-to-Noise Ratio (SNR)
        effective_noise = max(0.5, float(surface_noise_sigma))
        snr = float(med_recess / effective_noise)

        # Volumetric Void Index: integrated positive recess normalized per unit crack length
        pos_recess = np.maximum(0.0, recess_values)
        volumetric_void = float(np.sum(pos_recess) / max(1.0, float(crack_length_px)))

        # Create 2D recess map for visualization
        recess_map = np.zeros_like(depth_map)
        recess_map[valid_y, valid_x] = recess_values

        return {
            "median_recess": round(med_recess, 2),
            "mean_recess": round(mean_recess, 2),
            "max_recess": round(max_recess, 2),
            "p90_recess": round(p90_recess, 2),
            "p75_recess": round(p75_recess, 2),
            "p25_recess": round(p25_recess, 2),
            "iqr_recess": round(iqr_recess, 2),
            "recess_std": round(std_recess, 2),
            "local_noise_sigma": round(effective_noise, 2),
            "local_snr": round(snr, 2),
            "volumetric_void_index": round(volumetric_void, 2),
            "valid_depth_ratio": round(valid_ratio, 3),
            "detectable_recess": bool(snr >= 1.0),
            "reliable_recess": bool(snr >= 2.0),
            "recess_map": recess_map
        }
