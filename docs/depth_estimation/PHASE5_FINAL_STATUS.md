# Crack Detection — Final Software Status

## Project scope

The project operates under a permanent **Software-Only Scope** for RGB-D concrete crack analysis and relative sensor-depth prediction. All physical hardware dependencies, physical concrete test specimens, physical depth gauges, physical calibration procedures, and physical millimeter depth predictions are removed from the project scope. The software estimates relative crevice recess in dimensionless sensor-depth disparity units from raw optical observations.

---

## Dataset

- **Captures:** Exactly 508 original unaugmented RGB-D concrete crack captures from the Concrete Crack RGB-D dataset.
- **RGB Imagery:** High-resolution 3-channel color imagery ($1920 \times 1080 \times 3$, `uint8`).
- **Depth Imagery:** 1-channel 8-bit depth maps ($1920 \times 1080$, `uint8`, $[0, 255]$ discrete disparity counts stored as JPEG).
- **Crack Masks:** Binary segmentation masks ($1920 \times 1080$, `uint8`, where $1 = \text{crack interior}$, $0 = \text{intact concrete}$).
- **Augmentation Handling:** While augmented variants exist in raw folders, all 508 captures are strictly grouped by unique original capture stem (`stem`). Augmented copies are excluded from independent cross-validation splits to prevent optimistic data leakage.
- **Evaluation Methodology:** 5-Fold Grouped Cross-Validation (`GroupKFold` / `KFold` by stem) ensuring all frames from a physical location reside strictly in either train or test folds.

---

## Target

The target variable is the **median relative optical recess** ($\Delta Z$):
$$\Delta Z = \text{Median}_{(x, y) \in \text{Crack}} \left[ Z_{\text{surface}}(x, y) - Z_{\text{crevice}}(x, y) \right]$$
where $Z_{\text{surface}}$ is the local planar datum fitted by RANSAC on the intact annular concrete ring ($5\text{--}20\text{ px}$) outside the crack, and $Z_{\text{crevice}}$ is the bilateral-filtered depth within the crack interior.

> [!CAUTION]
> **This target is not physical crack depth in millimeters.**  
> It represents optical disparity displacement relative to the local concrete surface datum in dimensionless sensor units. No physical millimeter ground truth exists.

---

## Model

The exact verified final model is:
- **Model Family:** `ExtraTreesRegressor` (Extremely Randomized Trees)
- **Hyperparameters:**
  - `n_estimators`: 100 trees
  - `max_depth`: 8 levels
  - `min_samples_leaf`: 2 samples
  - `max_features`: `"sqrt"` (5 candidate features per split)
  - `random_state`: 42 (deterministic reproducibility)
  - `n_jobs`: -1 (parallel CPU execution)
- **Feature Preprocessing:** `StandardScaler` fitted on the 508-sample training population.
- **Feature Selection:** Top-25 independent features ranked by permutation importance stability selection:
  `hydraulic_radius_proxy`, `median_width_px`, `orientation_deg`, `log_median_width`, `mean_width_px`, `aperture_roughness_ratio`, `max_width_px`, `crack_area_px`, `compactness`, `p75_width_px`, `p25_width_px`, `plane_inlier_ratio`, `tortuosity`, `bbox_width_px`, `aspect_width_ratio`, `mean_luminance_contrast`, `local_surface_median`, `rgb_ring_mean_intensity`, `contrast_roughness_product`, `branch_points`, `rgb_edge_gradient`, `rgb_contrast_ratio`, `skeleton_length_px`, `branch_density_per_1kpx`, `main_component_ratio`.

---

## Validation

- **Evaluation Methodology:** 5-Fold GroupKFold partitioned strictly by original capture stem (`stem`).
- **Leakage Prevention:** 100% verified quarantine of all 15 target-derived features (`raw_median_depth`, `local_snr`, `volumetric_void_index`, `relative_recess_mean/p90/p75/min/max/std/iqr`, `deterministic_recess_units`).
- **Authoritative Generalization Metrics (Out-of-Fold Cross-Validation):**
  - **$R^2$:** **$0.428 \pm 0.090$**
  - **MAE:** **$9.161 \pm 1.368$ sensor units**
  - **RMSE:** **$14.278 \pm 2.000$ sensor units**
  - **MedAE:** **$5.456$ sensor units**
- **Baseline Comparison:**
  - Beats naive Mean Baseline ($12.712$ MAE) by **$-27.9\%$ MAE**.
  - Beats Univariate Width Baseline ($9.870$ MAE, $R^2 = 0.316$) by **$-7.2\%$ MAE** ($+35.4\%$ relative $R^2$).
- **Final Fit Training Performance:** $R^2 = 0.7555$, $\text{MAE} = 6.253$ sensor units. (Documented strictly as training fit; never substituted for generalization accuracy).

---

## Uncertainty

Software uncertainty is quantified using **Split Conformal Prediction** calibrated on out-of-fold grouped absolute residuals from the 508 captures:
$$\mathcal{C}_{1-\alpha}(\mathbf{x}) = \left[ \widehat{\Delta Z}(\mathbf{x}) - q_{1-\alpha}, \; \widehat{\Delta Z}(\mathbf{x}) + q_{1-\alpha} \right]$$
- **80% Prediction Interval:** $\pm 12.98$ sensor units (Empirical Coverage: **$79.9\%$**)
- **90% Prediction Interval:** $\pm 22.35$ sensor units (Empirical Coverage: **$90.0\%$**)
- **95% Prediction Interval:** $\pm 32.67$ sensor units (Empirical Coverage: **$94.9\%$**)

Intervals are distribution-free, expand monotonically with nominal coverage, and are expressed strictly in dimensionless sensor-depth units.

---

## Robustness

The frozen champion software model was evaluated under 6 standardized image and sensor perturbations:
- Image rotations ($90^\circ, 180^\circ, 270^\circ$): $\text{MAE} = 9.168$ ($+0.08\%$)
- Gaussian image blur ($5\times 5$): $\text{MAE} = 9.175$ ($+0.15\%$)
- Brightness shifts ($\pm 20\%$): $\text{MAE} = 9.162$ ($+0.01\%$)
- Horizontal & vertical flips: $\text{MAE} = 9.161$ ($0.00\%$)
- Salt & pepper noise ($1\%$): $\text{MAE} = 9.210$ ($+0.53\%$)
- Gaussian sensor noise ($\sigma = 2.0$): $\text{MAE} = 9.245$ ($+0.92\%$)

Degradation across all perturbations is strictly $< 1.5\%$, confirming that the model resides securely in the `HIGH` robustness tier.

---

## Limitations

1. **No Physical Crack-Depth Ground Truth:** Public datasets contain no certified contact gauge or microscopic depth labels.
2. **Depth is Sensor-Relative:** Predictions are dimensionless relative optical recess counts ($\Delta Z$), not physical millimeters.
3. **Narrow Cracks Poorly Observable:** For cracks $< 10\text{ px}$ (26.6% of dataset), the recess signal ($0.54\text{--}1.13$ units) is submerged below the sensor's physical noise floor ($\sigma \approx 2.24$ units).
4. **Severe Depth Inversion Artifacts:** $28.9\%$ of dataset captures ($147/508$) exhibit negative optical recess ($\Delta Z < 0$) due to passive stereo shadow clipping in narrow crevices.
5. **Information Ceiling Observed Around $R^2 \approx 0.43$:** 2D surface features cannot observe internal crevice depths beneath the surface; the remaining $57.2\%$ of variance represents aleatoric noise and unobservable 3D geometry.
6. **No Physical mm Calibration:** The system does not possess a verified physical calibration curve.

---

## 70% objective

> The target of R² ≥ 0.70 was not achieved. Under the tested leakage-controlled feature sets, target formulations, model families, and grouped-validation methodology, the best observed generalization performance was approximately R² ≈ 0.43.

---

## Calibration

- **System State:** `UNCALIBRATED`
- **Physical Prediction Lock:** `Physical millimeter prediction is disabled.` Calling `predict_depth_mm()` strictly raises `NotCalibratedError`.

---

## Final decision

```
SOFTWARE PROTOTYPE COMPLETE — PERFORMANCE LIMITED
```
The software prototype is complete, fully validated, reproducible, and packaged with production CLI and batch inference interfaces. Generalization performance is bounded at $R^2 \approx 0.428 \pm 0.090$ under strict zero-leakage conditions.
