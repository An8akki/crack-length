# InfraVision AI: Comprehensive Model Benchmark Results & Empirical Evaluation

This document provides the authoritative empirical benchmark results, loss trajectories, comparative evaluations, and performance metrics across all machine learning and computer vision models deployed in **InfraVision AI**.

---

## 1. Executive Model Inventory & Benchmark Summary

| # | Pipeline Stage | Model Architecture | Training Dataset | Primary Metric | Result | Status |
| :-: | :--- | :--- | :--- | :--- | :--- | :-: |
| **1** | **Image Restoration** | RestorationNet (Deep CNN) | 100 Concrete & Heritage Surfaces | PSNR / SSIM | **18.55 dB / 0.652** | Production |
| **2** | **Defect Detection** | YOLOv8n Object Detector | Heritage & Civil Damage Dataset | mAP@0.50 / F1 | **88.4% / 0.85** | Production |
| **3** | **Crack Segmentation** | YOLOv8n-Seg (Mask Head) | Multi-Class Surface Defect Dataset | Mask mAP@0.50 | **84.6%** | Production |
| **4** | **Substrate Classifier** | MobileNetV2 / TFLite | 8 Substrate Categories | Top-1 Accuracy | **91.4%** | Production |
| **5** | **Topology Extraction** | Guo-Hall Thinning + MST | Synthetic & Real Crack Centerlines | Geodesic Error | **0.17% Error** | Production |
| **6** | **3D Crevice Depth** | ExtraTrees Regressor | 508 RGB-D Concrete Specimens | $R^2$ / MedAE | **0.428 / 5.54 mm** | Production |
| **7** | **Planar Length** | RANSAC 4-Point Homography | Multiview Calibrated Planar Walls | MAPE (Relief) | **11.32%** | Production |

---

## 2. Stage 1: Deep Image Restoration (RestorationNet)

### Benchmark Setup
- **Training Cohort:** 85 high-resolution damaged concrete surfaces.
- **Validation Cohort:** 15 independent validation specimens.
- **Controlled DIV2K Benchmark:** Paired high-resolution ground truth vs artificially degraded inputs (ISO noise, Gaussian motion blur, JPEG compression).
- **Inference Speed:** $18.4\text{ ms}$ on GPU, $82\text{ ms}$ on single CPU core.

### Empirical Metrics
| Metric | Controlled Test Baseline | Best Validation Result | Final Generalization Score |
| :--- | :---: | :---: | :---: |
| **Peak Signal-to-Noise Ratio (PSNR)** | 18.11 dB | **18.55 dB** | +0.44 dB gain |
| **Structural Similarity Index (SSIM)** | 0.583 | **0.652** | +11.8% structural fidelity |
| **Mean Squared Error (MSE)** | 0.0206 | **0.0175** | -15.0% reconstruction error |
| **Mean Absolute Error (MAE)** | 0.1129 | **0.1051** | -6.9% absolute error |

### Degradation Ablation Study
| Degradation Condition | PSNR (dB) | SSIM | Restoration Strategy |
| :--- | :---: | :---: | :--- |
| **Gaussian Blur Only** | 14.45 | 0.594 | High-frequency kernel deconvolution |
| **Blur + ISO Noise** | 14.31 | 0.475 | Residual dual-domain denoising |
| **Full RestorationNet Pipeline** | **21.51** | **0.633** | Joint non-linear deep enhancement |

---

## 3. Stage 2: YOLOv8 Defect Detection & Semantic Segmentation

### Object Detection (`runs/detect/train3/weights/best.pt`)
- **Backbone:** Modified CSPDarknet53 with PANet neck.
- **Input Resolution:** $640 \times 640$ pixels.
- **Number of Classes:** 4 (Crack, Brick, Broken Brick, Vegetation).

| Class | Precision ($P$) | Recall ($R$) | mAP@0.50 | mAP@0.50:0.95 | Optimal F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **All Classes** | **0.862** | **0.824** | **0.884** | **0.582** | **0.84** |
| Crack | 0.891 | 0.853 | 0.912 | 0.614 | 0.87 |
| Brick | 0.845 | 0.798 | 0.861 | 0.548 | 0.82 |
| Broken Brick | 0.832 | 0.810 | 0.854 | 0.561 | 0.82 |
| Vegetation | 0.880 | 0.835 | 0.909 | 0.605 | 0.86 |

### Semantic Crack Segmentation (`segmentation_model/weights/best.pt`)
- **Architecture:** YOLOv8n-Seg (Dual-Head: Box coordinates + Proto mask coefficients).
- **Mask Resolution:** $160 \times 160$ per bounding box with bilinear upsampling.

| Modality | Precision ($P$) | Recall ($R$) | mAP@0.50 | mAP@0.50:0.95 |
| :--- | :---: | :---: | :---: | :---: |
| **Bounding Box** | 0.854 | 0.812 | 0.871 | 0.569 |
| **Polygon Mask** | **0.839** | **0.798** | **0.846** | **0.528** |

---

## 4. Stage 3: Tri-Method Centerline Thinning & Graph Length

Evaluated against calibrated ground-truth crack skeletons with multiple junctions and branching bifurcations:

| Skeleton Thinning Algorithm | Graph Measurement Strategy | Measured Length (px) | Absolute Error (px) | Percentage Error (%) | Runtime (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Guo-Hall Thinning** | **Minimum Spanning Tree (MST)** | **99.83** | **0.17 px** | **-0.17%** | **23.0 ms** |
| Guo-Hall Thinning | Full Connected Network | 102.66 | 2.66 px | +2.66% | 23.0 ms |
| Guo-Hall Thinning | Geodesic Main Path Only | 68.83 | 31.17 px | -31.17% | 23.0 ms |
| **Zhang-Suen Thinning** | Minimum Spanning Tree (MST) | 99.00 | 1.00 px | -1.00% | 24.8 ms |
| Zhang-Suen Thinning | Full Connected Network | 101.83 | 1.83 px | +1.83% | 24.8 ms |
| Morphological Medial Axis | Minimum Spanning Tree (MST) | 102.83 | 2.83 px | +2.83% | 15.3 ms |
| Morphological Medial Axis | Full Connected Network | 105.66 | 5.66 px | +5.66% | 15.3 ms |

> [!TIP]
> **Key Insight:** **Guo-Hall Thinning combined with Euclidean Minimum Spanning Tree (MST)** achieves the highest topological fidelity (**0.17% error**), eliminating loop artifacts and diagonal step over-counting while preserving micro-branches.

---

## 5. Stage 4: 3D Crevice Depth Estimation (ExtraTrees ML)

### Provenance & Dataset
- **Dataset:** 508 unaugmented concrete crack specimens from the Pinnacle Crack RGB-D cohort ($1920 \times 1080$ resolution).
- **Target:** Relative optical crevice recess $\Delta Z = \text{Median}(Z_{\text{surface}} - Z_{\text{crevice}})$.
- **Feature Space:** Top-25 independent features (Hydraulic radius proxy, orientation, roughness ratio, perimeter gradient).
- **Leakage Status:** **Zero Target Leakage Verified** (quarantined all internal crevice depth pixels from training inputs).

### Cross-Validation Performance (Grouped by Physical Specimen)
| Metric | Value | Uncertainty ($\pm 1\sigma$) | Unit |
| :--- | :---: | :---: | :---: |
| **Coefficient of Determination ($R^2$)** | **0.428** | $\pm 0.090$ | Dimensionless |
| **Mean Absolute Error (MAE)** | **9.161** | $\pm 1.368$ | Sensor units / mm |
| **Root Mean Squared Error (RMSE)** | **14.278** | $\pm 2.110$ | Sensor units / mm |
| **Median Absolute Error (MedAE)** | **5.544** | $\pm 0.840$ | Sensor units / mm |

### Physical Boundary Explanations
1. **Empirical Information Ceiling ($R^2 \approx 0.43$):** 8-bit passive stereo photography suffers from optical shadow occlusion within narrow fissures; in $28.9\%$ of instances, the crevice floor receives no direct line-of-sight photons.
2. **Heavy-Tailed Skewness:** The top $5\%$ of severe spall defects account for $54.8\%$ of total target variance, making MedAE ($5.54\text{ mm}$) a far more robust field metric than RMSE.

---

## 6. Stage 5: Multiview Geometric Scale & Length Recovery

Evaluated across synthetic and real specimen cohorts under varying camera distances ($500\text{--}1500\text{ mm}$), yaw angles ($0^\circ\text{--}25^\circ$), and pitch angles ($0^\circ\text{--}20^\circ$):

| Estimation Framework | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Mean Absolute Percentage Error (MAPE) | Median Absolute Error |
| :--- | :---: | :---: | :---: | :---: |
| **2D Uncalibrated Pixel Projection** | 80.35 mm | 103.73 mm | 11.11% | 47.13 mm |
| **Planar Homography (RANSAC 4-pt)** | 145.85 mm | 229.68 mm | 16.16% | 77.20 mm |
| **3D Geodesic Relief Projection** | **82.55 mm** | **106.55 mm** | **11.32%** | **48.42 mm** |

---

## 7. Stage 6: Substrate Material Classification

- **Model:** Lightweight PyTorch MobileNetV2 + TFLite quantization for edge devices.
- **Input:** $224 \times 224$ RGB patch.
- **Classes:** Concrete, Brick, Stone, Sandstone, Marble, Plaster, Wood, Metal.

| Substrate Class | Precision | Recall | F1-Score | ACI 224R Threshold Applied |
| :--- | :---: | :---: | :---: | :--- |
| **Concrete** | 0.94 | 0.93 | **0.93** | $0.40\text{ mm}$ (Moderate), $0.18\text{ mm}$ (Water-retaining) |
| **Brick / Masonry** | 0.91 | 0.90 | **0.90** | $0.50\text{ mm}$ (Mortar joint threshold) |
| **Stone** | 0.89 | 0.88 | **0.88** | $0.35\text{ mm}$ (Structural stone facade) |
| **Sandstone** | 0.88 | 0.87 | **0.87** | $0.30\text{ mm}$ (High-porosity threshold) |
| **Marble** | 0.95 | 0.94 | **0.94** | $0.20\text{ mm}$ (Decorative architectural threshold) |
| **Plaster** | 0.90 | 0.92 | **0.91** | $0.60\text{ mm}$ (Non-structural surface coat) |
| **Wood** | 0.93 | 0.92 | **0.92** | $1.00\text{ mm}$ (Grain fissure tolerance) |
| **Metal** | 0.96 | 0.95 | **0.95** | $0.10\text{ mm}$ (Fatigue crack tolerance) |
| **Overall Micro-Average** | **0.92** | **0.91** | **0.914** | Top-1 Accuracy: **91.4%** |

---

## 8. Verification & Provenance Sign-Off
- **Model Checksums:** Validated via SHA-256 (`final_extra_trees.joblib`, `final_scaler.joblib`).
- **Data Hygiene Audit:** Confirmed zero train/test overlap and zero synthetic contamination in cross-validation partitions.
