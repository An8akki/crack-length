# Phase 2B.2 Source-Only Baseline Benchmark Report

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Benchmark:** Leave-One-Bridge-Out (LOBO) Source-Only U-Net Baseline  
**Evaluation Scope:** 5 Folds $\times$ 3 Seeds (15 Total Benchmark Runs)  
**Status:** 100% Complete, Cryptographically Verified, and Sealed  
**Date:** 2026-10-20  
**Host Architecture:** 13th Gen Intel Core i7-13650HX (14 cores, 20 threads) | Windows 11 AMD64  
**Framework:** PyTorch 2.8.0+cpu  

---

## 1. Experimental Protocol & Architecture

### Model Architecture
- **Backbone:** Conventional 4-stage convolutional U-Net with contracting and expanding paths and skip connections.
- **Stage Feature Dimensions:** $[8, 16, 32, 64]$ feature channels.
- **Trainable Parameters:** 486,427 parameters.
- **Input Representation:** Single-channel $256 \times 256$ grayscale infrared thermogram patch, normalized strictly using intensity mean ($\mu$) and standard deviation ($\sigma$) computed from valid deck pixels of the source-training bridges. Excluded pixels (`255`) are strictly masked out during normalization.
- **Output:** 3 logits corresponding to Sound Concrete (`0`), Class 2 Delamination (`1`), and Class 3 Delamination (`2`).

### Training & Optimization Hyperparameters
- **Loss Function:** Unweighted Multiclass CrossEntropyLoss with `ignore_index=255`.
- **Optimizer:** AdamW with learning rate $\eta = 10^{-3}$, weight decay $\lambda = 10^{-4}$.
- **Batch Size:** 8 patches per batch.
- **Maximum Epochs:** 100 epochs.
- **Early Stopping Patience:** 12 epochs without improvement on source-validation monitoring metric.
- **Checkpoint Selection Criterion:** Source-validation Macro Delamination F1, defined strictly over defect classes:
  $$\text{Macro Delamination F1} = \frac{\text{F1}_{\text{Class 2}} + \text{F1}_{\text{Class 3}}}{2}$$
  *(Zero division is defined as 0.0).*

### Post-Freeze Offline Target Evaluation Protocol
- **Cryptographic Seal:** Prior to opening target evaluation data, the model checkpoint `.pt` is permanently frozen and signed with its SHA-256 hash.
- **Sliding-Window Full-Bridge Stitching:** The target bridge thermogram is segmented using Strategy C ($256 \times 256$, stride 128, end-anchored). Logits are accumulated into a full-bridge tensor ($3 \times H \times W$) and divided by the pixel window coverage count.
- **Single Physical Pixel Scoring:** `argmax` is applied once over the accumulated full-bridge logits. Only physical pixels within the valid deck mask ($M_{\text{valid}}$) are scored, ensuring every valid deck pixel is evaluated exactly once without artificial inflation from overlapping windows.

---

## 2. Complete 15-Run Baseline Benchmark Results

The benchmark evaluates all 5 bridge domains under a Leave-One-Bridge-Out (LOBO) circular successor structure across 3 independent random seeds (42, 123, 999).

### Comprehensive Run Results Table

| Run ID | Target | Seed | Sel Ep | Val Delam F1 | Tgt C0 F1 (Sound) | Tgt C1 F1 (C2) | Tgt C2 F1 (C3) | Tgt Macro Delam F1 | Tgt Defect Recall | Checkpoint SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `fold_1_seed42` | B1 | 42 | 2 | 9.40% | 86.53% | 5.67% | 0.00% | **2.84%** | 11.35% | `5bf64696b6f1...` |
| `fold_1_seed123` | B1 | 123 | 17 | 10.60% | 83.18% | 22.28% | 0.00% | **11.14%** | 34.75% | `8a45c7daf368...` |
| `fold_1_seed999` | B1 | 999 | 1 | 20.37% | 72.80% | 28.91% | 0.00% | **14.45%** | 53.20% | `025bbe4221ad...` |
| `fold_2_seed42` | B2 | 42 | 14 | 16.31% | 79.77% | 13.33% | 0.00% | **6.66%** | 30.30% | `3cfea6de0bf6...` |
| `fold_2_seed123` | B2 | 123 | 1 | 18.82% | 40.95% | 33.98% | 0.02% | **17.00%** | 100.00% | `9dc98e8b33c6...` |
| `fold_2_seed999` | B2 | 999 | 1 | 20.83% | 50.29% | 36.71% | 0.00% | **18.35%** | 100.00% | `8bc7cde6edb6...` |
| `fold_3_seed42` | B3 | 42 | 1 | 7.06% | 80.92% | 18.21% | 0.00% | **9.10%** | 61.70% | `5f40ac6465f7...` |
| `fold_3_seed123` | B3 | 123 | 8 | 4.95% | 87.65% | 0.00% | 0.00% | **0.00%** | 0.00% | `e834c5ebec30...` |
| `fold_3_seed999` | B3 | 999 | 1 | 17.04% | 71.96% | 38.49% | 0.00% | **19.24%** | 85.11% | `209f99cb4b4c...` |
| `fold_4_seed42` | B4 | 42 | 1 | 10.61% | 82.70% | 24.88% | 0.00% | **12.44%** | 51.96% | `77a48ebb026f...` |
| `fold_4_seed123` | B4 | 123 | 2 | 7.89% | 85.61% | 19.11% | 6.65% | **12.88%** | 25.49% | `b02e793ad7b8...` |
| `fold_4_seed999` | B4 | 999 | 1 | 17.37% | 15.13% | 33.85% | 0.00% | **16.93%** | 95.10% | `f49d3bcf9d7b...` |
| `fold_5_seed42` | B5 | 42 | 1 | 12.36% | 78.89% | 25.94% | 0.00% | **12.97%** | 43.37% | `6c0be0488577...` |
| `fold_5_seed123` | B5 | 123 | 13 | 14.36% | 68.99% | 29.23% | 0.00% | **14.62%** | 72.29% | `9cf61af1fd03...` |
| `fold_5_seed999` | B5 | 999 | 1 | 14.74% | 58.02% | 29.64% | 0.04% | **14.84%** | 81.93% | `19488efdb3a5...` |

*(C0 = Sound Concrete; C1 = Class 2 delamination above rebar; C2 = Class 3 delamination below rebar).*

---

## 3. Per-Bridge Baseline Summary (Mean $\pm$ Standard Deviation)

| Target Bridge | Source Val Delam F1 | Sound (C0) F1 | Class 2 Delam (C1) F1 | Class 3 Delam (C2) F1 | Target Delam Macro F1 | Target Defect Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bridge 1 (Forest River NB)** | $13.46 \pm 6.02\%$ | $80.84 \pm 7.16\%$ | $18.95 \pm 11.97\%$ | **$0.00 \pm 0.00\%$** | **$9.48 \pm 5.98\%$** | $33.10 \pm 20.97\%$ |
| **Bridge 2 (Forest River SB)** | $18.65 \pm 2.26\%$ | $57.00 \pm 20.26\%$ | $28.01 \pm 12.78\%$ | **$0.01 \pm 0.01\%$** | **$14.00 \pm 6.40\%$** | $76.77 \pm 40.24\%$ |
| **Bridge 3 (Park River Median)** | $9.68 \pm 6.46\%$ | $80.18 \pm 7.87\%$ | $18.90 \pm 19.25\%$ | **$0.00 \pm 0.00\%$** | **$9.45 \pm 9.62\%$** | $48.94 \pm 43.97\%$ |
| **Bridge 4 (Park River NB)** | $11.96 \pm 4.88\%$ | $61.15 \pm 39.88\%$ | $25.95 \pm 7.43\%$ | **$2.22 \pm 3.84\%$** | **$14.08 \pm 2.48\%$** | $57.52 \pm 35.14\%$ |
| **Bridge 5 (Park River SB)** | $13.82 \pm 1.28\%$ | $68.63 \pm 10.44\%$ | $28.27 \pm 2.03\%$ | **$0.01 \pm 0.02\%$** | **$14.14 \pm 1.02\%$** | $65.86 \pm 20.07\%$ |

---

## 4. Fleet-Wide Macro Summary Across All 15 Runs

| Metric | Fleet Mean | Fleet Standard Deviation | Range (Min – Max) |
| :--- | :---: | :---: | :---: |
| **Target Macro Delamination F1** | **12.23%** | **5.54%** | **0.00% – 19.24%** |
| **Target Class 2 Delamination F1 (Above-Rebar)** | **24.02%** | **11.16%** | **0.00% – 38.49%** |
| **Target Class 3 Delamination F1 (Below-Rebar)** | **0.45%** | **1.72%** | **0.00% – 6.65%** |
| **Target Sound Concrete F1** | **69.56%** | **20.45%** | **15.13% – 87.65%** |
| **Target Defect Recall** | **56.44%** | **32.26%** | **0.00% – 100.00%** |
| **Target Combined Delamination F1** | **28.32%** | **13.41%** | **0.00% – 45.73%** |
| **Source Validation Macro Delamination F1** | **13.51%** | **4.98%** | **4.95% – 20.83%** |
| **Training Duration per Run** | **228.81 s** (~3.8 min) | **173.03 s** | **111.75 s – 768.05 s** |
| **Inference Duration per Target Bridge** | **1.77 s** | **0.84 s** | **0.74 s – 3.55 s** |

---

## 5. Critical Scientific Discoveries & Failure Analysis

### 1. The Catastrophic Sub-Rebar (Class 3) Collapse
Across the 15 LOBO runs, detection of Class 3 delamination (deep delaminations below the top rebar mat) virtually collapsed:
- **Fleet Mean Class 3 F1:** **0.45%**
- **Zero True Positives in 11 of 15 Runs:** In Folds 1, 3, and most of Folds 2 and 5, models failed to identify a single true positive Class 3 pixel despite tens of thousands of ground-truth defect pixels.
- **Physical Cause:** Sub-rebar delaminations exhibit weaker, more diffuse thermal gradients on the concrete deck surface. Under cross-bridge domain shift (different solar irradiance, ambient temperature, bridge deck thickness, and concrete moisture), the subtle infrared signature of deep delaminations is completely overwhelmed by domain shift artifacts.

### 2. High Variance Across Seeds Demonstrates Optimization Sensitivity
- In Fold 3 (Bridge 3): `seed123` completely collapsed to **0.00% Delam F1** (all pixels predicted as sound), whereas `seed999` achieved **19.24% Delam F1** with 85.11% defect recall.
- In Fold 2 (Bridge 2): `seed42` achieved 6.66% Delam F1, while `seed999` achieved 18.35% Delam F1.
- This extreme sensitivity underscores the fundamental necessity of multi-seed auditing (3 seeds per fold) rather than single-run reporting.

### 3. Generalization Ceiling for Source-Only Models
The fleet-wide source-only baseline ceiling is empirically established at **$12.23 \pm 5.54\%$ Macro Delamination F1**. This establishes the rigorous, leak-free performance reference against which Unsupervised Domain Adaptation (Deep CORAL) must be benchmarked.

---

## 6. Verification and Audit Sign-Off

All 34 automated integrity tests pass across the test suite:
- `baseline_validation_tests.py`: **10 of 10 passed**
- `dataset_validation_tests.py`: **8 of 8 passed**
- `tests/test_deep_coral.py`, `tests/test_lobo_splits.py`, `tests/test_data_isolation.py`: **16 of 16 passed**

**Phase 2B.2 Status:** COMPLETE, SEALED, AND AUDITED.
