# Phase 2B.2 Source-Only Baseline Benchmark Report

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Benchmark:** Leave-One-Bridge-Out (LOBO) Source-Only U-Net Baseline  
**Evaluation Target:** 5 Folds $\times$ 3 Seeds (15 Total Benchmark Runs)  
**Date:** 2026-10-20  

---

## 1. Experimental Protocol & Architecture

### Model Architecture
- **Backbone:** Conventional 4-stage convolutional U-Net with skip connections.
- **Stage Feature Dimensions:** $[8, 16, 32, 64]$ channels.
- **Trainable Parameters:** 486,427 parameters.
- **Input Representation:** Single-channel $256 \times 256$ grayscale infrared thermogram patch, normalized exclusively using intensity mean ($\mu$) and standard deviation ($\sigma$) computed from valid deck pixels of the corresponding source-training bridges. Excluded pixels (`255`) are strictly masked out during normalization.
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
- **Sliding-Window Full-Bridge Stitching:** The target bridge thermogram is segmented using a sliding window ($256 \times 256$, stride 128). Logits are accumulated into a full-bridge tensor ($3 \times H \times W$) and divided by the pixel window coverage count.
- **Single Physical Pixel Scoring:** `argmax` is applied once over the accumulated full-bridge logits. Only physical pixels within the valid deck mask ($M_{\text{valid}}$) are scored, ensuring every valid deck pixel is evaluated exactly once without artificial inflation from overlapping windows.

---

## 2. Completed Baseline Runs: Bridge 1 (Target B1) Evaluation

Fold 1 trains on Bridges B3, B4, and B5 (241 source patches), validates on Bridge B2 (36 patches), and evaluates on held-out Bridge B1 (45 patches, 858,114 physical valid deck pixels).

### Numerical Performance Summary (Bridge 1)

| Run ID | Seed | Selected Epoch | Val Macro Delam F1 | Target C0 F1 (Sound) | Target C1 F1 (Class 2) | Target C2 F1 (Class 3) | Target Macro Delam F1 | Target Defect Recall | Checkpoint SHA-256 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| `fold_1_seed42` | 42 | 2 | 9.40% | 86.53% | 5.67% | **0.00%** | **2.84%** | 11.35% | `5bf64696b6f1...` |
| `fold_1_seed123` | 123 | 17 | 10.60% | 83.18% | 22.28% | **0.00%** | **11.14%** | 34.75% | `8a45c7daf368...` |
| `fold_1_seed999` | 999 | 1 | 20.37% | 72.80% | 28.91% | **0.00%** | **14.45%** | 53.20% | `025bbe4221ad...` |
| **Bridge 1 Mean $\pm$ SD** | — | — | **$13.46 \pm 6.01\%$** | **$80.84 \pm 7.16\%$** | **$18.95 \pm 11.97\%$** | **$0.00 \pm 0.00\%$** | **$9.48 \pm 5.99\%$** | **$33.10 \pm 20.97\%$** | — |

*(Class 1 = Class 2 delamination above rebar; Class 2 = Class 3 delamination below rebar).*

---

## 3. Empirical Confusion Matrices on Target Bridge 1

### Seed 42 Confusion Matrix:
- Sound Concrete (`0`): Actual Sound: 649,428 correct, 13,046 false positive C1, 3 false positive C2.
- Class 2 Delamination (`1`): Actual C1: 147,279 misclassified as Sound, 4,870 correct, 0 as C2.
- Class 3 Delamination (`2`): Actual C2: 41,916 misclassified as Sound, 1,572 misclassified as C1, **0 correct**.

### Seed 123 Confusion Matrix:
- Sound Concrete (`0`): 580,568 correct, 81,909 false positive C1, 0 false positive C2.
- Class 2 Delamination (`1`): 121,315 misclassified as Sound, 30,834 correct, 0 as C2.
- Class 3 Delamination (`2`): 31,626 misclassified as Sound, 11,862 misclassified as C1, **0 correct**.

### Seed 999 Confusion Matrix:
- Sound Concrete (`0`): 439,845 correct, 222,349 false positive C1, 283 false positive C2.
- Class 2 Delamination (`1`): 85,062 misclassified as Sound, 67,087 correct, 0 as C2.
- Class 3 Delamination (`2`): 20,915 misclassified as Sound, 22,573 misclassified as C1, **0 correct**.

---

## 4. Key Scientific Findings from the Source-Only Baseline

1. **Complete Cross-Bridge Failure on Deep Delamination (Class 3):**
   Across all three seeds on Bridge 1, **Class 3 delamination F1 was exactly 0.00%**. Source-only training completely fails to detect sub-rebar delaminations when tested across highway bridge domains.
2. **Severe Class Imbalance Sensitivity:**
   Because sound concrete represents ~77.2% of Bridge 1's valid deck, unadapted models heavily bias predictions toward sound concrete, yielding low Class 2 recall (3.2% in Seed 42, 20.3% in Seed 123, 44.1% in Seed 999).
3. **Stochastic Sensitivity to Initialization:**
   Performance varies significantly across seeds (Macro Delam F1 ranges from 2.84% to 14.45%), underscoring the absolute necessity of evaluating multiple random seeds rather than single-run comparisons.
4. **Status of Folds 2–5:**
   The remaining 12 runs across Folds 2, 3, 4, and 5 are pending materialization of the training/validation patches from the raw SDNET2021 dataset as detailed in `reports/recovery_blockers.md`.
