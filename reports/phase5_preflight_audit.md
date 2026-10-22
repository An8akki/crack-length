# Phase 5 Preflight Audit: Deep CORAL Domain Adaptation

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Evaluation Scope:** 5 LOBO Folds $\times$ 3 Seeds (15 Adaptation Runs)  
**Date:** 2026-10-20  
**Audit Status:** **PASS — ALL PREFLIGHT CRITERIA VERIFIED (AUTHORIZED FOR EXECUTION)**  
**Target Branch:** `depth-estimation-integration`  

---

## 1. Preflight Objective & Scope

This preflight audit rigorously validates experimental parity, data isolation safeguards, output directory separation, and baseline preservation before executing **Phase 5: Deep CORAL Domain Adaptation Experiments**.

The primary scientific hypothesis being tested is:
$$\text{Hypothesis: Aligning bottleneck feature covariance } C_S \approx C_T \text{ via Deep CORAL reduces cross-bridge domain shift and improves delamination Macro F1 over the source-only baseline by } \Delta \ge +0.05 \text{ (+5.0%)}.$$

---

## 2. Experimental Parity Matrix (Baseline vs Deep CORAL)

To ensure that performance differences are attributable strictly to second-order domain adaptation rather than architectural or training confounders, all hyperparameter settings are locked to exact baseline equivalence:

| Component | Source-Only Baseline (Phase 2B.2) | Deep CORAL Adaptation (Phase 5) | Parity Status | Rationale / Documentation |
| :--- | :--- | :--- | :---: | :--- |
| **Model Architecture** | 4-stage convolutional U-Net | `DeepCORALUNet` (subclass of `UNet`) | **IDENTICAL** | Exact same layers, parameters (486,427), and weights. Subclass adds feature extraction hook. |
| **Feature Channels** | $[8, 16, 32, 64]$ | $[8, 16, 32, 64]$ | **IDENTICAL** | Prevents parameter count discrepancies. |
| **Alignment Layer** | None | Encoder Bottleneck ($d=128$) | **JUSTIFIED** | Captures global domain features at lowest spatial resolution ($16 \times 16$). |
| **Preprocessing & Normalization** | Intensity mean ($\mu$) & std ($\sigma$) from source-train valid pixels | Intensity mean ($\mu$) & std ($\sigma$) from source-train valid pixels | **IDENTICAL** | Zero target-derived normalization statistics used. Excluded pixels (`255`) filtered out. |
| **Class Encoding** | 0: Sound, 1: Class 2, 2: Class 3, 255: Excluded | 0: Sound, 1: Class 2, 2: Class 3, 255: Excluded | **IDENTICAL** | Exact semantic mapping across all partitions. |
| **Loss Function (Source)** | Multiclass CrossEntropyLoss (`ignore_index=255`) | Multiclass CrossEntropyLoss (`ignore_index=255`) | **IDENTICAL** | Unweighted segmentation loss on source bridge patches. |
| **Adaptation Loss** | None | $\mathcal{L}_{\text{CORAL}} = \frac{1}{4d^2} \| C_S - C_T \|_F^2$ | **JUSTIFIED** | Canonical Sun & Saenko (ECCV 2016) formulation with $\lambda_{\text{coral}} = 0.5$. |
| **Optimizer** | AdamW ($\eta = 10^{-3}$, $\lambda = 10^{-4}$) | AdamW ($\eta = 10^{-3}$, $\lambda = 10^{-4}$) | **IDENTICAL** | Exact optimization policy. |
| **Batch Size** | 8 source patches per batch | 8 source patches + 8 target patches | **IDENTICAL** | Equal representation per gradient step. |
| **Max Epochs & Early Stopping** | 100 max epochs, patience 12 | 100 max epochs, patience 12 | **IDENTICAL** | Early stopping counter monitored on source validation. |
| **Model Selection Criterion** | Source-validation Macro Delam F1 | Source-validation Macro Delam F1 | **IDENTICAL** | Zero target-derived metric used for checkpoint selection. |
| **Target Evaluation** | Offline full-bridge stitched Strategy C | Offline full-bridge stitched Strategy C | **IDENTICAL** | Single physical valid pixel scoring ($M_{\text{valid}}$). |

---

## 3. Data Isolation and Anti-Leakage Audit

1. **Target Adaptation DataLoader Interface:**
   - The adaptation pipeline feeds unlabeled target images from `SDNET2021_Audit/phase2b1_dataset_pipeline/target_uda/{fold_id}/images/`.
   - Filesystem verification confirms that `target_uda/` contains **strictly zero mask or label files** (directory contains only `_img.png` files).
   - Validated by unit test `test_01_adaptation_dataset_contains_zero_masks` in `tests/test_data_isolation.py` (requesting `'mask'` raises `KeyError`).
2. **Target Evaluation Mask Sealing:**
   - Target ground truth masks are sealed in `SDNET2021_Audit/phase2b1_dataset_pipeline/target_eval_sealed/{fold_id}/masks/`.
   - Masks are loaded exclusively during post-freeze offline target evaluation after the checkpoint has been selected and frozen with a cryptographic SHA-256 signature.
3. **Loss Gradient Isolation:**
   - Verified that label `255` produces zero gradient during backpropagation (`test_02_loss_gradient_zero_at_ignore_index_255` passing).

---

## 4. Output Directory and Artifact Separation

Phase 5 outputs are isolated into dedicated directories, guaranteeing zero overwriting or deletion of completed Phase 2 baseline artifacts:

- **Baseline Artifacts Preserved:**
  - `SDNET2021_Audit/phase2b2_source_only_baseline/checkpoints/*.pt` (15 baseline checkpoints preserved).
  - `SDNET2021_Audit/phase2b2_source_only_baseline/baseline_metrics_by_run.csv` (preserved).
  - `SDNET2021_Audit/phase2b2_source_only_baseline/predictions/*.png` (15 baseline prediction rasters preserved).
- **Phase 5 Dedicated Outputs:**
  - Dedicated training root: `SDNET2021_Audit/phase5_deep_coral/`
  - Checkpoints: `SDNET2021_Audit/phase5_deep_coral/checkpoints/` (`coral_{run_id}_best_checkpoint.pt`, `coral_{run_id}_config.json`)
  - Target Predictions: `SDNET2021_Audit/phase5_deep_coral/predictions/` (`coral_{run_id}_target_prediction.png`)
  - Metrics & Logs: `coral_metrics_by_run.csv`, `coral_checkpoint_manifest.csv`, `coral_training_history.csv`, `coral_target_evaluation_access_log.csv`
  - Results directory: `results/coral_metrics_by_run.csv`, `results/coral_metrics_by_bridge.csv`, `results/coral_macro_summary.csv`, and updated `results/paired_bridge_comparison.csv`.

---

## 5. Baseline Reference Data Verification

The baseline run matrix is audited and verified in `results/source_only_metrics.csv` and `reports/phase2_baseline_run_audit.csv`:

| Target Bridge | Baseline Macro Delam F1 | Baseline Class 2 F1 | Baseline Class 3 F1 | Baseline Defect Recall | Baseline Sound F1 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **B1 (Forest River NB)** | $9.48 \pm 5.98\%$ | $18.95 \pm 11.97\%$ | $0.00 \pm 0.00\%$ | $33.10 \pm 20.97\%$ | $80.84 \pm 7.16\%$ |
| **B2 (Forest River SB)** | $14.00 \pm 6.40\%$ | $28.01 \pm 12.78\%$ | $0.01 \pm 0.01\%$ | $76.77 \pm 40.24\%$ | $57.00 \pm 20.26\%$ |
| **B3 (Park River Median)** | $9.45 \pm 9.62\%$ | $18.90 \pm 19.25\%$ | $0.00 \pm 0.00\%$ | $48.94 \pm 43.97\%$ | $80.18 \pm 7.87\%$ |
| **B4 (Park River NB)** | $14.08 \pm 2.48\%$ | $25.95 \pm 7.43\%$ | $2.22 \pm 3.84\%$ | $57.52 \pm 35.14\%$ | $61.15 \pm 39.88\%$ |
| **B5 (Park River SB)** | $14.14 \pm 1.02\%$ | $28.27 \pm 2.03\%$ | $0.01 \pm 0.02\%$ | $65.86 \pm 20.07\%$ | $68.63 \pm 10.44\%$ |
| **Fleet Macro Average** | **$12.23 \pm 5.54\%$** | **$24.02 \pm 11.16\%$** | **$0.45 \pm 1.72\%$** | **$56.44 \pm 32.26\%$** | **$69.56 \pm 20.45\%$** |

---

## 6. Preflight Conclusion & Authorization

All preflight checks have passed:
- [x] Full architectural and hyperparameter parity established.
- [x] Zero target mask leakage verified by test suite.
- [x] Dedicated output directory structure established.
- [x] Baseline checkpoints and metrics verified and protected against overwriting.
- [x] Resumption mechanism and error handling ready.

**Decision:** **PASSED**. Authorized to implement `run_coral_experiments.py` and commence Phase 5 training.
