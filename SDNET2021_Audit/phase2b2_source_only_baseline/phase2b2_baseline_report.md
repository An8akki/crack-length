# Phase 2B.2 Source-Only Baseline Benchmark Report
## AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

**Author / Runner:** Antigravity Research Assistant  
**Date:** 2026-10-20  
**Repository:** `https://github.com/An8akki/crack-length` (`depth-estimation-integration`)  
**Scope:** 15 Complete Benchmark Runs (5 LOBO Folds $\times$ 3 Seeds: 42, 123, 999)  
**Status:** 100% Complete, Cryptographically Verified, and Sealed  

---

### Executive Summary

Phase 2B.2 establishes the definitive empirical source-only baseline for semantic segmentation of concrete bridge deck delaminations under real-world cross-bridge domain shift using the SDNET2021 infrared thermography dataset.

Across 15 runs evaluated on strictly sealed held-out target bridge decks using Strategy C sliding-window raster reconstruction (single evaluation per physical valid pixel), the fleet-wide results demonstrate:
1. **Fleet Macro Delamination F1:** **$12.23\% \pm 5.54\%$** (Range: 0.00% – 19.24%).
2. **Class 2 Delamination (Above-Rebar) F1:** **$24.02\% \pm 11.16\%$** (Range: 0.00% – 38.49%).
3. **Class 3 Delamination (Below-Rebar) F1:** **$0.45\% \pm 1.72\%$** (Range: 0.00% – 6.65%).
4. **Target Defect Recall:** **$56.44\% \pm 32.26\%$** (Range: 0.00% – 100.00%).
5. **Sound Concrete (Class 0) F1:** **$69.56\% \pm 20.45\%$** (Range: 15.13% – 87.65%).

---

### Per-Bridge Baseline Breakdown

| Bridge Domain | Target Deck | Valid Pixels | Mean Macro Delam F1 | Class 2 F1 (Above-Rebar) | Class 3 F1 (Below-Rebar) | Sound F1 | Defect Recall |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1** (Forest River NB) | 45 patches | 858,114 px | **$9.48 \pm 5.98\%$** | $18.95 \pm 11.97\%$ | **$0.00 \pm 0.00\%$** | $80.84 \pm 7.16\%$ | $33.10 \pm 20.97\%$ |
| **B2** (Forest River SB) | 36 patches | 621,161 px | **$14.00 \pm 6.40\%$** | $28.01 \pm 12.78\%$ | **$0.01 \pm 0.01\%$** | $57.00 \pm 20.26\%$ | $76.77 \pm 40.24\%$ |
| **B3** (Park River Median) | 99 patches | 1,757,327 px | **$9.45 \pm 9.62\%$** | $18.90 \pm 19.25\%$ | **$0.00 \pm 0.00\%$** | $80.18 \pm 7.87\%$ | $48.94 \pm 43.97\%$ |
| **B4** (Park River NB) | 46 patches | 1,109,661 px | **$14.08 \pm 2.48\%$** | $25.95 \pm 7.43\%$ | **$2.22 \pm 3.84\%$** | $61.15 \pm 39.88\%$ | $57.52 \pm 35.14\%$ |
| **B5** (Park River SB) | 96 patches | 1,743,588 px | **$14.14 \pm 1.02\%$** | $28.27 \pm 2.03\%$ | **$0.01 \pm 0.02\%$** | $68.63 \pm 10.44\%$ | $65.86 \pm 20.07\%$ |

---

### Core Scientific Findings

#### 1. Severe Generalization Failure on Sub-Rebar Defects
In 11 out of 15 runs, Class 3 delamination detection dropped to exactly **0.00% F1**, with zero true positive pixels predicted. Even in the remaining 4 runs, the highest achieved F1 was only 6.65% (`fold_4_seed123`). This proves that without domain adaptation, deep convolutional features learned from source bridge infrared thermograms fail to capture the subtle thermal signatures of subsurface defects across different bridge geometries and diurnal heating cycles.

#### 2. High Stochastic Sensitivity to Random Initialization
In Fold 3, `seed123` suffered complete optimization collapse (predicting 100% sound concrete, yielding 0.00% defect recall), whereas `seed999` achieved 19.24% Macro F1 with 85.11% defect recall. This demonstrates that multi-seed Leave-One-Bridge-Out auditing is mandatory to avoid cherry-picking artifacts in civil infrastructure deep learning literature.

#### 3. Formal Benchmark Target for Unsupervised Domain Adaptation (UDA)
The source-only reference standard is established as:
$$\text{Baseline Target Macro Delamination F1} = 12.23\% \pm 5.54\%$$
Per the pre-declared statistical decision criteria, an unsupervised domain adaptation method (such as Deep CORAL) will be deemed successful if it achieves a paired effect size improvement:
$$\Delta \ge +5.0\% \text{ Macro Delamination F1}$$
across the 5 LOBO bridge folds.

---

### Verification and Rigor Safeguards
- **Zero Target Leakage:** Validated by `baseline_validation_tests.py` (10/10 tests passing).
- **Zero Label Corruption:** Ignored label `255` verified to yield zero loss and zero gradient.
- **Cryptographic Freezing:** Checkpoint SHA-256 hashes generated and logged prior to target evaluation.
- **Full Coverage Guarantee:** Validated by `dataset_validation_tests.py` (8/8 tests passing, 0 uncovered valid deck pixels across 6,089,851 fleet valid deck pixels).
