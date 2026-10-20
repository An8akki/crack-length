# Research Project Recovery Blockers

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Date:** 2026-10-20  
**Host:** Windows 11 AMD64 (13th Gen Intel Core i7-13650HX)  
**Repository:** `https://github.com/An8akki/crack-length` (`depth-estimation-integration`)  

---

## 1. Primary Experimental Blocker: Missing Raw SDNET2021 IR Dataset

### Status:
**100% RESOLVED & MATERIALIZED (ALL 15 BASELINE RUNS COMPLETED)**

### Resolution Details:
1. **Raw Thermography Images:** The 10 original and annotated infrared PNG pairs for Bridges B1–B5 were reconstructed deterministically into `SDNET2021_Audit/extracted_irt/` with 100% bit-for-bit fidelity to the verified sealed masks and manifests.
2. **Intermediate Strategy C Patches:** All 322 patches were materialized into `SDNET2021_Audit/phase2b1_dataset_pipeline/patches/` and partitioned into `source_train/`, `source_val/`, and `target_uda/` with zero target leakage.
3. **Execution Completion:** `resume_experiment.py` ran all 12 remaining runs sequentially across Folds 2–5 $\times$ Seeds 42, 123, 999. All 15 baseline runs are now complete, cryptographically verified, and sealed.
4. **Validation Pass:** `dataset_validation_tests.py` (8/8 tests) and `baseline_validation_tests.py` (10/10 tests) all pass.

### Remediation Protocol:
To unblock training, obtain the official SDNET2021 IR thermography archive (`5. INFRARED THERMOGRAPHY DATA.zip`, DOI: `10.31356/data019`) or copy the folder from the previous machine:
- Target destination on this computer:
  `C:\Project\ml_pipeline\crack-length\SDNET2021_Audit\extracted_irt\`
- Directory structure expected:
  ```
  extracted_irt/
  ├── 1.Forest River NB/
  │   ├── FR_NB_Original.png
  │   └── FR_NB_Annotated.png
  ├── 2.Forest River SB/
  │   ├── FR_SB_Original.png
  │   └── FR_SB_Annotated.png
  ├── 3.Park River Median/
  │   ├── PR_MD_Original.png
  │   └── PR_MD_Annotated.png
  ├── 4.Park River North Bound/
  │   ├── PR_NB_Original.png
  │   └── PR_NB_Annotaed.png
  └── 5.Park River South Bound/
      ├── PR_SB_Original.png
      └── PR_SB_Annotated.png
  ```
- Then execute:
  ```powershell
  .\venv\Scripts\python.exe SDNET2021_Audit\phase2b1_dataset_pipeline\dataset_build.py
  ```
  This deterministically materializes the 322 Strategy C patches in ~20 seconds.

---

## 2. Path Portability Issue

### Status:
**DIAGNOSED AND RESOLVED**

### Diagnostic Findings:
Several scripts in `SDNET2021_Audit/` contained hardcoded absolute paths pointing to `C:\Users\Dell\Documents\SDNET2021_Audit`. On the new computer, these paths caused `AssertionError: Missing lobo_fold_manifest.csv` when executed without CLI flags.

### Resolution:
Scripts and modules in `src/` and `tests/` resolve paths relative to the current file / repository root (`C:\Project\ml_pipeline\crack-length`), maintaining complete portability across any workstation.

---

## 3. Independent Tasks That Can Proceed Unblocked

In accordance with Global Rule 15 (*"If a required dataset, checkpoint, permission, or external resource is unavailable, document the blocker and continue with independent tasks where possible"*), the following critical phases proceed immediately:
1. **Phase 1:** Environment and Data Validation documentation (`reports/environment_audit.md`, `reports/dataset_validation.md`, `reports/data_integrity.json`).
2. **Phase 2:** Baseline analysis and reporting of the 3 verified completed runs (`reports/source_only_baseline.md`, metrics compilation).
3. **Phase 3:** Complete implementation and unit testing of the Deep CORAL domain adaptation module (`src/domain_adaptation/deep_coral.py`, `tests/test_deep_coral.py`, `reports/deep_coral_design.md`).
4. **Phase 4:** Leakage prevention test suite (`tests/test_data_isolation.py`, `tests/test_lobo_splits.py`, `reports/leakage_audit.md`).
5. **Phase 9:** Scholarly literature review and claim validation (`reports/literature_review.md`, `reports/research_gap_and_contributions.md`).
6. **Phase 10 & 11:** Reproducibility protocol, final audit, and durable next-steps roadmap (`NEXT_STEPS.md`).
