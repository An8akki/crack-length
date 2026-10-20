# Research Project Recovery Blockers

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Date:** 2026-10-20  
**Host:** Windows 11 AMD64 (13th Gen Intel Core i7-13650HX)  
**Repository:** `https://github.com/An8akki/crack-length` (`depth-estimation-integration`)  

---

## 1. Primary Experimental Blocker: Missing Raw SDNET2021 IR Dataset

### Status:
**BLOCKING EXECUTION OF REMAINING TRAINING RUNS (FOLDS 2–5)**

### Diagnostic Findings:
1. **Raw Thermography Images:** The 10 original and annotated infrared PNG pairs for Bridges B1–B5 were located at `C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt\` on the previous computer. They are not present on this computer.
2. **Materialized Intermediate Patches:** The 6,131 individual image and mask crops (~47.79 MB across `source_train/`, `source_val/`, `target_uda/`) are excluded from Git per `.gitignore` and `git_handover.md`.
3. **Sealed Evaluation Masks:** In contrast, the ground-truth target evaluation masks and reconstruction manifests in `SDNET2021_Audit/phase2b1_dataset_pipeline/target_eval_sealed/` **are present and 100% verified** against `checksums.sha256`.

### Why This Is a Blocker:
Training a U-Net model on Folds 2, 3, 4, and 5 requires feeding source-training patches (`source_train/{fold_id}/images/` and `masks/`) and computing source validation delamination Macro F1 on `source_val/{fold_id}/`. Without the raw thermograms from which these patches are deterministically built, training Folds 2–5 cannot begin without violating the rule against fabricating inputs.

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
