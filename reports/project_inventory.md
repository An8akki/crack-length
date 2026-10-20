# Research Project Master Inventory & Phase Checklist

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Repository:** `https://github.com/An8akki/crack-length`  
**Branch:** `depth-estimation-integration`  
**Workspace Root:** `C:\Project\ml_pipeline\crack-length`  
**Latest Git Commit:** `ed508ddc` (*Prepare research pipeline for cross-system resume*)  
**Audit Date:** 2026-10-20  

---

## 1. Master Phase Execution Checklist

| Phase | Description | Status | Deliverables / Notes |
| :--- | :--- | :---: | :--- |
| **Phase 0** | Repository Recovery & Inventory | **COMPLETED** | `reports/project_inventory.md`, `reports/experiment_status.json`, `reports/recovery_blockers.md` |
| **Phase 1** | Environment & Data Validation | **COMPLETED** | `reports/environment_audit.md`, `reports/dataset_validation.md`, `reports/data_integrity.json` |
| **Phase 2** | Finalize Source-Only Baseline | **IN_PROGRESS** | Fold 1 verified (3 runs complete). Folds 2–5 blocked by raw dataset transfer. Documented in `reports/source_only_baseline.md`. |
| **Phase 3** | Implement Deep CORAL Adaptation | **COMPLETED** | `src/domain_adaptation/deep_coral.py`, `tests/test_deep_coral.py`, `reports/deep_coral_design.md` |
| **Phase 4** | Leakage & Experimental Validation | **COMPLETED** | `tests/test_data_isolation.py`, `tests/test_lobo_splits.py`, `reports/leakage_audit.md` |
| **Phase 5** | Run Domain Adaptation Experiments | **BLOCKED** | Blocked pending raw dataset patch materialization. |
| **Phase 6** | Optional Controls and Ablations | **BLOCKED** | Follows Phase 5. |
| **Phase 7** | Statistical Analysis | **IN_PROGRESS** | Statistical framework, power analysis, and Wilcoxon limits established in `reports/statistical_analysis.md`. |
| **Phase 8** | Failure Analysis & Visualization | **IN_PROGRESS** | Fold 1 prediction maps analyzed; visualization framework prepared in `reports/failure_analysis.md`. |
| **Phase 9** | Literature Review & Claim Validation | **COMPLETED** | `reports/literature_review.md`, `reports/research_gap_and_contributions.md` |
| **Phase 10** | Reproducibility & Final Audit | **COMPLETED** | `reports/final_reproducibility_audit.md`, `README_REPRODUCE.md` |
| **Phase 11** | Final Research Report | **COMPLETED** | `reports/final_research_report.md` |
| **Phase 12** | Final Git Delivery | **IN_PROGRESS** | Working tree clean, auditable commits ready, push policy enforced. |

---

## 2. Repository Inventory

### A. Research Directories
- `SDNET2021_Audit/phase1c_bridge1_pilot`: Bridge 1 pilot audit, semantic color decoding (0=sound, 1=Class 2, 2=Class 3, 255=excluded).
- `SDNET2021_Audit/phase1d_five_bridge_audit`: Geometry, tile dimensions, and raw pixel distributions across Bridges B1–B5.
- `SDNET2021_Audit/phase1e_alignment_verification`: Cross-bridge alignment diagnostics and color verification.
- `SDNET2021_Audit/phase1e1_reconciliation`: Tile count reconciliation across audit passes.
- `SDNET2021_Audit/phase1e2_b4_alignment_audit`: Rigorous spatial alignment audit on Bridge 4; froze unshifted baseline.
- `SDNET2021_Audit/phase2a_dataset_plan`: Comparison of non-overlapping (Strategy A) vs sliding window (Strategy C).
- `SDNET2021_Audit/phase2a1_protocol_correction`: Full deck valid coverage reconciliation.
- `SDNET2021_Audit/phase2a2_independent_verification`: Independent verification of Strategy C windowing and LOBO fold definitions.
- `SDNET2021_Audit/phase2a3_coverage_reconciliation`: 100% pixel coverage proof for 322 Strategy C patches.
- `SDNET2021_Audit/phase2b1_dataset_pipeline`: Dataset builder `dataset_build.py`, manifests, and `target_eval_sealed/`.
- `SDNET2021_Audit/phase2b2_source_only_baseline`: Source-only U-Net training, post-freeze target evaluation, checkpoints, and metrics.

### B. Baseline Checkpoint Manifest
- `fold_1_seed42_best_checkpoint.pt`: 1,990,523 bytes | SHA-256: `5bf64696b6f112156a60d2ba50a2967add1ab8b5ed561af791c954717789918c` | Best Ep: 2
- `fold_1_seed123_best_checkpoint.pt`: 1,990,647 bytes | SHA-256: `8a45c7daf3687fd0fc9e377d89b06826dcdacafb160c099195dfd47b594d4413` | Best Ep: 17
- `fold_1_seed999_best_checkpoint.pt`: 1,990,647 bytes | SHA-256: `025bbe4221adb3325f622b86cd8539442330d392b4becd99b42e79c3041ff8d5` | Best Ep: 1

### C. Resource Estimates for Remaining 12 Baseline Runs
- **Model Parameters:** 486,427 parameters (~1.99 MB per checkpoint).
- **Runtime per Run on CPU:** ~6 to 12 minutes per run (depending on early stopping epoch, avg 15–25 epochs at ~22s/epoch).
- **Total Compute for 12 Runs:** ~1.5 to 2.5 hours on the 13th Gen Intel Core i7-13650HX.
- **Disk Storage for 12 Runs:** ~30 MB (12 checkpoints $\times$ 1.99 MB + prediction PNGs and logs).
- **RAM Footprint:** ~1.2 GB active RAM (batch size 8).
