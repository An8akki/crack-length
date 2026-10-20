# Phase 2B.2 Benchmark Shutdown and Recovery Audit

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Component:** Source-Only U-Net Baseline and Five-Fold LOBO Benchmark (Phase 2B.2)  
**Timestamp of Audit:** 2026-10-20T11:15:30+05:30  
**Machine / Host:** Windows 11 AMD64 (Host: Dell)  
**Python Environment:** Python 3.12.10, PyTorch 2.8.0+cpu  

---

## 1. Executive Summary & Objective

In preparation for a hardware transition (migrating to another machine), the active background execution of the Phase 2B.2 Leave-One-Bridge-Out (LOBO) benchmark was inspected, safely paused, and audited for complete reproducibility and zero artifact corruption.

- **Primary Action:** Graceful termination of the benchmark process (`task-908`, running `phase2b2_source_only_baseline\run_experiment.py`).
- **Completed & Verified Runs:** **3 of 15** runs (Fold 1: seeds `42`, `123`, `999` on Target Bridge B1) are **100% complete**, verified, and permanently frozen.
- **Interrupted / Cancelled Run:** Run 4 (`fold_2_seed42`) was actively executing at Epoch 15 when shutdown was initiated. Because early stopping and checkpoint freezing occur at the completion of a run, **no corrupt or partial checkpoint was written to disk**. This run is cleanly marked as `PENDING` and will restart cleanly from Epoch 1 upon resumption.
- **Pending Runs:** **12 of 15** runs remain to be executed on the new system.
- **Data Isolation:** All strict safeguards (SHA-256 pre-evaluation freezing, sealed target evaluation access logging, unweighted CrossEntropyLoss with `ignore_index=255`) remain 100% intact.

---

## 2. Process Inspection and Shutdown Timeline

1. **Process Discovery:**
   - Active task identified: `task-908` (`python -u c:\Users\Dell\.gemini\antigravity-ide\scratch\ml_pipeline\scratch\run_experiment.py`).
   - Associated monitoring timer: `task-979` (cancelled immediately to prevent asynchronous interrupts).
2. **Execution Inspection:**
   - Inspection of `task-908.log` at 11:08:10 confirmed:
     * `fold_1_seed42`: Completed in 383.3s. Best Epoch 2 (Val Macro Delam F1: 9.40%). Target B1 Delam Macro F1 = 2.84%, Defect Recall = 11.35%.
     * `fold_1_seed123`: Completed in 771.9s. Best Epoch 17 (Val Macro Delam F1: 10.60%). Target B1 Delam Macro F1 = 11.14%, Defect Recall = 34.75%.
     * `fold_1_seed999`: Completed in 348.3s. Best Epoch 1 (Val Macro Delam F1: 20.37%). Target B1 Delam Macro F1 = 14.45%, Defect Recall = 53.20%.
     * `fold_2_seed42`: In progress. Epoch 14 achieved a new best validation score of 18.26%, resetting early stopping patience to 0/12; Epoch 15 finished at 11:09:44.
3. **Termination Execution:**
   - In accordance with the shutdown protocol, `task-908` was cancelled cleanly via `manage_task(Action='kill')` at 11:09:54.
   - Verification confirmed that no orphan Python processes remain active on the system.

---

## 3. Checkpoint & Artifact Integrity Audit

Every preserved checkpoint was programmatically tested for:
1. File existence and size (~1.99 MB).
2. SHA-256 bitwise match against `checkpoint_manifest.csv` and `target_evaluation_access_log.csv`.
3. Valid state dictionary loading into PyTorch (`torch.load(..., map_location='cpu')`).
4. Prediction map raster integrity (`predictions/*.png`).

### Verification Table

| Run ID | Target | Seed | Status | File Size | SHA-256 Checksum | PyTorch State Dict Keys | Prediction Map | Access Log Verified |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `fold_1_seed42` | B1 | 42 | **VERIFIED** | 1,990,523 bytes | `5bf64696b6f112156a60d2ba50a2967add1ab8b5ed561af791c954717789918c` | 118 tensors | `fold_1_seed42_target_prediction.png` (17,040 B) | YES |
| `fold_1_seed123` | B1 | 123 | **VERIFIED** | 1,990,647 bytes | `8a45c7daf3687fd0fc9e377d89b06826dcdacafb160c099195dfd47b594d4413` | 118 tensors | `fold_1_seed123_target_prediction.png` (19,480 B) | YES |
| `fold_1_seed999` | B1 | 999 | **VERIFIED** | 1,990,647 bytes | `025bbe4221adb3325f622b86cd8539442330d392b4becd99b42e79c3041ff8d5` | 118 tensors | `fold_1_seed999_target_prediction.png` (12,909 B) | YES |
| `fold_2_seed42` | B2 | 42 | **INTERRUPTED** | *None written* | N/A | N/A | *None written* | N/A |
| `fold_2_seed123` – `fold_5_seed999` | B2–B5 | 123, 999 | **PENDING** | *None written* | N/A | N/A | *None written* | N/A |

---

## 4. Manifest and Log Verification

- `checkpoint_manifest.csv`: Contains exactly 3 frozen checkpoint records.
- `baseline_metrics_by_run.csv`: Contains complete target metrics for the 3 completed runs on Bridge 1 (precision, recall, F1, IoU per class, combined delamination metrics, defect recall, confusion matrices, and inference durations).
- `target_evaluation_access_log.csv`: Verifies post-freeze target access for the 3 completed runs with `integrity_verified=True`.
- `training_history.csv`: Contains epoch-by-epoch training and validation loss/F1 curves for the 3 completed runs (56 total epoch rows).

---

## 5. Resumption Strategy on New System

A dedicated resumption tool, `resume_experiment.py`, was created and tested. It enforces the following logic:
1. **Zero-Overwrite Verification:** Performs bitwise SHA-256 and PyTorch load checks before skipping any completed run.
2. **Safe Restart of Interrupted Runs:** Detects `fold_2_seed42` as pending, restarts it cleanly from Epoch 1, and appends its outputs upon completion.
3. **Atomic File Writes:** Writes all CSVs to `.tmp` files before replacing to eliminate corruption risks during system power-downs or interruptions.
4. **Fleet Aggregation:** Automatically compiles `baseline_metrics_by_bridge.csv` and `baseline_macro_summary.csv` as soon as all 15 runs finish.
