# Next-System Setup, Recovery, and Benchmark Resumption Guide

**Project:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Repository:** `https://github.com/An8akki/crack-length.git`  
**Branch:** `depth-estimation-integration`  
**Benchmark Target:** Phase 2B.2 (Source-Only U-Net Baseline & Five-Fold LOBO Benchmark)  

This guide provides exact, verified Windows PowerShell instructions to set up a new computer, restore the environment, verify existing artifacts, and seamlessly resume the remaining 12 benchmark runs.

---

## 1. System Requirements & Prerequisites

- **Operating System:** Windows 10/11 64-bit (or Linux AMD64).
- **Python Version:** **Python 3.12** (tested on 3.12.10). Ensure `python` and `pip` are on your system `PATH`.
- **Git:** Git 2.40+ installed.
- **Hardware Recommended:** CPU with $\ge 8$ logical threads (e.g. Intel Core i7 / AMD Ryzen 7) or compatible NVIDIA GPU with CUDA. (Baseline is CPU-optimized with 486k parameters, running at ~18–26s per epoch).

---

## 2. Step-by-Step PowerShell Setup

Open **Windows PowerShell** and execute the following commands in sequence.

### Step 2.1 — Clone the Repository
```powershell
# Set your target workspace path (customize if desired)
$WORKSPACE_DIR = "$HOME\Documents\Research"
New-Item -ItemType Directory -Path $WORKSPACE_DIR -Force
Set-Location $WORKSPACE_DIR

# Clone directly into the target branch
git clone -b depth-estimation-integration https://github.com/An8akki/crack-length.git
Set-Location crack-length

# Verify repository branch
git status
```

### Step 2.2 — Create & Activate Virtual Environment
```powershell
# Create fresh Python 3.12 virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

### Step 2.3 — Install Dependencies
```powershell
# Install exact pinned requirements
pip install -r SDNET2021_Audit\handover\environment_requirements.txt

# Verify key imports
python -c "import torch, torchvision, numpy, pandas, PIL, scipy, sklearn; print('PyTorch:', torch.__version__, '| NumPy:', numpy.__version__)"
```

---

## 3. Data Setup & Reconstruction

The research code expects the SDNET2021 raw files and audit folders to reside either in `C:\Users\Dell\Documents\SDNET2021_Audit\` or in your local repository under `SDNET2021_Audit/`.

### Step 3.1 — Configure Raw SDNET2021 IR Data
If the raw infrared thermography images are not already present:
1. Obtain the official SDNET2021 infrared thermography dataset (`5. INFRARED THERMOGRAPHY DATA.zip`).
2. Extract the 10 PNG image pairs (original and annotated for Bridges 1–5) to:
   `<AUDIT_DIR>\extracted_irt\`
   * Bridges: `PR_EB`, `PR_WB`, `PR_Median`, `PR_NB`, `PR_SB` (B1–B5).

### Step 3.2 — Rebuild Dataset Pipeline Patches (If Needed)
The 322 Strategy C patches ($256 \times 256$ with 50% overlap) are generated deterministically:
```powershell
Set-Location SDNET2021_Audit\phase2b1_dataset_pipeline

# Execute deterministic dataset builder
python dataset_build.py

# Run pipeline validation tests
python dataset_validation_tests.py
```
*Expected Result: 100% test pass (exit code 0).*

---

## 4. Checkpoint Verification & Status Audit

Before launching training, verify that the 3 completed Fold 1 checkpoints transferred without corruption.

```powershell
Set-Location ..\phase2b2_source_only_baseline

# Run the status audit without training
python resume_experiment.py --status
```

### Expected Audit Output:
```text
================================================================================
PHASE 2B.2 U-NET BASELINE BENCHMARK STATUS AUDIT
================================================================================
Run ID           | Target | Seed  | Status                 | Val Delam F1 | Tgt Delam F1 | Defect Rec
--------------------------------------------------------------------------------
fold_1_seed42    | B1     | 42    | VERIFIED_COMPLETE      | 9.40%        | 2.84%        | 11.3%     
fold_1_seed123   | B1     | 123   | VERIFIED_COMPLETE      | 10.60%       | 11.14%       | 34.8%     
fold_1_seed999   | B1     | 999   | VERIFIED_COMPLETE      | 20.37%       | 14.45%       | 53.2%     
fold_2_seed42    | B2     | 42    | PENDING                | N/A          | N/A          | N/A       
... (folds 2-5 seeds pending)
--------------------------------------------------------------------------------
TOTAL: 15 runs | Completed & Verified: 3 | Incomplete/Interrupted: 0 | Pending: 12
================================================================================
```

---

## 5. Resuming the Benchmark

To resume the benchmark from where the previous system left off:

```powershell
# Launch automated resumption runner
python resume_experiment.py
```

### What `resume_experiment.py` does:
1. **Skips Completed Runs:** It verifies `fold_1_seed42`, `fold_1_seed123`, and `fold_1_seed999` using SHA-256 and skips them without retraining.
2. **Restarts Interrupted Run:** It starts `fold_2_seed42` cleanly from Epoch 1.
3. **Sequential Execution:** Executes Folds 2, 3, 4, and 5 across seeds `42`, `123`, and `999` (12 remaining runs).
4. **Post-Freeze Target Evaluation:**
   * Trains model using only source training patches.
   * Selects best checkpoint based on source validation Macro Delamination F1.
   * Freezes checkpoint `.pt` and writes cryptographic SHA-256 signature.
   * Opens sealed target evaluation path (`target_eval_sealed/`) and logs access in `target_evaluation_access_log.csv`.
   * Stitches full-bridge sliding-window prediction map ($3 \times H \times W$), evaluating each physical valid pixel exactly once.
5. **Incremental & Atomic Saving:** Saves metrics after each run to `.tmp` files before atomically updating CSVs.
6. **Automatic Fleet Summaries:** When all 15 runs finish, it generates:
   * `baseline_metrics_by_bridge.csv` (per-bridge mean $\pm$ standard deviation).
   * `baseline_macro_summary.csv` (fleet-wide macro averages).

---

## 6. Post-Benchmark Verification

Once all 15 runs finish, execute the validation test suite:

```powershell
python baseline_validation_tests.py
```
This script verifies:
1. Fold assignments match the circular successor rule.
2. Zero target leakage in source training or validation.
3. Zero target masks in UDA partitions.
4. Loss function strictly masks label `255`.
5. Normalization parameters ($\mu, \sigma$) derived exclusively from source-training pixels.
6. Full-bridge sliding-window coverage with zero hole pixels.
7. Post-freeze target evaluation authorization.
8. Checkpoint SHA-256 integrity across all 15 runs.
9. Confusion matrix and metric reconciliation.
10. Deterministic seeding and reproducibility.

---

## 7. Research Protocol Safeguards

- **No Domain Adaptation:** Keep the baseline strictly source-only. Do not enable Deep CORAL or pseudo-labeling until Phase 2C.
- **Bridge 4 Alignment:** Do not shift Bridge 4 masks. Keep the documented unshifted baseline intact.
- **Ignore Index:** Ensure `ignore_index=255` is always active during loss computation and evaluation.
