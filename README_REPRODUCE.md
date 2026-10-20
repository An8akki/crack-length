# Exact Reproduction Guide: AI-Based Concrete Delamination Characterization

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Repository:** `https://github.com/An8akki/crack-length`  
**Branch:** `depth-estimation-integration`  

This document provides step-by-step PowerShell instructions to reproduce the entire environment, validation test suites, baseline benchmark, and Deep CORAL adaptation pipeline.

---

## 1. System Requirements & Prerequisites

- **Operating System:** Windows 10/11 64-bit (or Linux AMD64)
- **Python Version:** Python 3.12 (tested on 3.12.0)
- **Git:** Git 2.40+ installed
- **Hardware:** CPU with $\ge 8$ logical cores (tested on 13th Gen Intel Core i7-13650HX, 14 cores, 20 threads) or NVIDIA GPU.

---

## 2. Environment Setup

```powershell
# 1. Clone repository directly to research branch
git clone -b depth-estimation-integration https://github.com/An8akki/crack-length.git
cd crack-length

# 2. Create Python 3.12 virtual environment
py -3.12 -m venv venv

# 3. Activate environment
.\venv\Scripts\Activate.ps1

# 4. Install exact pinned requirements
pip install -r SDNET2021_Audit\handover\environment_requirements.txt

# 5. Verify core libraries
python -c "import torch, torchvision, numpy, pandas; print('PyTorch:', torch.__version__, '| Device:', 'cuda' if torch.cuda.is_available() else 'cpu')"
```

---

## 3. Data Restoration & Patch Materialization

The raw infrared thermography images from SDNET2021 (`5. INFRARED THERMOGRAPHY DATA.zip`, DOI: `10.31356/data019`) must be placed in `SDNET2021_Audit\extracted_irt\`:

```
SDNET2021_Audit/extracted_irt/
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

Once the 10 raw image pairs are present, execute the deterministic dataset builder:

```powershell
# Materialize 322 Strategy C patches (takes ~20 seconds)
.\venv\Scripts\python.exe SDNET2021_Audit\phase2b1_dataset_pipeline\dataset_build.py
```

---

## 4. Run Automated Validation Test Suites

Execute all independent validation test suites:

```powershell
# 1. Run dataset pipeline validation tests
.\venv\Scripts\python.exe SDNET2021_Audit\phase2b1_dataset_pipeline\dataset_validation_tests.py

# 2. Run LOBO split and boundary tests
.\venv\Scripts\python.exe tests\test_lobo_splits.py

# 3. Run adversarial data isolation and anti-leakage tests
.\venv\Scripts\python.exe tests\test_data_isolation.py

# 4. Run Deep CORAL unit tests
.\venv\Scripts\python.exe tests\test_deep_coral.py

# 5. Run baseline protocol and safeguard tests
.\venv\Scripts\python.exe SDNET2021_Audit\phase2b2_source_only_baseline\baseline_validation_tests.py
```

---

## 5. Source-Only Baseline Benchmark Execution

```powershell
# 1. Check current status without training (Fold 1 verified, Folds 2-5 ready)
.\venv\Scripts\python.exe SDNET2021_Audit\phase2b2_source_only_baseline\resume_experiment.py --status --baseline-dir SDNET2021_Audit\phase2b2_source_only_baseline

# 2. Launch automated resumption runner (completes remaining 12 runs)
.\venv\Scripts\python.exe SDNET2021_Audit\phase2b2_source_only_baseline\resume_experiment.py --baseline-dir SDNET2021_Audit\phase2b2_source_only_baseline --data-dir SDNET2021_Audit\phase2b1_dataset_pipeline --src-dir SDNET2021_Audit\extracted_irt
```

---

## 6. Checkpoint Integrity & Verification

All checkpoints can be verified against their SHA-256 signatures:

```powershell
# Compute SHA-256 for all checkpoints
Get-FileHash SDNET2021_Audit\phase2b2_source_only_baseline\checkpoints\*.pt
```
Expected hashes:
- `fold_1_seed42`: `5bf64696b6f112156a60d2ba50a2967add1ab8b5ed561af791c954717789918c`
- `fold_1_seed123`: `8a45c7daf3687fd0fc9e377d89b06826dcdacafb160c099195dfd47b594d4413`
- `fold_1_seed999`: `025bbe4221adb3325f622b86cd8539442330d392b4becd99b42e79c3041ff8d5`
