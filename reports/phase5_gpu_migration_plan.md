# Phase 5: GPU Migration & Safe Benchmark Recovery Plan

**Project Title:** AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift  
**Target Hardware:** NVIDIA GeForce RTX 3050 6GB Laptop GPU (Ampere, CC 8.6)  
**Host CPU:** 13th Gen Intel Core i7-13650HX (14 cores / 20 threads)  
**Dedicated CUDA Environment:** `C:\Project\ml_pipeline\crack-length\venv_cuda`  
**Date:** 2026-10-20  
**Migration Status:** **READY FOR GPU EXECUTION (7.73x Speedup Verified, Zero Data Loss)**  

---

## 1. Executive Summary

This report establishes the verified GPU migration and safe continuation plan for **Phase 5: Deep CORAL Domain Adaptation Benchmark**.

1. **Hardware & Environment:** The host possesses an NVIDIA GeForce RTX 3050 Laptop GPU (6 GB VRAM). A completely isolated virtual environment (`.\venv_cuda`) was constructed with Python 3.12, official PyTorch with CUDA 12.4 (`torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`), and all required dependencies. The original CPU environment (`.\venv`) remains preserved and unmodified.
2. **GPU Capability & Diagnostics:** `torch.cuda.is_available()` returns `True`. Full forward/backward passes on `DeepCORALUNet` (486,427 parameters) with real Strategy C patches ($8 \times 1 \times 256 \times 256$) were executed on CUDA.
3. **VRAM Footprint & Speedup:** Peak VRAM consumption is **711.18 MB** (<12% of the 6 GB physical capacity, zero OOM risk). GPU iteration speed is **0.064s / step** compared to **0.496s / step** on CPU, yielding a **7.73x speedup**.
4. **State of Completed CPU Runs:** All 3 runs for Bridge B1 (`coral_fold_1_seed42`, `coral_fold_1_seed123`, `coral_fold_1_seed999`) completed cleanly and are 100% verified across 5 required artifact criteria.
5. **Preliminary Bridge B1 Finding:** Deep CORAL achieved an average Macro Delamination F1 of **$12.65\% \pm 1.06\%$** on Bridge B1 (vs Baseline **$9.48\% \pm 5.98\%$**), yielding a **$+3.17\%$ improvement** and recovering sub-rebar delaminations (`Class 3`) from total collapse (0.00% in baseline).
6. **Recovery Plan:** The GPU runner will reuse all 3 completed Fold 1 runs and commence cleanly from Run 4 (`coral_fold_2_seed42`). Total estimated runtime for the remaining 12 runs drops from ~4.0 hours (CPU) to **~30 minutes (GPU)**.

---

## 2. Verification of Resume and Skip Conditions

To satisfy experimental rigor, the runner's resume mechanism in `run_coral_experiments.py` enforces a **strict 5-point verification standard** before any run can be skipped:

```python
# 5-Point Verification Standard
1. Metrics Row: Entry exists in coral_metrics_by_run.csv with non-null target_macro_delam_f1.
2. Checkpoint File: coral_{run_id}_best_checkpoint.pt exists, is non-empty, and loadable by PyTorch.
3. Config Integrity: coral_{run_id}_config.json exists with 'frozen': True, and its recorded
   checkpoint_sha256 exactly matches the cryptographic SHA-256 hash of the checkpoint file.
4. Prediction Rasters: coral_{run_id}_target_prediction.png exists in both
   SDNET2021_Audit/phase5_deep_coral/predictions/ AND results/visualizations/ with non-zero size.
5. Confusion Matrices: coral_{run_id}_cm.csv exists in both
   SDNET2021_Audit/phase5_deep_coral/confusion_matrices/ AND results/deep_coral_confusion_matrices/.
```

### Partial Run Handling
- If a run is interrupted mid-training, it possesses **no entry** in `coral_metrics_by_run.csv` and **no frozen checkpoint**.
- Under the 5-point check, any incomplete run fails verification and is **safely retried from epoch 1**.
- Partial or unverified temporary outputs are automatically overwritten upon retrying, preventing any duplicate or contaminated records.

---

## 3. Audit of Preserved Completed Runs (Bridge B1 Complete)

All 3 runs of Leave-One-Bridge-Out Fold 1 (Target: Bridge B1, Forest River NB) completed cleanly and have been verified against the 5-point standard:

| Run ID | Seed | Target Bridge | Selected Epoch | Total Epochs | Source-Val F1 | Target Delam Macro F1 | Class 2 F1 (Above Rebar) | Class 3 F1 (Sub-Rebar) | Defect Recall | Checkpoint SHA-256 | Verification Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `coral_fold_1_seed42` | 42 | B1 | 11 | 23 | 8.71% | **11.28%** | 22.57% | 0.00% | 36.17% | `ecadedd30373...` | **PASS (5/5)** |
| `coral_fold_1_seed123` | 123 | B1 | 58 | 70 | 13.40% | **13.86%** | 26.78% | **0.94%** | 46.81% | `48d319e03533...` | **PASS (5/5)** |
| `coral_fold_1_seed999` | 999 | B1 | 1 | 13 | 15.99% | **12.82%** | 25.28% | **0.36%** | 50.35% | `7257a3c382a6...` | **PASS (5/5)** |

### Comparison Against Phase 2B.2 Baseline on Bridge B1:
- **Baseline Mean Delamination F1:** $9.48\% \pm 5.98\%$ (Seed 42: 2.84%, Seed 123: 11.14%, Seed 999: 14.45%)
- **Deep CORAL Mean Delamination F1:** **$12.65\% \pm 1.06\%$** (Seed 42: 11.28%, Seed 123: 13.86%, Seed 999: 12.82%)
- **Empirical Gain on B1:** **$\Delta = +3.17\%$ (+33.4% relative gain)** with a **5.6x reduction in seed variance** ($\pm 1.06\%$ vs $\pm 5.98\%$).
- **Sub-Rebar Delamination Recovery:** Baseline suffered complete collapse (0.00% F1) across all seeds. Deep CORAL recovered true-positive sub-rebar delaminations in Seed 123 (237 pixels) and Seed 999 (82 pixels).

---

## 4. Hardware, Disk, and Environment Verification

```
Hardware Inventory:
  - Processor: 13th Gen Intel Core i7-13650HX (14 cores / 20 threads)
  - Dedicated GPU: NVIDIA GeForce RTX 3050 Laptop GPU (6144 MiB VRAM, Driver 581.86, System CUDA 13.0)
  - Host RAM: 15.71 GB Physical RAM
  - Free Disk Space on C:: 31.46 GB free (ample capacity)

Environment Separation:
  - Primary CPU Virtualenv: C:\Project\ml_pipeline\crack-length\venv (Python 3.12, torch==2.8.0+cpu) [PRESERVED]
  - Dedicated GPU Virtualenv: C:\Project\ml_pipeline\crack-length\venv_cuda (Python 3.12, torch==2.6.0+cu124) [READY]
```

### Installed Packages in `venv_cuda`:
- `torch`: `2.6.0+cu124`
- `torchvision`: `0.21.0+cu124`
- `numpy`: `2.5.2`
- `scipy`: `1.18.1`
- `pandas`: `3.0.6`
- `scikit-learn`: `1.9.1`
- `scikit-image`: `0.26.0`
- `pillow`: `12.3.0`
- `tqdm`: `4.70.1`

All 20 unit tests in the project test suite pass cleanly under `venv_cuda` (`Ran 20 tests in 2.58s — OK`).

---

## 5. Isolated GPU Smoke Test & Benchmark Results

The smoke test script (`tests/test_gpu_deep_coral.py`) verified real Strategy C patch dimensions ($8 \times 1 \times 256 \times 256$) without modifying benchmark outputs:

```json
{
  "timestamp": "2026-10-20 15:01:48",
  "pytorch_version": "2.6.0+cu124",
  "cuda_available": true,
  "cuda_version": "12.4",
  "device_name": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
  "tests": {
    "model_on_cuda": true,
    "tensors_on_cuda": true,
    "forward_backward_on_cuda": true,
    "total_loss_value": 1.1572
  },
  "vram_usage": {
    "allocated_mb": 225.77,
    "peak_mb": 711.18,
    "reserved_mb": 752.00
  },
  "benchmark": {
    "steps": 10,
    "cpu_per_step_sec": 0.4956,
    "gpu_per_step_sec": 0.0641,
    "speedup_factor": 7.73
  }
}
```

### Key Diagnostic Takeaways:
1. **Safety Margin:** Peak VRAM is **711.18 MB**, leaving over **5.3 GB of free VRAM** on the RTX 3050. Batch size 8 source + 8 target executes with zero memory strain.
2. **Speedup Factor:** 7.73x acceleration reduces epoch time from ~45 seconds to ~6 seconds.
3. **Total Benchmark Time:** The remaining 12 runs will execute in approximately **30 to 35 minutes total**.

---

## 6. Safe Continuation & Recovery Plan

### 6.1 Process State & Safe Transition
- Original task `task-849` (CPU PID 10740) was canceled by the task manager after finishing Run 3.
- All 3 completed Fold 1 runs were fully saved, hashed, and verified on disk prior to task exit.
- Zero partial files were generated for Fold 2.

### 6.2 Resumption Mapping Across 15 Runs

```
[Completed & Reused: 3] [To Execute on GPU: 12] (Total: 15)
```

1. `coral_fold_1_seed42`: **REUSED** (Verified checkpoint `ecadedd30373...`, Target F1: 11.28%)
2. `coral_fold_1_seed123`: **REUSED** (Verified checkpoint `48d319e03533...`, Target F1: 13.86%)
3. `coral_fold_1_seed999`: **REUSED** (Verified checkpoint `7257a3c382a6...`, Target F1: 12.82%)
4. `coral_fold_2_seed42` (Target B2): **EXECUTE ON GPU** (Run 4/15)
5. `coral_fold_2_seed123` (Target B2): **EXECUTE ON GPU** (Run 5/15)
6. `coral_fold_2_seed999` (Target B2): **EXECUTE ON GPU** (Run 6/15)
7. `coral_fold_3_seed42` (Target B3): **EXECUTE ON GPU** (Run 7/15)
8. `coral_fold_3_seed123` (Target B3): **EXECUTE ON GPU** (Run 8/15)
9. `coral_fold_3_seed999` (Target B3): **EXECUTE ON GPU** (Run 9/15)
10. `coral_fold_4_seed42` (Target B4): **EXECUTE ON GPU** (Run 10/15)
11. `coral_fold_4_seed123` (Target B4): **EXECUTE ON GPU** (Run 11/15)
12. `coral_fold_4_seed999` (Target B4): **EXECUTE ON GPU** (Run 12/15)
13. `coral_fold_5_seed42` (Target B5): **EXECUTE ON GPU** (Run 13/15)
14. `coral_fold_5_seed123` (Target B5): **EXECUTE ON GPU** (Run 14/15)
15. `coral_fold_5_seed999` (Target B5): **EXECUTE ON GPU** (Run 15/15)

### 6.3 Invariant Research Settings (100% Preserved)
- Model: `DeepCORALUNet(in_channels=1, num_classes=3, features=[8, 16, 32, 64])`
- Alignment layer: Bottleneck ($d=128$)
- Alignment loss: $\lambda_{\text{coral}} = 0.5$
- Optimization: AdamW ($\text{lr}=10^{-3}$, $\text{weight\_decay}=10^{-4}$)
- Batch size: 8 source patches + 8 unlabeled target patches
- Early stopping: Patience 12 on source-validation Macro Delamination F1
- Target labels: Strictly sealed (`target_eval_sealed/`) and evaluated offline post-freeze only
- Zero baseline modification: All baseline checkpoints, predictions, and metrics preserved untouched

### 6.4 Execution Command
```powershell
.\venv_cuda\Scripts\python.exe run_coral_experiments.py
```
