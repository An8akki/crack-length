# Git Repository Handover & Version Control Guide

**Repository URL:** `https://github.com/An8akki/crack-length.git`  
**Target Branch:** `depth-estimation-integration`  
**Tracking Remote:** `origin/depth-estimation-integration`  
**Commit Title:** `Prepare research pipeline for cross-system resume`  

---

## 1. Repository Structure & Branch Policy

- **Strict Branch Adherence:** All research artifacts, audit reports, dataset pipeline specifications, and baseline code are committed directly to `depth-estimation-integration`.
- **No Branch Overwrites:** Never force-push (`git push --force`).
- **No Discarding of Pre-Existing Code:** Pre-existing features in the repository (e.g. 3D depth estimation, planar calibration, crack quantification, webapp APIs) remain untouched.

---

## 2. Handled Files and Artifact Tracking Policy

### A. Tracked Research Artifacts
The following directories and files are included in version control:
1. **Audit Reports & Mathematical Reconciliations:**
   - `phase1c_bridge1_pilot/`
   - `phase1d_five_bridge_audit/`
   - `phase1e_alignment_verification/`
   - `phase1e1_reconciliation/`
   - `phase1e2_b4_alignment_audit/`
   - `phase2a_dataset_plan/`
   - `phase2a1_protocol_correction/`
   - `phase2a2_independent_verification/`
   - `phase2a3_coverage_reconciliation/`
2. **Dataset Pipeline Code & Manifests (Phase 2B.1):**
   - `phase2b1_dataset_pipeline/dataset_build.py`
   - `phase2b1_dataset_pipeline/dataset_validation_tests.py`
   - `phase2b1_dataset_pipeline/lobo_fold_manifest.csv`
   - `phase2b1_dataset_pipeline/patch_manifest.csv`
   - `phase2b1_dataset_pipeline/pipeline_verification_report.md`
   - `phase2b1_dataset_pipeline/verification_results.txt`
   - `phase2b1_dataset_pipeline/target_eval_sealed/*/eval_stitch_manifest.json`
3. **Source-Only Baseline Benchmark (Phase 2B.2):**
   - `phase2b2_source_only_baseline/unet_model.py`
   - `phase2b2_source_only_baseline/train_source_only.py`
   - `phase2b2_source_only_baseline/evaluate_frozen_target.py`
   - `phase2b2_source_only_baseline/run_experiment.py`
   - `phase2b2_source_only_baseline/resume_experiment.py`
   - `phase2b2_source_only_baseline/baseline_validation_tests.py`
   - `phase2b2_source_only_baseline/baseline_metrics_by_run.csv`
   - `phase2b2_source_only_baseline/checkpoint_manifest.csv`
   - `phase2b2_source_only_baseline/training_history.csv`
   - `phase2b2_source_only_baseline/target_evaluation_access_log.csv`
   - `phase2b2_source_only_baseline/environment_and_hardware.txt`
   - `phase2b2_source_only_baseline/checkpoints/*.json` (run configs)
   - `phase2b2_source_only_baseline/predictions/*.png` (stitched prediction maps)
4. **Handover Specifications (`handover/`):**
   - `README_NEXT_SYSTEM.md`
   - `PROJECT_STATE.md`
   - `shutdown_audit.md`
   - `git_handover.md`
   - `environment_requirements.txt`
   - `experiment_resume_status.json`
   - `checksums.sha256`

### B. Files Excluded from Version Control (Size & Policy Safeguards)
1. **Large External Datasets:**
   - `RGBD_Concrete_Crack_Dataset.zip` (**9.98 GB**): Stored locally in workspace root; strictly excluded from Git staging.
   - SDNET2021 raw infrared images (`extracted_irt/*.png`, 4.4 MB) and `5. INFRARED THERMOGRAPHY DATA.zip`: To adhere to data licensing policies, raw source datasets are maintained locally and documented via checksums.
2. **Materialized Intermediate Patches (Phase 2B.1):**
   - The 6,131 small tile PNGs (~47.79 MB) generated across `source_train/`, `source_val/`, and `target_uda/` are excluded from the Git tree. Running `python dataset_build.py` reproduces the identical bit-exact patches in ~20 seconds.
3. **Local Checkpoint Binaries (`*.pt`):**
   - The 3 completed checkpoints (`fold_1_seed42_best_checkpoint.pt`, `fold_1_seed123_best_checkpoint.pt`, `fold_1_seed999_best_checkpoint.pt`) are 1.99 MB each. They can be transferred via Git or secondary transfer. Their SHA-256 signatures are sealed in `checksums.sha256` and `checkpoint_manifest.csv`.

---

## 3. Git Commands for Next-System Setup

On the new computer, open **Windows PowerShell** and run:

```powershell
# 1. Clone the repository directly to the research branch
git clone -b depth-estimation-integration https://github.com/An8akki/crack-length.git

# 2. Enter repository directory
cd crack-length

# 3. Verify branch and remote tracking
git branch -vv
git remote -v

# 4. Check status to ensure working tree is clean
git status
```
