"""
Full 15-Run LOBO Benchmark Orchestrator for Phase 2B.2
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Executes:
  - 5 LOBO Folds x 3 Seeds (42, 123, 999) = 15 Baseline Runs
  - Checkpoint selection strictly on source-validation macro delamination F1
  - Checkpoint freezing with SHA-256 checksums
  - Offline post-freeze full-bridge stitched target evaluation
  - Generates master metrics CSVs and audit logs
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Local imports
from train_source_only import train_single_run
from evaluate_frozen_target import evaluate_frozen_target

DATA_DIR = r"C:\Users\Dell\Documents\SDNET2021_Audit\phase2b1_dataset_pipeline"
BASE_OUT_DIR = r"C:\Users\Dell\Documents\SDNET2021_Audit\phase2b2_source_only_baseline"
SRC_DIR = r"C:\Users\Dell\Documents\SDNET2021_Audit\extracted_irt"

FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
SEEDS = [42, 123, 999]


def run_all_experiments(
    max_epochs=100,
    patience=12,
    batch_size=8,
    lr=1e-3,
    weight_decay=1e-4,
    features=[8, 16, 32, 64],
    resume=True
):
    print("=" * 80)
    print("STARTING 15-RUN SOURCE-ONLY U-NET LOBO BENCHMARK (PHASE 2B.2)")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Features: {features} | LR: {lr} | Weight Decay: {weight_decay} | Batch Size: {batch_size}")
    print(f"Max Epochs: {max_epochs} | Patience: {patience} | Seeds: {SEEDS}")
    print("=" * 80)
    
    os.makedirs(os.path.join(BASE_OUT_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(BASE_OUT_DIR, "predictions"), exist_ok=True)
    
    metrics_run_csv = os.path.join(BASE_OUT_DIR, "baseline_metrics_by_run.csv")
    history_csv = os.path.join(BASE_OUT_DIR, "training_history.csv")
    checkpoint_manifest_csv = os.path.join(BASE_OUT_DIR, "checkpoint_manifest.csv")
    
    run_records = []
    all_histories = []
    checkpoint_records = []
    
    # Check for existing completed runs if resuming
    existing_runs = set()
    if resume and os.path.exists(metrics_run_csv):
        try:
            df_existing = pd.read_csv(metrics_run_csv)
            existing_runs = set(df_existing["run_id"].unique())
            run_records = df_existing.to_dict("records")
            print(f"Resuming: Found {len(existing_runs)} already completed runs in {metrics_run_csv}")
        except Exception:
            pass

    total_runs = len(FOLDS) * len(SEEDS)
    current_run_idx = 0
    t_suite_start = time.time()
    
    for f_id in FOLDS:
        for seed in SEEDS:
            current_run_idx += 1
            run_id = f"{f_id}_seed{seed}"
            
            print(f"\n{'='*70}")
            print(f"PROGRESS: Run {current_run_idx}/{total_runs} -> {run_id}")
            print(f"{'='*70}")
            
            if resume and run_id in existing_runs:
                print(f"[{run_id}] Already completed. Skipping training and target evaluation.")
                continue
                
            # 1. Train Model and Freeze Checkpoint
            t0 = time.time()
            train_res = train_single_run(
                fold_id=f_id,
                seed=seed,
                data_dir=DATA_DIR,
                out_dir=BASE_OUT_DIR,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                features=features
            )
            t_train = time.time() - t0
            
            # Record checkpoint
            checkpoint_records.append({
                "run_id": run_id,
                "fold_id": f_id,
                "seed": seed,
                "target_bridge": train_res["target_bridge"],
                "best_epoch": train_res["best_epoch"],
                "total_epochs": train_res["total_epochs"],
                "val_macro_delam_f1": train_res["best_val_delam_f1"],
                "checkpoint_path": train_res["checkpoint_path"],
                "checkpoint_sha256": train_res["checkpoint_sha256"],
                "frozen": True,
                "freeze_timestamp": datetime.now().isoformat()
            })
            
            # Save checkpoint manifest incrementally
            df_ckpts = pd.DataFrame(checkpoint_records)
            df_ckpts.to_csv(checkpoint_manifest_csv, index=False)
            
            # Record history
            all_histories.extend(train_res["history"])
            df_hist = pd.DataFrame(all_histories)
            df_hist.to_csv(history_csv, index=False)
            
            # 2. Offline Target Evaluation on Frozen Checkpoint
            t_eval_0 = time.time()
            eval_res = evaluate_frozen_target(
                config_path=train_res["config_path"],
                data_dir=DATA_DIR,
                src_dir=SRC_DIR,
                out_dir=BASE_OUT_DIR
            )
            t_eval = time.time() - t_eval_0
            
            # Combined record
            eval_res["training_duration_sec"] = train_res["training_duration_sec"]
            eval_res["checkpoint_sha256"] = train_res["checkpoint_sha256"]
            eval_res["val_delam_iou"] = round(train_res["best_val_delam_iou"], 4)
            run_records.append(eval_res)
            
            # Save metrics by run incrementally
            df_runs = pd.DataFrame(run_records)
            df_runs.to_csv(metrics_run_csv, index=False)
            
            print(f"[{run_id}] Finished in {time.time()-t0:.1f}s (Train: {t_train:.1f}s, Eval: {t_eval:.1f}s)")
            
    total_suite_time = time.time() - t_suite_start
    print(f"\nAll 15 runs finished in {total_suite_time/60:.2f} minutes!")
    
    # 3. Aggregate Metrics by Target Bridge (mean +/- std across 3 seeds)
    df_all_runs = pd.DataFrame(run_records)
    bridge_agg_records = []
    
    metric_cols = [
        "source_val_f1", "val_delam_iou",
        "target_c0_f1", "target_c0_iou",
        "target_c1_precision", "target_c1_recall", "target_c1_f1", "target_c1_iou",
        "target_c2_precision", "target_c2_recall", "target_c2_f1", "target_c2_iou",
        "target_macro_delam_f1", "target_macro_delam_iou",
        "target_comb_delam_precision", "target_comb_delam_recall", "target_comb_delam_f1", "target_comb_delam_iou",
        "target_defect_recall"
    ]
    
    for tgt_b in ["B1", "B2", "B3", "B4", "B5"]:
        sub = df_all_runs[df_all_runs["target_bridge"] == tgt_b]
        b_rec = {
            "target_bridge": tgt_b,
            "num_runs": len(sub),
            "target_valid_pixels": int(sub["target_valid_pixels"].iloc[0]),
            "target_c1_gt_pixels": int(sub["target_c1_pixels"].iloc[0]),
            "target_c2_gt_pixels": int(sub["target_c2_pixels"].iloc[0]),
            "target_total_defects": int(sub["target_total_defects_ge_50px"].iloc[0])
        }
        for col in metric_cols:
            vals = sub[col].values
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            b_rec[f"{col}_mean"] = round(mean_v, 4)
            b_rec[f"{col}_std"] = round(std_v, 4)
            b_rec[f"{col}_formatted"] = f"{mean_v*100:.2f} ± {std_v*100:.2f}%"
            
        bridge_agg_records.append(b_rec)
        
    df_bridge = pd.DataFrame(bridge_agg_records)
    bridge_csv_path = os.path.join(BASE_OUT_DIR, "baseline_metrics_by_bridge.csv")
    df_bridge.to_csv(bridge_csv_path, index=False)
    print(f"Saved: {bridge_csv_path}")
    
    # 4. Aggregate Overall Fleet Macro Summary
    macro_rec = {
        "fleet_bridges_count": 5,
        "total_benchmark_runs": 15,
        "seeds_evaluated": "42, 123, 999",
        "fleet_total_valid_pixels": 6089851
    }
    for col in metric_cols:
        means = df_bridge[f"{col}_mean"].values
        overall_mean = float(np.mean(means))
        overall_std = float(np.std(means))
        macro_rec[f"macro_{col}_mean"] = round(overall_mean, 4)
        macro_rec[f"macro_{col}_domain_std"] = round(overall_std, 4)
        macro_rec[f"macro_{col}_formatted"] = f"{overall_mean*100:.2f} ± {overall_std*100:.2f}%"
        
    df_macro = pd.DataFrame([macro_rec])
    macro_csv_path = os.path.join(BASE_OUT_DIR, "baseline_macro_summary.csv")
    df_macro.to_csv(macro_csv_path, index=False)
    print(f"Saved: {macro_csv_path}")
    
    print("\nBENCHMARK ORCHESTRATION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    run_all_experiments()
