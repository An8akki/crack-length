"""
Automated Resumption & Verification Runner for Phase 2B.2 U-Net Baseline LOBO Benchmark
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Ensures:
  1. Full integrity verification before skipping any run.
  2. Zero overwrites of valid, verified completed runs.
  3. Clean restart of interrupted runs from Epoch 1 with explicit logging.
  4. Incremental and atomic saving of manifests, history, access logs, and metrics.
  5. Automatic fleet aggregation (per-bridge mean +/- std, fleet macro summary) upon completion.
  6. Dry-run / status audit mode (--status or --dry-run) without launching training.
"""

import os
import sys
import time
import json
import shutil
import hashlib
import argparse
from datetime import datetime
import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AUDIT_ROOT = r"C:\Users\Dell\Documents\SDNET2021_Audit"
if not os.path.exists(DEFAULT_AUDIT_ROOT):
    DEFAULT_AUDIT_ROOT = os.path.dirname(SCRIPT_DIR)
AUDIT_ROOT = DEFAULT_AUDIT_ROOT
PIPELINE_DIR = os.path.join(AUDIT_ROOT, "phase2b1_dataset_pipeline")
BASELINE_DIR = os.path.join(AUDIT_ROOT, "phase2b2_source_only_baseline")
SRC_DIR = os.path.join(AUDIT_ROOT, "extracted_irt")

# Add to path for module imports
sys.path.insert(0, BASELINE_DIR)
sys.path.insert(0, SCRIPT_DIR)

try:
    from unet_model import UNet
    from train_source_only import train_single_run
    from evaluate_frozen_target import evaluate_frozen_target
except ImportError as e:
    sys.path.insert(0, BASELINE_DIR)
    from unet_model import UNet
    from train_source_only import train_single_run
    from evaluate_frozen_target import evaluate_frozen_target

ALL_FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
ALL_SEEDS = [42, 123, 999]

TARGET_BRIDGE_MAP = {
    "fold_1": "B1",
    "fold_2": "B2",
    "fold_3": "B3",
    "fold_4": "B4",
    "fold_5": "B5"
}


def verify_run_completion(run_id, fold_id, seed, baseline_dir):
    """
    Rigorously verifies whether a run is 100% complete, verified, and sealed.
    Returns: (is_complete, status_str, details_dict)
    """
    ckpt_manifest_csv = os.path.join(baseline_dir, "checkpoint_manifest.csv")
    metrics_csv = os.path.join(baseline_dir, "baseline_metrics_by_run.csv")
    access_log_csv = os.path.join(baseline_dir, "target_evaluation_access_log.csv")
    ckpts_dir = os.path.join(baseline_dir, "checkpoints")
    preds_dir = os.path.join(baseline_dir, "predictions")

    expected_ckpt = os.path.join(ckpts_dir, f"{run_id}_best_checkpoint.pt")
    expected_config = os.path.join(ckpts_dir, f"{run_id}_config.json")
    expected_pred = os.path.join(preds_dir, f"{run_id}_target_prediction.png")

    details = {
        "run_id": run_id,
        "fold_id": fold_id,
        "seed": seed,
        "target_bridge": TARGET_BRIDGE_MAP[fold_id],
        "checkpoint_exists": os.path.exists(expected_ckpt),
        "config_exists": os.path.exists(expected_config),
        "prediction_exists": os.path.exists(expected_pred),
        "checkpoint_sha256": None,
        "checkpoint_valid": False,
        "manifest_recorded": False,
        "target_evaluated": False,
        "access_logged": False,
        "selected_epoch": None,
        "val_macro_delam_f1": None,
        "target_macro_delam_f1": None,
        "target_defect_recall": None
    }

    # Check 1: Checkpoint file existence and size
    if not details["checkpoint_exists"] or os.path.getsize(expected_ckpt) < 1000:
        return False, "PENDING", details

    # Check 2: Checkpoint SHA-256 calculation & PyTorch loadability
    try:
        with open(expected_ckpt, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()
        details["checkpoint_sha256"] = computed_sha
        
        # Test loading torch state
        ckpt_data = torch.load(expected_ckpt, map_location="cpu")
        if "state_dict" in ckpt_data or "model_state_dict" in ckpt_data:
            details["checkpoint_valid"] = True
            details["selected_epoch"] = ckpt_data.get("best_epoch")
            details["val_macro_delam_f1"] = ckpt_data.get("val_macro_delam_f1")
            details["val_delam_f1"] = details["val_macro_delam_f1"]
        else:
            return False, "CORRUPTED_CHECKPOINT", details
    except Exception as e:
        details["error"] = str(e)
        return False, "CORRUPTED_CHECKPOINT", details

    # Check 3: Checkpoint manifest check
    if os.path.exists(ckpt_manifest_csv):
        try:
            df_ck = pd.read_csv(ckpt_manifest_csv)
            if run_id in df_ck["run_id"].values:
                rec = df_ck[df_ck["run_id"] == run_id].iloc[0]
                if rec["checkpoint_sha256"] == computed_sha and bool(rec["frozen"]):
                    details["manifest_recorded"] = True
        except Exception:
            pass

    # Check 4: Metrics record check
    if os.path.exists(metrics_csv):
        try:
            df_m = pd.read_csv(metrics_csv)
            if run_id in df_m["run_id"].values:
                m_rec = df_m[df_m["run_id"] == run_id].iloc[0]
                details["target_evaluated"] = True
                details["target_macro_delam_f1"] = float(m_rec.get("target_macro_delam_f1", 0.0))
                details["target_defect_recall"] = float(m_rec.get("target_defect_recall", 0.0))
        except Exception:
            pass

    # Check 5: Access log check
    if os.path.exists(access_log_csv):
        try:
            df_acc = pd.read_csv(access_log_csv)
            if run_id in df_acc["run_id"].values:
                acc_rec = df_acc[df_acc["run_id"] == run_id].iloc[0]
                if bool(acc_rec.get("integrity_verified", False)):
                    details["access_logged"] = True
        except Exception:
            pass

    # Complete determination
    if (details["checkpoint_exists"] and details["checkpoint_valid"] and
        details["manifest_recorded"] and details["target_evaluated"] and
        details["access_logged"] and details["prediction_exists"]):
        return True, "VERIFIED_COMPLETE", details
    else:
        return False, "INCOMPLETE_INTERRUPTED", details


def compile_fleet_summaries(baseline_dir):
    """Compiles per-bridge statistics and macro summary from baseline_metrics_by_run.csv."""
    metrics_csv = os.path.join(baseline_dir, "baseline_metrics_by_run.csv")
    bridge_csv = os.path.join(baseline_dir, "baseline_metrics_by_bridge.csv")
    macro_csv = os.path.join(baseline_dir, "baseline_macro_summary.csv")

    if not os.path.exists(metrics_csv):
        return

    df_runs = pd.read_csv(metrics_csv)
    if len(df_runs) == 0:
        return

    metric_cols = [
        "source_val_f1", "target_c0_f1", "target_c0_iou",
        "target_c1_precision", "target_c1_recall", "target_c1_f1", "target_c1_iou",
        "target_c2_precision", "target_c2_recall", "target_c2_f1", "target_c2_iou",
        "target_macro_delam_f1", "target_macro_delam_iou",
        "target_comb_delam_precision", "target_comb_delam_recall",
        "target_comb_delam_f1", "target_comb_delam_iou",
        "target_defect_recall", "training_duration_sec", "inference_duration_sec"
    ]

    avail_metric_cols = [c for c in metric_cols if c in df_runs.columns]

    # Group by target bridge
    bridge_records = []
    for bridge_id, group in df_runs.groupby("target_bridge"):
        rec = {
            "target_bridge": bridge_id,
            "num_seeds": len(group),
            "completed_seeds": ",".join(str(s) for s in sorted(group["seed"].tolist()))
        }
        for col in avail_metric_cols:
            rec[f"{col}_mean"] = round(float(group[col].mean()), 4)
            rec[f"{col}_std"] = round(float(group[col].std(ddof=1)) if len(group) > 1 else 0.0, 4)
        bridge_records.append(rec)

    df_bridge = pd.DataFrame(bridge_records)
    df_bridge.to_csv(bridge_csv, index=False)

    # Fleet macro summary across bridges
    macro_rec = {
        "num_bridges_evaluated": len(df_bridge),
        "total_runs_evaluated": len(df_runs),
        "aggregation_timestamp": datetime.now().isoformat()
    }
    for col in avail_metric_cols:
        macro_rec[f"fleet_mean_{col}"] = round(float(df_runs[col].mean()), 4)
        macro_rec[f"fleet_std_{col}"] = round(float(df_runs[col].std(ddof=1)) if len(df_runs) > 1 else 0.0, 4)

    df_macro = pd.DataFrame([macro_rec])
    df_macro.to_csv(macro_csv, index=False)
    print(f"[RECOVERY] Updated fleet summaries: {bridge_csv} and {macro_csv}")


def audit_experiment_status(baseline_dir):
    """Audits all 15 expected benchmark runs."""
    status_records = []
    completed_count = 0
    incomplete_count = 0
    pending_count = 0

    print("=" * 80)
    print("PHASE 2B.2 U-NET BASELINE BENCHMARK STATUS AUDIT")
    print("=" * 80)
    print(f"{'Run ID':<16} | {'Target':<6} | {'Seed':<5} | {'Status':<22} | {'Val Delam F1':<12} | {'Tgt Delam F1':<12} | {'Defect Rec':<10}")
    print("-" * 80)

    for f_id in ALL_FOLDS:
        for seed in ALL_SEEDS:
            run_id = f"{f_id}_seed{seed}"
            is_done, status_str, details = verify_run_completion(run_id, f_id, seed, baseline_dir)
            
            val_f1_val = details.get("val_macro_delam_f1")
            val_f1_str = f"{val_f1_val*100:.2f}%" if val_f1_val is not None else "N/A"
            tgt_f1_val = details.get("target_macro_delam_f1")
            tgt_f1_str = f"{tgt_f1_val*100:.2f}%" if tgt_f1_val is not None else "N/A"
            def_rec_val = details.get("target_defect_recall")
            def_rec_str = f"{def_rec_val*100:.1f}%" if def_rec_val is not None else "N/A"

            print(f"{run_id:<16} | {details['target_bridge']:<6} | {seed:<5} | {status_str:<22} | {val_f1_str:<12} | {tgt_f1_str:<12} | {def_rec_str:<10}")

            if status_str == "VERIFIED_COMPLETE":
                completed_count += 1
            elif "INCOMPLETE" in status_str or "CORRUPTED" in status_str:
                incomplete_count += 1
            else:
                pending_count += 1

            status_records.append({
                "run_id": run_id,
                "fold_id": f_id,
                "seed": seed,
                "target_bridge": details["target_bridge"],
                "status": status_str,
                "is_complete": is_done,
                "checkpoint_sha256": details["checkpoint_sha256"],
                "selected_epoch": details["selected_epoch"],
                "val_macro_delam_f1": details["val_macro_delam_f1"],
                "target_macro_delam_f1": details["target_macro_delam_f1"],
                "target_defect_recall": details["target_defect_recall"]
            })

    print("-" * 80)
    print(f"TOTAL: 15 runs | Completed & Verified: {completed_count} | Incomplete/Interrupted: {incomplete_count} | Pending: {pending_count}")
    print("=" * 80)
    return status_records, completed_count, incomplete_count, pending_count


def run_resume(baseline_dir, data_dir, src_dir, max_epochs=100, patience=12, batch_size=8, lr=1e-3, weight_decay=1e-4):
    """Executes safe resumption of outstanding benchmark runs."""
    status_records, n_comp, n_incomp, n_pend = audit_experiment_status(baseline_dir)

    if n_comp == 15:
        print("[RESUME] All 15 benchmark runs are already VERIFIED and COMPLETE. Nothing to do.")
        compile_fleet_summaries(baseline_dir)
        return

    print(f"\n[RESUME] Commencing execution of {n_incomp + n_pend} outstanding runs...")

    metrics_csv = os.path.join(baseline_dir, "baseline_metrics_by_run.csv")
    ckpt_manifest_csv = os.path.join(baseline_dir, "checkpoint_manifest.csv")
    history_csv = os.path.join(baseline_dir, "training_history.csv")

    for rec in status_records:
        run_id = rec["run_id"]
        f_id = rec["fold_id"]
        seed = rec["seed"]

        if rec["is_complete"]:
            print(f"[{run_id}] Verified complete. Skipping.")
            continue

        if rec["status"] == "INCOMPLETE_INTERRUPTED":
            print(f"\n{'!'*70}")
            print(f"[{run_id}] Interrupted run detected. Cleaning up and restarting cleanly from Epoch 1...")
            print(f"{'!'*70}")
        else:
            print(f"\n{'-'*70}")
            print(f"[{run_id}] Starting run...")
            print(f"{'-'*70}")

        # Train single run
        train_res = train_single_run(
            fold_id=f_id,
            seed=seed,
            data_dir=data_dir,
            out_dir=baseline_dir,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            features=[8, 16, 32, 64]
        )

        # Append to checkpoint manifest atomically
        ckpt_rec = {
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
        }
        if os.path.exists(ckpt_manifest_csv):
            df_ck = pd.read_csv(ckpt_manifest_csv)
            df_ck = df_ck[df_ck["run_id"] != run_id]
            df_ck = pd.concat([df_ck, pd.DataFrame([ckpt_rec])], ignore_index=True)
        else:
            df_ck = pd.DataFrame([ckpt_rec])
        df_ck.to_csv(ckpt_manifest_csv + ".tmp", index=False)
        shutil.move(ckpt_manifest_csv + ".tmp", ckpt_manifest_csv)

        # Append training history
        if os.path.exists(history_csv):
            df_hist = pd.read_csv(history_csv)
            df_hist = df_hist[df_hist["run_id"] != run_id]
            df_hist = pd.concat([df_hist, pd.DataFrame(train_res["history"])], ignore_index=True)
        else:
            df_hist = pd.DataFrame(train_res["history"])
        df_hist.to_csv(history_csv + ".tmp", index=False)
        shutil.move(history_csv + ".tmp", history_csv)

        # Run offline target evaluation
        eval_res = evaluate_frozen_target(
            config_path=train_res["config_path"],
            data_dir=data_dir,
            src_dir=src_dir,
            out_dir=baseline_dir
        )
        eval_res["training_duration_sec"] = train_res["training_duration_sec"]
        eval_res["checkpoint_sha256"] = train_res["checkpoint_sha256"]

        # Append metrics atomically
        if os.path.exists(metrics_csv):
            df_m = pd.read_csv(metrics_csv)
            df_m = df_m[df_m["run_id"] != run_id]
            df_m = pd.concat([df_m, pd.DataFrame([eval_res])], ignore_index=True)
        else:
            df_m = pd.DataFrame([eval_res])
        df_m.to_csv(metrics_csv + ".tmp", index=False)
        shutil.move(metrics_csv + ".tmp", metrics_csv)

        print(f"[{run_id}] COMPLETED & SEALED. Target Delam F1: {eval_res.get('target_macro_delam_f1', 0)*100:.2f}%")

    compile_fleet_summaries(baseline_dir)
    print("\n[ALL BENCHMARK RUNS COMPLETED SUCCESSFULLY]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2B.2 Baseline Resumption & Status Auditor")
    parser.add_argument("--status", action="store_true", help="Audit status of all 15 runs without training")
    parser.add_argument("--dry-run", action="store_true", help="Audit status of all 15 runs without training")
    parser.add_argument("--baseline-dir", type=str, default=BASELINE_DIR)
    parser.add_argument("--data-dir", type=str, default=PIPELINE_DIR)
    parser.add_argument("--src-dir", type=str, default=SRC_DIR)
    args = parser.parse_args()

    if args.status or args.dry_run:
        audit_experiment_status(args.baseline_dir)
    else:
        run_resume(args.baseline_dir, args.data_dir, args.src_dir)
