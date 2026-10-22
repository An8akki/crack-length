"""
Full 15-Run LOBO Deep CORAL Domain Adaptation Orchestrator (Phase 5)
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Executes:
  - 5 LOBO Folds x 3 Seeds (42, 123, 999) = 15 Deep CORAL Runs
  - Feature alignment via Deep CORAL (lambda_coral = 0.5 at bottleneck layer d=128)
  - Unlabeled target domain DataLoader from target_uda/ with zero target mask access
  - Model selection strictly on source-validation macro delamination F1 (patience = 12)
  - Checkpoint freezing with SHA-256 checksums
  - Offline post-freeze full-bridge stitched target evaluation on held-out bridge
  - Preserves Phase 2B.2 baseline artifacts intact (writes to phase5_deep_coral/ and results/)
  - Generates master metrics CSVs, paired bridge comparison, and confusion matrices
"""

import os
import sys
import time
import json
import hashlib
import random
import numpy as np
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Ensure project root and baseline directories are in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR
BASELINE_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "phase2b2_source_only_baseline")
DATA_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "phase2b1_dataset_pipeline")
SRC_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "extracted_irt")
CORAL_OUT_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "phase5_deep_coral")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from unet_model import UNet
from src.domain_adaptation.deep_coral import DeepCORALUNet, coral_loss
from train_source_only import (
    compute_source_train_normalization,
    NormalizedPatchDataset,
    compute_confusion_matrix,
    compute_metrics_from_cm,
    set_seed
)
from evaluate_frozen_target import evaluate_frozen_target

FOLDS = ["fold_1", "fold_2", "fold_3", "fold_4", "fold_5"]
SEEDS = [42, 123, 999]


class UnlabeledNormalizedPatchDataset(Dataset):
    """
    Dataset providing normalized single-channel unlabeled images for domain adaptation.
    Strictly contains NO masks or label attributes. Attempting to access 'mask' raises KeyError.
    Normalization statistics (mean, std) are derived strictly from source-training valid pixels.
    """
    def __init__(self, data_dir, fold_id, mean_val, std_val):
        self.data_dir = data_dir
        self.fold_id = fold_id
        self.mean_val = mean_val
        self.std_val = std_val

        self.img_dir = os.path.join(data_dir, "target_uda", fold_id, "images")
        if not os.path.exists(self.img_dir):
            raise FileNotFoundError(f"Target UDA image directory missing: {self.img_dir}")

        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith("_img.png")])
        if len(self.img_files) == 0:
            raise ValueError(f"No target UDA patches found in: {self.img_dir}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fn = self.img_files[idx]
        pid = fn.replace("_img.png", "")
        img_p = os.path.join(self.img_dir, fn)

        with Image.open(img_p) as img:
            arr_img = np.array(img, dtype=np.float32) / 255.0

        # Z-score normalization using source-training statistics
        arr_img_norm = (arr_img - self.mean_val) / self.std_val
        tensor_img = torch.from_numpy(arr_img_norm).unsqueeze(0)  # (1, H, W)

        return {
            "image": tensor_img,
            "patch_id": pid
        }


def colorize_prediction_map(pred_map):
    """
    Colorizes prediction raster using research palette:
      Sound (Class 0): Gray [128, 128, 128]
      Class 2 Above-Rebar (Class 1): Green [0, 255, 0]
      Class 3 Sub-Rebar (Class 2): Red [255, 0, 0]
      Excluded (255): White [255, 255, 255]
    """
    H, W = pred_map.shape
    colorized = np.full((H, W, 3), 255, dtype=np.uint8)
    colorized[pred_map == 0] = [128, 128, 128]
    colorized[pred_map == 1] = [0, 255, 0]
    colorized[pred_map == 2] = [255, 0, 0]
    return colorized


def train_single_coral_run(
    fold_id="fold_1",
    seed=42,
    data_dir=DATA_DIR,
    out_dir=CORAL_OUT_DIR,
    max_epochs=100,
    patience=12,
    batch_size=8,
    lr=1e-3,
    weight_decay=1e-4,
    features=[8, 16, 32, 64],
    lambda_coral=0.5,
    feature_layer="bottleneck"
):
    """
    Executes a single Deep CORAL domain adaptation training run.
    Model selection is performed strictly on source-validation macro delamination F1.
    """
    run_id = f"coral_{fold_id}_seed{seed}"
    print(f"\n{'='*70}")
    print(f"STARTING DEEP CORAL ADAPTATION RUN: {run_id}")
    print(f"Fold: {fold_id} | Seed: {seed} | Lambda CORAL: {lambda_coral} | Layer: {feature_layer}")
    print(f"{'='*70}")

    set_seed(seed)

    # 1. Load fold metadata and enforce strict LOBO isolation
    lobo_df = pd.read_csv(os.path.join(data_dir, "lobo_fold_manifest.csv"))
    fold_row = lobo_df[lobo_df["fold_id"] == fold_id].iloc[0]
    tgt_bridge = fold_row["target_bridge_id"]
    val_bridge = fold_row["val_bridge_id"]
    train_bridges = fold_row["train_bridge_ids"].split(",")

    assert tgt_bridge not in train_bridges, f"CRITICAL LEAK: Target {tgt_bridge} in train {train_bridges}"
    assert tgt_bridge != val_bridge, f"CRITICAL LEAK: Target {tgt_bridge} is val bridge"

    # 2. Compute normalization statistics strictly from source-training valid pixels
    t_norm_0 = time.time()
    mean_val, std_val = compute_source_train_normalization(data_dir, fold_id)
    t_norm = time.time() - t_norm_0
    print(f"[{run_id}] Source-Train Normalization: mean={mean_val:.4f}, std={std_val:.4f} (computed in {t_norm:.2f}s)")

    # 3. Build Datasets and DataLoaders
    # Source labeled datasets
    train_ds = NormalizedPatchDataset(data_dir, fold_id, "source_train", mean_val, std_val)
    val_ds = NormalizedPatchDataset(data_dir, fold_id, "source_val", mean_val, std_val)

    # Target unlabeled dataset (strictly images only)
    tgt_ds = UnlabeledNormalizedPatchDataset(data_dir, fold_id, mean_val, std_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    target_loader = DataLoader(tgt_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    print(f"[{run_id}] Dataset sizes: Train={len(train_ds)}, Val={len(val_ds)}, Target UDA={len(tgt_ds)}")

    # 4. Instantiate DeepCORALUNet, Optimizer, and Loss Criterion
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dev_desc = f"{torch.cuda.get_device_name(0)} (CUDA)" if device.type == "cuda" else "CPU"
    print(f"[{run_id}] Execution Device: {device} [{dev_desc}]")
    model = DeepCORALUNet(in_channels=1, num_classes=3, features=features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    best_val_f1 = -1.0
    best_epoch = -1
    best_val_metrics = {}
    best_state_dict = None

    patience_counter = 0
    history = []

    t_start = time.time()

    # Cycling target iterator reshuffled upon StopIteration
    target_iter = iter(target_loader)

    for epoch in range(1, max_epochs + 1):
        t_ep_start = time.time()

        # --- TRAINING WITH CORAL ALIGNMENT ---
        model.train()
        train_loss_sum = 0.0
        train_seg_loss_sum = 0.0
        train_coral_loss_sum = 0.0
        train_batches = 0

        for src_batch in train_loader:
            try:
                tgt_batch = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                tgt_batch = next(target_iter)

            src_imgs = src_batch["image"].to(device)
            src_masks = src_batch["mask"].to(device)
            tgt_imgs = tgt_batch["image"].to(device)

            optimizer.zero_grad()

            # Forward pass on source
            src_logits, src_feats = model(src_imgs, return_features=True, feature_layer=feature_layer)
            # Forward pass on target (unlabeled)
            _, tgt_feats = model(tgt_imgs, return_features=True, feature_layer=feature_layer)

            # Losses
            loss_seg = criterion(src_logits, src_masks)
            loss_coral = coral_loss(src_feats, tgt_feats)
            total_loss = loss_seg + lambda_coral * loss_coral

            total_loss.backward()
            optimizer.step()

            train_loss_sum += total_loss.item()
            train_seg_loss_sum += loss_seg.item()
            train_coral_loss_sum += loss_coral.item()
            train_batches += 1

        train_loss = train_loss_sum / train_batches if train_batches > 0 else 0.0
        train_seg_loss = train_seg_loss_sum / train_batches if train_batches > 0 else 0.0
        train_coral_loss = train_coral_loss_sum / train_batches if train_batches > 0 else 0.0

        # --- EVALUATE ON SOURCE-VAL (Model Selection) ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_true = []
        all_pred = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs, return_features=False)
                loss = criterion(logits, masks)

                val_loss_sum += loss.item()
                val_batches += 1

                preds = torch.argmax(logits, dim=1)
                all_true.append(masks.cpu().numpy().flatten())
                all_pred.append(preds.cpu().numpy().flatten())

        val_loss = val_loss_sum / val_batches if val_batches > 0 else 0.0
        y_true_all = np.concatenate(all_true)
        y_pred_all = np.concatenate(all_pred)

        cm = compute_confusion_matrix(y_true_all, y_pred_all, num_classes=3)
        metrics = compute_metrics_from_cm(cm)
        val_macro_f1 = metrics["val_macro_delam_f1"]

        t_ep = time.time() - t_ep_start

        is_best = (val_macro_f1 > best_val_f1)
        if is_best:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            best_val_metrics = metrics
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_mark = " [*BEST]"
        else:
            patience_counter += 1
            best_mark = f" [Patience {patience_counter}/{patience}]"

        print(
            f"[{run_id}] Ep {epoch:03d}/{max_epochs:03d} | TotalLoss: {train_loss:.4f} "
            f"(Seg: {train_seg_loss:.4f}, CORAL: {train_coral_loss:.4f}) | "
            f"ValLoss: {val_loss:.4f} | ValDelamF1: {val_macro_f1*100:.2f}% "
            f"(C1: {metrics['class1_f1']*100:.1f}%, C2: {metrics['class2_f1']*100:.1f}%){best_mark} in {t_ep:.1f}s"
        )

        history.append({
            "run_id": run_id,
            "fold_id": fold_id,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_seg_loss": train_seg_loss,
            "train_coral_loss": train_coral_loss,
            "val_loss": val_loss,
            "val_c0_f1": metrics["class0_f1"],
            "val_c1_f1": metrics["class1_f1"],
            "val_c2_f1": metrics["class2_f1"],
            "val_macro_delam_f1": val_macro_f1,
            "val_comb_delam_f1": metrics["comb_delam_f1"],
            "epoch_duration_sec": round(t_ep, 2)
        })

        if patience_counter >= patience:
            print(f"[{run_id}] Early stopping triggered at epoch {epoch} (Best Epoch: {best_epoch}, Best Val F1: {best_val_f1*100:.2f}%)")
            break

    total_train_time = time.time() - t_start

    # 5. Save and Freeze Checkpoint
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_filename = f"{run_id}_best_checkpoint.pt"
    ckpt_path = os.path.join(ckpt_dir, ckpt_filename)

    checkpoint_payload = {
        "run_id": run_id,
        "fold_id": fold_id,
        "seed": seed,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "val_macro_delam_f1": best_val_f1,
        "val_metrics": best_val_metrics,
        "normalization": {"mean": mean_val, "std": std_val},
        "model_architecture": {
            "name": "DeepCORALUNet",
            "in_channels": 1,
            "num_classes": 3,
            "features": features,
            "feature_layer": feature_layer,
            "lambda_coral": lambda_coral
        },
        "hyperparameters": {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "lambda_coral": lambda_coral,
            "feature_layer": feature_layer
        },
        "bridges": {
            "target": tgt_bridge,
            "val": val_bridge,
            "train": train_bridges
        },
        "state_dict": best_state_dict
    }

    torch.save(checkpoint_payload, ckpt_path)

    # Compute SHA-256 Checksum
    with open(ckpt_path, "rb") as f:
        ckpt_sha256 = hashlib.sha256(f.read()).hexdigest()

    # Save Run Config JSON
    config_filename = f"{run_id}_config.json"
    config_path = os.path.join(ckpt_dir, config_filename)
    config_payload = {
        "run_id": run_id,
        "fold_id": fold_id,
        "seed": seed,
        "target_bridge": tgt_bridge,
        "val_bridge": val_bridge,
        "train_bridges": train_bridges,
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "best_val_macro_delam_f1": best_val_f1,
        "best_val_metrics": best_val_metrics,
        "normalization": {"mean": mean_val, "std": std_val},
        "features": features,
        "hyperparameters": {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience,
            "lambda_coral": lambda_coral,
            "feature_layer": feature_layer
        },
        "checkpoint_file": ckpt_filename,
        "checkpoint_sha256": ckpt_sha256,
        "frozen": True,
        "training_duration_sec": round(total_train_time, 2)
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    print(f"[{run_id}] Saved FROZEN checkpoint: {ckpt_path} (SHA-256: {ckpt_sha256[:12]}...)")

    return {
        "run_id": run_id,
        "fold_id": fold_id,
        "seed": seed,
        "target_bridge": tgt_bridge,
        "val_bridge": val_bridge,
        "train_bridges": ",".join(train_bridges),
        "best_epoch": best_epoch,
        "total_epochs": epoch,
        "best_val_delam_f1": best_val_f1,
        "best_val_delam_iou": best_val_metrics["val_macro_delam_iou"],
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": ckpt_sha256,
        "config_path": config_path,
        "training_duration_sec": round(total_train_time, 2),
        "history": history
    }


def run_all_coral_experiments(
    max_epochs=100,
    patience=12,
    batch_size=8,
    lr=1e-3,
    weight_decay=1e-4,
    features=[8, 16, 32, 64],
    lambda_coral=0.5,
    feature_layer="bottleneck",
    resume=True
):
    """
    Orchestrates the full 15-run LOBO Deep CORAL domain adaptation benchmark.
    Performs post-freeze target evaluation, generates full artifact suites, and updates paired comparison.
    """
    print("=" * 80)
    print("STARTING 15-RUN DEEP CORAL DOMAIN ADAPTATION BENCHMARK (PHASE 5)")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    active_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    active_dev_name = torch.cuda.get_device_name(0) if active_dev.type == "cuda" else "CPU"
    print(f"Compute Device: {active_dev} [{active_dev_name}]")
    print(f"Features: {features} | LR: {lr} | Weight Decay: {weight_decay} | Batch: {batch_size}")
    print(f"Lambda CORAL: {lambda_coral} | Layer: {feature_layer} | Max Ep: {max_epochs} | Patience: {patience}")
    print(f"Output Directory: {CORAL_OUT_DIR}")
    print("=" * 80)

    os.makedirs(os.path.join(CORAL_OUT_DIR, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(CORAL_OUT_DIR, "predictions"), exist_ok=True)
    os.makedirs(os.path.join(CORAL_OUT_DIR, "confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "deep_coral_confusion_matrices"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "visualizations"), exist_ok=True)

    metrics_run_csv = os.path.join(CORAL_OUT_DIR, "coral_metrics_by_run.csv")
    history_csv = os.path.join(CORAL_OUT_DIR, "coral_training_history.csv")
    checkpoint_manifest_csv = os.path.join(CORAL_OUT_DIR, "coral_checkpoint_manifest.csv")

    run_records = []
    all_histories = []
    checkpoint_records = []

    # Check for existing completed runs if resuming with strict 5-point verification:
    # 1. Row in metrics CSV with valid target_macro_delam_f1
    # 2. Checkpoint .pt exists, non-empty, and loadable
    # 3. Config .json exists with frozen: True and SHA-256 matches actual file
    # 4. Prediction .png exists in both out_dir/predictions and results/visualizations
    # 5. Confusion matrix .csv exists in both confusion_matrices/ and results/deep_coral_confusion_matrices
    existing_runs = set()
    if resume and os.path.exists(metrics_run_csv):
        try:
            df_existing = pd.read_csv(metrics_run_csv)
            for _, r in df_existing.iterrows():
                r_id = r["run_id"]
                ckpt_p = os.path.join(CORAL_OUT_DIR, "checkpoints", f"{r_id}_best_checkpoint.pt")
                cfg_p = os.path.join(CORAL_OUT_DIR, "checkpoints", f"{r_id}_config.json")
                pred_p1 = os.path.join(CORAL_OUT_DIR, "predictions", f"{r_id}_target_prediction.png")
                pred_p2 = os.path.join(RESULTS_DIR, "visualizations", f"{r_id}_target_prediction.png")
                cm_p1 = os.path.join(CORAL_OUT_DIR, "confusion_matrices", f"{r_id}_cm.csv")
                cm_p2 = os.path.join(RESULTS_DIR, "deep_coral_confusion_matrices", f"{r_id}_cm.csv")

                # Verify all artifacts
                if not (os.path.exists(ckpt_p) and os.path.getsize(ckpt_p) > 0):
                    continue
                if not (os.path.exists(cfg_p) and os.path.getsize(cfg_p) > 0):
                    continue
                try:
                    with open(cfg_p, "r", encoding="utf-8") as f_cfg:
                        cfg_data = json.load(f_cfg)
                    if not cfg_data.get("frozen", False):
                        continue
                    with open(ckpt_p, "rb") as f_ck:
                        actual_h = hashlib.sha256(f_ck.read()).hexdigest()
                    if cfg_data.get("checkpoint_sha256") != actual_h:
                        continue
                except Exception:
                    continue

                if not (os.path.exists(pred_p1) and os.path.getsize(pred_p1) > 0 and os.path.exists(pred_p2)):
                    continue
                if not (os.path.exists(cm_p1) and os.path.getsize(cm_p1) > 0 and os.path.exists(cm_p2)):
                    continue
                if pd.isnull(r.get("target_macro_delam_f1")):
                    continue

                existing_runs.add(r_id)

            run_records = [r for r in df_existing.to_dict("records") if r["run_id"] in existing_runs]
            print(f"Resuming: Found {len(existing_runs)} fully verified completed runs in {metrics_run_csv}")
        except Exception as e:
            print(f"Warning during resume inspection: {e}")

    if resume and os.path.exists(checkpoint_manifest_csv):
        try:
            df_ck_ex = pd.read_csv(checkpoint_manifest_csv)
            checkpoint_records = [r for r in df_ck_ex.to_dict("records") if r["run_id"] in existing_runs]
        except Exception:
            pass

    if resume and os.path.exists(history_csv):
        try:
            df_h_ex = pd.read_csv(history_csv)
            all_histories = [r for r in df_h_ex.to_dict("records") if r["run_id"] in existing_runs]
        except Exception:
            pass

    total_runs = len(FOLDS) * len(SEEDS)
    current_run_idx = 0
    t_suite_start = time.time()

    for f_id in FOLDS:
        for seed in SEEDS:
            current_run_idx += 1
            run_id = f"coral_{f_id}_seed{seed}"

            print(f"\n{'='*75}")
            print(f"BENCHMARK PROGRESS: Run {current_run_idx}/{total_runs} -> {run_id}")
            print(f"{'='*75}")

            if resume and run_id in existing_runs:
                print(f"[{run_id}] Already completed with verified checkpoint. Skipping training and target evaluation.")
                continue

            # 1. Train Deep CORAL and Freeze Checkpoint
            t0 = time.time()
            train_res = train_single_coral_run(
                fold_id=f_id,
                seed=seed,
                data_dir=DATA_DIR,
                out_dir=CORAL_OUT_DIR,
                max_epochs=max_epochs,
                patience=patience,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                features=features,
                lambda_coral=lambda_coral,
                feature_layer=feature_layer
            )
            t_train = time.time() - t0

            # Record checkpoint manifest entry
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
                "freeze_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

            # Save checkpoint manifest incrementally
            df_ckpts = pd.DataFrame(checkpoint_records)
            df_ckpts.to_csv(checkpoint_manifest_csv, index=False)

            # Record training history incrementally
            all_histories.extend(train_res["history"])
            df_hist = pd.DataFrame(all_histories)
            df_hist.to_csv(history_csv, index=False)

            # 2. Offline Post-Freeze Full-Bridge Target Evaluation
            t_eval_0 = time.time()
            eval_res = evaluate_frozen_target(
                config_path=train_res["config_path"],
                data_dir=DATA_DIR,
                src_dir=SRC_DIR,
                out_dir=CORAL_OUT_DIR
            )
            t_eval = time.time() - t_eval_0

            # Combined record
            eval_res["training_duration_sec"] = train_res["training_duration_sec"]
            eval_res["checkpoint_sha256"] = train_res["checkpoint_sha256"]
            eval_res["val_delam_iou"] = round(train_res["best_val_delam_iou"], 4)
            run_records.append(eval_res)

            # Save metrics by run incrementally to both phase5_deep_coral and results/
            df_runs = pd.DataFrame(run_records)
            df_runs.to_csv(metrics_run_csv, index=False)
            results_run_csv = os.path.join(RESULTS_DIR, "coral_metrics_by_run.csv")
            df_runs.to_csv(results_run_csv, index=False)

            # 3. Export Confusion Matrix CSV
            # Parse confusion matrix string: "[[...],[...],[...]]"
            cm_raw = json.loads(eval_res["confusion_matrix"])
            cm_df = pd.DataFrame(
                cm_raw,
                index=["Actual_Sound_0", "Actual_Class2_1", "Actual_Class3_2"],
                columns=["Pred_Sound_0", "Pred_Class2_1", "Pred_Class3_2"]
            )
            cm_csv_path1 = os.path.join(CORAL_OUT_DIR, "confusion_matrices", f"{run_id}_cm.csv")
            cm_csv_path2 = os.path.join(RESULTS_DIR, "deep_coral_confusion_matrices", f"{run_id}_cm.csv")
            cm_df.to_csv(cm_csv_path1)
            cm_df.to_csv(cm_csv_path2)

            # 4. Copy Prediction Raster & Colorize for Visualizations
            pred_png_src = os.path.join(CORAL_OUT_DIR, "predictions", f"{run_id}_target_prediction.png")
            pred_png_dst = os.path.join(RESULTS_DIR, "visualizations", f"{run_id}_target_prediction.png")
            if os.path.exists(pred_png_src):
                with Image.open(pred_png_src) as img_pred:
                    arr_pred = np.array(img_pred)
                    img_pred.save(pred_png_dst)

                    # Generate colorized visualization
                    colorized_arr = colorize_prediction_map(arr_pred)
                    colorized_dst = os.path.join(RESULTS_DIR, "visualizations", f"{run_id}_target_prediction_colorized.png")
                    Image.fromarray(colorized_arr).save(colorized_dst)

            print(f"[{run_id}] Finished in {time.time()-t0:.1f}s (Train: {t_train:.1f}s, Eval: {t_eval:.1f}s)")

    total_suite_time = time.time() - t_suite_start
    print(f"\n{'='*80}")
    print(f"ALL 15 DEEP CORAL EXPERIMENTS FINISHED IN {total_suite_time/60:.2f} MINUTES!")
    print(f"{'='*80}")

    # 5. Aggregate Metrics by Target Bridge (mean +/- std across 3 seeds)
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
            "completed_seeds": ",".join(str(s) for s in sorted(sub["seed"].unique())),
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
    bridge_csv_path1 = os.path.join(CORAL_OUT_DIR, "coral_metrics_by_bridge.csv")
    bridge_csv_path2 = os.path.join(RESULTS_DIR, "coral_metrics_by_bridge.csv")
    df_bridge.to_csv(bridge_csv_path1, index=False)
    df_bridge.to_csv(bridge_csv_path2, index=False)
    print(f"Saved Bridge Summaries:\n  - {bridge_csv_path1}\n  - {bridge_csv_path2}")

    # 6. Aggregate Overall Fleet Macro Summary
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
    macro_csv_path1 = os.path.join(CORAL_OUT_DIR, "coral_macro_summary.csv")
    macro_csv_path2 = os.path.join(RESULTS_DIR, "coral_macro_summary.csv")
    df_macro.to_csv(macro_csv_path1, index=False)
    df_macro.to_csv(macro_csv_path2, index=False)
    print(f"Saved Fleet Macro Summaries:\n  - {macro_csv_path1}\n  - {macro_csv_path2}")

    # 7. Update Paired Bridge Comparison Table
    update_paired_bridge_comparison(df_all_runs, df_bridge)

    print("\nDEEP CORAL BENCHMARK ORCHESTRATION COMPLETED SUCCESSFULLY!")


def update_paired_bridge_comparison(df_all_runs, df_coral_bridge):
    """
    Updates results/paired_bridge_comparison.csv with paired baseline vs Deep CORAL deltas.
    Predeclared practical significance threshold: delta_macro_f1 >= +0.05 (+5.0%).
    """
    paired_csv_path = os.path.join(RESULTS_DIR, "paired_bridge_comparison.csv")
    baseline_bridge_csv = os.path.join(RESULTS_DIR, "source_only_metrics_by_bridge.csv")

    assert os.path.exists(paired_csv_path), f"Missing paired comparison CSV: {paired_csv_path}"
    assert os.path.exists(baseline_bridge_csv), f"Missing baseline bridge CSV: {baseline_bridge_csv}"

    df_paired = pd.read_csv(paired_csv_path)
    df_base_bridge = pd.read_csv(baseline_bridge_csv)

    updated_rows = []
    for idx, row in df_paired.iterrows():
        b_id = row["bridge_id"]
        b_name = row["bridge_name"]

        # Baseline stats
        sub_base = df_base_bridge[df_base_bridge["target_bridge"] == b_id].iloc[0]
        base_macro_f1 = float(sub_base["target_macro_delam_f1_mean"])
        base_macro_std = float(sub_base["target_macro_delam_f1_std"])
        base_c1_f1 = float(sub_base["target_c1_f1_mean"])
        base_c2_f1 = float(sub_base["target_c2_f1_mean"])

        # Coral stats
        sub_coral = df_coral_bridge[df_coral_bridge["target_bridge"] == b_id].iloc[0]
        coral_macro_f1 = float(sub_coral["target_macro_delam_f1_mean"])
        coral_macro_std = float(sub_coral["target_macro_delam_f1_std"])
        coral_c1_f1 = float(sub_coral["target_c1_f1_mean"])
        coral_c2_f1 = float(sub_coral["target_c2_f1_mean"])

        # Seed scores
        sub_runs = df_all_runs[df_all_runs["target_bridge"] == b_id]
        coral_s42 = float(sub_runs[sub_runs["seed"] == 42]["target_macro_delam_f1"].iloc[0])
        coral_s123 = float(sub_runs[sub_runs["seed"] == 123]["target_macro_delam_f1"].iloc[0])
        coral_s999 = float(sub_runs[sub_runs["seed"] == 999]["target_macro_delam_f1"].iloc[0])

        # Deltas (Coral - Baseline)
        delta_macro = round(coral_macro_f1 - base_macro_f1, 4)
        delta_c2 = round(coral_c1_f1 - base_c1_f1, 4) # SDNET Class 2 is model c1
        delta_c3 = round(coral_c2_f1 - base_c2_f1, 4) # SDNET Class 3 is model c2
        practical_met = bool(delta_macro >= 0.05)

        updated_rows.append({
            "bridge_id": b_id,
            "bridge_name": b_name,
            "status": "completed",
            "baseline_seed42_delam_f1": row["baseline_seed42_delam_f1"],
            "baseline_seed123_delam_f1": row["baseline_seed123_delam_f1"],
            "baseline_seed999_delam_f1": row["baseline_seed999_delam_f1"],
            "baseline_mean_delam_f1": round(base_macro_f1, 4),
            "baseline_std_delam_f1": round(base_macro_std, 4),
            "coral_mean_delam_f1": round(coral_macro_f1, 4),
            "coral_std_delam_f1": round(coral_macro_std, 4),
            "delta_macro_f1": delta_macro,
            "delta_class2_f1": delta_c2,
            "delta_class3_f1": delta_c3,
            "practical_threshold_met_ge_005": practical_met
        })

    df_updated = pd.DataFrame(updated_rows)
    df_updated.to_csv(paired_csv_path, index=False)
    print(f"\nUpdated Paired Comparison Table:\n{df_updated[['bridge_id', 'baseline_mean_delam_f1', 'coral_mean_delam_f1', 'delta_macro_f1', 'practical_threshold_met_ge_005']]}")


if __name__ == "__main__":
    run_all_coral_experiments()
