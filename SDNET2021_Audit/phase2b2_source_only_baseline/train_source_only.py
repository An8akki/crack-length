"""
Source-Only Baseline Training for SDNET2021 LOBO Benchmark
Phase 2B.2: AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Trains a conventional U-Net on source-training bridges and selects checkpoints
strictly on the designated source-validation bridge using macro delamination F1.
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

# Import local U-Net model
from unet_model import UNet

DEFAULT_AUDIT_ROOT = r"C:\Users\Dell\Documents\SDNET2021_Audit"
if not os.path.exists(DEFAULT_AUDIT_ROOT):
    DEFAULT_AUDIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "phase2b1_dataset_pipeline")
BASE_OUT_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "phase2b2_source_only_baseline")


def set_seed(seed):
    """Sets random seeds across random, numpy, and torch for maximum reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic mode where possible
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_source_train_normalization(data_dir, fold_id):
    """
    Computes intensity mean and standard deviation strictly from source-training valid pixels.
    Excluded pixels (label 255) are strictly filtered out to prevent canvas bias.
    """
    train_dir = os.path.join(data_dir, "source_train", fold_id)
    img_dir = os.path.join(train_dir, "images")
    mask_dir = os.path.join(train_dir, "masks")
    
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith("_img.png")])
    valid_intensities = []
    
    for fn in img_files:
        pid = fn.replace("_img.png", "")
        img_p = os.path.join(img_dir, fn)
        mask_p = os.path.join(mask_dir, f"{pid}_mask.png")
        
        with Image.open(img_p) as img, Image.open(mask_p) as m:
            arr_img = np.array(img, dtype=np.float32) / 255.0 # [0, 1]
            arr_mask = np.array(m)
            
        valid_px = arr_img[arr_mask != 255]
        if len(valid_px) > 0:
            valid_intensities.append(valid_px)
            
    all_valid = np.concatenate(valid_intensities)
    mean_val = float(np.mean(all_valid))
    std_val = float(np.std(all_valid))
    if std_val < 1e-4:
        std_val = 1.0
        
    return mean_val, std_val


class NormalizedPatchDataset(Dataset):
    """Dataset providing normalized 1-channel images and 3-class semantic masks."""
    def __init__(self, data_dir, fold_id, partition, mean_val, std_val):
        self.data_dir = data_dir
        self.fold_id = fold_id
        self.partition = partition
        self.mean_val = mean_val
        self.std_val = std_val
        
        part_dir = os.path.join(data_dir, partition, fold_id)
        self.img_dir = os.path.join(part_dir, "images")
        self.mask_dir = os.path.join(part_dir, "masks")
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith("_img.png")])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        fn = self.img_files[idx]
        pid = fn.replace("_img.png", "")
        
        img_p = os.path.join(self.img_dir, fn)
        mask_p = os.path.join(self.mask_dir, f"{pid}_mask.png")
        
        with Image.open(img_p) as img, Image.open(mask_p) as m:
            arr_img = np.array(img, dtype=np.float32) / 255.0
            arr_mask = np.array(m, dtype=np.int64)
            
        # Z-score normalization using source-training statistics
        arr_img_norm = (arr_img - self.mean_val) / self.std_val
        tensor_img = torch.from_numpy(arr_img_norm).unsqueeze(0) # (1, H, W)
        tensor_mask = torch.from_numpy(arr_mask) # (H, W)
        
        return {
            "image": tensor_img,
            "mask": tensor_mask,
            "patch_id": pid
        }


def compute_confusion_matrix(y_true, y_pred, num_classes=3):
    """Computes num_classes x num_classes confusion matrix ignoring 255."""
    valid_mask = (y_true >= 0) & (y_true < num_classes)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_t, y_p):
        cm[t, p] += 1
    return cm


def compute_metrics_from_cm(cm):
    """
    Computes per-class precision, recall, F1, IoU and macro metrics from a confusion matrix.
    Explicit zero-division rule: if denominator is 0, metric evaluates to 0.0.
    """
    num_classes = cm.shape[0]
    metrics = {}
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        metrics[f"class{c}_precision"] = float(precision)
        metrics[f"class{c}_recall"] = float(recall)
        metrics[f"class{c}_f1"] = float(f1)
        metrics[f"class{c}_iou"] = float(iou)
        metrics[f"class{c}_gt_pixels"] = int(cm[c, :].sum())
        
    # Macro F1 over Delamination Classes (Class 1 and Class 2)
    c1_f1 = metrics["class1_f1"]
    c2_f1 = metrics["class2_f1"]
    metrics["val_macro_delam_f1"] = float((c1_f1 + c2_f1) / 2.0)
    metrics["val_macro_delam_iou"] = float((metrics["class1_iou"] + metrics["class2_iou"]) / 2.0)
    
    # Combined Delamination (Classes 1 or 2 vs Class 0)
    tp_comb = cm[1:, 1:].sum()
    fp_comb = cm[0, 1:].sum()
    fn_comb = cm[1:, 0].sum()
    p_comb = tp_comb / (tp_comb + fp_comb) if (tp_comb + fp_comb) > 0 else 0.0
    r_comb = tp_comb / (tp_comb + fn_comb) if (tp_comb + fn_comb) > 0 else 0.0
    f1_comb = (2 * p_comb * r_comb) / (p_comb + r_comb) if (p_comb + r_comb) > 0 else 0.0
    iou_comb = tp_comb / (tp_comb + fp_comb + fn_comb) if (tp_comb + fp_comb + fn_comb) > 0 else 0.0
    
    metrics["comb_delam_precision"] = float(p_comb)
    metrics["comb_delam_recall"] = float(r_comb)
    metrics["comb_delam_f1"] = float(f1_comb)
    metrics["comb_delam_iou"] = float(iou_comb)
    
    return metrics


def train_single_run(
    fold_id="fold_1",
    seed=42,
    data_dir=DATA_DIR,
    out_dir=BASE_OUT_DIR,
    max_epochs=100,
    patience=12,
    batch_size=8,
    lr=1e-3,
    weight_decay=1e-4,
    features=[8, 16, 32, 64]
):
    """
    Executes a single source-only baseline training run with early stopping.
    """
    run_id = f"{fold_id}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"STARTING BASELINE TRAINING: {run_id}")
    print(f"{'='*60}")
    
    set_seed(seed)
    
    # Load fold metadata
    lobo_df = pd.read_csv(os.path.join(data_dir, "lobo_fold_manifest.csv"))
    fold_row = lobo_df[lobo_df["fold_id"] == fold_id].iloc[0]
    tgt_bridge = fold_row["target_bridge_id"]
    val_bridge = fold_row["val_bridge_id"]
    train_bridges = fold_row["train_bridge_ids"].split(",")
    
    # STRICT ISOLATION ASSERTION: Target bridge must NOT be in training or val
    assert tgt_bridge not in train_bridges, f"CRITICAL LEAK: Target {tgt_bridge} in train {train_bridges}"
    assert tgt_bridge != val_bridge, f"CRITICAL LEAK: Target {tgt_bridge} is val"
    
    # 1. Compute normalization statistics strictly from source-training valid pixels
    t_norm_0 = time.time()
    mean_val, std_val = compute_source_train_normalization(data_dir, fold_id)
    t_norm = time.time() - t_norm_0
    print(f"[{run_id}] Computed Source-Train Normalization: mean={mean_val:.4f}, std={std_val:.4f} in {t_norm:.2f}s")
    
    # 2. Build Datasets and DataLoaders
    train_ds = NormalizedPatchDataset(data_dir, fold_id, "source_train", mean_val, std_val)
    val_ds = NormalizedPatchDataset(data_dir, fold_id, "source_val", mean_val, std_val)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 3. Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, num_classes=3, features=features).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    
    best_val_f1 = -1.0
    best_epoch = -1
    best_cm = None
    best_val_metrics = {}
    best_state_dict = None
    
    patience_counter = 0
    history = []
    
    t_start = time.time()
    
    for epoch in range(1, max_epochs + 1):
        t_ep_start = time.time()
        
        # --- TRAIN EPOCH ---
        model.train()
        train_loss_sum = 0.0
        train_batches = 0
        
        for batch in train_loader:
            imgs = batch["image"].to(device)
            masks = batch["mask"].to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_batches += 1
            
        train_loss = train_loss_sum / train_batches if train_batches > 0 else 0.0
        
        # --- EVALUATE ON SOURCE-VAL ---
        model.eval()
        val_loss_sum = 0.0
        val_batches = 0
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                masks = batch["mask"].to(device)
                logits = model(imgs)
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
            best_cm = cm
            best_val_metrics = metrics
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            best_mark = " [*BEST]"
        else:
            patience_counter += 1
            best_mark = f" [Patience {patience_counter}/{patience}]"
            
        print(
            f"[{run_id}] Ep {epoch:03d}/{max_epochs:03d} | TrainLoss: {train_loss:.4f} | "
            f"ValLoss: {val_loss:.4f} | ValDelamF1: {val_macro_f1*100:.2f}% (C1: {metrics['class1_f1']*100:.1f}%, C2: {metrics['class2_f1']*100:.1f}%){best_mark} in {t_ep:.1f}s"
        )
        
        history.append({
            "run_id": run_id,
            "fold_id": fold_id,
            "seed": seed,
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_c0_f1": metrics["class0_f1"],
            "val_c1_f1": metrics["class1_f1"],
            "val_c2_f1": metrics["class2_f1"],
            "val_macro_delam_f1": val_macro_f1,
            "val_comb_delam_f1": metrics["comb_delam_f1"],
            "epoch_duration_sec": round(t_ep, 2)
        })
        
        if patience_counter >= patience:
            print(f"[{run_id}] Early stopping triggered after {epoch} epochs (Best Epoch: {best_epoch}, Best Val F1: {best_val_f1*100:.2f}%)")
            break
            
    total_train_time = time.time() - t_start
    
    # 4. Save and Freeze Checkpoint
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
        "model_architecture": {"name": "UNet", "in_channels": 1, "num_classes": 3, "features": features},
        "hyperparameters": {
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "max_epochs": max_epochs,
            "patience": patience
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
            "patience": patience
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
