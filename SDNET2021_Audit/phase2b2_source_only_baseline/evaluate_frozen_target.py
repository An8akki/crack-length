"""
Frozen Target Evaluation for SDNET2021 LOBO Benchmark
Phase 2B.2: AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Performs full-bridge stitched sliding-window inference on held-out target bridges
using verified frozen checkpoints. Enforces zero target leakage safeguards.
"""

import os
import sys
import time
import json
import hashlib
from datetime import datetime
import numpy as np
from PIL import Image
import pandas as pd
from scipy import ndimage
import torch

from unet_model import UNet

DEFAULT_AUDIT_ROOT = r"C:\Users\Dell\Documents\SDNET2021_Audit"
if not os.path.exists(DEFAULT_AUDIT_ROOT):
    DEFAULT_AUDIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "extracted_irt")
DATA_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "phase2b1_dataset_pipeline")
BASE_OUT_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "phase2b2_source_only_baseline")

BRIDGE_SOURCE_MAP = {
    "B1": {"folder": "1.Forest River NB", "orig": "FR_NB_Original.png", "annot": "FR_NB_Annotated.png"},
    "B2": {"folder": "2.Forest River SB", "orig": "FR_SB_Original.png", "annot": "FR_SB_Annotated.png"},
    "B3": {"folder": "3.Park River Median", "orig": "PR_MD_Original.png", "annot": "PR_MD_Annotated.png"},
    "B4": {"folder": "4.Park River North Bound", "orig": "PR_NB_Original.png", "annot": "PR_NB_Annotaed.png"},
    "B5": {"folder": "5.Park River South Bound", "orig": "PR_SB_Original.png", "annot": "PR_SB_Annotated.png"}
}


def log_target_evaluation_access(log_csv_path, record):
    """Appends access record to target_evaluation_access_log.csv."""
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    df = pd.DataFrame([record])
    if not os.path.exists(log_csv_path):
        df.to_csv(log_csv_path, index=False)
    else:
        df.to_csv(log_csv_path, mode="a", header=False, index=False)


def evaluate_frozen_target(
    config_path,
    data_dir=DATA_DIR,
    src_dir=SRC_DIR,
    out_dir=BASE_OUT_DIR
):
    """
    Evaluates a frozen checkpoint on its corresponding held-out target bridge.
    """
    # 1. Load and Verify Frozen Configuration
    assert os.path.exists(config_path), f"Configuration file missing: {config_path}"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    run_id = config["run_id"]
    fold_id = config["fold_id"]
    seed = config["seed"]
    tgt_bridge = config["target_bridge"]
    ckpt_file = config["checkpoint_file"]
    recorded_sha256 = config["checkpoint_sha256"]
    is_frozen = config.get("frozen", False)
    
    assert is_frozen, f"Run {run_id} is NOT marked as frozen! Evaluation blocked."
    
    ckpt_path = os.path.join(out_dir, "checkpoints", ckpt_file)
    assert os.path.exists(ckpt_path), f"Checkpoint missing: {ckpt_path}"
    
    # Compute SHA-256 checksum and verify integrity
    with open(ckpt_path, "rb") as f:
        actual_sha256 = hashlib.sha256(f.read()).hexdigest()
        
    assert actual_sha256 == recorded_sha256, (
        f"INTEGRITY VIOLATION in {run_id}: Checkpoint SHA-256 mismatch!\n"
        f"Recorded: {recorded_sha256}\nActual:   {actual_sha256}"
    )
    
    # Log access
    access_log_path = os.path.join(out_dir, "target_evaluation_access_log.csv")
    access_record = {
        "timestamp": datetime.now().isoformat(),
        "run_id": run_id,
        "fold_id": fold_id,
        "seed": seed,
        "target_bridge": tgt_bridge,
        "checkpoint_file": ckpt_file,
        "checkpoint_sha256": actual_sha256,
        "access_type": "post_freeze_target_inference",
        "integrity_verified": True
    }
    log_target_evaluation_access(access_log_path, access_record)
    
    print(f"\n[EVAL TARGET] Executing Target Evaluation for {run_id} on {tgt_bridge} (SHA-256 Verified)")
    
    # 2. Load Model and Frozen Weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = config.get("features", [8, 16, 32, 64])
    model = UNet(in_channels=1, num_classes=3, features=features).to(device)
    
    checkpoint_payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint_payload["state_dict"])
    model.eval()
    
    # 3. Load Target Stitching Metadata from Sealed Partition
    sealed_dir = os.path.join(data_dir, "target_eval_sealed", fold_id)
    manifest_path = os.path.join(sealed_dir, "eval_stitch_manifest.json")
    assert os.path.exists(manifest_path), f"Missing stitch manifest: {manifest_path}"
    
    with open(manifest_path, "r", encoding="utf-8") as f:
        stitch_manifest = json.load(f)
        
    W = stitch_manifest["dimensions"]["width"]
    H = stitch_manifest["dimensions"]["height"]
    patches = stitch_manifest["patches"]
    
    mean_val = config["normalization"]["mean"]
    std_val = config["normalization"]["std"]
    
    # 4. Full-Bridge Sliding Window Inference with Logit Accumulation
    t_inf_0 = time.time()
    logit_accumulator = torch.zeros((3, H, W), dtype=torch.float32)
    count_accumulator = torch.zeros((1, H, W), dtype=torch.float32)
    
    with torch.no_grad():
        for p in patches:
            y1, y2 = p["y_min"], p["y_max"]
            x1, x2 = p["x_min"], p["x_max"]
            img_fn = p["image_file"]
            img_p = os.path.join(sealed_dir, "images", img_fn)
            
            with Image.open(img_p) as img:
                arr_img = np.array(img, dtype=np.float32) / 255.0
                
            arr_norm = (arr_img - mean_val) / std_val
            tensor_in = torch.from_numpy(arr_norm).unsqueeze(0).unsqueeze(0).to(device) # (1, 1, 256, 256)
            
            logits = model(tensor_in).squeeze(0).cpu() # (3, 256, 256)
            
            logit_accumulator[:, y1:y2, x1:x2] += logits
            count_accumulator[:, y1:y2, x1:x2] += 1.0
            
    # Normalize logits by window overlap counts
    normalized_logits = logit_accumulator / count_accumulator
    final_pred_map = torch.argmax(normalized_logits, dim=0).numpy().astype(np.uint8) # (H, W)
    t_inf = time.time() - t_inf_0
    
    # 5. Load Target Ground-Truth Semantic Mask
    b_info = BRIDGE_SOURCE_MAP[tgt_bridge]
    annot_path = os.path.join(src_dir, b_info["folder"], b_info["annot"])
    with Image.open(annot_path) as img_a:
        arr_annot_rgb = np.array(img_a.convert("RGB"))
        
    R = arr_annot_rgb[:, :, 0]
    G = arr_annot_rgb[:, :, 1]
    B = arr_annot_rgb[:, :, 2]
    
    mask_c2 = (R == 0) & (G == 255) & (B == 0)
    mask_c3 = (R == 255) & (G == 0) & (B == 0)
    mask_sound = (R == G) & (G == B) & (R > 0) & (R < 255)
    
    gt_semantic = np.full((H, W), 255, dtype=np.uint8)
    gt_semantic[mask_sound] = 0
    gt_semantic[mask_c2] = 1
    gt_semantic[mask_c3] = 2
    
    valid_mask = (gt_semantic != 255)
    num_valid_eval = int(valid_mask.sum())
    
    # 6. Compute Target Metrics Strictly on Valid Physical Pixels
    y_true = gt_semantic[valid_mask]
    y_pred = final_pred_map[valid_mask]
    
    # Build 3x3 confusion matrix
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
        
    # Per-class metrics
    metrics = {}
    for c in range(3):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = (2 * p_val * r_val) / (p_val + r_val) if (p_val + r_val) > 0 else 0.0
        iou_val = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        metrics[f"class{c}_precision"] = float(p_val)
        metrics[f"class{c}_recall"] = float(r_val)
        metrics[f"class{c}_f1"] = float(f1_val)
        metrics[f"class{c}_iou"] = float(iou_val)
        metrics[f"class{c}_gt_pixels"] = int(cm[c, :].sum())
        
    # Delamination Macro (Class 1 & Class 2)
    c1_f1 = metrics["class1_f1"]
    c2_f1 = metrics["class2_f1"]
    macro_delam_f1 = float((c1_f1 + c2_f1) / 2.0)
    macro_delam_iou = float((metrics["class1_iou"] + metrics["class2_iou"]) / 2.0)
    
    # Combined Delamination (Classes 1 or 2 vs Class 0)
    tp_comb = cm[1:, 1:].sum()
    fp_comb = cm[0, 1:].sum()
    fn_comb = cm[1:, 0].sum()
    p_comb = tp_comb / (tp_comb + fp_comb) if (tp_comb + fp_comb) > 0 else 0.0
    r_comb = tp_comb / (tp_comb + fn_comb) if (tp_comb + fn_comb) > 0 else 0.0
    f1_comb = (2 * p_comb * r_comb) / (p_comb + r_comb) if (p_comb + r_comb) > 0 else 0.0
    iou_comb = tp_comb / (tp_comb + fp_comb + fn_comb) if (tp_comb + fp_comb + fn_comb) > 0 else 0.0
    
    # Defect-level recall via connected components (minimum area 50 px)
    gt_delam_binary = (gt_semantic == 1) | (gt_semantic == 2)
    labeled_defects, num_defects = ndimage.label(gt_delam_binary, structure=np.ones((3, 3)))
    pred_delam_binary = (final_pred_map == 1) | (final_pred_map == 2)
    
    detected_defects = 0
    evaluated_defects = 0
    for defect_idx in range(1, num_defects + 1):
        defect_mask = (labeled_defects == defect_idx)
        area = int(defect_mask.sum())
        if area >= 50: # Exclude tiny 1-2 pixel noise
            evaluated_defects += 1
            overlap = int((defect_mask & pred_delam_binary).sum())
            if (overlap / area) >= 0.10: # At least 10% overlap threshold
                detected_defects += 1
                
    defect_recall = float(detected_defects / evaluated_defects) if evaluated_defects > 0 else 0.0
    
    # Save Stitched Prediction Raster
    pred_dir = os.path.join(out_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    pred_save_path = os.path.join(pred_dir, f"{run_id}_target_prediction.png")
    
    # Preserve excluded pixels as 255 in the saved visualization raster
    final_pred_map_masked = final_pred_map.copy()
    final_pred_map_masked[~valid_mask] = 255
    Image.fromarray(final_pred_map_masked, mode="L").save(pred_save_path)
    
    # Confusion matrix string for reporting
    cm_str = f"[[{cm[0,0]},{cm[0,1]},{cm[0,2]}],[{cm[1,0]},{cm[1,1]},{cm[1,2]}],[{cm[2,0]},{cm[2,1]},{cm[2,2]}]]"
    
    eval_result = {
        "run_id": run_id,
        "fold_id": fold_id,
        "seed": seed,
        "target_bridge": tgt_bridge,
        "selected_epoch": config["best_epoch"],
        "source_val_f1": round(config["best_val_macro_delam_f1"], 4),
        "target_valid_pixels": num_valid_eval,
        "target_c0_pixels": metrics["class0_gt_pixels"],
        "target_c1_pixels": metrics["class1_gt_pixels"],
        "target_c2_pixels": metrics["class2_gt_pixels"],
        "target_c0_precision": round(metrics["class0_precision"], 4),
        "target_c0_recall": round(metrics["class0_recall"], 4),
        "target_c0_f1": round(metrics["class0_f1"], 4),
        "target_c0_iou": round(metrics["class0_iou"], 4),
        "target_c1_precision": round(metrics["class1_precision"], 4),
        "target_c1_recall": round(metrics["class1_recall"], 4),
        "target_c1_f1": round(metrics["class1_f1"], 4),
        "target_c1_iou": round(metrics["class1_iou"], 4),
        "target_c2_precision": round(metrics["class2_precision"], 4),
        "target_c2_recall": round(metrics["class2_recall"], 4),
        "target_c2_f1": round(metrics["class2_f1"], 4),
        "target_c2_iou": round(metrics["class2_iou"], 4),
        "target_macro_delam_f1": round(macro_delam_f1, 4),
        "target_macro_delam_iou": round(macro_delam_iou, 4),
        "target_comb_delam_precision": round(p_comb, 4),
        "target_comb_delam_recall": round(r_comb, 4),
        "target_comb_delam_f1": round(f1_comb, 4),
        "target_comb_delam_iou": round(iou_comb, 4),
        "target_defect_recall": round(defect_recall, 4),
        "target_total_defects_ge_50px": evaluated_defects,
        "target_detected_defects": detected_defects,
        "confusion_matrix": cm_str,
        "inference_duration_sec": round(t_inf, 2),
        "prediction_raster_file": f"{run_id}_target_prediction.png"
    }
    
    print(f"[{run_id}] Finished Target Evaluation on {tgt_bridge}:")
    print(f"  Target Delam Macro F1: {macro_delam_f1*100:.2f}% | C1 F1: {metrics['class1_f1']*100:.1f}% | C2 F1: {metrics['class2_f1']*100:.1f}% | Comb F1: {f1_comb*100:.1f}% | Defect Recall: {defect_recall*100:.1f}%")
    
    return eval_result
