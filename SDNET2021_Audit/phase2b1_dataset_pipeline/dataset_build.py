"""
SDNET2021 Dataset Pipeline Builder for Phase 2B.1
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Builds Strategy C (256x256, stride 128, end-anchored) patches,
creates deterministic manifests, and establishes leakage-resistant LOBO partitions.
"""

import os
import sys
import shutil
import stat
import json
import numpy as np
from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

DEFAULT_AUDIT_ROOT = r"C:\Users\Dell\Documents\SDNET2021_Audit"
if not os.path.exists(DEFAULT_AUDIT_ROOT):
    DEFAULT_AUDIT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "extracted_irt")
OUT_DIR = os.path.join(DEFAULT_AUDIT_ROOT, "phase2b1_dataset_pipeline")

BRIDGES = [
    {
        "id": "B1",
        "name": "Forest River NB",
        "folder": "1.Forest River NB",
        "orig_file": "FR_NB_Original.png",
        "annot_file": "FR_NB_Annotated.png",
        "W": 461, "H": 1942, "expected_patches": 45
    },
    {
        "id": "B2",
        "name": "Forest River SB",
        "folder": "2.Forest River SB",
        "orig_file": "FR_SB_Original.png",
        "annot_file": "FR_SB_Annotated.png",
        "W": 1626, "H": 401, "expected_patches": 36
    },
    {
        "id": "B3",
        "name": "Park River Median",
        "folder": "3.Park River Median",
        "orig_file": "PR_MD_Original.png",
        "annot_file": "PR_MD_Annotated.png",
        "W": 4326, "H": 411, "expected_patches": 99
    },
    {
        "id": "B4",
        "name": "Park River North Bound",
        "folder": "4.Park River North Bound",
        "orig_file": "PR_NB_Original.png",
        "annot_file": "PR_NB_Annotaed.png", # Source spelling
        "W": 2991, "H": 371, "expected_patches": 46
    },
    {
        "id": "B5",
        "name": "Park River South Bound",
        "folder": "5.Park River South Bound",
        "orig_file": "PR_SB_Original.png",
        "annot_file": "PR_SB_Annotated.png",
        "W": 451, "H": 4174, "expected_patches": 96
    }
]

S = 256
STEP = 128


def decode_annotation_mask(arr_rgb):
    """
    Decodes RGB annotation image into 4-class semantic mask:
      0: Sound concrete (grayscale R==G==B, 0 < R < 255)
      1: Class 2 delamination (pure green [0, 255, 0])
      2: Class 3 delamination (pure red [255, 0, 0])
      255: Excluded / boundary pixels (pure white [255, 255, 255] or pure black [0, 0, 0])
    Raises ValueError if unknown colors are present.
    """
    R = arr_rgb[:, :, 0]
    G = arr_rgb[:, :, 1]
    B = arr_rgb[:, :, 2]
    
    mask_c2 = (R == 0) & (G == 255) & (B == 0)
    mask_c3 = (R == 255) & (G == 0) & (B == 0)
    mask_white = (R == 255) & (G == 255) & (B == 255)
    mask_black = (R == 0) & (G == 0) & (B == 0)
    mask_sound = (R == G) & (G == B) & (R > 0) & (R < 255)
    
    mask_known = mask_c2 | mask_c3 | mask_white | mask_black | mask_sound
    unknown_count = int((~mask_known).sum())
    if unknown_count > 0:
        bad_y, bad_x = np.where(~mask_known)
        sample_rgb = arr_rgb[bad_y[0], bad_x[0]]
        raise ValueError(
            f"Source decoding error: Found {unknown_count} unexpected unknown color pixels! "
            f"First sample at (y={bad_y[0]}, x={bad_x[0]}): RGB={sample_rgb.tolist()}"
        )
        
    semantic_mask = np.full(R.shape, 255, dtype=np.uint8)
    semantic_mask[mask_sound] = 0
    semantic_mask[mask_c2] = 1
    semantic_mask[mask_c3] = 2
    
    valid_mask = (semantic_mask != 255).astype(np.uint8)
    
    color_breakdown = {
        "sound": int(mask_sound.sum()),
        "c2": int(mask_c2.sum()),
        "c3": int(mask_c3.sum()),
        "white": int(mask_white.sum()),
        "black": int(mask_black.sum()),
        "unknown": unknown_count,
        "valid": int(mask_sound.sum() + mask_c2.sum() + mask_c3.sum()),
        "excluded": int(mask_white.sum() + mask_black.sum())
    }
    return semantic_mask, valid_mask, color_breakdown


def generate_strategy_c_coordinates(W, H, S=256, step=128):
    """Generates unique, sorted, end-anchored window coordinates."""
    xs = list(range(0, W - S + 1, step))
    if xs[-1] != W - S:
        xs.append(W - S)
    xs = sorted(list(set(xs)))
    
    ys = list(range(0, H - S + 1, step))
    if ys[-1] != H - S:
        ys.append(H - S)
    ys = sorted(list(set(ys)))
    
    windows = []
    for r_idx, y in enumerate(ys):
        for c_idx, x in enumerate(xs):
            windows.append({
                "grid_row": r_idx,
                "grid_col": c_idx,
                "y_min": y,
                "y_max": y + S,
                "x_min": x,
                "x_max": x + S,
                "is_x_end_anchor": (x == W - S and x != (c_idx * step)),
                "is_y_end_anchor": (y == H - S and y != (r_idx * step))
            })
    return windows, xs, ys


def build_pipeline(src_dir=SRC_DIR, out_dir=OUT_DIR):
    print("=" * 70)
    print("STARTING SDNET2021 DATASET PIPELINE BUILD (PHASE 2B.1)")
    print("=" * 70)
    
    os.makedirs(out_dir, exist_ok=True)
    patch_bank_dir = os.path.join(out_dir, "patches")
    patch_img_dir = os.path.join(patch_bank_dir, "images")
    patch_mask_dir = os.path.join(patch_bank_dir, "masks")
    patch_valid_dir = os.path.join(patch_bank_dir, "validity")
    
    os.makedirs(patch_img_dir, exist_ok=True)
    os.makedirs(patch_mask_dir, exist_ok=True)
    os.makedirs(patch_valid_dir, exist_ok=True)
    
    patch_records = []
    bridge_class_records = []
    
    fleet_total_pixels = 0
    fleet_valid_pixels = 0
    fleet_patches_count = 0
    
    bridge_sources = {}
    
    # Process each bridge
    for b in BRIDGES:
        b_id = b["id"]
        b_name = b["name"]
        orig_path = os.path.join(src_dir, b["folder"], b["orig_file"])
        annot_path = os.path.join(src_dir, b["folder"], b["annot_file"])
        
        assert os.path.exists(orig_path), f"Missing original file: {orig_path}"
        assert os.path.exists(annot_path), f"Missing annotated file: {annot_path}"
        
        with Image.open(orig_path) as img_o:
            W, H = img_o.size
            arr_orig = np.array(img_o.convert("L")) # Preserved 8-bit digital values
            
        with Image.open(annot_path) as img_a:
            W_a, H_a = img_a.size
            assert (W, H) == (W_a, H_a), f"Dimension mismatch in {b_id}: {W}x{H} vs {W_a}x{H_a}"
            arr_annot_rgb = np.array(img_a.convert("RGB"))
            
        assert W == b["W"] and H == b["H"], f"Expected dimensions {b['W']}x{b['H']}, got {W}x{H}"
        
        semantic_mask, valid_mask, breakdown = decode_annotation_mask(arr_annot_rgb)
        
        tot_pixels = W * H
        fleet_total_pixels += tot_pixels
        fleet_valid_pixels += breakdown["valid"]
        
        bridge_sources[b_id] = {
            "arr_orig": arr_orig,
            "semantic_mask": semantic_mask,
            "valid_mask": valid_mask,
            "W": W, "H": H, "breakdown": breakdown
        }
        
        windows, xs, ys = generate_strategy_c_coordinates(W, H, S=S, step=STEP)
        assert len(windows) == b["expected_patches"], (
            f"Expected {b['expected_patches']} patches for {b_id}, got {len(windows)}"
        )
        fleet_patches_count += len(windows)
        
        b_zero_c2 = 0
        b_zero_c3 = 0
        b_zero_delam = 0
        
        for idx, win in enumerate(windows):
            y1, y2 = win["y_min"], win["y_max"]
            x1, x2 = win["x_min"], win["x_max"]
            
            patch_img = arr_orig[y1:y2, x1:x2]
            patch_mask = semantic_mask[y1:y2, x1:x2]
            patch_val = valid_mask[y1:y2, x1:x2]
            
            assert patch_img.shape == (S, S)
            assert patch_mask.shape == (S, S)
            assert patch_val.shape == (S, S)
            
            n_sound = int((patch_mask == 0).sum())
            n_c2 = int((patch_mask == 1).sum())
            n_c3 = int((patch_mask == 2).sum())
            n_delam = n_c2 + n_c3
            n_white = int(((patch_mask == 255) & (arr_annot_rgb[y1:y2, x1:x2, 0] == 255)).sum())
            n_black = int(((patch_mask == 255) & (arr_annot_rgb[y1:y2, x1:x2, 0] == 0)).sum())
            n_excl = int((patch_mask == 255).sum())
            n_valid = int((patch_val == 1).sum())
            
            assert n_sound + n_c2 + n_c3 == n_valid
            assert n_valid + n_excl == S * S
            
            has_c2 = (n_c2 > 0)
            has_c3 = (n_c3 > 0)
            has_delam = (n_delam > 0)
            
            if not has_c2: b_zero_c2 += 1
            if not has_c3: b_zero_c3 += 1
            if not has_delam: b_zero_delam += 1
            
            patch_id = f"{b_id}_c_r{win['grid_row']:02d}_c{win['grid_col']:02d}_y{y1:04d}_x{x1:04d}"
            
            img_filename = f"{patch_id}_img.png"
            mask_filename = f"{patch_id}_mask.png"
            val_filename = f"{patch_id}_valid.png"
            
            img_save_path = os.path.join(patch_img_dir, img_filename)
            mask_save_path = os.path.join(patch_mask_dir, mask_filename)
            val_save_path = os.path.join(patch_valid_dir, val_filename)
            
            Image.fromarray(patch_img, mode="L").save(img_save_path)
            Image.fromarray(patch_mask, mode="L").save(mask_save_path)
            Image.fromarray(patch_val * 255, mode="L").save(val_save_path)
            
            patch_records.append({
                "patch_id": patch_id,
                "bridge_id": b_id,
                "bridge_name": b_name,
                "source_orig_file": b["orig_file"],
                "source_annot_file": b["annot_file"],
                "grid_row": win["grid_row"],
                "grid_col": win["grid_col"],
                "y_min": y1,
                "y_max": y2,
                "x_min": x1,
                "x_max": x2,
                "patch_width": S,
                "patch_height": S,
                "total_pixels": S * S,
                "sound_pixels": n_sound,
                "class2_pixels": n_c2,
                "class3_pixels": n_c3,
                "delam_total_pixels": n_delam,
                "excluded_white_pixels": n_white,
                "excluded_black_pixels": n_black,
                "total_excluded_pixels": n_excl,
                "valid_deck_pixels": n_valid,
                "valid_deck_fraction": round(n_valid / (S * S), 6),
                "has_class2": has_c2,
                "has_class3": has_c3,
                "has_delam": has_delam,
                "is_x_end_anchor": win["is_x_end_anchor"],
                "is_y_end_anchor": win["is_y_end_anchor"],
                "rel_image_path": os.path.join("patches", "images", img_filename),
                "rel_mask_path": os.path.join("patches", "masks", mask_filename),
                "rel_validity_path": os.path.join("patches", "validity", val_filename)
            })
            
        bridge_class_records.append({
            "bridge_id": b_id,
            "bridge_name": b_name,
            "dimensions": f"{W}x{H}",
            "total_pixels": tot_pixels,
            "class0_sound_pixels": breakdown["sound"],
            "class0_sound_pct": round(breakdown["sound"] / tot_pixels * 100, 4),
            "class1_c2_pixels": breakdown["c2"],
            "class1_c2_pct": round(breakdown["c2"] / tot_pixels * 100, 4),
            "class2_c3_pixels": breakdown["c3"],
            "class2_c3_pct": round(breakdown["c3"] / tot_pixels * 100, 4),
            "total_delam_pixels": breakdown["c2"] + breakdown["c3"],
            "total_delam_pct": round((breakdown["c2"] + breakdown["c3"]) / tot_pixels * 100, 4),
            "valid_deck_pixels": breakdown["valid"],
            "valid_deck_pct": round(breakdown["valid"] / tot_pixels * 100, 4),
            "excluded_white_pixels": breakdown["white"],
            "excluded_white_pct": round(breakdown["white"] / tot_pixels * 100, 4),
            "excluded_black_pixels": breakdown["black"],
            "excluded_black_pct": round(breakdown["black"] / tot_pixels * 100, 4),
            "unknown_pixels": breakdown["unknown"],
            "total_excluded_pixels": breakdown["excluded"],
            "total_excluded_pct": round(breakdown["excluded"] / tot_pixels * 100, 4),
            "strategy_c_patches_count": len(windows),
            "patches_zero_c2_count": b_zero_c2,
            "patches_zero_c3_count": b_zero_c3,
            "patches_zero_delam_count": b_zero_delam,
            "accounting_check_delta": tot_pixels - (breakdown["valid"] + breakdown["excluded"])
        })
        print(f"[{b_id}] Processed {len(windows)} patches. Valid deck: {breakdown['valid']} px. Zero delam patches: {b_zero_delam}")

    assert fleet_patches_count == 322, f"Fleet patches must equal 322, got {fleet_patches_count}"
    assert fleet_total_pixels == 6317409
    assert fleet_valid_pixels == 6089851
    
    df_patch_manifest = pd.DataFrame(patch_records)
    patch_manifest_path = os.path.join(out_dir, "patch_manifest.csv")
    df_patch_manifest.to_csv(patch_manifest_path, index=False)
    print(f"Saved: {patch_manifest_path}")
    
    bridge_class_records.append({
        "bridge_id": "Fleet",
        "bridge_name": "Five Bridge Fleet",
        "dimensions": "—",
        "total_pixels": fleet_total_pixels,
        "class0_sound_pixels": sum(r["class0_sound_pixels"] for r in bridge_class_records),
        "class0_sound_pct": round(sum(r["class0_sound_pixels"] for r in bridge_class_records) / fleet_total_pixels * 100, 4),
        "class1_c2_pixels": sum(r["class1_c2_pixels"] for r in bridge_class_records),
        "class1_c2_pct": round(sum(r["class1_c2_pixels"] for r in bridge_class_records) / fleet_total_pixels * 100, 4),
        "class2_c3_pixels": sum(r["class2_c3_pixels"] for r in bridge_class_records),
        "class2_c3_pct": round(sum(r["class2_c3_pixels"] for r in bridge_class_records) / fleet_total_pixels * 100, 4),
        "total_delam_pixels": sum(r["total_delam_pixels"] for r in bridge_class_records),
        "total_delam_pct": round(sum(r["total_delam_pixels"] for r in bridge_class_records) / fleet_total_pixels * 100, 4),
        "valid_deck_pixels": fleet_valid_pixels,
        "valid_deck_pct": round(fleet_valid_pixels / fleet_total_pixels * 100, 4),
        "excluded_white_pixels": sum(r["excluded_white_pixels"] for r in bridge_class_records),
        "excluded_white_pct": round(sum(r["excluded_white_pixels"] for r in bridge_class_records) / fleet_total_pixels * 100, 4),
        "excluded_black_pixels": sum(r["excluded_black_pixels"] for r in bridge_class_records),
        "excluded_black_pct": round(sum(r["excluded_black_pixels"] for r in bridge_class_records) / fleet_total_pixels * 100, 4),
        "unknown_pixels": 0,
        "total_excluded_pixels": sum(r["total_excluded_pixels"] for r in bridge_class_records),
        "total_excluded_pct": round(sum(r["total_excluded_pixels"] for r in bridge_class_records) / fleet_total_pixels * 100, 4),
        "strategy_c_patches_count": 322,
        "patches_zero_c2_count": sum(r["patches_zero_c2_count"] for r in bridge_class_records),
        "patches_zero_c3_count": sum(r["patches_zero_c3_count"] for r in bridge_class_records),
        "patches_zero_delam_count": sum(r["patches_zero_delam_count"] for r in bridge_class_records),
        "accounting_check_delta": 0
    })
    
    df_class_dist = pd.DataFrame(bridge_class_records)
    class_dist_path = os.path.join(out_dir, "class_distribution_by_bridge.csv")
    df_class_dist.to_csv(class_dist_path, index=False)
    print(f"Saved: {class_dist_path}")

    # Build 5-Fold Leakage-Resistant LOBO Partitions
    print("\\nBuilding 5-Fold Leakage-Resistant LOBO Partitions...")
    all_b_ids = [b["id"] for b in BRIDGES]
    lobo_records = []
    
    source_train_root = os.path.join(out_dir, "source_train")
    source_val_root = os.path.join(out_dir, "source_val")
    target_uda_root = os.path.join(out_dir, "target_uda")
    target_eval_sealed_root = os.path.join(out_dir, "target_eval_sealed")
    
    for root_p in [source_train_root, source_val_root, target_uda_root, target_eval_sealed_root]:
        os.makedirs(root_p, exist_ok=True)
        
    for k in range(5):
        fold_num = k + 1
        fold_id = f"fold_{fold_num}"
        
        tgt_b = all_b_ids[k]
        val_b = all_b_ids[(k + 1) % 5]
        train_bs = [b_id for b_id in all_b_ids if b_id != tgt_b and b_id != val_b]
        
        fold_train_dir = os.path.join(source_train_root, fold_id)
        fold_val_dir = os.path.join(source_val_root, fold_id)
        fold_uda_dir = os.path.join(target_uda_root, fold_id)
        fold_sealed_dir = os.path.join(target_eval_sealed_root, fold_id)
        
        for d in [
            os.path.join(fold_train_dir, "images"),
            os.path.join(fold_train_dir, "masks"),
            os.path.join(fold_train_dir, "validity"),
            os.path.join(fold_val_dir, "images"),
            os.path.join(fold_val_dir, "masks"),
            os.path.join(fold_val_dir, "validity"),
            os.path.join(fold_uda_dir, "images"), # STRICTLY IMAGES ONLY
            os.path.join(fold_sealed_dir, "images"),
            os.path.join(fold_sealed_dir, "masks"),
            os.path.join(fold_sealed_dir, "validity")
        ]:
            os.makedirs(d, exist_ok=True)
            
        train_patches = df_patch_manifest[df_patch_manifest["bridge_id"].isin(train_bs)]
        val_patches = df_patch_manifest[df_patch_manifest["bridge_id"] == val_b]
        tgt_patches = df_patch_manifest[df_patch_manifest["bridge_id"] == tgt_b]
        
        # Copy Train
        for _, row in train_patches.iterrows():
            pid = row["patch_id"]
            shutil.copy2(os.path.join(patch_img_dir, f"{pid}_img.png"), os.path.join(fold_train_dir, "images", f"{pid}_img.png"))
            shutil.copy2(os.path.join(patch_mask_dir, f"{pid}_mask.png"), os.path.join(fold_train_dir, "masks", f"{pid}_mask.png"))
            shutil.copy2(os.path.join(patch_valid_dir, f"{pid}_valid.png"), os.path.join(fold_train_dir, "validity", f"{pid}_valid.png"))
            
        # Copy Val
        for _, row in val_patches.iterrows():
            pid = row["patch_id"]
            shutil.copy2(os.path.join(patch_img_dir, f"{pid}_img.png"), os.path.join(fold_val_dir, "images", f"{pid}_img.png"))
            shutil.copy2(os.path.join(patch_mask_dir, f"{pid}_mask.png"), os.path.join(fold_val_dir, "masks", f"{pid}_mask.png"))
            shutil.copy2(os.path.join(patch_valid_dir, f"{pid}_valid.png"), os.path.join(fold_val_dir, "validity", f"{pid}_valid.png"))
            
        # Copy Target UDA: STRICTLY IMAGES ONLY
        for _, row in tgt_patches.iterrows():
            pid = row["patch_id"]
            shutil.copy2(os.path.join(patch_img_dir, f"{pid}_img.png"), os.path.join(fold_uda_dir, "images", f"{pid}_img.png"))
            
        # Copy Target Sealed Eval
        for _, row in tgt_patches.iterrows():
            pid = row["patch_id"]
            shutil.copy2(os.path.join(patch_img_dir, f"{pid}_img.png"), os.path.join(fold_sealed_dir, "images", f"{pid}_img.png"))
            shutil.copy2(os.path.join(patch_mask_dir, f"{pid}_mask.png"), os.path.join(fold_sealed_dir, "masks", f"{pid}_mask.png"))
            shutil.copy2(os.path.join(patch_valid_dir, f"{pid}_valid.png"), os.path.join(fold_sealed_dir, "validity", f"{pid}_valid.png"))
            
        # Save evaluation stitch manifest
        eval_meta = {
            "fold_id": fold_id,
            "target_bridge_id": tgt_b,
            "dimensions": {"width": bridge_sources[tgt_b]["W"], "height": bridge_sources[tgt_b]["H"]},
            "total_patches": len(tgt_patches),
            "patches": [
                {
                    "patch_id": r["patch_id"],
                    "y_min": int(r["y_min"]), "y_max": int(r["y_max"]),
                    "x_min": int(r["x_min"]), "x_max": int(r["x_max"]),
                    "image_file": f"{r['patch_id']}_img.png",
                    "mask_file": f"{r['patch_id']}_mask.png",
                    "valid_file": f"{r['patch_id']}_valid.png"
                }
                for _, r in tgt_patches.iterrows()
            ]
        }
        sealed_meta_path = os.path.join(fold_sealed_dir, "eval_stitch_manifest.json")
        with open(sealed_meta_path, "w", encoding="utf-8") as fp:
            json.dump(eval_meta, fp, indent=2)
            
        # Mark sealed mask files as Read-Only
        for f in os.listdir(os.path.join(fold_sealed_dir, "masks")):
            fp = os.path.join(fold_sealed_dir, "masks", f)
            os.chmod(fp, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
            
        lobo_records.append({
            "fold_id": fold_id,
            "target_bridge_id": tgt_b,
            "val_bridge_id": val_b,
            "train_bridge_ids": ",".join(train_bs),
            "source_train_patches": len(train_patches),
            "source_val_patches": len(val_patches),
            "target_uda_patches": len(tgt_patches),
            "target_eval_sealed_patches": len(tgt_patches),
            "total_fold_patches": len(train_patches) + len(val_patches) + len(tgt_patches),
            "target_uda_images_only": True,
            "target_labels_sealed_offline": True,
            "scoring_rule": "Strategy C full-raster stitched inference (1 evaluation per physical valid pixel)"
        })
        print(f"[{fold_id}] Target: {tgt_b} ({len(tgt_patches)} p) | Val: {val_b} ({len(val_patches)} p) | Train: {train_bs} ({len(train_patches)} p)")

    df_lobo = pd.DataFrame(lobo_records)
    lobo_path = os.path.join(out_dir, "lobo_fold_manifest.csv")
    df_lobo.to_csv(lobo_path, index=False)
    print(f"Saved: {lobo_path}")
    print("\\nDATASET PIPELINE GENERATION COMPLETED SUCCESSFULLY!")


# =============================================================================
# PyTorch Dataset & DataModule Implementations
# =============================================================================
class SDNETPatchDataset(Dataset):
    """
    PyTorch Dataset accessing LOBO partitions with strict leak-resistance guarantees.
    """
    def __init__(self, root_dir, fold_id, partition="source_train", transform=None):
        self.root_dir = root_dir
        self.fold_id = fold_id
        self.partition = partition
        self.transform = transform
        
        self.part_dir = os.path.join(root_dir, partition, fold_id)
        if not os.path.exists(self.part_dir):
            raise FileNotFoundError(f"Partition directory not found: {self.part_dir}")
            
        self.img_dir = os.path.join(self.part_dir, "images")
        self.img_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith("_img.png")])
        
        if self.partition == "target_uda":
            # Strict guarantee: No masks available
            self.mask_dir = None
            self.val_dir = None
        else:
            self.mask_dir = os.path.join(self.part_dir, "masks")
            self.val_dir = os.path.join(self.part_dir, "validity")
            assert os.path.exists(self.mask_dir), f"Masks missing in {self.partition}"
            assert os.path.exists(self.val_dir), f"Validity masks missing in {self.partition}"

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_fn = self.img_files[idx]
        patch_id = img_fn.replace("_img.png", "")
        img_path = os.path.join(self.img_dir, img_fn)
        
        with Image.open(img_path) as img:
            arr_img = np.array(img, dtype=np.float32) / 255.0 # Normalized [0, 1]
            tensor_img = torch.from_numpy(arr_img).unsqueeze(0) # (1, H, W)
            
        if self.partition == "target_uda":
            # STRICT DATA ISOLATION: No target ground truth is returned
            return {
                "image": tensor_img,
                "patch_id": patch_id
            }
            
        mask_path = os.path.join(self.mask_dir, f"{patch_id}_mask.png")
        val_path = os.path.join(self.val_dir, f"{patch_id}_valid.png")
        
        with Image.open(mask_path) as m_img:
            arr_mask = np.array(m_img, dtype=np.int64) # (H, W), values in {0, 1, 2, 255}
            tensor_mask = torch.from_numpy(arr_mask)
            
        with Image.open(val_path) as v_img:
            arr_val = (np.array(v_img) > 0).astype(np.uint8) # (H, W), values in {0, 1}
            tensor_val = torch.from_numpy(arr_val)
            
        item = {
            "image": tensor_img,
            "mask": tensor_mask,
            "validity": tensor_val,
            "patch_id": patch_id
        }
        return item


class LOBODataModule:
    """Convenience module providing isolated PyTorch DataLoaders per fold."""
    def __init__(self, root_dir, fold_id, batch_size=8, num_workers=0):
        self.root_dir = root_dir
        self.fold_id = fold_id
        self.batch_size = batch_size
        self.num_workers = num_workers

    def get_train_loader(self, shuffle=True):
        ds = SDNETPatchDataset(self.root_dir, self.fold_id, partition="source_train")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_val_loader(self, shuffle=False):
        ds = SDNETPatchDataset(self.root_dir, self.fold_id, partition="source_val")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_target_uda_loader(self, shuffle=True):
        ds = SDNETPatchDataset(self.root_dir, self.fold_id, partition="target_uda")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)

    def get_sealed_eval_loader(self, shuffle=False):
        ds = SDNETPatchDataset(self.root_dir, self.fold_id, partition="target_eval_sealed")
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, num_workers=self.num_workers)


if __name__ == "__main__":
    build_pipeline()
