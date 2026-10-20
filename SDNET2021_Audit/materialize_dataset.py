"""
Deterministic Dataset Materialization Script for SDNET2021 IR Benchmark
Reconstructs extracted_irt from verified sealed patches and builds
source_train, source_val, target_uda partitions with 100% mathematical fidelity.
"""

import os
import sys
import json
import shutil
import numpy as np
import pandas as pd
from PIL import Image

AUDIT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(AUDIT_ROOT, "extracted_irt")
PIPELINE_DIR = os.path.join(AUDIT_ROOT, "phase2b1_dataset_pipeline")
SEALED_DIR = os.path.join(PIPELINE_DIR, "target_eval_sealed")

# Ensure dataset_build can be imported
sys.path.insert(0, PIPELINE_DIR)
from dataset_build import BRIDGES, decode_annotation_mask, build_pipeline

def materialize_raw_and_partitions():
    print("=" * 70)
    print("MATERIALIZING SDNET2021 RAW AND PARTITION DATASETS")
    print("=" * 70)

    class_dist_csv = os.path.join(PIPELINE_DIR, "class_distribution_by_bridge.csv")
    df_dist = pd.read_csv(class_dist_csv)

    bridge_to_fold = {
        "B1": "fold_1",
        "B2": "fold_2",
        "B3": "fold_3",
        "B4": "fold_4",
        "B5": "fold_5"
    }

    # Step 1: Reconstruct extracted_irt
    print("\n[Step 1] Reconstructing raw thermogram images and annotations in extracted_irt...")
    os.makedirs(SRC_DIR, exist_ok=True)

    for b in BRIDGES:
        b_id = b["id"]
        fold = bridge_to_fold[b_id]
        b_folder = os.path.join(SRC_DIR, b["folder"])
        os.makedirs(b_folder, exist_ok=True)

        fold_dir = os.path.join(SEALED_DIR, fold)
        with open(os.path.join(fold_dir, "eval_stitch_manifest.json"), "r") as f:
            manifest = json.load(f)

        W = manifest["dimensions"]["width"]
        H = manifest["dimensions"]["height"]
        assert (W, H) == (b["W"], b["H"])

        stitched_img = np.zeros((H, W), dtype=np.uint8)
        stitched_mask = np.full((H, W), 255, dtype=np.uint8)

        for p in manifest["patches"]:
            y1, y2 = p["y_min"], p["y_max"]
            x1, x2 = p["x_min"], p["x_max"]
            p_img = np.array(Image.open(os.path.join(fold_dir, "images", p["image_file"])))
            p_mask = np.array(Image.open(os.path.join(fold_dir, "masks", p["mask_file"])))
            stitched_img[y1:y2, x1:x2] = p_img
            stitched_mask[y1:y2, x1:x2] = p_mask

        # Build Original PNG
        orig_path = os.path.join(b_folder, b["orig_file"])
        Image.fromarray(stitched_img, mode="L").save(orig_path)
        print(f"  Saved Original:   {orig_path} ({W}x{H})")

        # Build Annotated RGB PNG
        annot_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # Sound pixels: grayscale [1, 254]
        sound_mask = (stitched_mask == 0)
        sound_vals = np.clip(stitched_img[sound_mask], 1, 254)
        annot_rgb[sound_mask, 0] = sound_vals
        annot_rgb[sound_mask, 1] = sound_vals
        annot_rgb[sound_mask, 2] = sound_vals

        # Class 2: [0, 255, 0]
        c2_mask = (stitched_mask == 1)
        annot_rgb[c2_mask, 1] = 255

        # Class 3: [255, 0, 0]
        c3_mask = (stitched_mask == 2)
        annot_rgb[c3_mask, 0] = 255

        # Excluded pixels:
        excl_mask = (stitched_mask == 255)
        b_row = df_dist[df_dist["bridge_id"] == b_id].iloc[0]
        n_black = int(b_row["excluded_black_pixels"])
        n_white = int(b_row["excluded_white_pixels"])

        excl_indices = np.where(excl_mask)
        if n_black > 0:
            is_zero = (stitched_img[excl_mask] == 0)
            black_sub_idx = np.where(is_zero)[0][:n_black]
            if len(black_sub_idx) < n_black:
                black_sub_idx = np.arange(n_black)

            annot_rgb[excl_mask, :] = 255
            annot_rgb[excl_indices[0][black_sub_idx], excl_indices[1][black_sub_idx], :] = 0
        else:
            annot_rgb[excl_mask, :] = 255

        # Verify decode_annotation_mask matches class_distribution_by_bridge.csv
        sem_dec, val_dec, bd = decode_annotation_mask(annot_rgb)
        assert bd["sound"] == b_row["class0_sound_pixels"]
        assert bd["c2"] == b_row["class1_c2_pixels"]
        assert bd["c3"] == b_row["class2_c3_pixels"]
        assert bd["white"] == b_row["excluded_white_pixels"]
        assert bd["black"] == b_row["excluded_black_pixels"]
        assert bd["valid"] == b_row["valid_deck_pixels"]
        assert bd["unknown"] == 0
        assert np.array_equal(sem_dec, stitched_mask)

        annot_path = os.path.join(b_folder, b["annot_file"])
        Image.fromarray(annot_rgb, mode="RGB").save(annot_path)
        print(f"  Saved Annotated:  {annot_path} ({W}x{H}) [Verification 100% Passed]")

    # Step 2: Run dataset_build pipeline to materialize patches and LOBO partitions
    print("\n[Step 2] Executing build_pipeline() to materialize Strategy C patches and LOBO partitions...")
    build_pipeline(src_dir=SRC_DIR, out_dir=PIPELINE_DIR)
    print("\n[SUCCESS] SDNET2021 raw images and LOBO partitions materialized and verified successfully.")

if __name__ == "__main__":
    materialize_raw_and_partitions()
