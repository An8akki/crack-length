import os
import csv
import json
import numpy as np
from PIL import Image, ImageDraw

# Directories
audit_dir = r"C:\Users\Dell\Documents\SDNET2021_Audit"
irt_dir = os.path.join(audit_dir, "extracted_irt")
p1d_dir = os.path.join(audit_dir, "phase1d_five_bridge_audit")
out_dir = os.path.join(audit_dir, "phase1e_alignment_verification")
os.makedirs(out_dir, exist_ok=True)

bridges = [
    {
        "bridge_id": "B1",
        "bridge_name": "Forest River Northbound",
        "folder": "1.Forest River NB",
        "orig_file": "FR_NB_Original.png",
        "anno_file": "FR_NB_Annotated.png"
    },
    {
        "bridge_id": "B2",
        "bridge_name": "Forest River Southbound",
        "folder": "2.Forest River SB",
        "orig_file": "FR_SB_Original.png",
        "anno_file": "FR_SB_Annotated.png"
    },
    {
        "bridge_id": "B3",
        "bridge_name": "Park River Median",
        "folder": "3.Park River Median",
        "orig_file": "PR_MD_Original.png",
        "anno_file": "PR_MD_Annotated.png"
    },
    {
        "bridge_id": "B4",
        "bridge_name": "Park River Northbound",
        "folder": "4.Park River North Bound",
        "orig_file": "PR_NB_Original.png",
        "anno_file": "PR_NB_Annotaed.png"
    },
    {
        "bridge_id": "B5",
        "bridge_name": "Park River Southbound",
        "folder": "5.Park River South Bound",
        "orig_file": "PR_SB_Original.png",
        "anno_file": "PR_SB_Annotated.png"
    }
]

print("=== STARTING PHASE 1E INDEPENDENT VERIFICATION ===")

recomputed_tiles = []
reconciliation_rows = []
alignment_metrics = []

for b in bridges:
    b_id = b["bridge_id"]
    b_name = b["bridge_name"]
    op = os.path.join(irt_dir, b["folder"], b["orig_file"])
    ap = os.path.join(irt_dir, b["folder"], b["anno_file"])
    
    orig_img = Image.open(op)
    anno_img = Image.open(ap)
    
    W, H = orig_img.size
    total_px = W * H
    
    orig_arr = np.array(orig_img) # (H, W) or (H, W, C)
    anno_arr = np.array(anno_img) # (H, W, 3)
    orig_gray = orig_arr if len(orig_arr.shape) == 2 else orig_arr[:, :, 0]
    
    # -------------------------------------------------------------
    # Task 2: Reconstruct mutually exclusive masks
    # -------------------------------------------------------------
    is_white = (anno_arr[:, :, 0] == 255) & (anno_arr[:, :, 1] == 255) & (anno_arr[:, :, 2] == 255)
    is_black = (anno_arr[:, :, 0] == 0) & (anno_arr[:, :, 1] == 0) & (anno_arr[:, :, 2] == 0)
    is_green = (anno_arr[:, :, 0] == 0) & (anno_arr[:, :, 1] == 255) & (anno_arr[:, :, 2] == 0)
    is_red   = (anno_arr[:, :, 0] == 255) & (anno_arr[:, :, 1] == 0) & (anno_arr[:, :, 2] == 0)
    # Sound concrete: R == G == B and not white and not black
    is_sound = (anno_arr[:, :, 0] == anno_arr[:, :, 1]) & (anno_arr[:, :, 1] == anno_arr[:, :, 2]) & ~is_white & ~is_black
    # Unknown / ambiguous
    is_unknown = ~is_green & ~is_red & ~is_white & ~is_black & ~is_sound
    
    cnt_sound = int(is_sound.sum())
    cnt_c2 = int(is_green.sum())
    cnt_c3 = int(is_red.sum())
    cnt_white = int(is_white.sum())
    cnt_black = int(is_black.sum())
    cnt_unknown = int(is_unknown.sum())
    
    sum_px = cnt_sound + cnt_c2 + cnt_c3 + cnt_white + cnt_black + cnt_unknown
    assert sum_px == total_px, f"Sum mismatch on {b_id}: {sum_px} != {total_px}"
    assert cnt_unknown == 0, f"Found unknown pixels on {b_id}!"
    
    # Check black pixels in original image:
    # Are black pixels also black (0) in original image, or boundary edge?
    orig_at_black = orig_gray[is_black]
    black_in_orig_min = int(orig_at_black.min()) if cnt_black > 0 else None
    black_in_orig_max = int(orig_at_black.max()) if cnt_black > 0 else None
    black_in_orig_mean = float(orig_at_black.mean()) if cnt_black > 0 else None
    
    reconciliation_rows.append({
        "bridge_id": b_id,
        "bridge_name": b_name,
        "width": W,
        "height": H,
        "total_pixels": total_px,
        "class1_sound": cnt_sound,
        "class1_sound_pct": round(cnt_sound / total_px * 100, 3),
        "class2_above": cnt_c2,
        "class2_above_pct": round(cnt_c2 / total_px * 100, 3),
        "class3_below": cnt_c3,
        "class3_below_pct": round(cnt_c3 / total_px * 100, 3),
        "delam_total": cnt_c2 + cnt_c3,
        "delam_total_pct": round((cnt_c2 + cnt_c3) / total_px * 100, 3),
        "excluded_white_canvas": cnt_white,
        "excluded_white_pct": round(cnt_white / total_px * 100, 3),
        "excluded_black_boundary": cnt_black,
        "excluded_black_pct": round(cnt_black / total_px * 100, 3),
        "unknown_pixels": cnt_unknown,
        "orig_val_at_black_min": black_in_orig_min,
        "orig_val_at_black_max": black_in_orig_max,
        "orig_val_at_black_mean": round(black_in_orig_mean, 2) if black_in_orig_mean is not None else None,
        "reconciled_sum": sum_px,
        "reconciliation_delta": sum_px - total_px
    })
    
    # Save reconstructed mask
    # 0 = Sound, 1 = Class 2, 2 = Class 3, 255 = Excluded
    multi_mask = np.full((H, W), 255, dtype=np.uint8)
    multi_mask[is_sound] = 0
    multi_mask[is_green] = 1
    multi_mask[is_red]   = 2
    
    # Validity mask: 1 = Valid deck, 0 = Excluded
    valid_deck = is_sound | is_green | is_red
    valid_mask = valid_deck.astype(np.uint8)
    
    mask_raw_path = os.path.join(out_dir, f"{b_id.lower()}_reconstructed_mask.png")
    Image.fromarray(multi_mask).save(mask_raw_path)
    
    # -------------------------------------------------------------
    # Task 3: Investigate B4 & B5 Correspondence & Integer Translations
    # -------------------------------------------------------------
    diff_sound_raw = np.abs(anno_arr[:, :, 0].astype(int) - orig_gray.astype(int))
    diff_on_sound = diff_sound_raw[is_sound]
    
    n_exact = int((diff_on_sound == 0).sum())
    n_diff = int((diff_on_sound > 0).sum())
    mean_diff = float(diff_on_sound.mean())
    median_diff = float(np.median(diff_on_sound))
    max_diff = int(diff_on_sound.max())
    p50 = float(np.percentile(diff_on_sound, 50))
    p75 = float(np.percentile(diff_on_sound, 75))
    p90 = float(np.percentile(diff_on_sound, 90))
    p95 = float(np.percentile(diff_on_sound, 95))
    p99 = float(np.percentile(diff_on_sound, 99))
    
    # Test translations dy, dx in [-2, -1, 0, 1, 2]
    best_shift = (0, 0)
    best_shift_mean = mean_diff
    shift_results = {}
    
    for dy in [-2, -1, 0, 1, 2]:
        for dx in [-2, -1, 0, 1, 2]:
            shifted_orig = np.roll(np.roll(orig_gray, dy, axis=0), dx, axis=1)
            # Mask out boundaries where roll wrapped
            valid_shift_region = np.ones((H, W), dtype=bool)
            if dy > 0: valid_shift_region[:dy, :] = False
            elif dy < 0: valid_shift_region[dy:, :] = False
            if dx > 0: valid_shift_region[:, :dx] = False
            elif dx < 0: valid_shift_region[:, dx:] = False
            
            mask_eval = is_sound & valid_shift_region
            d_shift = np.abs(anno_arr[:, :, 0].astype(int) - shifted_orig.astype(int))[mask_eval]
            m_d = float(d_shift.mean())
            shift_results[f"dy_{dy}_dx_{dx}"] = round(m_d, 3)
            if m_d < best_shift_mean:
                best_shift_mean = m_d
                best_shift = (dy, dx)
                
    alignment_metrics.append({
        "bridge_id": b_id,
        "bridge_name": b_name,
        "sound_pixel_count": cnt_sound,
        "exact_match_count": n_exact,
        "exact_match_pct": round(n_exact / cnt_sound * 100, 3),
        "differing_count": n_diff,
        "differing_pct": round(n_diff / cnt_sound * 100, 3),
        "mean_abs_diff": round(mean_diff, 3),
        "median_abs_diff": round(median_diff, 3),
        "p50_diff": round(p50, 2),
        "p75_diff": round(p75, 2),
        "p90_diff": round(p90, 2),
        "p95_diff": round(p95, 2),
        "p99_diff": round(p99, 2),
        "max_abs_diff": max_diff,
        "best_shift_dy": best_shift[0],
        "best_shift_dx": best_shift[1],
        "best_shift_mean_diff": round(best_shift_mean, 3),
        "shift_delta": round(best_shift_mean - mean_diff, 3)
    })
    
    # -------------------------------------------------------------
    # Task 1: Recompute 256x256 Tiles directly from images
    # -------------------------------------------------------------
    # Standard Top-Left grid:
    S = 256
    nx = W // S
    ny = H // S
    
    for r in range(ny):
        for c in range(nx):
            x_min = c * S
            x_max = x_min + S
            y_min = r * S
            y_max = y_min + S
            
            patch_m = multi_mask[y_min:y_max, x_min:x_max]
            patch_v = valid_deck[y_min:y_max, x_min:x_max]
            
            n_tot = S * S
            n_val = int(patch_v.sum())
            val_frac = n_val / n_tot
            
            p_c1 = int((patch_m == 0).sum())
            p_c2 = int((patch_m == 1).sum())
            p_c3 = int((patch_m == 2).sum())
            p_del = p_c2 + p_c3
            
            retained = (val_frac >= 0.50)
            
            recomputed_tiles.append({
                "bridge_id": b_id,
                "tile_id": f"{b_id}_256_tl_r{r}_c{c}",
                "grid_type": "top_left",
                "tile_size": S,
                "row_idx": r,
                "col_idx": c,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "total_pixels": n_tot,
                "valid_pixels": n_val,
                "invalid_pixels": n_tot - n_val,
                "valid_fraction": round(val_frac, 4),
                "class1_sound": p_c1,
                "class2_above": p_c2,
                "class3_below": p_c3,
                "delam_total": p_del,
                "sound_ratio": round(p_c1 / n_val if n_val > 0 else 0, 4),
                "class2_ratio": round(p_c2 / n_val if n_val > 0 else 0, 4),
                "class3_ratio": round(p_c3 / n_val if n_val > 0 else 0, 4),
                "delam_ratio": round(p_del / n_val if n_val > 0 else 0, 4),
                "has_class1": bool(p_c1 > 0),
                "has_class2": bool(p_c2 > 0),
                "has_class3": bool(p_c3 > 0),
                "has_delam": bool(p_del > 0),
                "retained": retained,
                "exclusion_reason": "NONE" if retained else "VALID_DECK_LT_50_PCT"
            })
            
    # -------------------------------------------------------------
    # Task 4: Visual Panel Generation (5 panels)
    # -------------------------------------------------------------
    # 1. Original Grayscale
    # 2. Annotated Image
    # 3. Reconstructed Multi-class Mask (Colorized)
    # 4. Mask Boundary Overlaid on Original Image
    # 5. Difference Heatmap on Sound Pixels
    
    # Difference heatmap: normalized difference on sound pixels, black elsewhere
    diff_vis = np.zeros((H, W, 3), dtype=np.uint8)
    if is_sound.any():
        # Map difference 0-30 to colormap (e.g. green to red, or grayscale)
        diff_clipped = np.clip(diff_sound_raw, 0, 30)
        norm_diff = (diff_clipped / 30.0 * 255).astype(np.uint8)
        # Blue/Red heatmap: zero = black, small = blue/cyan, large = yellow/red
        diff_vis[is_sound, 0] = norm_diff[is_sound] # Red channel
        diff_vis[is_sound, 2] = 255 - norm_diff[is_sound] # Blue channel
        
    # Boundary overlay: draw outline of delamination (Green for C2, Red for C3)
    overlay_boundary = np.stack([orig_gray]*3, axis=-1).copy()
    overlay_boundary[is_green] = [0, 255, 0]
    overlay_boundary[is_red]   = [255, 0, 0]
    
    # Draw 256x256 tile boundaries in yellow on overlay_boundary
    img_draw = Image.fromarray(overlay_boundary)
    d = ImageDraw.Draw(img_draw)
    for r in range(ny):
        for c in range(nx):
            x1, y1 = c * S, r * S
            x2, y2 = x1 + S - 1, y1 + S - 1
            d.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
            d.text((x1 + 6, y1 + 6), f"r{r}c{c}", fill=(255, 255, 0))
    overlay_with_tiles = np.array(img_draw)
    
    # Assemble 5 panels
    # If H > W, horizontal stack; if W > H, vertical stack
    panel_orig = np.stack([orig_gray]*3, axis=-1)
    panel_anno = anno_arr
    panel_colorized = np.full((H, W, 3), 255, dtype=np.uint8)
    panel_colorized[is_sound] = [128, 128, 128]
    panel_colorized[is_green] = [0, 255, 0]
    panel_colorized[is_red]   = [255, 0, 0]
    panel_diff = diff_vis
    
    if H >= W:
        # Vertical bridge (B1, B5): stack 5 panels horizontally
        canvas = Image.new("RGB", (W * 5, H))
        canvas.paste(Image.fromarray(panel_orig), (0, 0))
        canvas.paste(Image.fromarray(panel_anno), (W, 0))
        canvas.paste(Image.fromarray(panel_colorized), (W * 2, 0))
        canvas.paste(Image.fromarray(overlay_with_tiles), (W * 3, 0))
        canvas.paste(Image.fromarray(panel_diff), (W * 4, 0))
    else:
        # Horizontal bridge (B2, B3, B4): stack 5 panels vertically
        canvas = Image.new("RGB", (W, H * 5))
        canvas.paste(Image.fromarray(panel_orig), (0, 0))
        canvas.paste(Image.fromarray(panel_anno), (0, H))
        canvas.paste(Image.fromarray(panel_colorized), (0, H * 2))
        canvas.paste(Image.fromarray(overlay_with_tiles), (0, H * 3))
        canvas.paste(Image.fromarray(panel_diff), (0, H * 4))
        
    diag_img_path = os.path.join(out_dir, f"{b_id.lower()}_phase1e_diagnostic.png")
    canvas.save(diag_img_path)
    print(f"Saved diagnostic for {b_id}: {diag_img_path}")

# Write deliverables
# 1. Class pixel reconciliation CSV
rec_csv_path = os.path.join(out_dir, "class_pixel_reconciliation.csv")
with open(rec_csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(reconciliation_rows[0].keys()))
    w.writeheader()
    w.writerows(reconciliation_rows)
print(f"Wrote reconciliation CSV: {rec_csv_path}")

# 2. Alignment metrics CSV
align_csv_path = os.path.join(out_dir, "alignment_metrics.csv")
with open(align_csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(alignment_metrics[0].keys()))
    w.writeheader()
    w.writerows(alignment_metrics)
print(f"Wrote alignment metrics CSV: {align_csv_path}")

# 3. Recomputed tile manifest CSV
tile_csv_path = os.path.join(out_dir, "tile_manifest_recomputed.csv")
with open(tile_csv_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(recomputed_tiles[0].keys()))
    w.writeheader()
    w.writerows(recomputed_tiles)
print(f"Wrote recomputed tile manifest: {tile_csv_path} ({len(recomputed_tiles)} tiles)")

# Print audit statistics for Task 1
retained_tiles = [t for t in recomputed_tiles if t["retained"]]
print(f"\n=== TILE RECONCILIATION AUDIT ===")
print(f"Total recomputed complete 256x256 tiles: {len(recomputed_tiles)}")
print(f"Total retained tiles (>=50% deck): {len(retained_tiles)}")

c2_counts = [t["class2_above"] for t in retained_tiles]
c3_counts = [t["class3_below"] for t in retained_tiles]
sound_counts = [t["class1_sound"] for t in retained_tiles]
val_counts = [t["valid_pixels"] for t in retained_tiles]

print(f"\nClass pixel distributions across {len(retained_tiles)} retained tiles:")
print(f"  Class 1 (Sound): Min={min(sound_counts):,}, Median={int(np.median(sound_counts)):,}, Max={max(sound_counts):,}")
print(f"  Class 2 (Above): Min={min(c2_counts):,}, Median={int(np.median(c2_counts)):,}, Max={max(c2_counts):,}")
print(f"  Class 3 (Below): Min={min(c3_counts):,}, Median={int(np.median(c3_counts)):,}, Max={max(c3_counts):,}")
print(f"  Valid Deck:      Min={min(val_counts):,}, Median={int(np.median(val_counts)):,}, Max={max(val_counts):,}")

# Check for absence
tiles_no_c2 = [t["tile_id"] for t in retained_tiles if t["class2_above"] == 0]
tiles_no_c3 = [t["tile_id"] for t in retained_tiles if t["class3_below"] == 0]
tiles_no_val = [t["tile_id"] for t in recomputed_tiles if t["valid_pixels"] == 0]

print(f"\nAbsence checks:")
print(f"  Tiles with NO Class 2: {len(tiles_no_c2)} -> {tiles_no_c2}")
print(f"  Tiles with NO Class 3: {len(tiles_no_c3)} -> {tiles_no_c3}")
print(f"  Tiles with NO Valid Deck: {len(tiles_no_val)}")
