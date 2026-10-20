import os
import csv
import json
import numpy as np
from PIL import Image, ImageDraw

audit_dir = r"C:\Users\Dell\Documents\SDNET2021_Audit"
irt_dir = os.path.join(audit_dir, "extracted_irt")
out_dir = os.path.join(audit_dir, "phase1d_five_bridge_audit")
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

bridge_summaries = []
all_tiles_records = []

for b in bridges:
    b_id = b["bridge_id"]
    b_name = b["bridge_name"]
    op = os.path.join(irt_dir, b["folder"], b["orig_file"])
    ap = os.path.join(irt_dir, b["folder"], b["anno_file"])
    
    orig_img = Image.open(op)
    anno_img = Image.open(ap)
    
    W, H = orig_img.size
    total_px = W * H
    dim_match = (orig_img.size == anno_img.size)
    
    orig_arr = np.array(orig_img)
    anno_arr = np.array(anno_img)
    
    # Masks
    is_green = (anno_arr[:, :, 0] == 0) & (anno_arr[:, :, 1] == 255) & (anno_arr[:, :, 2] == 0)
    is_red   = (anno_arr[:, :, 0] == 255) & (anno_arr[:, :, 1] == 0) & (anno_arr[:, :, 2] == 0)
    is_white = (anno_arr[:, :, 0] == 255) & (anno_arr[:, :, 1] == 255) & (anno_arr[:, :, 2] == 255)
    is_black = (anno_arr[:, :, 0] == 0) & (anno_arr[:, :, 1] == 0) & (anno_arr[:, :, 2] == 0)
    is_sound = (anno_arr[:, :, 0] == anno_arr[:, :, 1]) & (anno_arr[:, :, 1] == anno_arr[:, :, 2]) & ~is_white & ~is_black
    is_ambig = ~is_green & ~is_red & ~is_white & ~is_black & ~is_sound
    
    cnt_sound = int(is_sound.sum())
    cnt_c2    = int(is_green.sum())
    cnt_c3    = int(is_red.sum())
    cnt_white = int(is_white.sum())
    cnt_black = int(is_black.sum())
    cnt_ambig = int(is_ambig.sum())
    
    # Valid deck mask: Class 1, 2, 3
    valid_deck = is_sound | is_green | is_red
    cnt_valid_deck = int(valid_deck.sum())
    cnt_excluded = cnt_white + cnt_black
    
    # Correspondence on sound deck
    orig_gray = orig_arr if len(orig_arr.shape) == 2 else orig_arr[:, :, 0]
    diff_sound = np.abs(anno_arr[:, :, 0].astype(int) - orig_gray.astype(int))[is_sound]
    exact_matches = int((diff_sound == 0).sum())
    diff_matches = int((diff_sound > 0).sum())
    max_diff = int(diff_sound.max()) if len(diff_sound) > 0 else 0
    mean_diff = float(diff_sound.mean()) if len(diff_sound) > 0 else 0.0
    
    # Multi-class semantic mask: 0=Sound, 1=Class 2, 2=Class 3, 255=Excluded
    multiclass = np.full((H, W), 255, dtype=np.uint8)
    multiclass[is_sound] = 0
    multiclass[is_green] = 1
    multiclass[is_red]   = 2
    
    # Generate visual diagnostic:
    # 4-panel composite: Original (RGB), Annotated, Multiclass Colorized, Overlay with Grid
    colorized = np.full((H, W, 3), 255, dtype=np.uint8)
    colorized[multiclass == 0] = [128, 128, 128]
    colorized[multiclass == 1] = [0, 255, 0]
    colorized[multiclass == 2] = [255, 0, 0]
    
    # Overlay: Blend Class 2 (Green 50%) and Class 3 (Red 60%) over original
    overlay = np.stack([orig_gray]*3, axis=-1).copy()
    overlay[is_green, 1] = np.clip(orig_gray[is_green] * 0.5 + 255 * 0.5, 0, 255).astype(np.uint8)
    overlay[is_red, 0]   = np.clip(orig_gray[is_red] * 0.4 + 255 * 0.6, 0, 255).astype(np.uint8)
    
    # Save per-bridge diagnostic image
    # If image is very wide (H < W), stack vertically; if tall (H > W), stack horizontally
    if H > W:
        diag_canvas = Image.new("RGB", (W * 4, H))
        diag_canvas.paste(Image.fromarray(np.stack([orig_gray]*3, axis=-1)), (0, 0))
        diag_canvas.paste(anno_img, (W, 0))
        diag_canvas.paste(Image.fromarray(colorized), (W * 2, 0))
        diag_canvas.paste(Image.fromarray(overlay), (W * 3, 0))
    else:
        diag_canvas = Image.new("RGB", (W, H * 4))
        diag_canvas.paste(Image.fromarray(np.stack([orig_gray]*3, axis=-1)), (0, 0))
        diag_canvas.paste(anno_img, (0, H))
        diag_canvas.paste(Image.fromarray(colorized), (0, H * 2))
        diag_canvas.paste(Image.fromarray(overlay), (0, H * 3))
        
    diag_path = os.path.join(out_dir, f"{b_id.lower()}_diagnostic.png")
    diag_canvas.save(diag_path)
    
    # Tiling evaluation helper
    def evaluate_bridge_tiling(tile_size, anchor_mode):
        if anchor_mode == "top_left":
            x0, y0 = 0, 0
        else: # centered
            x0 = (W % tile_size) // 2
            y0 = (H % tile_size) // 2
            
        nx = (W - x0) // tile_size
        ny = (H - y0) // tile_size
        n_complete = nx * ny
        
        covered_area = n_complete * (tile_size * tile_size)
        uncovered_area = total_px - covered_area
        uncovered_pct = uncovered_area / total_px * 100
        
        t_records = []
        for r in range(ny):
            for c in range(nx):
                x_min = x0 + c * tile_size
                x_max = x_min + tile_size
                y_min = y0 + r * tile_size
                y_max = y_min + tile_size
                
                patch_m = multiclass[y_min:y_max, x_min:x_max]
                patch_v = valid_deck[y_min:y_max, x_min:x_max]
                
                n_tot = tile_size * tile_size
                n_val = int(patch_v.sum())
                val_frac = n_val / n_tot
                
                p_c1 = int((patch_m == 0).sum())
                p_c2 = int((patch_m == 1).sum())
                p_c3 = int((patch_m == 2).sum())
                p_del = p_c2 + p_c3
                
                retained = (val_frac >= 0.50)
                reason = "NONE" if retained else "VALID_DECK_LT_50_PCT"
                
                rec = {
                    "bridge_id": b_id,
                    "tile_size": tile_size,
                    "anchor": anchor_mode,
                    "tile_id": f"{b_id}_{tile_size}_{anchor_mode}_{r}_{c}",
                    "row": r,
                    "col": c,
                    "x_min": x_min,
                    "x_max": x_max,
                    "y_min": y_min,
                    "y_max": y_max,
                    "total_pixels": n_tot,
                    "valid_pixels": n_val,
                    "valid_fraction": round(val_frac, 4),
                    "class1_sound": p_c1,
                    "class2_above": p_c2,
                    "class3_below": p_c3,
                    "delam_total": p_del,
                    "has_class1": bool(p_c1 > 0),
                    "has_class2": bool(p_c2 > 0),
                    "has_class3": bool(p_c3 > 0),
                    "has_delam": bool(p_del > 0),
                    "retained": retained,
                    "exclusion_reason": reason
                }
                t_records.append(rec)
                all_tiles_records.append(rec)
                
        # Stats
        ret_count = sum(1 for t in t_records if t["retained"])
        rej_count = len(t_records) - ret_count
        c2_count = sum(1 for t in t_records if t["retained"] and t["has_class2"])
        c3_count = sum(1 for t in t_records if t["retained"] and t["has_class3"])
        del_count = sum(1 for t in t_records if t["retained"] and t["has_delam"])
        no_val_count = sum(1 for t in t_records if t["valid_pixels"] == 0)
        no_c2_count = sum(1 for t in t_records if t["retained"] and not t["has_class2"])
        no_c3_count = sum(1 for t in t_records if t["retained"] and not t["has_class3"])
        
        return {
            "tile_size": tile_size,
            "anchor": anchor_mode,
            "grid_nx": nx,
            "grid_ny": ny,
            "complete_tiles": n_complete,
            "retained_tiles": ret_count,
            "rejected_tiles": rej_count,
            "uncovered_area_px": uncovered_area,
            "uncovered_pct": round(uncovered_pct, 2),
            "tiles_with_c2": c2_count,
            "tiles_with_c3": c3_count,
            "tiles_with_any_delam": del_count,
            "tiles_no_valid": no_val_count,
            "tiles_no_c2": no_c2_count,
            "tiles_no_c3": no_c3_count,
            "tile_records": t_records
        }

    # Evaluate all 4 tile configs
    t256_tl  = evaluate_bridge_tiling(256, "top_left")
    t256_cen = evaluate_bridge_tiling(256, "centered")
    t384_tl  = evaluate_bridge_tiling(384, "top_left")
    t384_cen = evaluate_bridge_tiling(384, "centered")
    
    # Save per-bridge tile CSV (all 4 configs for this bridge)
    b_csv_path = os.path.join(out_dir, f"{b_id.lower()}_tiles_manifest.csv")
    b_tile_rows = t256_tl["tile_records"] + t256_cen["tile_records"] + t384_tl["tile_records"] + t384_cen["tile_records"]
    if len(b_tile_rows) > 0:
        with open(b_csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(b_tile_rows[0].keys()))
            w.writeheader()
            w.writerows(b_tile_rows)
            
    summary_item = {
        "bridge_id": b_id,
        "bridge_name": b_name,
        "orig_file": b["orig_file"],
        "anno_file": b["anno_file"],
        "width": W,
        "height": H,
        "aspect_ratio": round(max(W, H) / min(W, H), 2),
        "orientation": "Vertical" if H > W else "Horizontal",
        "total_pixels": total_px,
        "dim_match": dim_match,
        "sound_pixels": cnt_sound,
        "sound_pct": round(cnt_sound / total_px * 100, 2),
        "class2_pixels": cnt_c2,
        "class2_pct": round(cnt_c2 / total_px * 100, 2),
        "class3_pixels": cnt_c3,
        "class3_pct": round(cnt_c3 / total_px * 100, 2),
        "delam_total_pixels": cnt_c2 + cnt_c3,
        "delam_total_pct": round((cnt_c2 + cnt_c3) / total_px * 100, 2),
        "white_canvas_pixels": cnt_white,
        "white_canvas_pct": round(cnt_white / total_px * 100, 2),
        "black_boundary_pixels": cnt_black,
        "black_boundary_pct": round(cnt_black / total_px * 100, 2),
        "ambiguous_pixels": cnt_ambig,
        "exact_matches_sound": exact_matches,
        "exact_matches_sound_pct": round(exact_matches / cnt_sound * 100, 2) if cnt_sound > 0 else 0,
        "differing_pixels_sound": diff_matches,
        "max_abs_diff_sound": max_diff,
        "mean_abs_diff_sound": round(mean_diff, 2),
        # Tiling summaries
        "t256_tl_complete": t256_tl["complete_tiles"],
        "t256_tl_retained": t256_tl["retained_tiles"],
        "t256_tl_rejected": t256_tl["rejected_tiles"],
        "t256_tl_uncovered_pct": t256_tl["uncovered_pct"],
        "t256_tl_with_c2": t256_tl["tiles_with_c2"],
        "t256_tl_with_c3": t256_tl["tiles_with_c3"],
        "t256_cen_complete": t256_cen["complete_tiles"],
        "t256_cen_retained": t256_cen["retained_tiles"],
        "t256_cen_rejected": t256_cen["rejected_tiles"],
        "t256_cen_uncovered_pct": t256_cen["uncovered_pct"],
        "t384_tl_complete": t384_tl["complete_tiles"],
        "t384_tl_retained": t384_tl["retained_tiles"],
        "t384_tl_rejected": t384_tl["rejected_tiles"],
        "t384_tl_uncovered_pct": t384_tl["uncovered_pct"],
        "t384_tl_with_c2": t384_tl["tiles_with_c2"],
        "t384_tl_with_c3": t384_tl["tiles_with_c3"],
        "t384_cen_complete": t384_cen["complete_tiles"],
        "t384_cen_retained": t384_cen["retained_tiles"],
        "t384_cen_rejected": t384_cen["rejected_tiles"],
        "t384_cen_uncovered_pct": t384_cen["uncovered_pct"]
    }
    bridge_summaries.append(summary_item)

# Save combined bridges summary CSV
comb_summary_path = os.path.join(out_dir, "five_bridge_summary.csv")
with open(comb_summary_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(bridge_summaries[0].keys()))
    w.writeheader()
    w.writerows(bridge_summaries)
print(f"Wrote combined summary: {comb_summary_path}")

# Save combined all tiles CSV
comb_tiles_path = os.path.join(out_dir, "five_bridge_all_tiles.csv")
with open(comb_tiles_path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(all_tiles_records[0].keys()))
    w.writeheader()
    w.writerows(all_tiles_records)
print(f"Wrote all tiles manifest: {comb_tiles_path} ({len(all_tiles_records)} tiles)")
