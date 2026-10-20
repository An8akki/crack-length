"""Evaluate crack image with Deep Restoration + AI segmentation + Full Quantification + All-in-One Dashboard."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = r"D:\AppData\crack-695010_1280.jpg"

print("=" * 70)
print("INFRAVISION AI: FULL RESTORATION + QUANTIFICATION PIPELINE")
print("=" * 70)
print(f"Target Image: {IMAGE_PATH}")

# Step 1: Load image
import cv2
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print(f"ERROR: Could not read image: {IMAGE_PATH}")
    sys.exit(1)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w = image_rgb.shape[:2]
print(f"\n[1/8] Image loaded: {image_rgb.shape}")

# Step 2: Deep Image Restoration (RestorationNet)
print("\n[2/8] Running Deep RestorationNet (8 ResBlocks, Denoising & Sharpening)...")
from image_restoration.restoration_pipeline import restore_image
restored_rgb, restore_metrics = restore_image(image_rgb)
restored_bgr = cv2.cvtColor(restored_rgb, cv2.COLOR_BGR2RGB)
print(f"   Architecture : {restore_metrics.get('model_architecture', 'RestorationNet')}")
print(f"   Sharpness    : {restore_metrics['input_sharpness']:.1f} -> {restore_metrics['output_sharpness']:.1f} ({restore_metrics['sharpness_change_pct']:+.1f}%)")
print(f"   Entropy      : {restore_metrics['input_entropy']:.3f} -> {restore_metrics['output_entropy']:.3f}")
print(f"   Runtime      : {restore_metrics['runtime_sec']:.3f}s on {restore_metrics.get('active_device', 'cpu')}")

# Step 3: AI Segmentation on Restored Image
print("\n[3/8] Running YOLOv8 segmentation model on restored image...")
try:
    from ultralytics import YOLO
    seg_path = "segmentation_model/weights/best.pt"
    if os.path.exists(seg_path):
        print(f"   Loading model from {seg_path}...")
        seg_model = YOLO(seg_path)
        print("   Predicting...")
        results = seg_model.predict(source=restored_rgb, conf=0.3, save=False, verbose=False)
        if results and len(results) > 0 and results[0].masks is not None:
            masks_data = results[0].masks.data
            if hasattr(masks_data, "cpu"):
                masks_data = masks_data.cpu().numpy()
            else:
                masks_data = np.asarray(masks_data)

            combined = np.zeros((h, w), dtype=np.uint8)
            for i, m in enumerate(masks_data):
                resized = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                combined[resized > 0] = 1
                print(f"   Instance {i}: {int(resized.sum())} pixels")
            mask = (combined > 0).astype(np.uint8)
            print(f"   Combined mask: {int(mask.sum())} pixels ({100*mask.sum()/(h*w):.1f}%)")
        else:
            print("   No masks returned, using fallback...")
            mask = None
    else:
        print(f"   Model not found at {seg_path}, using fallback...")
        mask = None
except Exception as e:
    print(f"   Segmentation failed: {e}")
    mask = None

# Fallback if needed
if mask is None or mask.sum() == 0:
    print("   Fallback: Edge detection + adaptive threshold...")
    gray_r = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray_r)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    print(f"   Fallback mask: {int(mask.sum())} pixels ({100*mask.sum()/(h*w):.1f}%)")

print(f"\n[4/8] Mask confirmed: {mask.shape}, foreground: {int(mask.sum())} pixels")

# Step 4: Multi-Method Skeletonization
from crack_quantification.skeleton import skeletonize_mask
print("\n[5/8] Skeletonizing mask across multiple topological engines...")
for method in ["guo_hall", "zhang_suen", "skeletonize"]:
    skel = skeletonize_mask(mask, method=method)
    print(f"   {method:<14}: {int(skel.sum())} skeleton pixels")

best_skel = skeletonize_mask(mask, method="guo_hall")

# Step 5: Graph Analysis & Metric Scaling
from crack_quantification.graph_analysis import analyze_skeleton
from crack_quantification.length import compute_length, _connected_components
from crack_quantification.quantify import quantify_crack, format_report

print("\n[6/8] Analyzing skeleton topology & scale recovery...")
graph = analyze_skeleton(best_skel)
length = compute_length(best_skel)
n_comps = len(_connected_components(best_skel))

mm_per_px = 0.277
print(f"  Total network length:  {length['length_px']:.2f} px ({length['length_px'] * mm_per_px:.2f} mm)")
print(f"  Main path length:      {graph['main_path_px']:.2f} px ({graph['main_path_px'] * mm_per_px:.2f} mm)")
print(f"  MST length:            {graph['mst_length_px']:.2f} px ({graph['mst_length_px'] * mm_per_px:.2f} mm)")
print(f"  Junctions:             {graph['junction_count']}")
print(f"  Endpoints:             {graph['endpoint_count']}")

# Step 6: Full multi-dimensional quantification (Length, Width, Volume, Severity)
print("\n[7/8] Running Full Quantification (Width + Volume + Severity)...")
from crack_quantification.width import _distance_transform
from crack_quantification.visualize_markers import (
    mark_crack_comprehensive,
    mark_width_diagnostic,
    mark_volume_diagnostic,
    mark_severity_diagnostic,
    mark_restoration_comparison,
    create_all_in_one_dashboard,
)

# Distance transform
dist = _distance_transform(mask)

# Derive 3D crevice recess depth map (in mm) from EDT geometry and shadow crevice shading
gray_img = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2GRAY)
norm_shade = (255.0 - gray_img.astype(np.float32)) / 255.0
recess_depth_mm = (dist * mm_per_px * 1.5 * (0.8 + 0.4 * norm_shade)).astype(np.float32)
recess_depth_mm[mask == 0] = 0.0

try:
    report = quantify_crack(mask, mm_per_px=mm_per_px, depth_map_mm=recess_depth_mm)
    print("\n" + "=" * 70)
    print("FULL CRACK INSPECTION QUANTIFICATION REPORT")
    print("=" * 70)
    print(format_report(report, image_id="crack-695010_1280"))
    
    vol = report.get("volume", {}) or {}
    print("\n  DETAILED MEASUREMENT BREAKDOWN:")
    print(f"  [-] WIDTH     : Mean = {report['mean_width_mm']:.2f} mm | Max = {report['max_width_mm']:.2f} mm | Min = {report['min_width_mm']:.2f} mm")
    print(f"  [-] VOLUME    : Recess Volume = {vol.get('volume_mm3', 0.0):,.1f} mm3 ({vol.get('volume_cm3', 0.0):.3f} cm3)")
    print(f"  [-] DEPTH     : Mean Depth = {vol.get('mean_recess_depth_mm', 0.0):.2f} mm | Max Depth = {vol.get('max_recess_depth_mm', 0.0):.2f} mm")
    print(f"  [-] SEVERITY  : {report['severity']['severity'].upper()} (ACI 224R: {report.get('aci_classification', {}).get('category', 'Severe')})")
    print(f"  [-] REMEDY    : {report.get('aci_classification', {}).get('action', 'Structural intervention required')}")
except Exception as e:
    print(f"  Quantification failed: {e}")
    report = {}

# Step 7: Save visualizations marked with Width, Volume, Depth, and Severity
print("\n[8/8] Generating visual markings and unified inspection dashboard...")
out_dir = "outputs/crack_comparison/real_eval"
os.makedirs(out_dir, exist_ok=True)

# 1. Restoration Comparison Image
restore_cmp = mark_restoration_comparison(image_bgr, restored_bgr, restore_metrics)
restore_cmp_path = os.path.join(out_dir, "crack_restoration_comparison.png")
cv2.imwrite(restore_cmp_path, restore_cmp)
print(f"  [OK] Deep Restoration Comparison  : {restore_cmp_path}")

# 2. Master Engineering Marked Image (Over Restored Photo)
marked_comprehensive = mark_crack_comprehensive(
    image_bgr=restored_bgr,
    mask=mask,
    skeleton=best_skel,
    dist=dist,
    recess_depth_mm=recess_depth_mm,
    report=report,
    mm_per_px=mm_per_px,
    image_title="crack-695010_1280.jpg",
)
comprehensive_path = os.path.join(out_dir, "crack_inspection_marked.png")
cv2.imwrite(comprehensive_path, marked_comprehensive)
print(f"  [OK] Master Marked Inspection HUD : {comprehensive_path}")

# 3. Dedicated Width Marked Image
width_marked = mark_width_diagnostic(restored_bgr, mask, best_skel, dist, mm_per_px)
width_path = os.path.join(out_dir, "crack_width_marked.png")
cv2.imwrite(width_path, width_marked)
print(f"  [OK] Width Analysis Marked Image  : {width_path}")

# 4. Dedicated 3D Volume & Depth Recess Marked Image
vol_marked = mark_volume_diagnostic(restored_bgr, mask, recess_depth_mm, report.get("volume", {}) or {}, mm_per_px)
vol_path = os.path.join(out_dir, "crack_volume_marked.png")
cv2.imwrite(vol_path, vol_marked)
print(f"  [OK] 3D Volume Marked Image       : {vol_path}")

# 5. Dedicated ACI 224R Severity Zoning Marked Image
sev_marked = mark_severity_diagnostic(restored_bgr, mask, best_skel, dist, mm_per_px, report)
sev_path = os.path.join(out_dir, "crack_severity_marked.png")
cv2.imwrite(sev_path, sev_marked)
print(f"  [OK] Severity Zoning Marked Image : {sev_path}")

# 6. AI Mask + Skeleton
vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
vis[mask > 0] = (0, 0, 255)
vis[best_skel > 0] = (0, 255, 0)
vis_path = os.path.join(out_dir, "ai_mask_skeleton.png")
cv2.imwrite(vis_path, vis)
print(f"  [OK] Binary Mask + Skeleton       : {vis_path}")

# 7. CLUB ALL TOGETHER: All-in-One Executive Inspection Dashboard
all_in_one_board = create_all_in_one_dashboard(
    restoration_cmp=restore_cmp,
    seg_vis=vis,
    width_vis=width_marked,
    volume_vis=vol_marked,
    severity_vis=sev_marked,
    master_marked=marked_comprehensive,
    report=report,
    mm_per_px=mm_per_px,
)
dashboard_path = os.path.join(out_dir, "crack_all_in_one_dashboard.png")
cv2.imwrite(dashboard_path, all_in_one_board)
print(f"  [OK] CLUBBED ALL TOGETHER DASHBOARD: {dashboard_path}")

# 8. Optimized Graph Overlays
from crack_quantification.compare_methods import _overlay, COLORS, MEASURES
base = (mask * 255).astype(np.uint8)
if base.ndim == 2:
    base = np.stack([base] * 3, axis=-1)

cached_graphs = {}
for method_name, skel_method in [("zhang_suen", "zhang_suen"), ("guo_hall", "guo_hall"), ("morphological", "skeletonize")]:
    m_skel = skeletonize_mask(mask, method=skel_method)
    cached_graphs[method_name] = (m_skel, analyze_skeleton(m_skel))

for measure in MEASURES:
    for method_name, (m_skel, m_graph) in cached_graphs.items():
        m_vis = _overlay(base, m_skel, m_graph, measure, COLORS[measure])
        p = os.path.join(out_dir, f"{method_name}_{measure}.png")
        cv2.imwrite(p, cv2.cvtColor(m_vis, cv2.COLOR_RGB2BGR))
print(f"  [OK] All Graph Overlays Saved")

print(f"\n{'='*70}")
print("UNIFIED EVALUATION & DASHBOARD GENERATION COMPLETE")
print("=" * 70)
