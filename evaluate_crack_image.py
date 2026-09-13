"""Evaluate crack image with AI segmentation + length estimation."""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

IMAGE_PATH = r"D:\AppData\crack-695010_1280.jpg"

print("=" * 70)
print("CRACK LENGTH ESTIMATION - AI SEGMENTATION EVALUATION")
print("=" * 70)
print(f"Image: {IMAGE_PATH}")

# Step 1: Load image
import cv2
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    print(f"ERROR: Could not read image: {IMAGE_PATH}")
    sys.exit(1)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
h, w = image_rgb.shape[:2]
print(f"\n[1/7] Image loaded: {image_rgb.shape}")

# Step 2: AI Segmentation
print("\n[2/7] Running YOLOv8 segmentation model...")
try:
    from ultralytics import YOLO
    seg_path = "segmentation_model/weights/best.pt"
    if os.path.exists(seg_path):
        print(f"   Loading model from {seg_path}...")
        seg_model = YOLO(seg_path)
        print("   Predicting...")
        results = seg_model.predict(source=image_rgb, conf=0.3, save=False, verbose=False)
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

# Fallback
if mask is None or mask.sum() == 0:
    print("\n[2/7] Fallback: Edge detection + adaptive threshold...")
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    # Cracks are dark - use inverted adaptive threshold
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    print(f"   Fallback mask: {int(mask.sum())} pixels ({100*mask.sum()/(h*w):.1f}%)")

print(f"\n[3/7] Mask: {mask.shape}, foreground: {int(mask.sum())} pixels")

# Step 4: Skeletonize
from crack_quantification.skeleton import skeletonize_mask
print("\n[4/7] Skeletonizing mask...")
for method in ["guo_hall", "zhang_suen", "skeletonize"]:
    skel = skeletonize_mask(mask, method=method)
    print(f"   {method}: {int(skel.sum())} skeleton pixels")

best_skel = skeletonize_mask(mask, method="guo_hall")

# Step 5: Graph Analysis
from crack_quantification.graph_analysis import analyze_skeleton
from crack_quantification.length import compute_length, _connected_components
from crack_quantification.quantify import quantify_crack, format_report

print("\n[5/7] Analyzing skeleton graph...")
graph = analyze_skeleton(best_skel)
length = compute_length(best_skel)
n_comps = len(_connected_components(best_skel))

print(f"\n{'='*70}")
print("CRACK LENGTH RESULTS (guo_hall skeleton)")
print(f"{'='*70}")
print(f"  Total network length:  {length['length_px']:.2f} px")
print(f"  Main path length:      {graph['main_path_px']:.2f} px")
print(f"  MST length:            {graph['mst_length_px']:.2f} px")
print(f"  Connected components:  {length['components']}")
print(f"  Junctions:             {graph['junction_count']}")
print(f"  Endpoints:             {graph['endpoint_count']}")
print(f"  Skeleton pixels:       {int(best_skel.sum())}")

# Per-method comparison
print(f"\n  Per-method comparison:")
for method_name, skel_method in [("zhang_suen", "zhang_suen"), ("guo_hall", "guo_hall"), ("morphological", "skeletonize")]:
    m_skel = skeletonize_mask(mask, method=skel_method)
    m_graph = analyze_skeleton(m_skel)
    print(f"    {method_name:<12}  network={m_graph['network_length_px']:.1f}px  main={m_graph['main_path_px']:.1f}px  mst={m_graph['mst_length_px']:.1f}px  comps={len(m_graph['components'])}")

# Scale conversion
mm_per_px = 0.277
print(f"\n  With scale {mm_per_px} mm/px:")
print(f"    Network:  {length['length_px'] * mm_per_px:.2f} mm")
print(f"    Main:     {graph['main_path_px'] * mm_per_px:.2f} mm")
print(f"    MST:      {graph['mst_length_px'] * mm_per_px:.2f} mm")

# Step 6: Full quantification
print(f"\n[6/7] Full quantification report:")
try:
    report = quantify_crack(mask, mm_per_px=0.277)
    print(format_report(report, image_id="crack-695010_1280"))
except Exception as e:
    print(f"  Quantification: {e}")

# Step 7: Save visualizations
print("\n[7/7] Saving visualizations...")
os.makedirs("outputs/crack_comparison/real_eval", exist_ok=True)
vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
vis[mask > 0] = (0, 0, 255)
vis[best_skel > 0] = (0, 255, 0)
vis_path = "outputs/crack_comparison/real_eval/ai_mask_skeleton.png"
cv2.imwrite(vis_path, vis)
print(f"  {vis_path}")

# Save per-method graph overlays
from crack_quantification.compare_methods import _overlay, COLORS, MEASURES
base = (mask * 255).astype(np.uint8)
if base.ndim == 2:
    base = np.stack([base] * 3, axis=-1)
for measure in MEASURES:
    for method_name, skel_method in [("zhang_suen", "zhang_suen"), ("guo_hall", "guo_hall"), ("morphological", "skeletonize")]:
        m_skel = skeletonize_mask(mask, method=skel_method)
        m_graph = analyze_skeleton(m_skel)
        m_vis = _overlay(base, m_skel, m_graph, measure, COLORS[measure])
        p = f"outputs/crack_comparison/real_eval/{method_name}_{measure}.png"
        cv2.imwrite(p, cv2.cvtColor(m_vis, cv2.COLOR_RGB2BGR))
print(f"  All graph overlays saved")

print(f"\n{'='*70}")
print("EVALUATION COMPLETE")
print(f"{'='*70}")
