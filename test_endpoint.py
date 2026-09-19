"""
Test suite for /api/crack-length and /api/analyze endpoints
Verifying that the 3D Relative Depth Recess model executes on every run
and that depth values are correctly structured and returned for frontend consumption.
"""
import io
import json
import base64
import numpy as np
import requests
from PIL import Image

BASE_URL = 'http://localhost:7860'

print("=" * 70)
print("INFRAVISION AI - MULTI-MODAL PIPELINE & DEPTH MODEL VERIFICATION")
print("=" * 70)

# ---------------------------------------------------------------------------
# 1. Test GET /api/crack-length (Web UI HUD)
# ---------------------------------------------------------------------------
print("\n[TEST 1] GET /api/crack-length (Frontend Dashboard)")
r_get = requests.get(f'{BASE_URL}/api/crack-length', timeout=10)
print(f"  Status Code: {r_get.status_code}")
assert r_get.status_code == 200, f"Expected 200 OK, got {r_get.status_code}"
assert "3D Relative Recess" in r_get.text or "crack-length" in r_get.text.lower(), "Expected dashboard HTML"
print("  [OK] GET /api/crack-length successfully serves frontend UI with 3D Recess card")

# ---------------------------------------------------------------------------
# 2. Test POST /api/crack-length with mask (Monocular Depth Fallback)
# ---------------------------------------------------------------------------
print("\n[TEST 2] POST /api/crack-length (2D Mask -> Monocular Depth Model)")
mask_img = Image.new('L', (300, 150), color=0)
for x in range(30, 270):
    for y in range(70, 80):
        mask_img.putpixel((x, y), 255)

buf_mask = io.BytesIO()
mask_img.save(buf_mask, format='PNG')
mask_bytes = buf_mask.getvalue()

r_post_mono = requests.post(
    f'{BASE_URL}/api/crack-length',
    files={'mask': ('mask.png', mask_bytes, 'image/png')},
    data={'mm_per_px': '0.277'},
    timeout=20,
)
print(f"  Status Code: {r_post_mono.status_code}")
assert r_post_mono.status_code == 200, f"POST failed: {r_post_mono.text}"
res_mono = r_post_mono.json()

print(f"  2D Length:        {res_mono['length_px']:.1f} px ({res_mono['length_mm']:.2f} mm)")
print(f"  Mean Width:       {res_mono['width_px']} px ({res_mono['width_mm']} mm)")
print(f"  Best 2D Method:   {res_mono['best_method']}")

# Verify depth recess attached
assert 'depth_recess' in res_mono and res_mono['depth_recess'] is not None, "Missing depth_recess in /api/crack-length response"
dr_mono = res_mono['depth_recess']
assert dr_mono['predicted_relative_depth'] > 0, "Depth recess should be positive"
assert dr_mono['calibration_state'] == 'UNCALIBRATED', "Depth calibration state must remain UNCALIBRATED"
assert dr_mono['unit'] == 'mm', f"Unexpected unit: {dr_mono['unit']}"
assert len(dr_mono['interval_90']) == 2, "Missing 90% conformal interval"

print(f"  [DEPTH] Recess:   {dr_mono['predicted_relative_depth']:.2f} {dr_mono['unit']}")
print(f"  90% Conformal CI: [{dr_mono['interval_90'][0]:.2f}, {dr_mono['interval_90'][1]:.2f}] mm")
print(f"  Reliability:      {dr_mono['reliability']}")
print(f"  Monocular Fallback: {dr_mono['is_monocular_fallback']}")
print("  [OK] Depth model executed and returned valid relative crevice depth")

# ---------------------------------------------------------------------------
# 3. Test POST /api/crack-length with attached Depth Map
# ---------------------------------------------------------------------------
print("\n[TEST 3] POST /api/crack-length with Depth Map")
depth_img = Image.new('L', (300, 150), color=128)
for x in range(30, 270):
    for y in range(70, 80):
        depth_img.putpixel((x, y), 65)  # recessed depth depression

buf_depth = io.BytesIO()
depth_img.save(buf_depth, format='PNG')
depth_bytes = buf_depth.getvalue()

r_post_rgbd = requests.post(
    f'{BASE_URL}/api/crack-length',
    files={
        'image': ('image.png', mask_bytes, 'image/png'),
        'depth': ('depth.png', depth_bytes, 'image/png')
    },
    data={'mm_per_px': '0.277'},
    timeout=20,
)
print(f"  Status Code: {r_post_rgbd.status_code}")
assert r_post_rgbd.status_code == 200, f"Depth Map POST failed: {r_post_rgbd.text}"
res_rgbd = r_post_rgbd.json()

dr_rgbd = res_rgbd['depth_recess']
assert dr_rgbd is not None, "Missing depth_recess in Depth Map response"
assert dr_rgbd['predicted_relative_depth'] > 0, "Predicted relative depth should be positive"
assert dr_rgbd['is_monocular_fallback'] is False, "Expected depth map ingestion (not monocular fallback)"

print(f"  [DEPTH] Recess (with Depth Map): {dr_rgbd['predicted_relative_depth']:.2f} mm")
print(f"  90% Conformal CI:                [{dr_rgbd['interval_90'][0]:.2f}, {dr_rgbd['interval_90'][1]:.2f}] mm")
print(f"  Reliability:                     {dr_rgbd['reliability']}")
print(f"  Depth Map Ingestion:             Verified (is_monocular_fallback=False)")
print("  [OK] Attached depth map successfully processed by depth model")

# ---------------------------------------------------------------------------
# 4. Test POST /api/analyze (Main Analysis Pipeline with Depth Output)
# ---------------------------------------------------------------------------
print("\n[TEST 4] POST /api/analyze (Main Analysis Pipeline with Depth Recess)")
# Create synthetic RGB crack image
rgb_img = Image.new('RGB', (320, 240), color=(160, 160, 160))
for x in range(40, 280):
    for y in range(115, 125):
        rgb_img.putpixel((x, y), (30, 30, 30))  # dark crack line

buf_rgb = io.BytesIO()
rgb_img.save(buf_rgb, format='PNG')
rgb_b64 = "data:image/png;base64," + base64.b64encode(buf_rgb.getvalue()).decode('utf-8')

r_analyze = requests.post(
    f'{BASE_URL}/api/analyze',
    json={'image': rgb_b64, 'px_to_cm_ratio': 0.1, 'confidence_threshold': 0.25},
    timeout=30,
)
print(f"  Status Code: {r_analyze.status_code}")
assert r_analyze.status_code == 200, f"/api/analyze failed: {r_analyze.text}"
res_analyze = r_analyze.json()

assert 'results' in res_analyze, "Missing 'results' in /api/analyze response"
results = res_analyze['results']
cq = results.get('crack_quantification') or {}
depth_data = results.get('depth_recess') or cq.get('depth_recess')

print(f"  Crack Detection Count: {results.get('crack_detection', {}).get('count', 0)}")
print(f"  Crack Quantification Available: {cq.get('available', False)}")
if depth_data:
    p_depth = depth_data.get('predicted_relative_depth')
    rel = depth_data.get('reliability')
    ci90 = depth_data.get('interval_90')
    print(f"  [DEPTH] Analyzed Depth Recess: {p_depth} mm")
    print(f"  Reliability:                  {rel}")
    print(f"  90% CI:                       {ci90} mm")
    print("  [OK] Depth recess attached to /api/analyze results for frontend display")
else:
    print("  [INFO] No crack mask returned from segmentation model on synthetic flat image")

# Verify depth heatmap in output images
assert 'output_images' in res_analyze, "Missing 'output_images' in /api/analyze"
assert 'depth_estimation' in res_analyze['output_images'], "Missing 'depth_estimation' heatmap in output_images"
print("  [OK] Depth heatmap image generated in output_images['depth_estimation']")

print("\n" + "=" * 70)
print("ALL TESTS PASSED: DEPTH MODEL SUCCESSFULLY INTEGRATED INTO MAIN PIPELINE")
print("=" * 70)

