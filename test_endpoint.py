"""Test /api/crack-length endpoint."""
import io
import json
import numpy as np
import requests
from PIL import Image

# 1. Test GET (405 Method Not Allowed expected)
r = requests.get('http://localhost:7860/api/crack-length', timeout=10)
print(f"GET Status: {r.status_code} (Expected 405)")

# 2. Test POST with a generated binary mask
mask_img = Image.new('L', (200, 100), color=0)
for x in range(20, 180):
    for y in range(45, 55):
        mask_img.putpixel((x, y), 255)

buf = io.BytesIO()
mask_img.save(buf, format='PNG')
png_bytes = buf.getvalue()

r = requests.post(
    'http://localhost:7860/api/crack-length',
    files={'mask': ('mask.png', png_bytes, 'image/png')},
    data={'mm_per_px': '0.277'},
    timeout=15,
)
print(f"\nPOST Mask Status: {r.status_code}")
if r.status_code == 200:
    res = r.json()
    print(f"  Length: {res['length_px']:.1f} px ({res['length_mm']:.2f} mm)")
    print(f"  Width:  {res['width_px']} px ({res['width_mm']} mm)")
    print(f"  Method: {res['best_method']}")
else:
    print(f"  Error: {r.text}")

