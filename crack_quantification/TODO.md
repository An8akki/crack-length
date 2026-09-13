# Crack Quantification Package - Task Progress

## Steps from approved plan

- [x] 1. Create crack_quantification package skeleton (folder + __init__)
- [x] 2. Implement skeleton.py (binary mask -> 1-pixel skeleton)
- [x] 3. Implement length.py (skeleton graph traversal -> length in px)
- [x] 4. Implement width.py (distance transform -> width stats + profile)
- [x] 5. Implement area.py (count pixels x pixel area)
- [x] 6. Implement calibration.py (px <-> mm, 3 methods)
- [x] 7. Implement severity.py (rule-based severity + orientation + branches)
- [x] 8. Implement run_quantification.py (CLI demo, mask input)
- [x] 9. Test with synthetic mask (test script)
- [x] 10. Scale recovery module (metadata / distance / depth / fusion)
- [x] 11. Benchmark: compare_scale_methods.py
- [x] 12. Integrate with segmentation model: integrate_pipeline.py
- [x] 13. XAI explanation of px->mm conversion
- [x] 14. End-to-end verification (self-test + manual synthetic run)

## Stage 2 - FULLY AUTOMATIC CALIBRATION (user supplies only --image)
- [x] 15. autocalibrate.py: EXIF extraction (PIL) with camera model + focal + image size
- [x] 16. Camera sensor database (iPhone / Samsung / Pixel / Canon / Nikon / Sony / full-frame)
- [x] 17. Distance estimation hook (register_depth_provider for Metric3D / UniDepth / Depth Anything)
- [x] 18. Staged fallback: EXIF -> camera DB -> assumed default (with confidence + XAI)
- [x] 19. integrate_pipeline.py updated: `python -m ... --image crack.jpg` works automatically
- [x] 20. Verified: no-EXIF -> assumed (low conf); EXIF iPhone 15 Pro -> exif_camera_geometry

## Future Stages (per user roadmap)
- [ ] Stage 3: auto distance via metric depth model (Metric3D / UniDepth / DepthAnything)
- [ ] Stage 4: direct mm/pixel regression (Vision Transformer) - no camera geometry at inference
