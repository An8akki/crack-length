# System Module Inventory & Provenance Guide

This document maps all architectural components across the repository, delineating:
1. **Depth Estimation** (sourced from `An8akki/crack-length` on branch `depth-estimation-integration`)
2. **Length Estimation** (sourced from `An8akki/crack-length` on branch `length-estimation`)
3. **Image Restoration** (currently added locally in the workspace)
4. **Base Modules from `Rijja-explore/AI-Powered_Civil_Infrastructure`** (the upstream baseline)

---

## 1. Provenance Matrix

| Module | Source / Branch | Status | Primary Purpose |
| :--- | :--- | :--- | :--- |
| **Depth Estimation** | `personal/depth-estimation-integration` | On Git Remote (`An8akki/crack-length`) | 3D crevice depth recess estimation & ML regression (mm) |
| **Length Estimation** | `personal/length-estimation` & `main` | Local & Remote Synced | Tri-method skeleton thinning, planar homography & physical crack length (mm) |
| **Candidate A: Crack Width** | `crack_quantification/width.py` | Local Workspace Addition | Normal-ray cross-sections, boundary-reliability under restoration shifts, ACI 224R code |
| **Candidate B: Crack Volume**| `crack_quantification/volume.py` | Local Workspace Addition | Physical crevice recess volume (mm³/cm³) via 3D surface baseline depth integration |
| **Candidate C: Crack Area** | `crack_quantification/area.py` | Local Workspace Addition | Perspective-corrected area (mm²) & boundary uncertainty tolerance bounds (±Δ mm²) |
| **Candidate D: Temporal Progression**| `crack_quantification/temporal.py`| Local Workspace Addition | Uncertainty-aware change detection, annualized growth rates (mm/yr), structural alerts |
| **Image Restoration** | Local Workspace (`restoration_data/`, `restoration_results/`) | Untracked Local Assets | RestorationNet deep image denoising, super-resolution & visual enhancement |
| **Defect Detection** | `origin/main` (`Rijja-explore`) | Base Repository | YOLOv8 object detection for civil surface defects |
| **Defect Segmentation** | `origin/main` (`Rijja-explore`) | Base Repository | YOLOv8 instance segmentation & bounding box localisation |
| **Material Classification** | `origin/main` (`Rijja-explore`) | Base Repository | TFLite edge model classifying surface substrate (concrete, asphalt, brick) |
| **3D Heightmap Generation**| `origin/main` (`Rijja-explore`) | Base Repository | Monocular grayscale disparity/topography visualization |
| **Automated PDF Reporting** | `origin/main` (`Rijja-explore`) | Base Repository | Engineering-grade structural inspection PDF generator |
| **React & Flask Platform** | `origin/main` (`Rijja-explore`) | Base Repository | Full-stack application container & Hugging Face deployment pipeline |


---

## 2. Depth Estimation Module (From Git: `An8akki/crack-length`)

Located on branch `depth-estimation-integration` (commit `decc8b66`), this module models crack recess profile depth from 2D pixel intensity gradients and calibrated geometry.

### Directory Structure & Files
```
crack_depth/
├── __init__.py
├── predictor.py                     # High-level depth inference coordinator
├── core/
│   ├── __init__.py
│   ├── calibration_models.py       # Sensor & camera intrinsic/extrinsic geometry
│   ├── calibration_schema.py       # Validation schemas for depth calibration inputs
│   ├── ml_estimator.py             # Feature extractor & ExtraTrees depth regressor wrapper
│   ├── morphology.py               # Crack profile cross-section slicing & normal vectors
│   ├── preprocessor.py             # Image conditioning, contrast enhancement & normalization
│   ├── recess_extractor.py         # Crevice shadow, intensity trough, and recess profile metrics
│   └── surface_estimator.py        # Surrounding plane fit & baseline height reference
├── models/
│   ├── final_config.json           # Model hyperparameter configuration
│   ├── final_extra_trees.joblib    # Pre-trained ExtraTrees regressor for depth estimation
│   ├── final_scaler.joblib         # StandardScaler feature transformation matrix
│   └── step5_14_model_manifest.json# Model provenance, validation metrics & schema hash
docs/depth_estimation/
├── PHASE5_FINAL_STATUS.md          # Architectural validation report
├── phase5_15_leakage_audit.md      # Data hygiene and zero-leakage verification
└── step5_14_final_software_report.md# Final depth engineering benchmark report
crack_quantification/
└── recess_depth.py                 # Glue adapter connecting depth model to quantification API
test_crack_depth.py                 # Unit test suite verifying depth prediction accuracy
```

### Key Capabilities
- Slices orthogonal cross-sections along the crack skeleton.
- Extracts photometric and morphological recess features (shadow troughs, intensity variance, gradient steepness).
- Uses a trained `ExtraTreesRegressor` calibrated against ground-truth laser crevice depth profiles.
- Outputs physical crevice depth statistics: `max_depth_mm`, `mean_depth_mm`, `depth_profile_curve`.

---

## 3. Length Estimation Module (From `An8akki/crack-length`)

Located on branch `length-estimation` (commit `d713c708`), this module provides sub-millimeter crack length quantification using skeleton topology and planar calibration.

### Directory Structure & Files
```
crack_quantification/
├── __init__.py
├── api_adapter.py                  # Normalizes measurement inputs for REST endpoints
├── integrate_pipeline.py           # End-to-end integration: segmentation -> skeleton -> metric length
├── quantify.py                     # Tri-method skeleton thinning & path length calculations
└── test_quantification.py          # Validation tests for length quantification routines
planar_calibration/
├── __init__.py
├── camera_pose.py                  # PnP camera pitch/yaw/roll angle compensation
├── fiducial_detector.py            # ArUco / checkerboard / coin fiducial scale detection
├── planar_measurement.py           # Metric scale factor calculation (mm/pixel)
└── rectification.py                # Homography matrix derivation & perspective distortion correction
synthetic_specimens/
├── __init__.py
├── crack_fracture_engine.py        # Procedural crack generation with branch junctions
├── generator.py                    # Synthetic dataset generation pipeline
├── material_appearance.py          # Concrete roughness, aggregate, and texture synthesis
└── surface_models.py               # 3D surface mesh generation
validation_platform/
├── __init__.py
├── benchmark_runner.py             # Batch benchmark orchestrator
├── real_specimen_schema.py         # Ground truth benchmark dataset schema
├── reliability_model.py            # Confidence scoring & error bounds estimation
└── tri_method_benchmark.py         # Comparison harness for Zhang-Suen, Guo-Hall, & Morphological
outputs/crack_comparison/
├── comparison_report.json          # Metric comparison results across thinning algorithms
├── length_results.csv              # Tabular benchmark run logs
├── length_results.json             # JSON serialized benchmark metrics
├── all_methods_comparison.png      # Visual side-by-side comparison
└── mst_marked_crack.png            # Minimum Spanning Tree (MST) pruned graph overlay
evaluate_crack_image.py             # Standalone CLI evaluation script
test_planar_calibration.py          # Test suite for homography & fiducial scaling
test_synthetic_platform.py          # Test suite for synthetic specimen generation
templates/crack_length_ui.html      # Interactive standalone web inspection portal
docs/PHYSICAL_MEASUREMENT_GUIDE.md  # Engineering protocol for field calibration
CRACK_LENGTH_RESEARCH_PROTOCOL.md   # Academic research methodology & accuracy analysis
```

### Key Capabilities
- **Tri-Method Thinning**: Benchmark comparison between Zhang-Suen, Guo-Hall, and Morphological skeletonization.
- **Topological Graph Pruning**: Solves shortest/longest geodesic crack path and Minimum Spanning Tree (MST) to eliminate spurious 8-connected cycle loops.
- **Planar Calibration**: Corrects oblique camera perspectives via planar homography, yielding true millimeter-scale dimensions.

---

## 4. Image Restoration Module (Local Workspace Addition)

Located in the local workspace, this module handles deep image restoration and super-resolution for degraded civil infrastructure imagery.

### Directory Structure & Files
```
restoration_data/
├── concrete_cracks/                # 100 sample benchmark images (crack_0000.jpg to crack_0099.jpg)
└── uploaded_real_images/           # Real-world field inspection test inputs
restoration_results/
├── best_restoration_model.pt       # Trained RestorationNet checkpoint (~2.39 MB PyTorch model)
├── real_image_outputs/             # Restored infrastructure evaluation outputs (0000 to 0099)
├── uploaded_real_outputs/          # Restored field sample outputs
├── ablation_results.csv            # Network ablation performance metrics
├── baseline_comparison.csv         # Comparative metrics against bicubic/bilinear baselines
├── external_model_benchmark.csv    # Benchmark against external Super-Resolution algorithms
├── external_model_benchmark_summary.csv # High-level benchmark summary
├── real_infrastructure_image_test.csv   # No-reference evaluation on real-world structures
├── uploaded_real_image_test.csv    # Field test score records
├── training_history.csv            # Loss curve, PSNR & SSIM training trajectories
├── qualitative_comparison.png      # Visual comparison figure (Input vs Restored)
└── final_summary.json              # Aggregate performance metrics (PSNR: ~18.55 dB, SSIM: ~0.652)
```

### Key Capabilities
- Custom **RestorationNet** PyTorch architecture trained from scratch.
- Cleans sensor noise, motion blur, and compression artifacts on rough concrete textures prior to crack segmentation and quantification.

---

## 5. Base Modules from `Rijja-explore/AI-Powered_Civil_Infrastructure`

The upstream repository provides the complete base civil inspection system:

### 1. Object Detection Engine (YOLOv8)
* **Directory**: `runs/detect/`
* **Weights**: `runs/detect/train/weights/best.pt`, `runs/detect/train3/weights/best.pt`
* **Function**: Bounding-box detection for civil infrastructure defects (cracks, spalling, efflorescence, exposed rebar).
* **Metrics & Diagnostics**: PR curves, F1 curves, confusion matrices, and batch validation outputs.

### 2. Semantic Crack Segmentation & Localisation
* **Directory**: `segmentation_model/`
* **Weights**: `segmentation_model/weights/best.pt`
* **Script**: `segmentation_with_localisation.py`
* **Function**: Pixel-level polygon mask segmentation of cracks with spatial bounding coordinates.

### 3. Substrate Material Classification
* **Directory**: `materialclassification_model/`
* **Model**: `material_classifier.tflite`
* **Function**: Lightweight edge neural network identifying substrate surface material (Concrete, Asphalt, Masonry/Brick) to select appropriate structural degradation thresholds.

### 4. 3D Monocular Heightmap Reconstruction
* **Script**: `image_3d_heightmap.py`
* **Frontend View**: `frontend/src/pages/Heightmap3D.jsx` & `frontend/src/styles/heightmap3d.css`
* **Function**: Generates an interactive 3D topography heightfield mesh from single-view luminance and gradient cues using Three.js.

### 5. Automated Engineering PDF Report Generation
* **Script**: `pdf_report.py`
* **Function**: Compiles defect detections, severity ratings, spatial coordinates, and visual overlays into a formal engineering PDF report.

### 6. Full-Stack Web Application & Deployment Framework
* **Backend API**:
  * `finalwebapp.py` — Primary Flask application serving detection endpoints and inspection workflows.
  * `finalwebapp_api.py` — RESTful interface connecting React client to AI pipelines.
  * `hf_model_loader.py` & `model_downloader.py` — Auto-downloads model weights from Hugging Face Hub during boot.
* **Frontend Application**:
  * `frontend/` — Full React SPA containing:
    * `MainDashboard.jsx`: Infrastructure asset status & summary KPIs.
    * `ImageAnalysis.jsx`: Interactive upload, segmentation overlay, defect breakdown.
    * `Heightmap3D.jsx`: Interactive 3D visualizer.
    * `RealTimeMonitoring.jsx`: Live camera feed inspection.
    * `Environmental.jsx`: Environmental factor analysis (humidity, temperature wear).
    * `About.jsx`: Project specification & SDGs (UN SDG 9, 11, 13).
* **Container & CI/CD**:
  * `Dockerfile`, `startup.sh`, `.dockerignore`, `BUILD_TRIGGER`, `requirements.txt`.
  * `.github/workflows/deploy.yml` — Automated deployment pipeline to Hugging Face Spaces.

---

## 6. Advanced Candidate Research Modules (`crack_quantification/`)

In response to the candidate-by-candidate civil infrastructure research assessment, four specialized research modules are implemented and integrated with the segmentation, length, depth, and restoration pipelines:

### Candidate A: Advanced Crack-Width Estimation (`crack_quantification/width.py`)
- **Beyond Basic EDT**: Implements orthogonal normal-ray cross-sections along skeleton points, tracing perpendicular vectors to true mask boundaries.
- **Restoration & Boundary Reliability**: Evaluates width stability under 1-pixel boundary shifts (morphological erosion/dilation) simulating restoration edge-smoothing and segmentation uncertainty.
- **Civil Structural Code Classification**: Automates ACI 224R-01 classification (Hairline $<0.1$mm, Fine $0.1-0.2$mm, Moderate $0.2-0.3$mm, Severe $>0.3$mm).
- **Benchmark Evaluation**: Computes MAE, RMSE, bias, percentage error, and range-specific errors (<0.2mm, 0.2-0.5mm, >0.5mm) aligned with the 2025 Nature crack-width benchmark dataset.

### Candidate B: Physical Crack-Volume Estimation (`crack_quantification/volume.py`)
- **Physics-Based Surface Recess Integration**: Calculates true 3D crevice recess volume ($V$ in mm³ and cm³) by integrating depth relative to the surrounding healthy concrete baseline ($Z_{\text{baseline}}(p) - Z_{\text{depth}}(p)$), rather than naive length × width × depth multiplication.
- **Surface Baseline Reconstruction**: Automatically estimates baseline healthy substrate geometry via morphological grey closing or local neighborhood sampling.
- **Depth Uncertainty Modeling**: Accounts for sparse, noisy, or irregular depth samples to provide defensible 95% confidence bounds $[V_{\text{low}}, V_{\text{high}}]$.
- **Benchmark Evaluation**: Computes volume MAE, RMSE in mm³, and relative volume error %.

### Candidate C: Perspective-Aware Crack-Area Quantification (`crack_quantification/area.py`)
- **Perspective Distortion Correction**: Computes physical crack surface area (mm²) corrected for camera tilt angles and planar foreshortening via homography matrix rectification.
- **Defensible Boundary Uncertainty**: Employs inner and outer morphological tolerances to generate empirical error bounds (e.g. $A \pm \Delta$ mm²), modeling segmentation fuzziness.
- **Benchmark Evaluation**: Evaluates area MAE (mm²), relative percentage error, and Mean Intersection-over-Union (mIoU) against reference specimen masks.

### Candidate D: Uncertainty-Aware Temporal Crack Progression (`crack_quantification/temporal.py`)
- **Uncertainty-Aware Change Detection**: Distinguishes genuine physical crack expansion from measurement artifacts (camera positioning, lighting, moisture, or restoration changes) by only flagging growth when $\Delta > k \cdot \sigma_{\text{measurement}}$ ($k = 2.0$ for 95% confidence).
- **Annualized Growth Velocity**: Quantifies $\Delta L$ (mm/year), $\Delta W$ (mm/year), $\Delta A$ (mm²/year), and $\Delta V$ (mm³/year).
- **Structural Alert Classification**: `STABLE`, `MINOR_PROGRESSION`, `ACTIVE_GROWTH`, `CRITICAL_ACCELERATION`.
- **Spatial Progression Segmentation**: Classifies pixels into Stable Core, Newly Propagated Tips, Lateral Widening, and Transient Noise.

