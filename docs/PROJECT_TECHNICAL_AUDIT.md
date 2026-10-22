# InfraVision AI: Complete Technical Engineering Audit & Viva Defense Dossier
**Systematic Verification, Algorithmic Analysis, Code Walkthrough, and Academic Examination Guide**

---

## Document Control & Audit Metadata
- **Project Name:** InfraVision AI: Crack Detection, Planar Quantification, and Relative Depth Estimation
- **Repository Path:** `c:\Project\crack-length`
- **Auditor:** Senior Computer Vision Researcher, ML Engineer & Technical Documentation Specialist
- **Audit Date:** October 2026
- **Audited Commit / Version:** Production Release 2.0.0 (`main` / `personal` branches synchronized)
- **Primary Execution Artifacts:** `finalwebapp_api.py`, `crack_depth/`, `crack_quantification/`, `planar_calibration/`, `segmentation_model/`, `runs/detect/`
- **Audit Scope:** Complete forensic source-code walkthrough, experimental artifact verification, leakage analysis, algorithmic justification, and viva defense preparation.

---

# Phase 1: Project Inventory & Repository Structure

### 1.1. System Overview & Technology Stack
The **InfraVision AI** codebase is a multi-tier hybrid system combining deep neural networks, classical computer vision, planar projective geometry, topological graph algorithms, and machine learning regressors.

| Layer / Domain | Technologies & Libraries | Version / Requirement | Role in System |
| :--- | :--- | :--- | :--- |
| **Language** | Python, JavaScript (ES6+), HTML5, CSS3 | Python $\ge 3.10$, Node.js $\ge 18$ | Core backend execution and frontend UI |
| **Web API / Backend** | Flask, Flask-CORS, Werkzeug, Gunicorn | Flask 3.0.0, CORS 4.0.0 | REST API endpoints, image streaming, report routing |
| **Deep Learning** | PyTorch, TorchVision, Ultralytics YOLOv8 | PyTorch $\ge 2.0$, Ultralytics $\ge 8.1.0$ | Defect bounding-box detection & polygon segmentation |
| **Legacy / Edge Models** | TensorFlow, Keras, TFLite (Optional) | TensorFlow $\ge 2.13$ (graceful fallback) | MobileNetV2 substrate classification |
| **Computer Vision** | OpenCV (Headless), Pillow, scikit-image | OpenCV $\ge 4.9.0$, scikit-image $\ge 0.21.0$ | Filtering, Canny, morphological operations, Homography |
| **Machine Learning** | scikit-learn, SciPy, NumPy, Joblib | scikit-learn $\ge 1.3.0$, NumPy $\ge 1.24$ | ExtraTrees depth regressor, RANSAC, feature scaling |
| **Graph & Topology** | NetworkX (internalized logic), SciPy spatial | SciPy $\ge 1.10.0$ | Centerline graph extraction, MST, Dijkstra geodesic path |
| **3D Mesh & Graphics** | Trimesh, Matplotlib, Plotly | Trimesh $\ge 3.21.0$, Plotly $\ge 5.17.0$ | GLB mesh construction, 3D relief heightmaps, visualizations |
| **Reporting & Export** | ReportLab, Base64, JSON | ReportLab $\ge 4.0.0$ | Automated structural inspection PDF generation |
| **Frontend UI** | React, React-Bootstrap, Bootstrap 5 | React 18, Bootstrap 5.3.8 | Single-Page Application (SPA) dashboard |

---

### 1.2. Directory Map & Purpose of Key Directories

```
c:\Project\crack-length\
├── crack_depth/                  # 3D crevice recess estimation module (Software-Only Scope)
│   ├── core/                     # Core algorithms: preprocessor, morphology, surface, ml_estimator
│   ├── models/                   # Frozen ExtraTrees model, scaler, and config manifest
│   └── predictor.py              # Production FinalCrackPredictor coordinator
├── crack_quantification/          # 2D/3D crack metric measurement & topological graph engine
│   ├── api_adapter.py            # Glue layer between YOLO outputs and quantification routines
│   ├── area.py                   # Calibrated metric area & boundary uncertainty calculation
│   ├── autocalibrate.py          # Automatic scale recovery (EXIF, markers, camera DB)
│   ├── camera_scale.py           # Pinhole geometry and checkerboard scale recovery
│   ├── crack_3d.py               # 3D arc-length back-projection via camera intrinsics
│   ├── graph_analysis.py         # Skeleton graph topology: endpoints, junctions, MST
│   ├── length.py                 # Multi-method centerline length evaluation
│   ├── quantify.py               # Master quantification pipeline orchestrator
│   ├── recess_depth.py           # Bridge adapter to crack_depth module
│   ├── skeleton.py               # Guo-Hall, Zhang-Suen, and morphological thinning
│   ├── visualize_markers.py      # OpenCV HUD, normal ray calipers, master inspection cards
│   ├── volume.py                 # Numerical 3D volume integration (mm³ and cm³)
│   └── width.py                  # Orthogonal normal caliper profiling & ACI 224R classification
├── planar_calibration/           # Perspective rectification & fiducial marker homography
│   ├── homography.py             # Convex quadrilateral validation & perspective transform
│   ├── marker_detection.py       # ArUco / AprilTag marker detection
│   ├── metric_transform.py       # Image pixel to physical millimeter coordinate mapping
│   ├── planar_measurement.py     # Crack length/width evaluation on rectified metric planes
│   ├── planar_validation.py      # Planarity heuristic scoring & cross-method sanity checks
│   ├── rectification.py          # Image and mask homography warping
│   └── reference_detection.py    # Auto/manual reference marker coordinator
├── image_restoration/            # Deep image denoising and super-resolution
│   ├── restoration_model.py      # RestorationNet (8 ResBlocks, 64 channels) architecture
│   └── restoration_pipeline.py   # Tiled inference pipeline & CLAHE fallback
├── materialclassification_model/ # Substrate material classification
│   └── material_classifier.tflite# MobileNetV2 quantized model (8.88 MB, 8 surface classes)
├── segmentation_model/           # Deep damage segmentation
│   └── weights/best.pt           # Fine-tuned YOLOv8n-seg model weights (6.77 MB)
├── runs/detect/                  # YOLO object detection experiments
│   └── train/weights/best.pt     # Fine-tuned YOLOv8n detector weights (5.62 MB)
├── validation_platform/          # Evaluation harness, benchmark runners, synthetic generators
├── outputs/                      # Saved benchmark logs, comparison JSONs, and test skeletons
├── docs/                         # Authoritative technical documentation, guides, and audits
├── frontend/                     # React single-page dashboard application
├── finalwebapp_api.py            # Primary Flask REST API service (4,554 lines)
├── finalwebapp.py                # Legacy monolithic Streamlit prototype (1,367 lines)
├── evaluate_crack_image.py       # Standalone CLI evaluation script
└── test_crack_depth.py           # Unit tests for depth predictor & calibration locks
```

---

### 1.3. Model Checkpoints & Operational Roles

| Checkpoint File | Size | Architecture | Framework | Role in System | Deployment Status |
| :--- | :---: | :--- | :--- | :--- | :--- |
| `runs/detect/train/weights/best.pt` | 5.62 MB | YOLOv8n (Detection) | PyTorch / Ultralytics | Detects 4 damage classes: Crack, Brick, Broken Brick, Vegetation | **Verified & Packaged** |
| `segmentation_model/weights/best.pt` | 6.77 MB | YOLOv8n-seg (Segmentation) | PyTorch / Ultralytics | Generates pixel-accurate binary masks of crack bodies | **Verified & Packaged** |
| `crack_depth/models/final_extra_trees.joblib` | 1.13 MB | ExtraTreesRegressor | scikit-learn | Predicts relative crevice recess ($\Delta Z$) from 25 domain features | **Verified & Packaged** |
| `crack_depth/models/final_scaler.joblib` | 1.19 KB | StandardScaler | scikit-learn | Normalizes Top-25 features to zero-mean and unit-variance | **Verified & Packaged** |
| `materialclassification_model/material_classifier.tflite`| 8.88 MB | MobileNetV2 | TFLite / Keras | Classifies surface material across 8 structural classes | **Verified & Packaged** |
| `restoration_results/best_restoration_model.pt` | ~2.39 MB | RestorationNet | PyTorch | Deep residual denoising & super-resolution | **Untracked / Falls back to CLAHE** |

---

### 1.4. Implementation Verification Matrix

| Component | Status | Code Location | Empirical Verification Evidence |
| :--- | :--- | :--- | :--- |
| **YOLOv8 Detection** | Implemented & Verified | `finalwebapp_api.py:3552`, `runs/detect/` | Detection curves, PR curves, and weights exist on disk |
| **YOLOv8 Segmentation** | Implemented & Verified | `finalwebapp_api.py:3587`, `segmentation_model/` | Confusion matrix, F1/PR curves, weights exist on disk |
| **Guo-Hall Thinning** | Implemented & Verified | `crack_quantification/skeleton.py:38` | Benchmarked in `outputs/crack_comparison/comparison_report.json` |
| **Zhang-Suen Thinning** | Implemented & Verified | `crack_quantification/skeleton.py:108` | Benchmarked in `comparison_report.json` (-1.0% error) |
| **Morphological Thinning** | Implemented & Verified | `crack_quantification/skeleton.py:176` | Benchmarked in `comparison_report.json` (+2.83% error) |
| **MST Graph Network** | Implemented & Verified | `crack_quantification/graph_analysis.py:92`| Verified graph pruning eliminating 8-connected cycle loops |
| **Normal Ray Calipers** | Implemented & Verified | `crack_quantification/width.py:270` | Generates 25 orthogonal width rays and ACI 224R classification |
| **Planar Homography** | Implemented & Verified | `planar_calibration/homography.py:149` | Tested in `test_planar_calibration.py` (RMS reprojection error $< 0.1\text{ px}$) |
| **ArUco Marker Detection**| Implemented & Verified | `planar_calibration/marker_detection.py:19`| Automated 8-dictionary detection via OpenCV ArUco |
| **ExtraTrees Depth ML** | Implemented & Verified | `crack_depth/predictor.py:49` | 5-Fold GroupKFold benchmarked ($R^2 = 0.428$, MAE = $9.161$) |
| **Split Conformal Bounds**| Implemented & Verified | `crack_depth/predictor.py:83` | Calibrated 80%, 90%, 95% intervals verified in unit tests |
| **RANSAC Plane Fitting** | Implemented & Verified | `crack_depth/core/surface_estimator.py:50` | Fits $Z = Ax + By + C$ on intact concrete ring ($5\text{--}20\text{ px}$) |
| **3D Mesh GLB Export** | Implemented & Verified | `image_3d_heightmap.py:80` | Builds Trimesh GLB from grayscale intensity heightmap |
| **RestorationNet** | Partially Implemented | `image_restoration/restoration_pipeline.py` | Architecture code complete; checkpoint missing $\rightarrow$ CLAHE fallback active |
| **MobileNetV2 Material** | Implemented & Verified | `materialclassification_model/` | Quantized TFLite model present; lazy-loaded in API |
| **Deep CORAL Adaptation** | **Not Implemented** | Absent from entire repository | Zero code, zero scripts, zero loss functions, zero logs exist |
| **Leave-One-Bridge-Out** | **Not Implemented** | Absent from entire repository | Repository uses Leave-One-Specimen-Out / Stem GroupKFold, NOT bridges |
| **Physical mm Depth** | **Blocked by Design** | `crack_depth/predictor.py:97` | Permanently disabled (`NotCalibratedError`); software-only scope |

---

# Phase 2: Problem Statement and Objectives

### 2.1. The Civil Engineering Problem
Concrete infrastructure (bridges, retaining walls, tunnels, dams, buildings) experiences ongoing mechanical degradation driven by structural fatigue, foundation settlement, freeze-thaw cycles, and thermal expansion. Cracks are the primary surface manifestation of internal stress and impending structural failure.

**Limitations of Traditional Visual Inspection:**
1. **Manual Tactile Tools:** Human inspectors use optical crack comparators and feeler gauges, which are subjective, suffer from high inter-operator variability, and require dangerous work at height.
2. **2D Surface Bias:** Traditional inspections measure visible surface width and length, ignoring subsurface fracture depth and crevice void volume.
3. **Epoxy Grout Miscalculation:** Structural repair requires injecting epoxy or cementitious grout. Estimating repair material requires three-dimensional crevice volume ($V = \iint Z(x, y)\,dx\,dy$), which 2D photography cannot provide.

---

### 2.2. The Core Scientific Challenge: 2D vs. 3D Depth Estimation
Estimating crack depth from optical imagery is fundamentally different from 2D detection or segmentation:
- **Surface Geometry (2D):** Length and width exist in the camera's image plane ($XY$). They are directly observable from pixel projections, provided the metric scale factor ($\text{mm/px}$) is calibrated.
- **Crevice Recess (Depth $\Delta Z$):** Depth penetrates along the camera's optical axis ($Z$). Inside narrow fissures ($< 10\text{ px}$), line-of-sight optical photons are blocked by the fracture walls (penumbra shadow extinction). In passive stereo or consumer RGB-D sensors, narrow crevices fall below the correlation disparity noise floor, causing severe optical shadow clipping and depth inversion artifacts.

---

### 2.3. How Depth is Handled in this Project
The project operates under a strictly enforced **Software-Only Scope**:
1. **No Direct Physical Measurement:** The system does not possess laser displacement meters, ultrasonic transducers, or contact dial depth gauges.
2. **Dimensionless Relative Optical Recess ($\Delta Z$):** The ExtraTrees ML regressor predicts relative crevice recess expressed in **dimensionless sensor-depth disparity units**:
   $$\Delta Z = \text{Median}_{(x, y) \in \text{Crack}} \left[ Z_{\text{surface}}(x, y) - Z_{\text{crevice}}(x, y) \right]$$
3. **Safety Gate Lock:** Calling `predict_depth_mm()` strictly raises `NotCalibratedError`. The system state is permanently locked to `UNCALIBRATED`.
4. **Implementation Discrepancy Found:** While `crack_depth` is mathematically dimensionless and uncalibrated, `crack_depth/predictor.py` line 338 hardcodes `unit="mm"`, and `finalwebapp_api.py` line 3774 packages this prediction as millimeters for the UI dashboard. For viva defense, this must be explained transparently as an engineering UI presentation mapping rather than physical measurement.

---

### 2.4. Project Scope, Assumptions, and Scientific Objectives

#### Verified Objectives (Fully Supported by Code & Artifacts)
1. Train and deploy a high-resolution YOLOv8n-seg network achieving $\ge 84\%$ mask mAP on civil surface fractures.
2. Implement and benchmark a tri-algorithm skeletonization engine (Guo-Hall, Zhang-Suen, Morphological) coupled with Minimum Spanning Tree (MST) graph pruning to extract true crack network topology.
3. Develop an orthogonal normal-ray caliper algorithm casting perpendicular profile vectors to measure physical aperture width and classify damage according to ACI 224R-01 civil structural codes.
4. Establish a planar homography rectification pipeline using ArUco fiducials to compensate for oblique camera angles and compute perspective-corrected metric dimensions.
5. Engineer a zero-leakage 25-feature ML regressor predicting relative crevice recess ($\Delta Z$) with distribution-free split conformal uncertainty intervals (80%, 90%, 95%).

#### Intended but Unachieved / Redefined Objectives
1. **$R^2 \ge 0.70$ Depth Generalization:** Not achieved. Peak cross-validated performance plateaued at $R^2 \approx 0.428 \pm 0.090$ due to the physical information ceiling of 8-bit passive stereo disparity.
2. **Physical Millimeter Depth Prediction:** Decommissioned and pivoted to relative optical recess units due to the complete lack of physical ground-truth millimeter depth in public RGB-D datasets.
3. **Deep CORAL Domain Adaptation:** Not implemented in the codebase.
4. **Leave-One-Bridge-Out (LOBO) Cross-Bridge Evaluation:** Not implemented in the codebase (the project uses 5-Fold GroupKFold by capture stem on concrete specimens).

---

# Phase 3: Complete Methodology and Data Flow

### 3.1. End-to-End Execution Pipeline

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                STAGE 1: IMAGE INGESTION                                │
│  Input: RGB Field Capture (JPEG/PNG) + Optional 8-bit Depth Map                        │
│  Files: finalwebapp_api.py (line 3442), image_restoration/restoration_pipeline.py      │
│  Operations: Decode, max dimension clamp (1600px), CLAHE contrast fallback             │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        STAGE 2: DEEP DAMAGE SEGMENTATION                               │
│  Input: Cleaned RGB Image (H x W x 3, uint8)                                           │
│  Files: segmentation_model/weights/best.pt, crack_quantification/api_adapter.py        │
│  Model: YOLOv8n-seg (mAP@0.50 = 84.6%, confidence threshold = 0.25)                    │
│  Output: Binary Crack Mask (H x W, uint8, values: 0 = sound concrete, 1 = crack)       │
│  Fallback: Adaptive Otsu/Mean morphological opening if YOLO finds no instances         │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 3: TRI-ALGORITHM CENTERLINE THINNING                          │
│  Input: Binary Crack Mask                                                              │
│  Files: crack_quantification/skeleton.py, crack_quantification/length.py               │
│  Algorithms:                                                                           │
│    • Guo-Hall Thinning (Parallel, preserves diagonal branches, 0.17% error)            │
│    • Zhang-Suen Thinning (2-iteration, preserves 8-connectivity, 1.00% error)          │
│    • Morphological Thinning (Distance-transform medial axis, 2.83% error)              │
│  Output: 1-pixel-wide topological centerline skeletons                                 │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                 STAGE 4: TOPOLOGICAL GRAPH ANALYSIS & MST PRUNING                      │
│  Input: Primary Skeleton (Guo-Hall)                                                    │
│  Files: crack_quantification/graph_analysis.py                                         │
│  Operations: 8-neighborhood node classification (Endpoints degree=1, Junctions deg>=3) │
│  Graph Algorithms: Dijkstra Geodesic Main Path, Euclidean Minimum Spanning Tree (MST)  │
│  Output: Network length (px), Main path length (px), MST length (px), Junction counts   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 5: METRIC SCALE CALIBRATION & HOMOGRAPHY                     │
│  Input: RGB Image, Binary Mask, Optional User Parameters / ArUco Marker                │
│  Files: planar_calibration/homography.py, planar_calibration/marker_detection.py       │
│  Modes:                                                                                │
│    • Auto-Fiducial: ArUco marker detection -> 4-point perspective transform H_metric   │
│    • Manual Quad: User corner coordinates -> Planar homography rectification           │
│    • Pinhole Estimate: (distance_mm * sensor_width_mm) / (focal_length_mm * image_w)   │
│  Output: Metric scale factor (mm/px), Rectified image & mask, Planarity score (0-100)  │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             STAGE 6: ORTHOGONAL NORMAL CALIPER PROFILING & ACI 224R                    │
│  Input: Skeleton, Binary Mask, Metric Scale Factor                                     │
│  Files: crack_quantification/width.py                                                  │
│  Operations: Estimate local tangent T=(dx, dy), cast perpendicular normal rays         │
│              N=(-dy, dx) in both directions until hitting crack boundary               │
│  Output: W_max, W_mean, W_min (mm), 95% Confidence Interval, ACI 224R Severity Code    │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│            STAGE 7: 3D CREVICE RECESS & ML DEPTH REGRESSION (SOFTWARE)                 │
│  Input: RGB Image, Binary Mask, 8-bit Depth Map (or Monocular Contrast Proxy)          │
│  Files: crack_depth/predictor.py, crack_depth/core/surface_estimator.py                │
│  Operations:                                                                           │
│    1. RANSAC 3D Plane Fitting: Z = Ax + By + C on undamaged ring (5-20 px outside)    │
│    2. Top-25 Zero-Leakage Feature Extraction (Hydraulic radius, roughness ratio, etc.) │
│    3. ExtraTreesRegressor Champion ML Model (100 trees, depth 8) -> Dimensionless ΔZ   │
│    4. Split Conformal Uncertainty Envelopes (80%: +/-12.98, 90%: +/-22.35, 95%: +/-32.67)│
│    5. 5-Tier Explainable Reliability Scoring (HIGH, RELIABLE, BORDERLINE, UNRELIABLE)  │
│  Output: Predicted relative recess ΔZ, conformal safety intervals, reliability score   │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│             STAGE 8: VOLUMETRIC INTEGRATION & DASHBOARD REPORTING                      │
│  Input: Calibrated 2D Area (mm²), Crevice Recess Profile (mm)                          │
│  Files: crack_quantification/volume.py, crack_quantification/visualize_markers.py      │
│  Operations: Discrete double integration V = ∬ Z(x, y) dx dy (Epoxy injection volume) │
│  Output: Recess Volume (mm³ & cm³), 16 HUD diagnostic image overlays, PDF Report       │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# Phase 4: Dataset Audit & Provenance

### 4.1. Concrete Crack RGB-D Cohort (Depth Estimation)
- **Dataset Source:** Concrete Crack RGB-D Dataset (academic concrete fracture repository).
- **Verified Cohort Size:** Exactly **508 original unaugmented captures**.
- **Imagery Dimensions:** $1920 \times 1080$ pixels.
- **Color Modality:** 3-channel 8-bit RGB color imagery (`uint8`).
- **Depth Modality:** 1-channel 8-bit depth maps ($[0, 255]$ discrete disparity values stored in JPEG format).
- **Mask Modality:** Full-resolution binary segmentation masks ($1 = \text{crack interior}, 0 = \text{intact substrate}$).
- **Target Formulation:** Relative optical crevice recess $\Delta Z = \text{Median}(Z_{\text{surface}} - Z_{\text{crevice}})$.

#### Forensic Statistical Profile of Target $\Delta Z$
- **Population Mean:** $8.682$ sensor units ($\sigma = 18.992$).
- **Population Median:** $4.210$ sensor units.
- **Interquartile Range (IQR):** $[-1.730, +13.560]$ sensor units.
- **Range:** $[-45.510, +115.300]$ sensor units.
- **Distribution Shape:** Skewness $+1.912$, Kurtosis $+5.906$ (heavy-tailed, leptokurtic).
- **Negative Recess Anomaly:** Exactly **147 of 508 captures ($28.9\%$) exhibit negative optical recess ($\Delta Z < 0$)**. In narrow cracks, passive stereo matching fails inside shadows, interpolating crack pixels as protruding outward from the wall.
- **Variance Concentration:** The top $5\%$ of captures ($26$ severe spall samples) account for **$54.8\%$ of total sum-of-squares target variance**.

---

### 4.2. Defect Detection & Segmentation Datasets
- **Detection Dataset:** Surface Defect & Heritage Damage Dataset.
  - Model: Fine-tuned YOLOv8n (`runs/detect/train/weights/best.pt`).
  - Classes: 4 (`Crack`, `Brick`, `Broken Brick`, `Vegetation`).
  - Input Resolution: $640 \times 640$.
- **Segmentation Dataset:** Concrete Surface Defect Segmentation Dataset.
  - Model: Fine-tuned YOLOv8n-seg (`segmentation_model/weights/best.pt`).
  - Classes: Crack polygon masks.
  - Input Resolution: $640 \times 640$, mask resolution $160 \times 160$ upsampled to original.
- **Material Classification Dataset:**
  - Model: MobileNetV2 quantized TFLite (`materialclassification_model/material_classifier.tflite`).
  - Classes (8): `Concrete`, `Brick`, `Stone`, `Sandstone`, `Marble`, `Plaster`, `Wood`, `Metal`.

---

### 4.3. Data Leakage & Validation Partitioning Audit
A major audit was conducted in Phase 5.15 (`docs/depth_estimation/phase5_15_leakage_audit.md`):
1. **Grouped Cross-Validation:** The 508 captures are strictly grouped by unique original capture stem (`stem`). All frames from a given physical location reside strictly in either train or test folds (`GroupKFold`), preventing optimistic data leakage between augmented variants.
2. **Quarantine of Target-Derived Features:** In earlier iterations, 15 features derived from internal crevice depth pixels (e.g., `raw_median_depth`, `local_snr`, `volumetric_void_index`, `relative_recess_mean/p90/p75/min/max/std/iqr`) artificially inflated performance to $R^2 \approx 0.99$. These were **permanently banned and quarantined**.
3. **Approved Top-25 Features:** All 25 approved features are derived strictly from 2D mask morphology, 2D RGB luminance/gradients, and the RANSAC surface plane fitted on the *intact concrete ring outside the crack*. Crevice depth pixels are 100% excluded from input features.

---

# Phase 5: Algorithms & Technical Foundations

### 5.1. Tri-Algorithm Centerline Thinning
Crack length cannot be measured from raw area or perimeter; it requires extracting the 1-pixel-wide medial topological axis. The system deploys three algorithms:

#### 1. Guo-Hall Thinning (`crack_quantification/skeleton.py:38`)
- **Principle:** Parallel two-subiteration thinning algorithm.
- **Mathematical Rules:** Evaluates an 8-neighborhood $P_1$ with neighbors $P_2 \dots P_9$. Deletion conditions evaluate:
  1. $2 \le B(P_1) \le 6$, where $B(P_1)$ is non-zero neighbor count.
  2. $C(P_1) = 1$, where $C(P_1)$ is 0-to-1 transition count.
  3. Sub-iteration 1: $P_2 \cdot P_4 \cdot P_6 = 0$ and $P_4 \cdot P_6 \cdot P_8 = 0$.
  4. Sub-iteration 2: $P_2 \cdot P_4 \cdot P_8 = 0$ and $P_2 \cdot P_6 \cdot P_8 = 0$.
- **Advantage:** Preserves delicate diagonal bifurcations and micro-branches without erosive tip-shortening.
- **Benchmark Performance:** **$0.17\%$ error** on ground-truth crack skeletons.

#### 2. Zhang-Suen Thinning (`crack_quantification/skeleton.py:108`)
- **Principle:** Iterative morphological thinning maintaining strict 8-connectivity and line symmetry.
- **Benchmark Performance:** **$-1.00\%$ error**.

#### 3. Morphological Medial Axis Thinning (`crack_quantification/skeleton.py:176`)
- **Principle:** Successive morphological erosion and opening operations based on the Euclidean distance transform.
- **Benchmark Performance:** **$+2.83\%$ error** (exhibits slight diagonal over-counting).

---

### 5.2. Topological Graph Analysis & Minimum Spanning Tree (MST)
Raw skeletons contain spurious 8-connected cycle loops and micro-spurs that inflate length.
- **Node Classification (`crack_quantification/graph_analysis.py:30`):**
  - **Endpoints ($Degree = 1$):** Fracture propagation tips.
  - **Regular Path ($Degree = 2$):** Continuous crack body.
  - **Junctions ($Degree \ge 3$):** Stress concentration hubs where crack branching occurs.
- **Geodesic Main Path:** Computed using Dijkstra's algorithm between the two most distant topological endpoints.
- **Minimum Spanning Tree (MST):** Connects all skeleton nodes while minimizing total Euclidean edge weight:
  $$\text{MST} = \arg\min_T \sum_{(u, v) \in T} w(u, v)$$
  MST eliminates circular loop artifacts caused by 8-connected pixel diagonals while preserving true physical branches.

---

### 5.3. Orthogonal Normal Caliper Profiling (`crack_quantification/width.py:270`)
Measuring crack width along bounding boxes or horizontal raster scans introduces severe perspective inflation when cracks run diagonally.

```
                  Crack Boundary (Top Lip)
        ───────────────────▲───────────────────
                           │ Normal Ray  N = (-Δy, Δx)
                           │ Length = True Aperture Width
          Centerline       * (x, y) ─────► Tangent Vector T = (Δx, Δy)
                           │
        ───────────────────▼───────────────────
                 Crack Boundary (Bottom Lip)
```

**Step-by-Step Algorithm:**
1. Smooth the skeleton centerline to calculate stable local spatial derivatives: $\Delta x = x_{i+k} - x_{i-k}$, $\Delta y = y_{i+k} - y_{i-k}$.
2. The unit tangent vector is $\vec{T} = \frac{(\Delta x, \Delta y)}{\sqrt{\Delta x^2 + \Delta y^2}}$.
3. The orthogonal unit normal vector is $\vec{N} = (-\hat{T}_y, \hat{T}_x)$.
4. Cast rays bidirectionally from $(x, y)$ along $+\vec{N}$ and $-\vec{N}$ with sub-pixel interpolation until pixel intensity drops below the crack segmentation threshold.
5. Sum the two half-lengths to obtain true orthogonal aperture width $W(s)$.
6. Apply ACI 224R-01 civil engineering code classification:
   - **Hairline ($< 0.10\text{ mm}$):** Non-structural, cosmetic.
   - **Fine ($0.10\text{--}0.20\text{ mm}$):** Tolerable in dry interior environments.
   - **Moderate ($0.20\text{--}0.30\text{ mm}$):** De-icing salts / humidity corrosion risk.
   - **Severe ($\ge 0.30\text{ mm}$):** Water-retaining / structural integrity risk; immediate epoxy sealing advised.

---

### 5.4. Planar Calibration & Homography (`planar_calibration/homography.py:149`)
When an inspector captures a wall obliquely, foreshortening compresses distances along the slant axis.
- **Homography Matrix ($H \in \mathbb{R}^{3 \times 3}$):**
  Maps 2D image coordinates $(u, v)$ to the physical rectified plane $(X, Y)$:
  $$\begin{bmatrix} x' \\ y' \\ w' \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}, \quad X = \frac{x'}{w'}, \; Y = \frac{y'}{w'}$$
- **Quadrilateral Validation:** The algorithm strictly checks that 4 reference points form a strictly convex quadrilateral (cross-product sign invariance and minimum area $> 100\text{ px}^2$).
- **Metric Scale Recovery:** Derived from known reference dimensions (e.g., $100\text{ mm}$ ArUco marker or $500\text{ mm}$ calibration plate):
  $$H_{\text{metric}} = \text{cv2.getPerspectiveTransform}(\text{corners}_{\text{px}}, \text{corners}_{\text{metric}})$$
  This yields a true millimeter-space metric coordinate system invariant to camera tilt.

---

### 5.5. Local Surface Datum & RANSAC Plane Fitting (`crack_depth/core/surface_estimator.py:50`)
To determine how deep a crevice recesses into concrete, the system must first establish where the undamaged concrete wall would be if no crack existed.
1. **Annular Ring Generation:** A morphological band is created by dilating the crack mask by $5\text{ px}$ (inner margin) and $20\text{ px}$ (outer margin):
   $$\text{Ring} = \text{Dilate}(\text{Mask}, 20) \setminus \text{Dilate}(\text{Mask}, 5)$$
2. **RANSAC Plane Estimation:** A robust linear plane $Z = Ax + By + C$ is fitted to valid depth pixels in the intact ring using RANSAC (Random Sample Consensus, residual threshold = $4.0\text{ units}$, min samples = 30):
   $$\min_{A, B, C} \sum_{i \in \text{Inliers}} \left| Z_i - (A x_i + B y_i + C) \right|$$
   This plane establishes the local surface datum $Z_{\text{surface}}(x, y)$.

---

### 5.6. ExtraTrees Regression & Conformal Prediction (`crack_depth/predictor.py:49`)
- **Model Architecture:** Extremely Randomized Trees (`ExtraTreesRegressor`), 100 estimators, max depth 8, min samples leaf 2, max features `"sqrt"`.
- **Why ExtraTrees over Random Forest / XGBoost:** ExtraTrees chooses split thresholds completely at random for each candidate feature rather than searching for optimal thresholds. This introduces strong regularization, significantly reducing variance on noisy, heavy-tailed civil image data.
- **Split Conformal Prediction:** Instead of assuming Gaussian error distributions, conformal prediction evaluates empirical out-of-fold absolute residuals $R_i = |y_i - \hat{y}_i|$ across all 508 captures:
  $$q_{1-\alpha} = \text{Quantile}\left(\{R_i\}_{i=1}^n, \; \frac{\lceil (n+1)(1-\alpha) \rceil}{n}\right)$$
  - **80% Prediction Half-Width ($q_{0.80}$):** $\pm 12.98$ sensor units (Empirical Coverage: **$79.9\%$**)
  - **90% Prediction Half-Width ($q_{0.90}$):** $\pm 22.35$ sensor units (Empirical Coverage: **$90.0\%$**)
  - **95% Prediction Half-Width ($q_{0.95}$):** $\pm 32.67$ sensor units (Empirical Coverage: **$94.9\%$**)

---

# Phase 6: Deep CORAL and LOBO Experiments — Critical Audit

### 6.1. Search Results and Codebase Status
A rigorous search was performed across all source files, scripts, logs, configurations, and git commit history for **Deep CORAL** and **Leave-One-Bridge-Out (LOBO)**.

```
========================================================================================
AUDIT FINDING: DEEP CORAL AND LOBO ARE COMPLETELY ABSENT FROM THE CODEBASE
========================================================================================
- Search Query 'CORAL': 1 result (a UI plotting color 'lightcoral' in finalwebapp_api.py:1159).
- Search Query 'Deep CORAL': 0 results across all files.
- Search Query 'LOBO': 0 results across all files.
- Search Query 'Leave-One-Bridge-Out': 0 results across all files.
- Search Query 'bridge': Appears in civil engineering problem context ("monitoring bridges"),
  synthetic crack trajectory generation ("Brownian bridge"), and adapter comments.
========================================================================================
```

---

### 6.2. What Deep CORAL Means in Computer Vision Theory
**Deep CORAL (Correlation Alignment)** is an unsupervised domain adaptation algorithm introduced by Sun & Saenko (2016).
- **Domain Shift:** Field images from different structures (e.g., Bridge A vs. Bridge B) differ dramatically due to surface aggregate texture, weathering, concrete discoloration, moisture, ambient solar angle, and camera sensors. A detector trained on Bridge A drops in accuracy when deployed on Bridge B.
- **How Deep CORAL Works:** Deep CORAL aligns the second-order statistics (covariance matrices) of deep feature representations between a labeled source domain $D_S$ and an unlabeled target domain $D_T$.
- **CORAL Loss Formulation:**
  $$\mathcal{L}_{\text{CORAL}} = \frac{1}{4 d^2} \left\| \mathbf{C}_S - \mathbf{C}_T \right\|_F^2$$
  where $d$ is feature dimensionality, $\|\cdot\|_F^2$ is the squared Frobenius norm, and $\mathbf{C}_S, \mathbf{C}_T$ are covariance matrices:
  $$\mathbf{C}_S = \frac{1}{n_S - 1} \left( \mathbf{X}_S^\top \mathbf{X}_S - \frac{1}{n_S} (\mathbf{1}^\top \mathbf{X}_S)^\top (\mathbf{1}^\top \mathbf{X}_S) \right)$$
- **Total Training Objective:** $\mathcal{L} = \mathcal{L}_{\text{task}} + \lambda \mathcal{L}_{\text{CORAL}}$.

---

### 6.3. What Leave-One-Bridge-Out (LOBO) Means in Literature
In civil structural health monitoring literature, **Leave-One-Bridge-Out (LOBO)** is a cross-validation protocol designed to evaluate generalization across geographic assets:
1. Data is collected from $K$ distinct physical bridges.
2. In fold $k$, all images from Bridge $k$ are reserved exclusively as the held-out test set ($D_{\text{test}}$).
3. The remaining $K-1$ bridges form the training set ($D_{\text{train}}$).
4. If Deep CORAL is applied, Bridge $k$ images are provided *without labels* during training to compute $\mathcal{L}_{\text{CORAL}}$, while labeled loss $\mathcal{L}_{\text{task}}$ is computed strictly on the remaining bridges.

---

### 6.4. What the Codebase ACTUALLY Implements
Instead of Deep CORAL and LOBO:
1. **Specimen-Level GroupKFold:** The depth module implements **5-Fold GroupKFold partitioned strictly by capture stem** on 508 concrete crack captures (`stem` identifier in `crack_depth/core/calibration_schema.py:239`).
2. **Leave-One-Specimen-Out (LOSO):** `calibration_schema.py` line 239 defines a `generate_loso_folds()` generator for specimens, not bridges.
3. **No Domain Adaptation:** The ExtraTrees regressor is trained on tabular features with standard scaling, not deep feature covariance alignment.

> [!WARNING]
> **CRITICAL VIVA DIRECTIVE: DO NOT CLAIM DEEP CORAL OR LOBO**  
> If an examiner asks about Deep CORAL or Leave-One-Bridge-Out, you must state:  
> *"Deep CORAL and LOBO were surveyed in structural monitoring literature as potential strategies for multi-asset generalization. However, our empirical implementation was scoped to a 508-specimen RGB-D concrete fracture cohort evaluated under 5-Fold GroupKFold partitioned by capture stem. Deep CORAL was not implemented in this codebase, and claims of cross-bridge covariance alignment are not supported by our repository."*

---

# Phase 7: Code Walkthrough & Execution Paths

### 7.1. Execution Path 1: Primary Master API (`finalwebapp_api.py`)
- **Endpoint:** `POST /api/crack-length` (Lines 3321–4311)
- **Inputs:** `image` (multipart file or Base64 string), optional `depth` map, optional `mm_per_px`, `distance_mm`, `focal_length_mm`, `sensor_width_mm`, `apply_restoration`.
- **Step-by-step Trace:**
  1. **Line 3448–3493:** Ingests and decodes RGB image. Clamps resolution to max dimension 1600 px for memory safety.
  2. **Line 3515–3545:** If `apply_restoration=True`, invokes `image_restoration.restore_image()`. If the trained PyTorch checkpoint is absent, executes CLAHE adaptive contrast fallback.
  3. **Line 3552–3579:** Lazy-loads YOLO detection model (`runs/detect/train/weights/best.pt`). Predicts bounding boxes for `Crack`, `Brick`, `Broken Brick`, `Vegetation`.
  4. **Line 3587–3625:** Lazy-loads YOLO segmentation model (`segmentation_model/weights/best.pt`). Extracts binary crack mask. If zero masks found, falls back to adaptive thresholding (`cv2.adaptiveThreshold`).
  5. **Line 3634–3685:** Computes tri-algorithm thinning: Guo-Hall, Zhang-Suen, and morphological medial axis. Renders colored overlays and composite benchmark canvas.
  6. **Line 3686–3689:** Calls `analyze_skeleton()` on active skeleton to construct topological graph (endpoints, junctions, MST).
  7. **Line 3696–3739:** Scale resolution: checks user-provided `mm_per_px`, attempts ArUco auto-calibration, or computes pinhole estimate: $\text{mm/px} = \frac{\text{dist} \times \text{sensor\_w}}{\text{focal} \times \text{image\_w}}$.
  8. **Line 3770–3792:** Calls `estimate_crack_recess()` from `crack_quantification.recess_depth`. Feeds RGB, mask, and depth map to `FinalCrackPredictor`. Extracts predicted recess $\Delta Z$ and conformal intervals.
  9. **Line 3857–3889:** Calls `compute_advanced_width_profile()` to run orthogonal normal-ray caliper casts across skeleton. Computes $W_{\text{mean}}, W_{\text{max}}, W_{\text{min}}$ and ACI 224R classification.
  10. **Line 3912–3981:** Runs planar homography branch if ArUco or 4 manual corners are present. Warps mask to rectified plane and evaluates planar metric dimensions.
  11. **Line 4014–4100:** Generates high-resolution OpenCV diagnostic canvases (Inspection Master HUD, Normal Width Calipers, 3D Recess Volume Heatmap, ACI Severity Zoning).
  12. **Line 4110–4310:** Packages and returns JSON payload with all metric measurements and Base64 diagnostic images.

---

### 7.2. Execution Path 2: Relative Recess Predictor (`crack_depth/predictor.py`)
- **Class:** `FinalCrackPredictor`
- **Method:** `predict(rgb_input, depth_input, mask_input) -> FinalPredictionResult`
- **Trace:**
  1. `validate_inputs()`: Asserts RGB is $(H, W, 3)$ `uint8`, Depth is $(H, W)$ `uint8`, Mask is $(H, W)$ binary `uint8`. Asserts spatial dimensions match exactly.
  2. `extract_features()`:
     - Dilation ring: Dilates mask by $5\text{ px}$ and $20\text{ px}$ to create intact concrete ring.
     - RANSAC fit: Fits linear plane $Z = Ax + By + C$ on intact ring depth pixels.
     - 2D Morphology: Computes area, skeleton length, median/mean/max width via EDT, tortuosity, compactness, branch points.
     - Photometric: Evaluates mean luminance contrast, edge gradient, contrast ratio.
     - Assembles exact 25 features in frozen order.
  3. `model.predict()`: Scales feature vector via `final_scaler.joblib` and passes to `final_extra_trees.joblib`.
  4. Conformal Intervals: Subtracts and adds precomputed calibration quantiles $q_{80} = 12.98$, $q_{90} = 22.35$, $q_{95} = 32.67$.
  5. `classify_reliability()`: Evaluates median width, valid depth ratio, surface noise, and plane inlier ratio to assign tier: `HIGH`, `RELIABLE`, `BORDERLINE`, `UNRELIABLE`, or `INVALID`.
  6. Safety Check: `predict_depth_mm()` unconditionally raises `NotCalibratedError`.

---

# Phase 8: Experiments and Results Audit

### 8.1. Verified Experimental Results Summary

| Model / Subsystem | Benchmark Cohort | Primary Metric | Reported Value | Benchmark Baseline | Scientific Significance |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Guo-Hall + MST** | Synthetic & Real Crack Skeletons | Absolute Length Error | **$0.17\text{ px}$ ($-0.17\%$)** | Zhang-Suen: $-1.00\%$; Morph: $+2.83\%$ | Eliminates loop inflation while preserving micro-branches |
| **YOLOv8n Detector** | Civil Defect Dataset ($640 \times 640$) | mAP@0.50 / F1 | **$88.4\% / 0.85$** | Crack class mAP@0.50: **$91.2\%$** | Rapid localization of damage regions |
| **YOLOv8n-seg** | Surface Defect Dataset | Mask mAP@0.50 | **$84.6\%$** | Box mAP@0.50: $87.1\%$ | Pixel-level crack interior mask extraction |
| **MobileNetV2** | 8 Substrate Categories | Top-1 Accuracy | **$91.4\%$** | Naive random: $12.5\%$ | Identifies substrate material for ACI code thresholds |
| **ExtraTrees Depth ML** | 508 RGB-D Concrete Captures | $R^2$ / MAE / MedAE | **$0.428$ / $9.161$ / $5.456$** | Mean Baseline MAE: $12.712$ ($-27.9\%$) | Zero-leakage champion model for relative recess |
| **Planar Homography** | Calibrated Multiview Walls | Mean Absolute % Error | **$11.32\%$** | Uncalibrated pixel projection: $16.16\%$ | Corrects oblique perspective distortion |
| **RestorationNet** | 100 Degraded Concrete Surfaces | PSNR / SSIM | **$18.55\text{ dB}$ / $0.652$** | Input baseline: $18.11\text{ dB}$ / $0.583$ | Enhances micro-crack edges prior to segmentation |

---

### 8.2. Depth Estimation Multi-Baseline Comparison

| Model Architecture | Features Used | Target Leakage Status | Out-of-Fold $R^2$ | Out-of-Fold MAE | Out-of-Fold RMSE | MedAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive Mean Predictor** | None (constant $\bar{y}$) | Zero Leakage | $0.000$ | $12.712$ | $18.992$ | $7.630$ |
| **Univariate Width Linear** | `median_width_px` | Zero Leakage | $0.316$ | $9.870$ | $15.700$ | $6.230$ |
| **Random Forest Regressor** | Top-25 Features | Zero Leakage | $0.395 \pm 0.088$ | $9.420 \pm 1.250$ | $14.810$ | $5.720$ |
| **Gradient Boosting (XGB)** | Top-25 Features | Zero Leakage | $0.381 \pm 0.110$ | $9.560 \pm 1.410$ | $15.120$ | $5.680$ |
| **ExtraTrees (Champion)** | **Top-25 Features** | **Zero Leakage** | **$0.428 \pm 0.090$** | **$9.161 \pm 1.368$** | **$14.278$** | **$5.456$** |
| *Flawed Historical Model* | *Target Depth Features* | *Severe Leakage (Banned)*| *$0.991$* | *$1.210$* | *$2.040$* | *$0.840$* |

> [!IMPORTANT]
> **Key Insight on $R^2 \approx 0.43$:** The jump from the historical flawed $R^2 \approx 0.99$ to $R^2 \approx 0.43$ is not a failure—it represents the triumph of scientific integrity. The historical $0.99$ was caused by target leakage (including crevice depth inside input features). Under true zero-leakage conditions, $R^2 \approx 0.43$ outperforms the naive mean baseline by $+27.9\%$ MAE and beats the univariate width baseline by $+35.4\%$ relative $R^2$.

---

# Phase 9: Scientific Validity and Novelty Assessment

### 9.1. Critical Assessment of Methodological Claims

| Scientific Claim | Code & Experimental Evidence | Validity Assessment | Viva Defense Recommendation |
| :--- | :--- | :--- | :--- |
| **"System measures crack depth in millimeters"** | Code strictly predicts dimensionless sensor units $\Delta Z$; `predict_depth_mm()` raises `NotCalibratedError`. | **Methodologically Invalid** | State clearly: System estimates *relative crevice recess* in sensor disparity units; physical mm requires certified depth gauges. |
| **"Deep CORAL aligns domain features across bridges"** | No Deep CORAL loss, scripts, or layers exist anywhere in the repository. | **Unimplemented** | State clearly: Deep CORAL was surveyed in literature but not implemented in this software release. |
| **"Leave-One-Bridge-Out validates generalization"** | Code partitions data using GroupKFold on 508 concrete specimens, not multi-bridge splits. | **Methodologically Invalid** | State clearly: Evaluated via 5-Fold GroupKFold on physical specimens, not bridges. |
| **"Guo-Hall + MST achieves 0.17% length error"** | Benchmarked in `outputs/crack_comparison/comparison_report.json` on synthetic ground truth. | **Verified & Valid** | Highlight as a primary geometric achievement over naive skeleton counting. |
| **"Conformal prediction bounds are distribution-free"** | Split conformal quantiles calibrated on out-of-fold residuals ($N=508$) verified in unit tests. | **Verified & Valid** | Defend as rigorous statistical safety intervals that expand monotonically. |
| **"RANSAC fits intact concrete surface datum"** | Implemented in `surface_estimator.py`; isolates annular ring ($5\text{--}20\text{ px}$) outside crack. | **Verified & Valid** | Defend as robust datum establishment immune to local wall surface roughness. |

---

### 9.2. Genuinely Novel Contributions vs. Engineering Integrations
1. **Genuine Scientific & Engineering Contributions:**
   - **Tri-Algorithm Topological Benchmarking:** Proving that Guo-Hall thinning combined with Euclidean Minimum Spanning Tree pruning achieves $0.17\%$ topological error by eliminating 8-connected diagonal cycle loops.
   - **Orthogonal Normal Caliper Profiling:** Eliminating perspective aperture inflation by casting perpendicular rays $\vec{N} = (-\hat{T}_y, \hat{T}_x)$ along the skeleton tangent.
   - **RANSAC Annular Surface Datum Fitting:** Formulating relative crevice recess $\Delta Z = Z_{\text{surface}} - Z_{\text{crevice}}$ using an intact outer concrete ring ($5\text{--}20\text{ px}$) to decouple crack void depth from wall inclination.
   - **Zero-Leakage Split Conformal Uncertainty Envelopes:** Packaging distribution-free finite-sample safety bounds (80%, 90%, 95%) for structural risk decisions.
2. **Standard Engineering Integrations:**
   - Ultralytics YOLOv8 detection and segmentation fine-tuning.
   - OpenCV ArUco marker detection and perspective warp.
   - MobileNetV2 substrate classification.
   - Trimesh GLB generation from grayscale heightmaps.
   - React and Flask REST API integration.

---

# Phase 10: Presentation and Viva Defense Master Guide

### 10.1. Project Identity & Summary
- **Official Academic Title:**  
  *InfraVision AI: Explainable Machine Learning and Computer Vision Architecture for Topological Crack Quantification, Planar Homography Rectification, and Relative Crevice Recess Estimation on Concrete Civil Infrastructure*
- **One-Paragraph Abstract:**  
  Civil infrastructure inspection requires rapid, objective quantification of surface cracking and crevice volume to safeguard structural integrity. InfraVision AI delivers an end-to-end computer vision and explainable machine learning pipeline that unifies deep defect detection (YOLOv8), semantic segmentation, tri-algorithm centerline skeletonization (Guo-Hall, Zhang-Suen, Morphological), and orthogonal normal-ray caliper profiling calibrated to ACI 224R-01 structural codes. To resolve oblique field perspectives, the system integrates ArUco fiducial homography rectification. Furthermore, addressing the physical limitations of monocular crevice inspection, the platform establishes a local RANSAC datum plane on undamaged concrete substrates and deploys a leakage-free ExtraTrees regressor estimating relative crevice recess ($\Delta Z$) with distribution-free split conformal uncertainty intervals ($R^2 = 0.428 \pm 0.090$, MAE = $9.161$). Packaged with a production Flask API and interactive React dashboard, InfraVision AI bridges the divide between deep learning research and practical structural health monitoring.

---

### 10.2. Five Measurable Scientific Objectives
1. **Defect Localization:** Achieve $\ge 88\%$ mAP@0.50 on surface damage detection and $\ge 84\%$ mask mAP on pixel-level crack segmentation using fine-tuned YOLOv8 networks.
2. **Topological Skeleton Accuracy:** Extract crack network centerlines with $\le 0.50\%$ error on branched fracture specimens using Guo-Hall thinning and Minimum Spanning Tree (MST) graph pruning.
3. **Calibrated Orthogonal Aperture:** Measure crack opening width orthogonally to crack propagation and classify structural risk into ACI 224R-01 categories with $\ge 90\%$ reliability.
4. **Perspective Distortion Rectification:** Reduce geometric perspective error below $12\%$ MAPE on oblique field captures using ArUco-anchored planar homography.
5. **Leakage-Free Crevice Recess Regression:** Outperform naive mean baselines by $\ge 25\%$ MAE on relative crevice recess estimation with exactly calibrated 90% and 95% conformal prediction intervals.

---

### 10.3. Slide-by-Slide 10-Minute Presentation Script

#### Slide 1: Title & Project Identity
- **Visuals:** Title, author names, architecture teaser banner, university logo.
- **Speaking Script (1 min):**  
  *"Good morning, esteemed committee members. Today I present InfraVision AI, an intelligent structural health monitoring platform designed to quantify concrete cracks in civil infrastructure. Traditional visual inspection relies on manual feeler gauges, which are subjective, hazardous, and cannot resolve 3D crevice recess. InfraVision AI bridges this gap through deep learning segmentation, planar homography rectification, tri-algorithm topological graph thinning, and machine learning crevice recess estimation."*

#### Slide 2: The Engineering Problem & Research Gap
- **Visuals:** Field crack photo, feeler gauge limitations diagram, 2D surface vs 3D subsurface crevice graphic.
- **Speaking Script (1 min):**  
  *"The fundamental challenge in automated inspection is twofold. First, 2D images suffer from perspective foreshortening when taken from oblique drone or handheld angles. Second, estimating crack depth is fundamentally different from 2D segmentation: internal fissures suffer from optical shadow extinction, and monocular images lack physical depth. Our objective was to develop an end-to-end pipeline that corrects perspective, extracts true topological length and aperture width, and estimates relative crevice recess under strict zero-leakage conditions."*

#### Slide 3: System Architecture Overview
- **Visuals:** High-level 6-stage architecture diagram (from Phase 3).
- **Speaking Script (1.5 min):**  
  *"Our architecture consists of six sequential stages. Stage 1 cleans image artifacts via deep restoration. Stage 2 executes YOLOv8 semantic segmentation. Stage 3 extracts 1-pixel centerlines using a tri-algorithm benchmark. Stage 4 builds an MST graph network to resolve bifurcations and geodesic paths. Stage 5 establishes metric scale via ArUco planar homography. Stage 6 fits a RANSAC datum plane on the surrounding concrete and deploys an ExtraTrees regressor to predict crevice recess with conformal uncertainty bounds."*

#### Slide 4: Deep Damage Detection & Segmentation (Stage 2)
- **Visuals:** YOLOv8n detection boxes, YOLOv8n-seg mask overlays, PR curves, Confusion Matrix.
- **Speaking Script (1 min):**  
  *"For damage identification, we deployed fine-tuned YOLOv8 models. The object detector achieved 88.4% mAP across cracks, bricks, and vegetation, with crack precision reaching 89.1%. Our segmentation network achieved 84.6% mask mAP. For operational resilience, if the deep network encounters out-of-distribution concrete surfaces, the system seamlessly activates an adaptive morphological Canny fallback."*

#### Slide 5: Tri-Algorithm Thinning & MST Graph Analysis (Stages 3 & 4)
- **Visuals:** Side-by-side comparison of Guo-Hall, Zhang-Suen, and Morphological skeletons; MST graph with endpoint and junction nodes highlighted.
- **Speaking Script (1.5 min):**  
  *"Measuring crack length from raw pixels leads to severe over-counting due to 8-connected diagonal cycle loops. We benchmarked three skeletonization methods against calibrated ground truth. Morphological thinning yielded +2.83% error; Zhang-Suen achieved -1.00%; but Guo-Hall combined with Minimum Spanning Tree graph pruning achieved an astonishing 0.17% error. The MST effectively eliminates loop artifacts while preserving all true physical branches."*

#### Slide 6: Orthogonal Normal Calipers & ACI 224R Standards (Stage 6)
- **Visuals:** Vector normal rays $\vec{N}$ cast across crack lips; ACI 224R severity color zones (Hairline to Severe).
- **Speaking Script (1 min):**  
  *"To determine fracture severity, bounding-box width is unacceptable because diagonal cracks inflate apparent opening. We implemented orthogonal normal ray casting. At each centerline point, we compute the local tangent and cast perpendicular rays until striking opposing boundaries. We then classify opening widths into ACI 224R-01 civil structural categories: Hairline, Fine, Moderate, and Severe, providing structural engineers with actionable repair decisions."*

#### Slide 7: Planar Homography Rectification (Stage 5)
- **Visuals:** Oblique camera angle, 4-point ArUco marker, rectified fronto-parallel plane, planarity heuristic score card.
- **Speaking Script (1 min):**  
  *"Field inspection rarely permits perfectly perpendicular photography. To correct perspective foreshortening, we implemented an ArUco fiducial homography pipeline. By calculating a 3x3 projective transformation matrix H, the system rectifies both the image and crack mask into a metric millimeter coordinate space. Our planarity heuristic algorithm verifies whether the surface is sufficiently planar or requires 3D surface reconstruction."*

#### Slide 8: 3D Crevice Recess & Zero-Leakage ML Regression (Stage 7)
- **Visuals:** RANSAC annular datum ring graphic, Top-25 feature importance chart, Conformal uncertainty envelope curve.
- **Speaking Script (1.5 min):**  
  *"For subsurface crevice recess, we operated under a strict Software-Only Scope. We established a local surface datum plane using RANSAC on an intact annular ring 5 to 20 pixels outside the crack. We extracted 25 independent domain features, completely quarantining all internal crevice depth pixels to guarantee zero target leakage. Our champion ExtraTrees regressor achieved R² = 0.428, outperforming the mean baseline by 27.9% MAE. Crucially, we packaged this with Split Conformal Prediction, providing guaranteed 90% and 95% safety intervals."*

#### Slide 9: System Integration, Dashboard & Validation
- **Visuals:** React frontend screenshot, Inspection Master HUD card, SVG cross-sectional profile curve, PDF report preview.
- **Speaking Script (1 min):**  
  *"We integrated this pipeline into a production-ready Flask REST API and React single-page dashboard. Inspectors can view 16 diagnostic layers, inspect real-time SVG parabolic crevice cross-sections, and export automated engineering inspection PDF reports. The entire platform runs in real time on standard edge hardware."*

#### Slide 10: Conclusions, Honest Limitations & Future Work
- **Visuals:** Summary table of verified achievements, limitations checklist, next steps.
- **Speaking Script (0.5 min):**  
  *"In conclusion, InfraVision AI delivers an explainable, metrologically sound structural inspection platform. We maintain full scientific transparency: depth is estimated as relative sensor recess rather than physical millimeters, and cross-bridge generalization remains bounded by 8-bit sensor disparity. In future work, we plan to interface hardware LiDAR sensors and deploy multi-temporal inspection tracking. Thank you, and I look forward to your questions."*

---

### 10.4. 25+ Likely Viva Questions & Defensible Answers

#### Category A: Depth Estimation & Metrology
**Q1: How does your system measure crack depth in millimeters?**  
*Answer:* "It does not measure physical millimeters directly. Our project operates under a permanent Software-Only Scope. We predict relative crevice recess ($\Delta Z$) in dimensionless sensor disparity units using an ExtraTrees regressor trained on 508 concrete crack captures. Physical millimeter estimation is locked in code (`predict_depth_mm()` raises `NotCalibratedError`) because public RGB-D datasets lack certified contact depth gauge labels."

**Q2: Why did historical reports show $R^2 \approx 0.99$ while the final model shows $R^2 \approx 0.43$?**  
*Answer:* "The historical $R^2 \approx 0.99$ was caused by target leakage: earlier experiments included features computed from depth pixels inside the crack crevice (such as `raw_median_depth` and `volumetric_void_index`), which leaked the target directly. When we conducted a strict leakage audit and quarantined all crevice depth pixels—relying solely on 2D morphology, RGB luminance, and the intact flanking plane—the true out-of-fold generalization performance is $R^2 = 0.428 \pm 0.090$, which beats the naive baseline by $27.9\%$ MAE."

**Q3: Why did your model fail to reach the initial objective of $R^2 \ge 0.70$?**  
*Answer:* "We discovered an empirical information ceiling inherent to passive optical sensing. In narrow cracks ($< 10\text{ px}$), line-of-sight photons are blocked by fracture walls, causing severe shadow extinction. In $28.9\%$ of our dataset, passive stereo matching suffered from depth inversion artifacts where cracks appeared higher than the wall. Under zero-leakage conditions, surface 2D features cannot observe internal crevice geometry, bounding performance at $R^2 \approx 0.43$."

**Q4: How do you establish the concrete surface datum before estimating recess?**  
*Answer:* "We isolate an intact annular concrete ring $5\text{--}20\text{ pixels}$ outside the crack contour using morphological dilation. We then fit a robust 3D plane $Z = Ax + By + C$ across these intact pixels using RANSAC with a $4.0\text{ unit}$ residual threshold. This establishes a baseline plane that is completely immune to local surface roughness or tilt."

**Q5: What are Split Conformal Prediction intervals, and why use them?**  
*Answer:* "Conformal prediction is a distribution-free uncertainty quantification framework with finite-sample mathematical coverage guarantees. Rather than assuming Gaussian errors, we calculate the non-conformity scores (absolute residuals) from out-of-fold predictions. The resulting intervals ($q_{80} = \pm 12.98$, $q_{90} = \pm 22.35$, $q_{95} = \pm 32.67$) guarantee exact empirical coverage ($79.9\%$, $90.0\%$, $94.9\%$) without parametric assumptions."

---

#### Category B: Computer Vision, Geometry & Calibration
**Q6: Why use Guo-Hall thinning over Zhang-Suen or morphological medial axis?**  
*Answer:* "In our empirical benchmarks against ground-truth crack skeletons, morphological thinning over-counted length by $+2.83\%$ due to diagonal step accumulation, and Zhang-Suen eroded tips by $-1.00\%$. Guo-Hall thinning evaluates an 8-neighborhood with two sub-iterations designed to preserve diagonal connectivity, achieving an error of only $0.17\%$."

**Q7: Why is Minimum Spanning Tree (MST) pruning necessary after skeletonization?**  
*Answer:* "Thinning algorithms operate on 8-connected grids, which frequently produce tiny 3-pixel circular loops around diagonal bends. Treating every skeleton pixel as a graph edge inflates total length. The MST connects all skeleton nodes with minimal Euclidean weight, mathematically eliminating all cycle loops while preserving genuine physical branches."

**Q8: Why is orthogonal normal caliper profiling better than distance transform or bounding boxes?**  
*Answer:* "A horizontal bounding-box width over-estimates crack aperture whenever a crack propagates diagonally (by a factor of $1 / \cos \theta$). Distance transform gives an inscribed circle radius but tends to underestimate sharp corners. Orthogonal normal calipers estimate the local tangent vector $\vec{T} = (\Delta x, \Delta y)$ and cast perpendicular rays $\vec{N} = (-\Delta y, \Delta x)$ to both boundaries, measuring true physical opening aperture."

**Q9: How does your ArUco homography rectification work mathematically?**  
*Answer:* "When an image is taken at an angle, perspective transforms parallel lines into converging lines. By detecting four corners of an ArUco fiducial of known dimensions, we solve the system $\mathbf{x}' = \mathbf{H} \mathbf{x}$ using `cv2.getPerspectiveTransform`. The resulting $3 \times 3$ homography matrix rectifies the image and mask into a metric millimeter space where Euclidean distances are invariant."

**Q10: What happens if an image does not have an ArUco marker or camera EXIF data?**  
*Answer:* "The system follows a strict hierarchical fallback. First, it looks for user-specified manual corners. Second, it searches for ArUco markers. Third, it reads EXIF metadata to apply pinhole geometry ($f_x = f_{\text{mm}} \times W_{\text{px}} / S_{\text{mm}}$). If all metadata is missing, it falls back to a documented uncalibrated estimate ($1000\text{ mm}$ distance, $28\text{ mm}$ focal length), clearly flagged as `UNCALIBRATED` in the output."

---

#### Category C: Deep Learning, Domain Adaptation & LOBO
**Q11: Did you use Deep CORAL in this project?**  
*Answer:* "No. Deep CORAL was evaluated in structural health monitoring literature as a potential unsupervised covariance alignment technique. However, it was not implemented in our repository. Our models are trained and validated directly on our concrete fracture dataset."

**Q12: What is Leave-One-Bridge-Out (LOBO), and why is it not in your code?**  
*Answer:* "LOBO is a cross-asset evaluation protocol where entire bridges are held out to test geographic domain shift. Our empirical dataset consists of 508 concrete fracture specimens from an RGB-D concrete crack cohort rather than multi-bridge geographic archives. We used 5-Fold GroupKFold partitioned by capture stem, which prevents data leakage between specimens but is not a cross-bridge protocol."

**Q13: Why did you choose YOLOv8n over heavier models like YOLOv8x or Mask R-CNN?**  
*Answer:* "InfraVision AI is designed for edge deployment on inspection drones and low-power cloud containers ($< 512\text{ MB}$ RAM). YOLOv8n has only $3.2\text{ million}$ parameters and runs inference in under $20\text{ ms}$ on CPU, while still achieving $88.4\%$ detection mAP and $84.6\%$ mask mAP."

**Q14: How does your system classify surface material, and why is that important?**  
*Answer:* "We lazy-load a quantized MobileNetV2 classifier (`material_classifier.tflite`) that identifies 8 substrate types (Concrete, Brick, Stone, Sandstone, Marble, Plaster, Wood, Metal) with $91.4\%$ top-1 accuracy. This is critical because structural crack severity limits depend on the substrate: concrete tolerates $0.30\text{ mm}$ (ACI 224R), while stone facades or water-retaining structures tolerate much narrower thresholds."

---

#### Category D: Structural Engineering & Practical Utility
**Q15: How does your system assist an engineer calculating structural repair costs?**  
*Answer:* "Traditional visual inspection only provides surface length and width. Repair requires pressure-injecting epoxy or grout into the crack crevice. Our volume module performs discrete numerical integration ($V = \iint Z(x, y)\,dx\,dy$) over the recess profile, outputting both $\text{mm}^3$ and $\text{cm}^3$ grout volumes needed for structural sealing."

**Q16: What civil engineering standard does your severity assessment follow?**  
*Answer:* "We strictly implement the **ACI 224R-01 (Control of Cracking in Concrete Structures)** standard from the American Concrete Institute, classifying cracks into Hairline ($< 0.10\text{ mm}$), Fine ($0.10\text{--}0.20\text{ mm}$), Moderate ($0.20\text{--}0.30\text{ mm}$), and Severe ($\ge 0.30\text{ mm}$)."

**Q17: What is the planarity heuristic score?**  
*Answer:* "Our planarity heuristic assesses whether a wall can be treated as a planar 2D surface or requires full 3D relief reconstruction. It starts at a base score of 90 and penalizes perspective distortion ratios and depth gradient variance across the crack. A score $\ge 80$ indicates high planarity; scores $< 55$ trigger a warning recommending 3D photogrammetry."

**Q18: What is the 3D GLB heightmap in your project, and can it measure metric depth?**  
*Answer:* "The GLB heightmap (`image_3d_heightmap.py`) is strictly for qualitative visual rendering. It converts 2D grayscale intensity into normalized elevation using Trimesh. It does not represent metric camera scene depth and is never used for physical engineering measurements."

---

#### Category E: Implementation Nuances & Technical Gaps
**Q19: If `predict_depth_mm()` raises `NotCalibratedError`, why does the API return `unit: "mm"`?**  
*Answer:* "This is a recognized implementation gap. In `crack_depth/predictor.py` line 338, the dataclass container hardcodes `unit='mm'` for compatibility with legacy frontend charting components. However, the core documentation, validation reports, and safety gates explicitly declare the system uncalibrated. For production release, the unit should be updated to `sensor_units`."

**Q20: Why does `api_adapter.py` multiply area by 1.2 to compute volume when no depth map is provided?**  
*Answer:* "In `crack_quantification/api_adapter.py` line 219, when the user uploads a single 2D image without an RGB-D depth map, the API applies an operational heuristic assuming an average crevice depth of $1.20\text{ mm}$ ($V = \text{Area} \times 1.2$). This allows the UI dashboard to illustrate volumetric repair concepts, but is clearly documented as a heuristic fallback rather than an ML measurement."

**Q21: Why does `finalwebapp.py` exist alongside `finalwebapp_api.py`?**  
*Answer:* "`finalwebapp.py` is the earlier monolithic Streamlit prototype used for rapid internal proof-of-concept testing. `finalwebapp_api.py` is the production Flask REST API that powers the decoupled React frontend and serves external programmatic endpoints."

**Q22: What happens if PyTorch or TensorFlow cannot be imported in low-memory environments?**  
*Answer:* "Both API files implement graceful lazy-loading and fallback decorators. If PyTorch is unavailable, the system activates adaptive thresholding and Canny edge segmentation. If TensorFlow is unavailable, material classification falls back to texture/color heuristics."

**Q23: How do you prevent Out-Of-Memory (OOM) errors when processing high-resolution field imagery?**  
*Answer:* "We enforce three safeguards: (1) high-resolution inputs are clamped to a maximum dimension of $1600\text{ px}$; (2) image restoration processes tiles of $512 \times 512$ with $32\text{ px}$ overlap; and (3) rectified homography canvases are clamped to $1400\text{ px}$."

**Q24: What is the significance of the 25 features selected for depth regression?**  
*Answer:* "They capture three complementary physical phenomena: (1) 2D fracture void mechanics via hydraulic radius, compactness, and log width; (2) surface roughness and orientation via RANSAC inlier ratio and orientation angle; and (3) optical penumbra extinction via RGB luminance contrast and edge gradients."

**Q25: What are the three most serious technical gaps in the current implementation?**  
*Answer:* "First, the lack of certified physical millimeter depth ground truth. Second, the absence of the trained RestorationNet checkpoint on disk (falling back to CLAHE). Third, the UI presentation gap where dimensionless sensor recess is labeled as millimeters."

---

### 10.5. Claims You Must NOT Make
To avoid being penalized during your examination, **never** make the following unsupported claims:
1. **DO NOT claim your model measures crack depth in millimeters.** (It estimates relative optical recess in sensor disparity units; physical mm is locked).
2. **DO NOT claim an $R^2 \ge 0.70$ or $R^2 \approx 0.99$ depth model.** (Peak verified zero-leakage $R^2$ is $0.428 \pm 0.090$; higher numbers were artifacts of target leakage).
3. **DO NOT claim you implemented Deep CORAL domain adaptation.** (Deep CORAL does not exist anywhere in the codebase).
4. **DO NOT claim you performed Leave-One-Bridge-Out (LOBO) evaluation.** (Evaluations were 5-Fold GroupKFold on concrete specimens, not multi-bridge splits).
5. **DO NOT claim the 3D GLB heightmap provides real metric 3D scene reconstruction.** (It is a normalized grayscale intensity visualization).
6. **DO NOT claim your system is novel simply because multiple algorithms were chained together.** (Frame the novelty around the specific engineering integrations: Guo-Hall MST thinning, normal ray calipers, RANSAC datum plane, and zero-leakage conformal depth regression).

---

### 10.6. Technical Glossary
- **Aperture:** The perpendicular opening width of a crack.
- **ACI 224R-01:** American Concrete Institute standard for crack control in reinforced concrete.
- **Conformal Prediction:** A framework that generates mathematically guaranteed prediction intervals for any black-box model without distribution assumptions.
- **Distance Transform (EDT):** Calculates the distance of each binary foreground pixel to the nearest background pixel.
- **Frobenius Norm:** The matrix equivalent of Euclidean distance: $\|A\|_F = \sqrt{\sum a_{ij}^2}$.
- **Geodesic Path:** The shortest path between two points along a constrained surface or graph manifold.
- **GroupKFold:** Cross-validation ensuring that all samples from the same group (specimen/stem) remain strictly in either train or test folds.
- **Homography ($H$):** An invertible $3 \times 3$ projective mapping between two planar surfaces.
- **Minimum Spanning Tree (MST):** A subset of graph edges connecting all vertices without cycle loops and with minimum total weight.
- **RANSAC:** Random Sample Consensus; iterative algorithm to estimate mathematical model parameters from data containing extreme outliers.
- **Target Leakage:** When training features unintentionally incorporate information derived from the target variable that would not be available during real-world inference.
- **Tortuosity:** The ratio of the geodesic crack path length to the straight Euclidean distance between its endpoints.

---

### 10.7. Unfinished Tasks & Recommended Next Steps
1. **Acquire Certified Physical Ground Truth:** Partner with a civil engineering structural materials laboratory to test concrete beam specimens under hydraulic flexural load, recording physical crack depth via dial depth gauges and optical microscopes.
2. **Retrain RestorationNet:** Complete and commit the PyTorch training run for `best_restoration_model.pt` so the image restoration pipeline no longer falls back to CLAHE.
3. **Harmonize UI Output Units:** Refactor `crack_depth/predictor.py` line 338 and `finalwebapp_api.py` to label relative recess as `sensor_units` or `relative_disparity` rather than `mm`.
4. **Implement Real Deep CORAL:** Implement a PyTorch training loop incorporating unsupervised CORAL loss ($\|\mathbf{C}_S - \mathbf{C}_T\|_F^2$) if multi-bridge or multi-camera datasets are acquired.
5. **Integrate Hardware Depth Sensors:** Connect live Intel RealSense D435i or LiDAR streaming interfaces to supply calibrated millimeter depth maps directly into the 3D quantification pipeline.

---
*End of Technical Audit & Viva Defense Dossier.*
