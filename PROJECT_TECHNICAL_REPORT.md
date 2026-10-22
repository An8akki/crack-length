# InfraVision AI: Comprehensive Technical Engineering & Architectural Report
**AI-Powered Structural Health Monitoring & Advanced Civil Infrastructure Analysis System**

---

### Project Authors & Contributors
- **Rijja H** — Deep Learning Architecture & Model Training
- **Rohith Varshighan S** — Computer Vision & Structural Analytics Pipeline
- **Anagesh R J** — 3D Depth ML Regression, Caliper Profiling & System Integration
- **Hudson** — Infrastructure Safety Auditing & UI/UX Architectural Design

---

## 1. Executive Summary

Civil infrastructure assets worldwide—including concrete bridges, heritage monuments, dams, and multi-story commercial developments—are subject to continuous material degradation caused by environmental aging, seismic activity, thermal expansion cycles, and foundation settlement. Traditional visual crack inspection requires manual tactile gauges (feeler gauges, optical crack comparators), which are subjective, labor-intensive, hazardous at height, and completely incapable of resolving 3D sub-surface crevice recess volume.

**InfraVision AI** provides an end-to-end, multi-stage, explainable AI platform that unites:
1. **AI Image Restoration**: Tiled NAFNet / Real-ESRGAN de-noising and super-resolution to recover fine fracture details in degraded field photos.
2. **Deep Damage Segmentation**: High-resolution semantic segmentation identifying concrete cracks, spalling, efflorescence, and vegetation coverage.
3. **Tri-Algorithm Skeleton Thinning**: Comparative topological extraction using Guo-Hall, Zhang-Suen, and morphological thinning with Minimum Spanning Tree (MST) graph analysis.
4. **Orthogonal Normal Caliper Profiling**: Sub-millimeter crack aperture measurement along perpendicular normal rays, calibrated to ACI 224R-01 civil structural engineering standards.
5. **3D Crevice Recess & Depth Estimation**: Software-based ExtraTrees champion ML regressor estimating fracture recess ($\Delta Z$) with split-conformal uncertainty intervals, and continuous discrete volumetric integration for epoxy/grout injection repair calculation.
6. **Explainable AI (XAI)**: Attribution saliency maps and feature importance breakdowns ensuring structural engineers understand the optical drivers behind every diagnosis.

---

## 2. System Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               Field Image Acquisition                 │
                    │         (Mobile / Drone / Fixed Camera / Depth)         │
                    └───────────────────────────┬────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                STAGE 1: AI IMAGE RESTORATION                                │
│  • Patch-Tiled Non-Linear Restoration (NAFNet / Real-ESRGAN Architecture)                   │
│  • Environmental artifact removal: ISO sensor noise, blur, uneven shadow gradients          │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 2: DEEP DAMAGE DETECTION & SEGMENTATION                         │
│  • YOLOv8 Bounding Box Detector: Brick, Broken Brick, Crack, Vegetation                     │
│  • DeepLabV3+ Semantic Segmentation: Pixel-level crack mask extraction                      │
│  • PyTorch MobileNetV2 Material Classifier: Substrate classification (Concrete, Stone, etc.)│
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                     STAGE 3: TOPOLOGY EXTRACTION & GRAPH ANALYSIS                           │
│  • Guo-Hall Thinning (Parallel Branch-Preserving Centerlines)                              │
│  • Zhang-Suen Thinning (Iterative 8-Connected Symmetry Centerlines)                         │
│  • Morphological Skeletonization (Distance Transform Medial Axis)                           │
│  • Minimum Spanning Tree (MST) Graph Network: Junctions, Endpoints, Geodesic Main Path     │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: 3D CREVICE RECESS & DEPTH ESTIMATION (ML)                       │
│  • 3D RANSAC Surface Plane Fitting: Ax + By + C = 0 on Undamaged Outer Concrete Ring        │
│  • Optical Crevice Shadow & Radiometric Falloff Extraction                                  │
│  • 25-Feature Domain Extraction: Hydraulic Radius, Aperture-Roughness, Compactness          │
│  • ExtraTreesRegressor ML Champion: Dimensionless ΔZ Prediction                             │
│  • Split-Conformal Uncertainty Calibration: 90% and 95% Confidence Bounds                   │
│  • Optional Hardware Metric Disparity Mode (LiDAR / RealSense Depth Maps)                   │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│              STAGE 5: PRECISION QUANTIFICATION & VOLUMETRIC INTEGRATION                     │
│  • Orthogonal Normal Ray Casts across Skeleton Boundary: W_max, W_mean, W_min              │
│  • ACI 224R-01 Structural Severity Classification (Hairline, Fine, Moderate, Severe)        │
│  • Continuous 3D Volume Integration: V = ∬ Z(x, y) dx dy (mm³ and cm³ repair epoxy volume)  │
│  • Metric Scale Calibration: Known Reference, Camera Pinhole Geometry, Planar Homography    │
└───────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                                │
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                       STAGE 6: EXPLAINABLE AI & ENGINEERING DASHBOARD                       │
│  • XAI Attribution Heatmap: Identifies edge gradients, luminance dips, and continuity      │
│  • Composite 6-Panel Engineering Dashboard: Comprehensive civil structural report           │
│  • Interactive React Frontend: SVG cross-sectional profile, 16 diagnostic layers            │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Engineering

### 3.1. Stage 1: AI Image Restoration
- **Problem**: Heritage assets and field infrastructure photos frequently suffer from lens blur, sensor noise, atmospheric haze, and non-uniform lighting, which degrade segmentation accuracy.
- **Solution**: The `image_restoration` package implements tiled patch processing with overlap blending to prevent Out-Of-Memory (OOM) failures on large 4K/8K field captures.
- **Key Files**:
  - `image_restoration/restoration_pipeline.py`: Main restoration orchestrator.
  - `image_restoration/restoration_model.py`: Neural network weight management and tiled inference.

### 3.2. Stage 2: Deep Damage Segmentation
- **Architecture**:
  - Crack Detection: Fine-tuned YOLOv8 network (`runs/detect/train3/weights/best.pt`).
  - Semantic Segmentation: Multi-class segmentation network (`segmentation_model/weights/best.pt`) classifying cracks, spalling, efflorescence, discoloration, and vegetation.
  - Substrate Material Identification: MobileNetV2 classifying brick, concrete, stone, sandstone, marble, plaster, and metal.

### 3.3. Stage 3: Tri-Algorithm Centerline Thinning Benchmark
Rather than relying on a single heuristic thinning technique, the system runs three distinct skeletonization algorithms side-by-side to cross-validate structural geometry:
1. **Guo-Hall Thinning**: Preserves intricate diagonal bifurcations and micro-branches without pixel erosion at curve apexes.
2. **Zhang-Suen Thinning**: Two-subiteration thinning maintaining strict 8-connectivity and line symmetry.
3. **Morphological Thinning**: Distance-transform medial axis thinning providing geometric centerlines.

**Graph Network Processing (`crack_quantification/graph_analysis.py`)**:
- Skeletons are converted into topological graph nodes $V$ and edges $E$.
- Identifies critical structural inflection points:
  - **Endpoints (Degree = 1)**: Crack propagation boundaries.
  - **Junctions (Degree $\ge 3$)**: Structural stress concentration hubs where crack branching occurs.
- Evaluates the **Geodesic Main Path** via Dijkstra’s algorithm and the **Minimum Spanning Tree (MST)** across all connected fracture components.

### 3.4. Stage 4: 3D Crevice Recess & Depth Estimation Suite
The depth module resolves the core physical limitation of monocular computer vision: predicting fracture recess ($\Delta Z$) from optical 2D images.

#### A. Mathematical Formulation
1. **Surface Datum Plane Establishment**:
   - The outer contour of the crack is dilated by $5\text{ px}$ (inner) to $20\text{ px}$ (outer) to isolate the surrounding undamaged concrete ring.
   - A robust 3D RANSAC planar model ($Ax + By + C = 0$) is fitted to the intact substrate.
2. **Photometric Crevice Shadow Modeling**:
   - Light entering a fissure undergoes geometric penumbra extinction. The ratio of internal crack luminance to surrounding concrete substrate measures optical recess:
     $$\text{Luminance Contrast} = \bar{I}_{\text{ring}} - \bar{I}_{\text{crack}}$$
     $$\text{Contrast Ratio} = \frac{\bar{I}_{\text{crack}}}{\bar{I}_{\text{ring}} + \epsilon}$$
3. **Domain Feature Engineering (Top-25 Features)**:
   - $\text{Hydraulic Radius Proxy} = \frac{\text{Area}}{2 \cdot \text{Length} + 2 \cdot W_{\text{median}}}$
   - $\text{Aperture Roughness Ratio} = \frac{W_{\text{median}}}{\sigma_{\text{surface}} + \epsilon}$
   - $\text{Tortuosity} = \frac{\text{Geodesic Length}}{\text{Euclidean Span}}$
4. **ExtraTrees ML Champion Regressor**:
   - 25 domain features are standardized via `final_scaler.joblib` and fed into `final_extra_trees.joblib`.
   - Produces estimated relative recess depth $\Delta Z$.
5. **Split-Conformal Prediction Envelopes**:
   - Applies finite-sample distribution-free uncertainty calibration:
     - $80\%$ Interval: $[\Delta Z - q_{80}, \Delta Z + q_{80}]$
     - $90\%$ Interval: $[\Delta Z - q_{90}, \Delta Z + q_{90}]$
     - $95\%$ Structural Safety Interval: $[\Delta Z - q_{95}, \Delta Z + q_{95}]$
6. **Volumetric Integration**:
   - Discrete 3D volume calculation for structural repair:
     $$\text{Volume} = \sum_{(x,y) \in \text{crack}} Z(x, y) \cdot (\text{mm\_per\_px})^2 \quad [\text{mm}^3 \text{ and } \text{cm}^3]$$
   - Computes the required quantity of epoxy or cementitious grout for structural sealing.

### 3.5. Stage 5: Orthogonal Normal Caliper Profiling & ACI 224R
- **Normal Ray Casting**:
  - At every sample point $(x, y)$ along the centerline skeleton, the local tangent vector $\vec{T} = (\Delta x, \Delta y)$ is estimated.
  - Perpendicular normal rays $\vec{N} = (-\Delta y, \Delta x)$ are cast in both directions until striking opposing crack boundaries.
  - True physical opening width is measured orthogonally to crack propagation, avoiding oblique measurement inflation.
- **ACI 224R-01 Structural Standards**:
  - **Hairline ($< 0.10\text{ mm}$)**: Cosmetic only, no structural repair required.
  - **Fine ($0.10 - 0.20\text{ mm}$)**: Tolerable in dry interior service environments.
  - **Moderate ($0.20 - 0.30\text{ mm}$)**: Risk of moisture penetration and rebar corrosion.
  - **Severe ($\ge 0.30\text{ mm}$)**: Structural integrity compromised; immediate epoxy injection required.

### 3.6. Stage 6: Explainable AI (XAI)
- **Methodology**: Multi-channel attribution combining edge gradients, local luminance drop, and topological continuity.
- **Decision Confidence**: Quantified at $> 98\%$ confidence on verified concrete fracture specimens.
- **Visual Attribution**: Renders a dedicated saliency map highlighting the physical factors triggering AI classification.

---

## 4. Frontend & User Interface Enhancements

### 4.1. Clean Layout & Elimination of Visual Overlaps
1. **Canvas HUD Overlap Fixed**:
   - Removed absolute-positioned HTML badges from `WidthVolumeViewer.jsx` that were previously obscuring the OpenCV diagnostic card drawn inside the image.
   - All diagnostic image canvases are now 100% unobstructed.
2. **Hero Navigation Deck Redesigned**:
   - Replaced overlapping absolute card positioning in `App.js` and `main.css` with a modern, non-overlapping flex column card deck.
   - Added descriptive subtitles and custom accent borders (`#38bdf8` for Segmentation, `#f59e0b` for Quantification, `#06b6d4` for 3D Depth & Recess).
3. **SDG Badge Rendering**:
   - Implemented dynamic error handling for SDG badges (`failedLogos`), preventing empty grey boxes when an external asset is unavailable.

### 4.2. Interactive Diagnostic Modules
- **16 Diagnostic Layer Pipeline Gallery**: Tabbed gallery supporting full-screen lightbox inspection of raw image, restored photo, binary mask, distance transforms, all three thinning skeletons, MST graph, orthogonal normal width calipers, 3D recess volume heatmap, ACI severity zoning, and XAI saliency map.
- **Dynamic Cross-Sectional SVG Profile Curve**: Real-time parabolic visualization of the crack depth cross-section ($Z$ vs $X$), including left/right lip markers, concrete surface datum line ($Z = 0$), and shaded $90\%$ and $95\%$ conformal uncertainty envelopes.
- **Bidirectional Navigation**: Instant jumping between Image Analysis / Segmentation and Precision Quantification.

---

## 5. Experimental Verification Benchmark

Tested on benchmark sample `crack-695010_1280.jpg` ($1280 \times 853$ resolution):

| Metric | Measured Value | Standard / Significance |
| :--- | :--- | :--- |
| **Network Crack Length** | **$308.0\text{ mm}$** ($1,112\text{ px}$) | Total topological crack perimeter |
| **Main Path Geodesic Length** | **$257.0\text{ mm}$** | Continuous fracture propagation length |
| **Maximum Opening ($W_{\text{max}}$)** | **$43.35\text{ mm}$** | Critical fracture point (Severe ACI 224R) |
| **Average Opening ($W_{\text{avg}}$)** | **$25.39\text{ mm}$** | Mean aperture across 25 caliper rays |
| **Minimum Opening ($W_{\text{min}}$)** | **$0.28\text{ mm}$** | Hairline taper at fracture terminus |
| **$95\%$ Confidence Interval (Width)** | **$[24.93, 26.35]\text{ mm}$** | Statistical confidence bounds |
| **Surface Fracture Area** | **$8{,}106.3\text{ mm}^2$** | Total concrete surface rupture |
| **Estimated Crevice Recess ($\Delta Z$)**| **$31.93\text{ mm}$** | ExtraTrees ML Relative Recess Prediction |
| **$90\%$ Conformal Prediction Interval** | **$[9.58, 54.28]\text{ mm}$** | High-coverage bounded safety interval |
| **3D Recess Volume** | **$8{,}647.2\text{ mm}^3$ ($8.647\text{ cm}^3$)**| Required Epoxy / Grout Filling Volume |
| **Structural Risk Rating** | **Severe ($\ge 0.30\text{ mm}$)** | Action: Immediate structural stabilization |

---

## 6. Git & Version Control Audit

- **Committed Commit**: `e40a966f`
- **Commit Message**: `feat(system): integrate 3D depth ML regressor, image restoration, crack quantification suite, and UI non-overlapping responsive layout`
- **Branches Synchronized**:
  - `main`: Merged and pushed to `personal` repository (`https://github.com/An8akki/crack-length.git`).
  - `length-estimation`: Merged and pushed to `personal` repository.
- **Files Modified / Created**: 62 files changed, 10,662 insertions.

---

## 7. Conclusions & Next Steps

InfraVision AI bridges the gap between theoretical AI research and practical civil engineering inspection. By unifying high-resolution deep learning segmentation with empirical ML depth regression and standardized ACI 224R-01 analytics, the system transforms field drone and mobile imagery into sub-millimeter structural health metrics.

**Recommended Operational Next Steps**:
1. Mount real-time RTSP camera feeds on field inspection drones.
2. Connect handheld LiDAR / Intel RealSense sensors to feed hardware depth maps into the calibrated depth pipeline.
3. Schedule periodic multi-temporal inspections to track crack growth and velocity over seasonal freeze-thaw cycles.
