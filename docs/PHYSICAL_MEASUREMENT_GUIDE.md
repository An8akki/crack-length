# Physical Crack Measurement & Calibration Subsystem

## Comprehensive Engineering Guide & Scientific Documentation

The **Physical Crack Measurement & Calibration Subsystem** in **InfraVision AI** provides metrologically rigorous, physical quantification of structural cracks on concrete and civil infrastructure surfaces. It bridges the gap between raw 2D pixel detections and millimeter-accurate engineering measurements.

---

## 🏗️ 1. Tri-Method Architecture Overview

Rather than relying on a single heuristic or uncalibrated pixel-to-millimeter ratio, the system implements a **tri-pathway comparative benchmark**:

```
                         RAW INPUT PHOTOGRAPH
                                  │
                       YOLOv8 Detection
                                  │
                      YOLOv8 Segmentation
                                  │
                    Crack Damage Binary Mask
                                  │
                3 Centerline Skeletonization Engines
             (Guo-Hall / Zhang-Suen / Morphological)
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
  2D METHOD             PLANAR HOMOGRAPHY              3D METHOD
(Global Scale)          (Metric Plane Quad)       (Surface Arc-Length)
         │                        │                        │
Pinhole camera model     Perspective unwarping     3D back-projection
Pixel scale factor       Closed-form H matrix      Depth relief integration
Network / MST length     Frontal metric length     3D geodesic arc-length
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
                Empirical Reliability & Decision Engine
                     (Predicted Error ± mm, 95% CI)
                                  │
               Final Calibrated Crack Assessment HUD
```

| Pathway | Mathematical Principle | Primary Strength | Limitation |
| :--- | :--- | :--- | :--- |
| **2D Method** | Orthogonal pinhole projection: $L_{\text{mm}} = L_{\text{px}} \cdot s$ | Extremely fast, zero setup | Distorts under oblique camera angles ($> 10^\circ$) |
| **Planar Homography** | Projective transformation: $\mathbf{x}' \sim H \mathbf{x}$ | Rectifies perspective skew and foreshortening | Requires 4-point reference frame or marker |
| **3D Arc-Length** | Back-projection & geodesic sum: $\sum \|\mathbf{P}_{i+1} - \mathbf{P}_i\|_2$ | Captures non-planar surface relief & curvature | Requires working distance and camera focal length |

---

## 🦴 2. Skeleton Thinning Engines & Topological Graph Analysis

Extracting the true physical centerline from a segmented crack mask requires topology-preserving skeletonization. The system computes and compares three algorithms:

### Algorithms Compared

1. **Guo-Hall (Recommended Baseline ⭐)**
   - **Formulation**: Iterative parallel thinning with directional connectivity weighting ($C(P_1) = 1$ and $2 \le \min(n_1, n_2) \le 3$).
   - **Performance**: Produces a strictly 1-pixel-wide, 8-connected medial axis. Completely eliminates $2 \times 2$ diagonal staircase artifacts and false branch spurs.
   - **MST Consistency**: The Minimum Spanning Tree (MST) length matches the Network length within $< 1\%$, confirming virtually zero artificial noise loops.

2. **Zhang-Suen (Legacy Parallel)**
   - **Formulation**: Classical two-pass deletion based on 8-neighborhood transitions $A(P_1) = 1$ and $2 \le B(P_1) \le 6$.
   - **Defect**: Cannot thin diagonal segments without violating its naive connectivity rules, leaving alternating **$2 \times 2$ pixel clusters**.
   - **Artifact**: In topological graph construction, every $2 \times 2$ cluster generates artificial micro-cycles and multiple branch nodes. This causes **spurious junction explosion (e.g., 400+ false junctions)** and inflates raw network length by **+45% to +90%**.

3. **Morphological Thinning (Medial Axis / Hit-or-Miss)**
   - **Formulation**: Iterative erosion using directional Golay structuring elements.
   - **Defect**: Concrete cracks feature micro-roughness and sharp aggregate notches. The structuring elements get caught in these concavities, generating spurious side dendrites (**+10% to +15% length inflation**).

### Graph Representation
The thinned skeleton is transformed into an undirected weighted graph $G = (V, E)$:
- **Endpoints ($V_{\text{end}}$)**: Pixels with exactly 1 neighbor ($\text{degree} = 1$).
- **Junctions ($V_{\text{junc}}$)**: Bifurcation pixels with $\ge 3$ neighbors.
- **Edges ($E$)**: Continuous 8-connected pixel chains weighted by Euclidean pixel distance ($1.0$ for orthogonal, $\sqrt{2} \approx 1.414$ for diagonal).
- **Dominant Path**: Diameter / longest geodesic path between any pair of endpoints in the crack network.
- **MST Length**: Kruskal's / Prim's Minimum Spanning Tree resolving cyclic loops.

---

## 📐 3. Planar Homography Calibration

When an inspector photographs a crack obliquely, perspective foreshortening shrinks distant pixels. Planar homography rectifies this distortion onto a calibrated frontal metric plane:

### Mathematical Model
A point $\mathbf{x} = [u, v, 1]^T$ in the image plane maps to a metric point $\mathbf{X} = [X, Y, 1]^T$ on the physical surface via the $3 \times 3$ homography matrix $H$:
$$\begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} \sim H \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

1. **Manual 4-Corner Mode**: The user selects four non-collinear points representing a rectangular reference frame (e.g., $500 \times 500\text{ mm}$ grid). The system computes $H$ via Singular Value Decomposition (SVD) of the Direct Linear Transformation (DLT) matrix.
2. **ArUco Marker Mode**: Automatically detects standard ArUco markers (e.g., 100mm size) to compute metric homography without human intervention.
3. **Metric Rectification**: The crack mask is unwarped into a front-parallel metric coordinate system where scale is constant throughout the frame.

---

## 🌐 4. 3D Surface Arc-Length Estimation

For non-planar, curved, or spalled surfaces, the 2D plane assumption breaks down. The 3D pathway computes arc-length along the physical concrete manifold:

1. **Camera Intrinsics**:
   $$f_x = \frac{f_{\text{mm}}}{\text{sensor\_width}_{\text{mm}}} \cdot W_{\text{px}}, \quad c_x = \frac{W}{2}, \quad c_y = \frac{H}{2}$$
2. **Surface Relief Reconstruction**:
   A relative depth elevation map $\Delta Z(u, v)$ is generated from concrete texture gradients and combined with the working distance $Z_0$:
   $$Z(u, v) = Z_0 + \Delta Z(u, v)$$
3. **Back-Projection**:
   Each skeleton pixel $(u_i, v_i)$ is back-projected into 3D world coordinates:
   $$X_i = \frac{(u_i - c_x) Z_i}{f_x}, \quad Y_i = \frac{(v_i - c_y) Z_i}{f_y}, \quad Z_i = Z(u_i, v_i)$$
4. **3D Arc-Length Integration**:
   Along each skeleton branch, 3D arc-length is integrated as:
   $$L_{3\text{D}} = \sum_{i=1}^{N-1} \sqrt{(X_{i+1} - X_i)^2 + (Y_{i+1} - Y_i)^2 + (Z_{i+1} - Z_i)^2}$$

---

## 🛡️ 5. Empirical Reliability & Adaptive Decision Engine

The system features an automated **decision and confidence scoring engine** (`validation_platform/`):
- **Feature Vector**: Extracted from camera viewpoint angle, working distance, reprojection error, surface planarity heuristic, and crack complexity.
- **Predicted Error Bound**: Computes expected physical error $\pm \epsilon\text{ mm}$ based on empirical regression calibrated across synthetic validation trials.
- **95% Confidence Interval**:
  $$\text{CI}_{95\%} = \left[ L_{\text{meas}} - 1.96 \cdot \sigma_e, \;\; L_{\text{meas}} + 1.96 \cdot \sigma_e \right]$$
- **Adaptive Method Recommendation**: Recommends `PLANAR` for flat surfaces under angle, `3D` for curved surfaces/columns, and flags measurements as `ACCEPTABLE` or `UNACCEPTABLE`.

---

## 🔬 6. Synthetic Ground-Truth Research Studio

To scientifically validate measurement accuracy without manual ground-truth error, the platform includes a **Synthetic Validation Studio**:
- **Parametric Crack Generators**:
  - Straight diagonal cracks
  - Sinusoidal undulating cracks
  - Arc-segment cracks
  - Multi-branch fracture trees
  - Irregular spalled fractures
- **Camera Simulation**:
  - Realistic pinhole projection with variable camera yaw ($0^\circ - 60^\circ$) and pitch ($0^\circ - 30^\circ$).
  - Concrete surface manifolds (Planar wall, inclined plane, cylindrical column, spalled relief).
  - Illumination simulation (Daylight, shadows, specular glare, underexposure, moisture).
- **Factorial Benchmark Suite**:
  - Executes batch trials across variable camera distances and angles.
  - Automatically calculates **Mean Absolute Error (MAE)**, **Root Mean Square Error (RMSE)**, and **Mean Absolute Percentage Error (MAPE)**.

---

## 🚀 7. How to Use

### A. Via Interactive Web Dashboard
1. Start the server:
   ```bash
   python finalwebapp_api.py
   ```
2. Navigate in your browser to:
   ```
   http://localhost:7860/api/crack-length
   ```
3. **Field Photo Mode**:
   - Upload an uncropped infrastructure photo.
   - Adjust the 4 planar calibration corners on the canvas (or click "Auto 4 Corners").
   - Specify reference frame width/height (default 500 mm).
   - Enter camera distance and focal length (optional).
   - Click **"Run Tri-Method Crack Quantification Pipeline"**.
   - Review side-by-side measurements, skeleton algorithm comparison table, and annotated HUD report.
4. **Synthetic Validation Studio Mode**:
   - Switch to the "Synthetic Research & Validation Studio" tab.
   - Choose crack morphology and surface type.
   - Click "Generate Synthetic Specimen" to inspect analytical ground truth against simulated projection.
   - Click "Launch Factorial Batch Experiments" to execute multi-condition benchmarks.

---

### B. Via REST API (Python Example)

```python
import requests

url = "http://localhost:7860/api/crack-length"

# 1. Inspect image with planar homography calibration
with open("crack_photo.jpg", "rb") as f:
    files = {"image": f}
    data = {
        "calibration_mode": "manual_4point",
        "ref_width_mm": "500.0",
        "ref_height_mm": "500.0",
        "corners": "[[120, 80], [740, 95], [780, 520], [105, 510]]",
        "distance_mm": "1000",
        "focal_length_mm": "28"
    }
    response = requests.post(url, files=files, data=data)
    result = response.json()

# Access results
print("Planar Length:", result["measurements"]["planar"]["network_length_mm"], "mm")
print("Guo-Hall Length:", result["per_method"]["guo_hall"]["network_mm"], "mm")
print("Zhang-Suen Junctions:", result["per_method"]["zhang_suen"]["junctions"])
print("Reliability:", result["reliability_prediction"]["measurement_reliability"])
print("Recommended:", result["method_recommendation"]["recommended_method"])
```

### C. Via cURL

```bash
curl -X POST http://localhost:7860/api/crack-length \
  -F "image=@concrete_crack.jpg" \
  -F "calibration_mode=manual_4point" \
  -F "ref_width_mm=500" \
  -F "ref_height_mm=500" \
  -F "corners=[[100,100],[700,100],[700,500],[100,500]]"
```
