---
title: InfraVision AI API
emoji: 🏗️
colorFrom: blue
colorTo: purple
sdk: docker
app_file: finalwebapp_api.py
pinned: false
---

# InfraVision AI 🏗️

**AI-Powered Infrastructure Monitoring & Analysis System**

An intelligent computer vision platform for detecting, analyzing, and monitoring civil infrastructure damage including cracks, material degradation, and environmental impact assessment.

---

## 🎯 Overview

InfraVision AI leverages advanced deep learning models (YOLOv8, MobileNetV2) to automatically:
- **Detect** structural cracks and defects in infrastructure
- **Segment** damaged regions with pixel-level precision
- **Classify** building materials (8 types: Brick, Concrete, Stone, Sandstone, Marble, Plaster, Wood, Metal)
- **Generate** 3D heightmaps from 2D images for visual analysis
- **Monitor** real-time video streams for infrastructure degradation
- **Assess** environmental impact and sustainability metrics
- **Generate** automated PDF reports with analytics

---

## 📊 Key Performance Metrics

| Metric | Performance |
|--------|-------------|
| **Crack Detection mAP50** | 79.14% |
| **Segmentation mAP50** | 78.33% |
| **Detection Precision** | 76.61% |
| **Average Inference Time** | ~80ms (detection), ~100-150ms (segmentation) |
| **Full Analysis Pipeline** | 2-5 seconds per image (CPU) |

---

## ✨ Features

### 🖼️ Image Analysis
- Upload infrastructure images for crack detection and classification
- Real-time severity assessment and confidence scoring
- Pixel-level damage visualization
- Automatic material classification

### 📏 Physical Crack Measurement & Tri-Method Calibration
- **Tri-Method Benchmark**: Side-by-side comparison of 2D Global Scale, Perspective-Rectified Planar Homography, and 3D Surface Arc-Length
- **Centerline Skeletonization**: Benchmark 3 skeleton engines (Guo-Hall vs. Zhang-Suen vs. Morphological) with graph-theoretic metrics (Network, MST, Dominant Path, Bifurcations)
- **Planar Homography Rectification**: Interactive 4-point quad or automatic ArUco marker detection to eliminate perspective distortion
- **Empirical Reliability Engine**: Automated error bound estimation ($\pm \text{mm}$) and 95% Confidence Intervals
- **Synthetic Validation Studio**: Generate parametric crack specimens with exact analytical ground truth curves and run factorial batch benchmarks
- **Interactive UI Dashboard**: Dedicated HUD available at `http://localhost:7860/api/crack-length`
- **Detailed Documentation**: See [docs/PHYSICAL_MEASUREMENT_GUIDE.md](docs/PHYSICAL_MEASUREMENT_GUIDE.md)

### 🎨 3D Heightmap Generation
- Convert 2D damage maps to interactive 3D models
- Export as GLB or STL formats for CAD software
- Customizable colormap visualization (JET colormap with edge enhancement)
- Real-time 3D rendering in browser

### 🌍 Environmental Impact Assessment
- Carbon footprint estimation
- Water consumption analysis
- Energy efficiency scoring
- Sustainability metrics dashboard

### 📹 Real-Time Monitoring
- Live camera feed processing
- Continuous infrastructure monitoring
- Stream-based analysis with configurable confidence thresholds
- Time-series tracking of changes

### 📈 Advanced Analytics
- Health trend analysis with confidence intervals
- Risk matrix visualization
- Statistical forecasting (ANOVA, time-series prediction)
- Exportable dashboard metrics

---

## 🏗️ Architecture

### Backend Stack
- **Framework**: Flask 3.0 with CORS support
- **Detection**: YOLOv8 (trained on infrastructure damage dataset)
- **Segmentation**: YOLOv8-Segmentation with instance masking
- **Classification**: MobileNetV2 (8 material classes)
- **3D Generation**: Trimesh-based heightmap engine
- **Analytics**: NumPy, Pandas, SciPy, Scikit-learn
- **Visualization**: Matplotlib, Seaborn, Plotly
- **Reporting**: ReportLab for PDF generation

### Frontend Stack
- **Framework**: React with Tailwind CSS
- **3D Rendering**: Three.js/Babylon.js for heightmap visualization
- **State Management**: React Context API
- **HTTP Client**: Axios with API configuration
- **Responsive Design**: Mobile-optimized UI

### Model Details
| Model | Framework | Purpose | Input Size | Accuracy |
|-------|-----------|---------|-----------|----------|
| YOLOv8 Detection | PyTorch | Crack detection | 640×640 | 79.14% mAP50 |
| YOLOv8 Segmentation | PyTorch | Damage segmentation | 640×640 | 78.33% mAP50 |
| MobileNetV2 Classifier | TensorFlow/TFLite | Material classification | 224×224 | 8 classes |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Node.js 16+ (for frontend)
- 4GB+ RAM recommended
- GPU support optional (CUDA for faster inference)

### Backend Installation

```bash
# Clone and navigate to project
cd d:\AI-Powered_Civil_Infrastructure

# Install Python dependencies
pip install -r requirements.txt

# Run the Flask API server
python finalwebapp_api.py
```

The API will be available at `http://localhost:7860`

### Frontend Installation

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The frontend will be available at `http://localhost:3000`

---

## 📡 API Endpoints

### Image Analysis
- `POST /api/analyze-image` - Upload and analyze infrastructure image
- `POST /api/detect-cracks` - Advanced crack detection with confidence threshold
- `POST /api/segment-damage` - Instance segmentation of damaged regions
- `POST /api/classify-material` - Material type classification
### Physical Crack Measurement & Calibration Platform
The scientific physical measurement branch of InfraVision AI. Converts 2D pixel detections into metric millimeter measurements via 3 distinct physical pathways:

- `GET /api/crack-length` - Interactive browser dashboard for field photo analysis, planar calibration, skeleton comparison, and synthetic validation studio.
- `POST /api/crack-length` - Full 6-stage physical quantification pipeline:
  - **Stage 1**: YOLOv8 Crack Detection (Bounding boxes)
  - **Stage 2**: YOLOv8 Segmentation (Pixel damage mask)
  - **Stage 3**: 3 Topology Centerline Skeletons (`guo_hall`, `zhang_suen`, `morphological`)
  - **Stage 4**: Planar Rectification (Homography $H$) & 3D Surface Arc-Length
  - **Stage 5**: Minimum Spanning Tree (MST) Graph Network
  - **Stage 6**: Final Annotated Measurement HUD Overlay Image
  - **Outputs**: Side-by-side comparison of 2D, Planar, and 3D measurements, error estimation bound ($\pm \text{mm}$), 95% Confidence Interval, and empirical reliability status.
- `POST /api/research/synthetic-specimen` - Generate synthetic crack specimens with exact analytical ground truths under variable camera angles ($0^\circ - 60^\circ$ yaw) and lighting conditions.
- `POST /api/research/run-batch-trials` - Execute automated factorial benchmark suites across multiple camera geometries (MAE, RMSE, MAPE).

### Video Processing
- `POST /api/process-video` - Analyze video file frame-by-frame
- `POST /api/stream-analysis` - Real-time streaming analysis endpoint

### Analytics
- `GET /api/analytics/health-trend` - Infrastructure health over time
- `GET /api/analytics/risk-matrix` - Risk assessment visualization
- `POST /api/analytics/statistics` - Statistical analysis with confidence intervals
- `GET /api/analytics/forecast` - Time-series forecasting

### Reporting
- `POST /api/generate-report` - Create comprehensive PDF report
- `GET /api/report/<report_id>` - Download generated report

### System
- `GET /api/health` - API health check
- `GET /api/models/status` - Model loading status

---

## 📋 Project Structure

```
AI-Powered_Civil_Infrastructure/
├── crack_quantification/             # Crack length estimation package (3rd branch)
│   ├── length.py                   # Skeleton graph -> crack length (px)
│   ├── skeleton.py                 # Binary mask -> 1-pixel skeleton
│   ├── graph_analysis.py           # Network / main_path / MST measurements
│   ├── quantify.py                 # Full pipeline (length+width+area+severity)
│   ├── api_adapter.py              # Segmentation results -> length report
│   ├── research_experiment.py      # Factorial evaluation runner
│   ├── compare_methods.py          # Method comparison with visual outputs
│   ├── compare_skeletons.py        # Skeletonization algorithm benchmark
│   ├── camera_scale.py             # Camera-scale recovery
│   ├── autocalibrate.py            # Fully automatic EXIF+DB+depth calibration
│   ├── scale_recovery.py           # 4 scale methods + fusion
│   ├── depth_3d.py                 # 3D crack length from depth/mesh/PCD
│   ├── reference_scale.py          # Homography + reference segment scale
│   ├── calibration.py              # px <-> mm conversions (3 methods)
│   ├── integrate_pipeline.py       # End-to-end: image -> mask -> report
│   ├── run_quantification.py       # CLI: mask -> report
│   ├── width.py                    # Per-point width from distance transform
│   ├── area.py                     # Pixel count area
│   ├── severity.py                 # Rule-based severity + orientation
│   ├── crack_3d.py                 # 3D crack length from reconstruction
│   ├── test_quantification.py      # Self-test with synthetic mask
│   ├── test_compare_methods.py     # MST/compare artifact tests
│   └── test_research_pipeline.py   # Research primitive tests
│
├── finalwebapp_api.py              # Flask API with 21+ endpoints
├── finalwebapp.py                  # Core analysis functions
├── image_3d_heightmap.py           # 3D model generation (GLB)
├── segmentation_with_localisation.py # Advanced segmentation
├── pdf_report.py                   # PDF report generation
├── hf_model_loader.py              # Hugging Face model auto-download
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker configuration
│
├── runs/                           # YOLOv8 training outputs
│   └── detect/train3/weights/      # Detection model weights (best.pt)
│
├── segmentation_model/
│   └── weights/                    # Segmentation model weights (best.pt)
│
├── materialclassification_model/
│   ├── material_classifier.h5      # TensorFlow model (21.7MB)
│   └── material_classifier.tflite  # TFLite optimized model (8.9MB)
│
└── frontend/                       # React application
    ├── public/
    │   ├── index.html
    │   └── dataset_analytics.json
    └── src/
        ├── pages/
        │   ├── ImageAnalysis.jsx
        │   ├── Heightmap3D.jsx
        │   ├── Environmental.jsx
        │   ├── RealTimeMonitoring.jsx
        │   └── About.jsx
        ├── components/
        │   ├── MainDashboard.jsx
        │   └── Navbar.jsx
        ├── contexts/
        │   └── AnalysisContext.js
        └── styles/
            ├── main.css
            ├── heightmap3d.css
            ├── mobile-responsive.css
            └── ...
```

---

## 🔧 Configuration

### Backend Configuration
Edit `finalwebapp_api.py` to configure:
- Detection confidence threshold (default: 0.3)
- Segmentation parameters
- Model paths
- API port and host

### Frontend Configuration
Edit `frontend/src/config/apiConfig.js` to set:
- Backend API base URL
- Timeout settings
- Request headers

### Environment Variables
Create a `.env` file:
```
FLASK_ENV=production
API_PORT=7860
MODEL_PATH=./runs/detect/train3/weights/best.pt
SEGMENTATION_MODEL_PATH=./segmentation_model/weights/best.pt
HF_TOKEN=your_hugging_face_token  # Optional for model download
```

---

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t infravision-ai:latest .
```

### Run Container
```bash
docker run -p 7860:7860 \
  -e FLASK_ENV=production \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/segmentation_model:/app/segmentation_model \
  infravision-ai:latest
```

---

## 🌐 Deployment

### Backend Deployment (Hugging Face Spaces)
1. Create a Hugging Face Space with Docker runtime
2. Upload repository or connect GitHub
3. Configure HF_TOKEN for model auto-download
4. Container starts Flask API on port 7860

### Frontend Deployment (Vercel)
1. Connect GitHub repository to Vercel
2. Set build command: `cd frontend && npm install && npm run build`
3. Set output directory: `frontend/build`
4. Configure environment variables for API endpoint
5. Deploy automatically on git push

### Live Deployment Status
- **Frontend**: Deployed to Vercel
- **Backend**: Deployed to Hugging Face Spaces
- **Models**: Auto-downloaded on startup via `hf_model_loader.py`

---

## 📝 Usage Examples

### Python Benchmark
```python
from finalwebapp import predict_cracks, classify_material, segment_image
import cv2

# Load image
image = cv2.imread('infrastructure.jpg')

# Detect cracks
results = predict_cracks(image, confidence_threshold=0.5)
print(f"Cracks detected: {len(results)}")

# Classify material
material = classify_material(image)
print(f"Material type: {material['class']}")

# Segment damage
mask = segment_image(image)
print(f"Mask shape: {mask.shape}")
```

### API Usage
```bash
# Upload and analyze image
curl -X POST http://localhost:7860/api/analyze-image \
  -F "file=@infrastructure.jpg" \
  -F "confidence_threshold=0.5"

# Generate 3D heightmap
curl -X POST http://localhost:7860/api/generate-heightmap \
  -F "file=@damage_map.png" \
  -F "format=glb" \
  -o model.glb

# Generate report
curl -X POST http://localhost:7860/api/generate-report \
  -H "Content-Type: application/json" \
  -d '{"title": "Infrastructure Report", "include_3d": true}' \
  -o report.pdf
```

---

## 🎓 Model Training

### Training YOLOv8 Detection
```bash
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model.train(
    data='path/to/dataset.yaml',
    epochs=100,
    imgsz=640,
    device=0
)
```

### Training Material Classifier
```bash
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(input_shape=(224, 224, 3), weights='imagenet')
# Add custom classification head for 8 material classes
# Train on annotated material dataset
model.save('material_classifier.h5')
```

---

## 🔍 Troubleshooting

### Model Download Issues
```python
# Manually trigger model download
python hf_model_loader.py --force
```

### CUDA/GPU Issues
```bash
# Run on CPU if GPU unavailable
export CUDA_VISIBLE_DEVICES=-1
python finalwebapp_api.py
```

### API Connection Issues
- Ensure Flask server is running: `python finalwebapp_api.py`
- Check firewall allows port 7860
- Verify API URL in frontend config matches backend host

### Memory Issues
- Reduce image input size in configuration
- Enable model quantization for faster inference
- Use smaller YOLOv8 variants (nano, small)

---

## 📊 Performance Optimization

### Inference Speedup
- **GPU Acceleration**: Use CUDA-enabled PyTorch (5-10x faster)
- **Model Quantization**: Convert to ONNX or TFLite (~3x faster, minimal accuracy loss)
- **Batch Processing**: Process multiple images in one batch
- **Caching**: Cache classification results for identical images

### Memory Optimization
- Load models once during startup (`hf_model_loader.py`)
- Use TFLite instead of full TensorFlow when available
- Stream processing for large video files

---

## 📦 Dependencies Management

### Requirements
- Python: 3.11+
- PyTorch: 2.9.0+
- TensorFlow: 2.13.0+
- OpenCV: 4.9.0+
- CUDA: 11.8+ (optional, for GPU)

### Virtual Environment Setup
```bash
# Create venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Code style: PEP 8 for Python, ESLint for JavaScript
- Test coverage: Aim for >80% coverage on new code
- Documentation: Update README and docstrings for new features
- Commit messages: Use clear, descriptive messages

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 📞 Support & Contact

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@infravision-ai.com

---

## 🔮 Roadmap

- [ ] Multi-GPU inference support
- [ ] Real-time 3D model streaming
- [ ] Mobile app (iOS/Android)
- [ ] Advanced ML pipeline optimization
- [ ] Integration with construction management tools
- [ ] Cloud storage for analysis history
- [ ] Advanced anomaly detection algorithms
- [ ] Industry-specific model variants

---

## 📚 References & Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [TensorFlow Deployment Guide](https://www.tensorflow.org/deploy)
- [React Best Practices](https://react.dev/)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [Civil Infrastructure Standards](https://www.astm.org/)

---

## ⭐ Acknowledgments

- YOLOv8 by Ultralytics
- TensorFlow/Keras by Google
- React community and Three.js contributors
- OpenCV for computer vision algorithms

---

**Last Updated**: April 2026
**Version**: 1.0.0
