FROM python:3.11-slim

WORKDIR /app

# Force rebuild - ensure runs/, segmentation_model/, frontend/ directories are properly included
# BuildKit cache bust: 2026-03-26T23:47:00
ARG BUILD_DATE=2026-03-26T23:47:00
LABEL builder.date="${BUILD_DATE}"

# Install system dependencies for image processing and OpenGL support
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (saves 2.5GB of CUDA packages and 350MB+ RAM on CPU servers)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install remaining packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code and core packages
COPY finalwebapp_api.py .
COPY finalwebapp.py .
COPY pdf_report.py .
COPY segmentation_with_localisation.py .
COPY image_3d_heightmap.py .
COPY hf_model_loader.py .
COPY model_downloader.py .
COPY *.py ./

# Copy complete engineering modules
COPY crack_depth /app/crack_depth
COPY crack_quantification /app/crack_quantification
COPY image_restoration /app/image_restoration
COPY planar_calibration /app/planar_calibration
COPY validation_platform /app/validation_platform
COPY templates /app/templates

# Copy frontend application and compiled production build for unified single-container serving
COPY frontend /app/frontend

# Create directory structure for models and runtime outputs
RUN mkdir -p /app/runs/detect/train/weights /app/runs/detect/train3/weights && \
    mkdir -p /app/segmentation_model/weights && \
    mkdir -p /app/materialclassification_model && \
    mkdir -p /app/uploads /app/outputs && \
    mkdir -p /app/__pycache__

# Expose port
EXPOSE 7860

# Set Flask app environment variable
ENV FLASK_APP=finalwebapp_api.py

# Copy startup script and make it executable
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Run the app through startup script
# Models will load if available, or app will use defaults with graceful fallback
CMD ["/app/startup.sh"]
