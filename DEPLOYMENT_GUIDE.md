# InfraVision AI: Deployment Guide
**Complete Step-by-Step Instructions for Cloud & Production Deployment**

---

## Deployment Architecture Overview

InfraVision AI is designed with dual-mode deployment flexibility:

```
┌────────────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT OPTIONS MATRIX                         │
├──────────────────┬──────────────────────┬──────────────────────────────┤
│ Strategy         │ Frontend             │ Backend & ML Models          │
├──────────────────┼──────────────────────┼──────────────────────────────┤
│ 1. Unified Cloud │ In-Container React   │ Docker / Flask (Port 7860)   │
│    (Recommended) │ (Served via Flask)   │ (Hugging Face Spaces/Render) │
├──────────────────┼──────────────────────┼──────────────────────────────┤
│ 2. Split Cloud   │ Vercel / Netlify CDN │ Render / HF Spaces / CloudRun│
├──────────────────┼──────────────────────┼──────────────────────────────┤
│ 3. Self-Hosted   │ Docker Compose       │ Docker Compose               │
│    (VPS / Local) │ (Port 7860)          │ (Port 7860)                  │
└──────────────────┴──────────────────────┴──────────────────────────────┘
```

---

## Method 1: Hugging Face Spaces (Recommended & Free Tier)

Hugging Face Spaces provides up to 16 GB RAM and optional T4 GPU hardware, making it ideal for PyTorch, YOLOv8, NAFNet restoration, and ExtraTrees regression.

### Steps:
1. **Create a Space**:
   - Go to [huggingface.co/new-space](https://huggingface.co/new-space).
   - Enter Space name (e.g., `infravision-ai`).
   - Space SDK: Select **Docker** (Blank).
   - Hardware: Select **CPU Basic (Free - 16GB RAM)** or **T4 GPU**.
   - Visibility: Public or Private.

2. **Connect Repository**:
   - Under Space **Settings** -> **Git Repository**, add your GitHub repository as a mirror:
     ```bash
     git remote add hf https://huggingface.co/spaces/<YOUR-USERNAME>/infravision-ai
     git push hf main
     ```
   - Alternatively, link the GitHub repository directly via Hugging Face GitHub integration.

3. **Automatic Build & Launch**:
   - Hugging Face automatically reads the [Dockerfile](file:///d:/Works/AI-Powered_Civil_Infrastructure/Dockerfile).
   - [startup.sh](file:///d:/Works/AI-Powered_Civil_Infrastructure/startup.sh) downloads the 7 trained models from `RijjaExplore/InfraVision-Models`.
   - The compiled React application and Flask API launch together on port `7860`.
   - Your live URL will be: `https://<YOUR-USERNAME>-infravision-ai.hf.space`

---

## Method 2: Vercel (Frontend) + Hugging Face / Render (Backend)

For ultra-low latency worldwide CDN delivery of the React user interface, host the frontend on Vercel and the Python ML backend on Render or Hugging Face.

### Step 1: Deploy Backend (Render or Hugging Face)
1. Push your code to GitHub.
2. On [render.com](https://render.com), create a new **Web Service**.
3. Select your repository.
4. Environment: **Docker** (Render reads `Dockerfile` automatically).
5. Plan: Starter or Standard (recommended for ML RAM).
6. Note your deployed Backend URL (e.g., `https://infravision-backend.onrender.com`).

### Step 2: Deploy Frontend (Vercel)
1. Go to [vercel.com/new](https://vercel.com/new).
2. Import your GitHub repository.
3. Configure project settings:
   - **Framework Preset**: `Create React App`
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `build`
4. In **Environment Variables**, add:
   ```env
   REACT_APP_API_URL = https://infravision-backend.onrender.com
   ```
5. Click **Deploy**. Vercel will deploy your frontend with global edge caching.

---

## Method 3: Docker & Docker Compose (Self-Hosted VPS / AWS EC2 / DigitalOcean)

For dedicated server hosting on Ubuntu / Debian / CentOS / Windows Server:

### Prerequisites:
- Docker Engine $\ge 24.0$
- Docker Compose $\ge 2.20$

### 1-Command Deployment:
```bash
# Clone the repository
git clone https://github.com/An8akki/crack-length.git infravision-ai
cd infravision-ai

# Start the complete system in background
docker compose up -d --build
```

### Accessing the System:
- Open your browser to `http://<YOUR-SERVER-IP>:7860`
- Check container logs:
  ```bash
  docker logs -f infravision-ai
  ```
- Stop the container:
  ```bash
  docker compose down
  ```

---

## Method 4: Local Production Run (Without Docker)

To run the unified production build on a local machine or server directly:

```bash
# 1. Build the React frontend production bundle
cd frontend
npm install
npm run build
cd ..

# 2. Activate Python environment
# Windows:
.\venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. Start the production server
python finalwebapp_api.py
```

Open `http://localhost:7860` in any web browser. Flask will serve both the React interface and the `/api/` endpoints on the same port.

---

## Verification & Health Check

After deploying to any platform, verify server status via the health check endpoint:

```bash
curl https://<YOUR-APP-URL>/api/health
```

Expected JSON response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "message": "AI-Powered Structural Health Monitoring API is running",
  "models_status": {
    "yolo": "✅ Trained crack detection model loaded",
    "segmentation": "✅ Segmentation model loaded",
    "material": "✅ Material classifier loaded"
  }
}
```
