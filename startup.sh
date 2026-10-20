#!/bin/bash
set -e

echo "🚀 Starting InfraVision AI Server..."
cd /app

# Ensure PORT is set (Render provides $PORT, default to 7860)
export PORT=${PORT:-7860}
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="-1"
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=3

echo "📍 Binding to port: $PORT"
exec python3 finalwebapp_api.py
