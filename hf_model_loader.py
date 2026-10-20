"""
Hugging Face Hub model loader for InfraVision
Downloads trained models from HF Hub at container startup
"""
import os
from pathlib import Path
from huggingface_hub import hf_hub_download

# Your HF Hub repo - Update this with your actual repo
HF_REPO_ID = "RijjaExplore/InfraVision-Models"  # Correct repo ID with proper capitalization

MODEL_FILES = {
    'crack_detection': {
        'filename': 'train3_best.pt',
        'dest_dir': 'runs/detect/train3/weights',
        'dest_filename': 'best.pt'
    },
    'segmentation': {
        'filename': 'segmentation_best.pt',
        'dest_dir': 'segmentation_model/weights',
        'dest_filename': 'best.pt'
    },
    'material_classifier_h5': {
        'filename': 'material_classifier.h5',
        'dest_dir': 'materialclassification_model',
        'dest_filename': 'material_classifier.h5'
    },
    'material_classifier_tflite': {
        'filename': 'material_classifier.tflite',
        'dest_dir': 'materialclassification_model',
        'dest_filename': 'material_classifier.tflite'
    },
    'sdg_image': {
        'filename': 'sdg.png',
        'dest_dir': 'frontend/src/assets',
        'dest_filename': 'sdg.png'
    },
    'sdg9_image': {
        'filename': 'sdg9.png',
        'dest_dir': 'frontend/src/assets',
        'dest_filename': 'sdg9.png'
    },
    'sdg11_image': {
        'filename': 'sdg11.png',
        'dest_dir': 'frontend/src/assets',
        'dest_filename': 'sdg11.png'
    }
}

def download_models_from_hf():
    """Download trained models from Hugging Face Hub"""
    print("\n🤗 Fetching models from Hugging Face Hub...")
    print(f"   Repository: {HF_REPO_ID}")
    
    models_loaded = 0
    total_models = len(MODEL_FILES)
    
    for model_type, file_info in MODEL_FILES.items():
        dest_dir = file_info['dest_dir']
        dest_filename = file_info['dest_filename']
        hf_filename = file_info['filename']
        
        # Create destination directory
        Path(dest_dir).mkdir(parents=True, exist_ok=True)
        dest_path = os.path.join(dest_dir, dest_filename)
        
        # Skip heavy crack detection download if the lightweight Nano model exists
        if model_type == 'crack_detection' and os.path.exists('runs/detect/train/weights/best.pt'):
            size_mb = os.path.getsize('runs/detect/train/weights/best.pt') / (1024 * 1024)
            print(f"✅ Using bundled lightweight YOLO defect model runs/detect/train/weights/best.pt ({size_mb:.1f}MB)")
            models_loaded += 1
            continue

        # Check if already exists
        if os.path.exists(dest_path):
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"✅ {model_type.replace('_', ' ')} model already exists ({size_mb:.1f}MB)")
            models_loaded += 1
            continue
        
        try:
            print(f"📥 Downloading {model_type}...")
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_filename,
                cache_dir=None,  # Don't use cache, download directly
                force_download=False
            )
            
            # Copy to expected location
            import shutil
            shutil.copy(downloaded_path, dest_path)
            
            size_mb = os.path.getsize(dest_path) / (1024 * 1024)
            print(f"✅ Downloaded {model_type}: {size_mb:.1f}MB")
            models_loaded += 1
            
        except Exception as e:
            print(f"⚠️ Could not download {model_type}: {e}")
            print(f"   App will use default YOLOv8 model")
    
    print(f"\n📊 Models loaded: {models_loaded}/{total_models}")
    return models_loaded > 0

if __name__ == '__main__':
    download_models_from_hf()
