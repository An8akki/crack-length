"""
restoration_pipeline.py
=======================

Inference pipeline and quality metric diagnostics for deep civil infrastructure
image restoration via RestorationNet.
"""

import time
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:  # pragma: no cover
    CV2_AVAILABLE = False

from .restoration_model import get_restoration_model, RestorationNet


def calculate_image_sharpness(img_gray: np.ndarray) -> float:
    """Calculate image sharpness via Laplacian variance."""
    if CV2_AVAILABLE:
        return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())
    else:
        gy, gx = np.gradient(img_gray.astype(float))
        return float(np.var(np.hypot(gx, gy)))


def calculate_image_entropy(img_gray: np.ndarray) -> float:
    """Calculate Shannon entropy of grayscale pixel distribution."""
    hist, _ = np.histogram(img_gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def compute_restoration_metrics(
    original_rgb: np.ndarray,
    restored_rgb: np.ndarray,
    runtime_sec: float = 0.0,
) -> dict:
    """
    Compute objective restoration quality metrics between original and restored images.

    Returns:
    - input_sharpness: Laplacian variance before restoration
    - output_sharpness: Laplacian variance after restoration
    - sharpness_change_pct: % sharpness change (+ indicates edge sharpening/enhancement)
    - input_entropy: Shannon entropy before restoration
    - output_entropy: Shannon entropy after restoration
    - input_brightness: Normalized mean brightness before restoration [0, 1]
    - output_brightness: Normalized mean brightness after restoration [0, 1]
    - runtime_sec: Model execution time in seconds
    """
    if CV2_AVAILABLE:
        orig_gray = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2GRAY)
        rest_gray = cv2.cvtColor(restored_rgb, cv2.COLOR_RGB2GRAY)
    else:
        orig_gray = np.mean(original_rgb, axis=2).astype(np.uint8)
        rest_gray = np.mean(restored_rgb, axis=2).astype(np.uint8)

    in_sharp = calculate_image_sharpness(orig_gray)
    out_sharp = calculate_image_sharpness(rest_gray)
    sharp_pct = ((out_sharp - in_sharp) / in_sharp * 100.0) if in_sharp > 0 else 0.0

    in_ent = calculate_image_entropy(orig_gray)
    out_ent = calculate_image_entropy(rest_gray)

    in_bright = float(np.mean(orig_gray) / 255.0)
    out_bright = float(np.mean(rest_gray) / 255.0)

    return {
        "input_sharpness": round(in_sharp, 2),
        "output_sharpness": round(out_sharp, 2),
        "sharpness_change_pct": round(sharp_pct, 2),
        "input_entropy": round(in_ent, 3),
        "output_entropy": round(out_ent, 3),
        "input_brightness": round(in_bright, 3),
        "output_brightness": round(out_bright, 3),
        "runtime_sec": round(runtime_sec, 4),
        "model_architecture": "RestorationNet (8 ResBlocks, 64 Channels)",
    }


def restore_image(
    image_rgb: np.ndarray,
    model: Optional[RestorationNet] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    max_dimension: int = 1024,
) -> Tuple[np.ndarray, dict]:
    """
    Run deep RestorationNet restoration on an input image.

    Parameters
    ----------
    image_rgb : np.ndarray
        Input RGB image (H, W, 3) with values in [0, 255] uint8.
    model : RestorationNet, optional
        Pre-loaded model. If None, loads from checkpoint.
    checkpoint_path : str, optional
        Path to best_restoration_model.pt.
    device : str, optional
        'cuda' or 'cpu'.
    max_dimension : int
        Max height or width for inference to prevent memory exhaustion.

    Returns
    -------
    Tuple[np.ndarray, dict]
        (Restored RGB uint8 image, quality metrics dictionary).
    """
    t_start = time.perf_counter()

    if model is None:
        model, active_device = get_restoration_model(checkpoint_path=checkpoint_path, device=device)
    else:
        active_device = next(model.parameters()).device.type

    if model is None:
        # Fallback if checkpoint cannot be loaded: CLAHE enhancement
        print("[WARNING] Model unavailable, executing adaptive contrast fallback...")
        if CV2_AVAILABLE:
            lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            restored = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
        else:
            restored = image_rgb.copy()
        metrics = compute_restoration_metrics(image_rgb, restored, time.perf_counter() - t_start)
        metrics["model_architecture"] = "CLAHE Adaptive Fallback (Checkpoint Missing)"
        return restored, metrics

    orig_h, orig_w = image_rgb.shape[:2]

    # Resize if image exceeds max_dimension to preserve memory and run in real-time
    scaled = False
    scale_factor = 1.0
    if max(orig_h, orig_w) > max_dimension:
        scale_factor = max_dimension / max(orig_h, orig_w)
        new_w = int(orig_w * scale_factor)
        new_h = int(orig_h * scale_factor)
        if CV2_AVAILABLE:
            inf_img = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            inf_img = image_rgb
        scaled = True
    else:
        inf_img = image_rgb

    # Memory-safe tiled inference
    tile_size = 512
    overlap = 32
    step = tile_size - overlap
    
    out_accum = np.zeros_like(inf_img, dtype=np.float32)
    weight_map = np.zeros((inf_img.shape[0], inf_img.shape[1]), dtype=np.float32)

    for y in range(0, inf_img.shape[0], step):
        for x in range(0, inf_img.shape[1], step):
            y_end = min(y + tile_size, inf_img.shape[0])
            x_end = min(x + tile_size, inf_img.shape[1])
            y_start = max(0, y_end - tile_size)
            x_start = max(0, x_end - tile_size)

            patch = inf_img[y_start:y_end, x_start:x_end]
            tensor_in = torch.from_numpy(patch).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor_in = tensor_in.to(active_device)

            with torch.no_grad():
                tensor_out = model(tensor_in)
                tensor_out = torch.clamp(tensor_out, 0.0, 1.0)

            p_np = (tensor_out.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.float32)
            out_accum[y_start:y_end, x_start:x_end] += p_np
            weight_map[y_start:y_end, x_start:x_end] += 1.0

    weight_map = np.maximum(weight_map, 1.0)
    restored_scaled = (out_accum / weight_map[:, :, None]).clip(0, 255).astype(np.uint8)

    if scaled and CV2_AVAILABLE:
        restored_net = cv2.resize(restored_scaled, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    else:
        restored_net = restored_scaled

    # Civil Infrastructure Engineering Enhancement & Crack Clarity Post-Processing:
    # 1. CLAHE local adaptive luminance equalization (illuminates crevices, equalizes glare/shadows)
    if CV2_AVAILABLE:
        lab = cv2.cvtColor(restored_net, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
        l_enh = clahe.apply(l)
        rgb_clahe = cv2.cvtColor(cv2.merge((l_enh, a, b)), cv2.COLOR_LAB2RGB)

        # 2. Edge-preserving bilateral filter (smooths sensor grain and gravel speckle while strictly preserving crack lips)
        denoised = cv2.bilateralFilter(rgb_clahe, d=7, sigmaColor=35, sigmaSpace=35)

        # 3. Targeted high-frequency crack edge unsharp masking
        blur = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.2)
        restored_final = cv2.addWeighted(denoised, 1.25, blur, -0.25, 0)
    else:
        restored_final = restored_net

    runtime = time.perf_counter() - t_start
    metrics = compute_restoration_metrics(image_rgb, restored_final, runtime_sec=runtime)
    metrics["active_device"] = active_device

    return restored_final, metrics
