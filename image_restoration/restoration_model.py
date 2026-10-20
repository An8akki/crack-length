"""
restoration_model.py
====================

PyTorch neural network architecture for deep civil infrastructure image restoration
(RestorationNet) loaded from trained checkpoint best_restoration_model.pt.

Features:
- Residual Network architecture with 8 residual blocks and 64 channels.
- Global skip-connection: output = clamp(x + residual, 0.0, 1.0).
- Super-resolution, denoising, and contrast enhancement on degraded concrete imagery.
"""

import os
from typing import Optional, Tuple
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual convolutional block with ReLU activation."""

    def __init__(self, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class RestorationNet(nn.Module):
    """Deep residual image restoration network for degraded concrete crack imagery."""

    def __init__(self, in_channels: int = 3, channels: int = 64, blocks: int = 8):
        super().__init__()
        self.head = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.tail = nn.Conv2d(channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.tail(self.body(self.head(x)))
        return torch.clamp(x + residual, 0.0, 1.0)


_GLOBAL_RESTORATION_MODEL: Optional[RestorationNet] = None
_MODEL_DEVICE: Optional[str] = None


def get_restoration_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
) -> Tuple[Optional[RestorationNet], str]:
    """
    Load or retrieve cached pre-trained RestorationNet model.

    Parameters
    ----------
    checkpoint_path : str, optional
        Path to best_restoration_model.pt.
    device : str, optional
        'cuda' or 'cpu'. If None, autodetects available GPU.

    Returns
    -------
    Tuple[Optional[RestorationNet], str]
        (Loaded model in eval mode, active device string).
    """
    global _GLOBAL_RESTORATION_MODEL, _MODEL_DEVICE

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if _GLOBAL_RESTORATION_MODEL is not None and _MODEL_DEVICE == device:
        return _GLOBAL_RESTORATION_MODEL, device

    if checkpoint_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_ckpt = os.path.join(base_dir, "restoration_results", "best_restoration_model.pt")
        if os.path.exists(default_ckpt):
            checkpoint_path = default_ckpt

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(f"[WARNING] Restoration checkpoint not found at: {checkpoint_path}")
        return None, device

    try:
        model = RestorationNet(channels=64, blocks=8)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()

        _GLOBAL_RESTORATION_MODEL = model
        _MODEL_DEVICE = device
        print(f"[RESTORE] RestorationNet model loaded successfully from {checkpoint_path} onto {device}")
        return model, device
    except Exception as e:
        print(f"[ERROR] Failed to load RestorationNet model: {e}")
        return None, device
