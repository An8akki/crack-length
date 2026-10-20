"""
Deep CORAL (Correlation Alignment) Domain Adaptation Module
AI-Based Concrete Delamination Characterization Under Cross-Bridge Domain Shift

Implements second-order covariance alignment between labeled source bridges
and unlabeled target bridges based on Sun & Saenko (ECCV/BMVC 2016).

Formulation:
    L_CORAL = (1 / (4 * d^2)) * ||C_s - C_t||_F^2
    L_total = L_seg(source) + lambda_coral * L_CORAL(source, target)

Safeguards:
    - Target bridge data is strictly unlabeled (images only).
    - Target ground truth masks are never loaded or accessed.
    - Source segmentation loss and model selection strictly preserved.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure base model import
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
BASELINE_DIR = os.path.join(PROJECT_ROOT, "SDNET2021_Audit", "phase2b2_source_only_baseline")
if BASELINE_DIR not in sys.path:
    sys.path.insert(0, BASELINE_DIR)

from unet_model import UNet, DoubleConv


def compute_covariance(features: torch.Tensor) -> torch.Tensor:
    """
    Computes sample covariance matrix of feature representations.
    
    Args:
        features: Tensor of shape (N, d), where N is number of feature vectors
                  and d is feature dimensionality.
                  
    Returns:
        cov: Tensor of shape (d, d), representing the unbiased sample covariance.
    """
    if features.dim() != 2:
        raise ValueError(f"Expected 2D tensor (N, d), got shape {features.shape}")
        
    n_samples, d_dim = features.shape
    if n_samples <= 1:
        raise ValueError(f"Covariance requires at least 2 samples, got N={n_samples}")
        
    # Subtract sample mean along observation axis
    mean = features.mean(dim=0, keepdim=True)
    centered = features - mean
    
    # Unbiased sample covariance: (1 / (N - 1)) * (X - mu)^T * (X - mu)
    cov = torch.matmul(centered.t(), centered) / float(n_samples - 1)
    return cov


def coral_loss(source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
    """
    Computes the Deep CORAL covariance alignment loss between source and target features.
    
    L_CORAL = (1 / (4 * d^2)) * ||C_s - C_t||_F^2
    
    Args:
        source_features: Tensor of shape (N_s, d) or (B_s, C, H, W)
        target_features: Tensor of shape (N_t, d) or (B_t, C, H, W)
        
    Returns:
        loss: Scalar differentiable torch.Tensor.
    """
    # If 4D convolutional feature maps (B, C, H, W), reshape to (B * H * W, C)
    if source_features.dim() == 4:
        bs, c, h, w = source_features.shape
        source_features = source_features.permute(0, 2, 3, 1).contiguous().view(-1, c)
    if target_features.dim() == 4:
        bt, c_t, ht, wt = target_features.shape
        target_features = target_features.permute(0, 2, 3, 1).contiguous().view(-1, c_t)
        
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch: source has d={source_features.shape[1]}, "
            f"target has d={target_features.shape[1]}"
        )
        
    d = source_features.shape[1]
    
    cov_s = compute_covariance(source_features)
    cov_t = compute_covariance(target_features)
    
    # Squared Frobenius norm: ||C_s - C_t||_F^2
    diff = cov_s - cov_t
    frobenius_sq = torch.sum(diff * diff)
    
    loss = frobenius_sq / (4.0 * float(d * d))
    return loss


class DeepCORALUNet(UNet):
    """
    U-Net extended with intermediate feature extraction for Deep CORAL domain adaptation.
    
    Supports extracting features from:
      - 'bottleneck': Lowest resolution encoder representation (128 channels, 16x16)
      - 'penultimate': Final decoder features prior to 1x1 classification head (8 channels, 256x256)
    """
    def __init__(self, in_channels=1, num_classes=3, features=[8, 16, 32, 64]):
        super().__init__(in_channels=in_channels, num_classes=num_classes, features=features)

    def forward(self, x, return_features=False, feature_layer='bottleneck'):
        """
        Forward pass with optional intermediate feature extraction.
        
        Args:
            x: Input tensor (B, 1, H, W)
            return_features: If True, returns tuple (logits, features)
            feature_layer: 'bottleneck' or 'penultimate'
            
        Returns:
            logits: (B, num_classes, H, W)
            features: (B, C_feat, H_feat, W_feat) if return_features=True
        """
        skip_connections = []

        # Encoder downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        bottleneck_feats = self.bottleneck(x)
        x = bottleneck_feats
        skip_connections = skip_connections[::-1]

        # Decoder upsampling with skip connections
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_conn = skip_connections[idx // 2]

            if x.shape != skip_conn.shape:
                x = F.interpolate(x, size=skip_conn.shape[2:], mode="bilinear", align_corners=True)

            concat_x = torch.cat((skip_conn, x), dim=1)
            x = self.ups[idx + 1](concat_x)

        penultimate_feats = x
        logits = self.final_conv(penultimate_feats)

        if not return_features:
            return logits

        if feature_layer == 'bottleneck':
            return logits, bottleneck_feats
        elif feature_layer == 'penultimate':
            return logits, penultimate_feats
        else:
            raise ValueError(f"Unknown feature_layer '{feature_layer}'. Choose 'bottleneck' or 'penultimate'.")
