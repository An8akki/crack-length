"""
Depth Preprocessor for Step 4C Pipeline
Pinnacle Concrete Crack-Depth Project

Implements and benchmarks multiple depth filtering algorithms:
- Raw (No filter)
- Median 3x3 and 5x5
- Bilateral Filter (Edge-Preserving)
- Robust Hybrid Filter (Bilateral + Invalid Hole Interpolation)

Computes quantitative filter metrics:
- Noise reduction ratio (surface variance reduction)
- Recess signal preservation
- JPEG 8x8 blockiness attenuation
- Crack boundary gradient preservation
"""

import numpy as np
import cv2


class DepthPreprocessor:
    def __init__(self):
        pass

    @staticmethod
    def filter_raw(depth):
        """Pass-through unfiltered depth."""
        return depth.copy().astype(np.float64)

    @staticmethod
    def filter_median(depth, ksize=5):
        """Standard median filter for impulse noise and salt-and-pepper suppression."""
        filtered = cv2.medianBlur(depth, ksize)
        return filtered.astype(np.float64)

    @staticmethod
    def filter_bilateral(depth, d=7, sigma_color=15.0, sigma_space=7.0):
        """Edge-preserving bilateral filter."""
        if depth.dtype not in (np.uint8, np.float32):
            depth_in = depth.astype(np.float32)
        else:
            depth_in = depth
        filtered = cv2.bilateralFilter(depth_in, d, sigma_color, sigma_space)
        return filtered.astype(np.float64)

    @staticmethod
    def filter_robust_hybrid(depth, mask=None):
        """
        Robust hybrid filter:
        1. Identifies invalid zero pixels.
        2. Applies bilateral smoothing to valid depth.
        3. Fills small invalid dropout holes in crack with localized median interpolation.
        """
        depth_f = depth.astype(np.float64)
        invalid_mask = (depth == 0)

        # Bilateral smooth
        smoothed = cv2.bilateralFilter(depth, 7, 15.0, 7.0).astype(np.float64)

        # If mask provided, interpolate invalid zero pixels inside crack using surrounding valid pixels
        if mask is not None and np.any(invalid_mask & (mask > 0)):
            hole_mask = (invalid_mask & (mask > 0)).astype(np.uint8)
            # Inpaint holes using Navier-Stokes based inpainting
            inpaint_src = depth.copy()
            inpainted = cv2.inpaint(inpaint_src, hole_mask, 3, cv2.INPAINT_NS)
            smoothed[hole_mask > 0] = inpainted[hole_mask > 0]

        return smoothed

    @staticmethod
    def compute_jpeg_blockiness(depth_array):
        """
        Computes horizontal 8x8 DCT block boundary discontinuity ratio.
        Values near 1.0 indicate absence of block artifacts.
        Values > 1.5 indicate severe JPEG block boundary discontinuities.
        """
        diff_h = np.abs(depth_array[:, 1:] - depth_array[:, :-1])
        cols = np.arange(diff_h.shape[1])
        boundary_cols = (cols % 8 == 7)
        internal_cols = (cols % 8 != 7)
        mean_boundary = float(np.mean(diff_h[:, boundary_cols]))
        mean_internal = float(np.mean(diff_h[:, internal_cols]))
        return float(mean_boundary / max(0.001, mean_internal))

    def evaluate_filters_on_sample(self, depth, mask, ring_mask):
        """
        Evaluates the 4 candidate preprocessing filters on a single capture.
        Returns:
            dict of filter_name -> metrics dict
        """
        filters = {
            "Raw": self.filter_raw(depth),
            "Median3": self.filter_median(depth, ksize=3),
            "Median5": self.filter_median(depth, ksize=5),
            "Bilateral": self.filter_bilateral(depth, d=7, sigma_color=15.0, sigma_space=7.0),
            "RobustHybrid": self.filter_robust_hybrid(depth, mask=mask)
        }

        eval_results = {}
        crack_idx = (mask > 0)
        ring_idx = (ring_mask > 0)

        # Precompute gradient magnitude of raw depth for edge comparison
        sobel_x = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(depth.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        raw_grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

        # Dilation of crack boundary for edge sharpness evaluation
        kernel_edge = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        boundary = (cv2.dilate(mask, kernel_edge) > 0) ^ crack_idx

        raw_edge_contrast = float(np.mean(raw_grad_mag[boundary])) if np.any(boundary) else 1.0

        for name, filtered in filters.items():
            # Noise on surrounding intact concrete
            ring_depths = filtered[ring_idx]
            ring_std = float(np.std(ring_depths)) if len(ring_depths) > 0 else 0.0

            # Median depth inside crack vs ring (apparent recess)
            crack_depths = filtered[crack_idx]
            recess_signal = float(np.median(ring_depths) - np.median(crack_depths)) if (len(ring_depths) > 0 and len(crack_depths) > 0) else 0.0

            # Blockiness ratio
            blockiness = self.compute_jpeg_blockiness(filtered)

            # Edge gradient preservation
            gx = cv2.Sobel(filtered.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(filtered.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx**2 + gy**2)
            filt_edge_contrast = float(np.mean(grad_mag[boundary])) if np.any(boundary) else 1.0
            edge_preservation_ratio = float(filt_edge_contrast / max(0.001, raw_edge_contrast))

            eval_results[name] = {
                "ring_std": round(ring_std, 3),
                "recess_signal": round(recess_signal, 3),
                "blockiness_ratio": round(blockiness, 3),
                "edge_preservation_ratio": round(edge_preservation_ratio, 3),
                "filtered_array": filtered
            }

        return eval_results
