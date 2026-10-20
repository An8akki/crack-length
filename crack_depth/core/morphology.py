"""
Crack Morphology Processing for Step 4C Pipeline
Pinnacle Concrete Crack-Depth Project

Extracts 2D geometric and morphological features from crack masks:
- Medial skeleton thinning
- Euclidean distance transform width profiling
- Width distribution statistics (median, mean, max, 90th percentile)
- Crack length, area, connected components, tortuosity, and orientation
"""

import numpy as np
import cv2


class CrackMorphologyExtractor:
    def __init__(self):
        pass

    @staticmethod
    def extract_skeleton(binary_mask):
        """Morphological thinning to extract 1-pixel wide medial skeleton."""
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        skeleton = np.zeros_like(binary_mask)
        temp_mask = binary_mask.copy()
        max_iters = max(binary_mask.shape) + 10
        iters = 0
        while iters < max_iters:
            eroded = cv2.erode(temp_mask, element, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            opened = cv2.dilate(eroded, element)
            skeleton = cv2.bitwise_or(skeleton, cv2.subtract(temp_mask, opened))
            temp_mask = eroded.copy()
            if cv2.countNonZero(temp_mask) == 0:
                break
            iters += 1
        return skeleton

    @staticmethod
    def count_branch_points(skeleton):
        """Identifies skeleton junction points where 8-neighborhood connections >= 3."""
        kernel = np.array([
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1]
        ], dtype=np.uint8)
        skel_norm = (skeleton > 0).astype(np.uint8)
        neighbor_count = cv2.filter2D(skel_norm, -1, kernel)
        branch_points = (skel_norm == 1) & (neighbor_count >= 3)
        return int(np.sum(branch_points))

    def analyze_crack(self, binary_mask):
        """
        Extracts comprehensive morphological parameters from a binary mask (H, W).
        Returns:
            dict of morphological metrics and skeleton mask
        """
        area_px = int(np.sum(binary_mask > 0))
        if area_px == 0:
            return {
                "area_px": 0,
                "length_px": 0,
                "median_width_px": 0.0,
                "mean_width_px": 0.0,
                "p90_width_px": 0.0,
                "max_width_px": 0.0,
                "width_std_px": 0.0,
                "num_components": 0,
                "main_component_ratio": 0.0,
                "tortuosity": 1.0,
                "orientation_deg": 0.0,
                "branch_points": 0,
                "skeleton": np.zeros_like(binary_mask)
            }

        # Connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        # stats: [x, y, w, h, area]
        comp_areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background (index 0)
        num_components = len(comp_areas)
        main_component_ratio = float(np.max(comp_areas) / max(1, area_px)) if num_components > 0 else 1.0

        # Skeletonization
        skeleton = self.extract_skeleton(binary_mask)
        skel_idx = (skeleton > 0)
        length_px = int(np.sum(skel_idx))

        # Distance Transform for continuous width profiling
        dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)

        if length_px > 0:
            # Width along skeleton is 2 * distance to closest background pixel
            raw_widths = dist_transform[skel_idx].astype(np.float64) * 2.0
            finite_widths = raw_widths[np.isfinite(raw_widths)]
            if len(finite_widths) > 0:
                median_w = float(np.median(finite_widths))
                mean_w = float(np.mean(finite_widths))
                p90_w = float(np.percentile(finite_widths, 90))
                max_w = float(np.max(finite_widths))
                std_w = float(np.std(finite_widths))
            else:
                median_w = mean_w = p90_w = max_w = std_w = 1.0
        else:
            # Fallback for degenerate single-pixel blobs
            median_w = mean_w = p90_w = max_w = std_w = 1.0

        # Tortuosity: skeleton length / bounding box diagonal
        y_indices, x_indices = np.where(binary_mask > 0)
        dy = np.max(y_indices) - np.min(y_indices) + 1
        dx = np.max(x_indices) - np.min(x_indices) + 1
        bbox_diag = float(np.sqrt(dx**2 + dy**2))
        tortuosity = float(length_px / max(1.0, bbox_diag))

        # Orientation via PCA on coordinates
        coords = np.column_stack([x_indices, y_indices]).astype(np.float32)
        if len(coords) >= 5:
            mean_c, eig_vec = cv2.PCACompute(coords, mean=None)
            primary_axis = eig_vec[0]
            angle_rad = np.arctan2(primary_axis[1], primary_axis[0])
            orientation_deg = float(np.degrees(angle_rad)) % 180.0
        else:
            orientation_deg = 0.0

        branch_pts = self.count_branch_points(skeleton)

        return {
            "area_px": area_px,
            "length_px": length_px,
            "median_width_px": round(median_w, 2),
            "mean_width_px": round(mean_w, 2),
            "p90_width_px": round(p90_w, 2),
            "max_width_px": round(max_w, 2),
            "width_std_px": round(std_w, 2),
            "num_components": int(num_components),
            "main_component_ratio": round(main_component_ratio, 3),
            "tortuosity": round(tortuosity, 3),
            "orientation_deg": round(orientation_deg, 1),
            "branch_points": int(branch_pts),
            "skeleton": skeleton
        }
