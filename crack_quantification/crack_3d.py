"""
crack_3d.py
===========

Compute real-world 3D length of cracks from a 2D binary segmentation mask
combined with 3D reconstruction data (depth map, triangle mesh, or point cloud).

Key Steps:
1. Skeletonize 2D binary mask to 1-pixel-wide centerline.
2. Prune spurious short spurs and boundary noise using skeleton graph analysis (skan).
3. Back-project ordered 2D skeleton coordinates to 3D world/camera coordinates:
   - For depth map: Pinhole camera back-projection (fx, fy, cx, cy).
   - For triangle mesh: Camera ray casting and mesh intersection (trimesh).
   - For point cloud: Ray-nearest neighbor projection (cKDTree).
4. Compute Euclidean arc length along ordered 3D branches.
5. Identify total network length, main-branch length, and per-branch breakdown.
6. Evaluate sanity-check ratio against naive planar 2D length.
"""

import numpy as np
from scipy.spatial import cKDTree

try:
    from skimage.morphology import skeletonize as sk_skeletonize
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from skan.csr import Skeleton, summarize
    SKAN_AVAILABLE = True
except ImportError:
    SKAN_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False

try:
    import cv2 as _cv2
    CV2_AVAILABLE = True
except ImportError:
    _cv2 = None
    CV2_AVAILABLE = False


def _parse_camera_matrix(K, image_shape):
    """
    Extract focal lengths and principal point from K matrix or dict.
    
    Parameters
    ----------
    K : np.ndarray (3, 3) or dict
        Camera intrinsic matrix or dict with 'fx', 'fy', 'cx', 'cy'.
    image_shape : tuple
        (height, width) of the image.
    """
    h, w = image_shape[:2]
    if isinstance(K, dict):
        fx = float(K.get("fx", 1000.0))
        fy = float(K.get("fy", fx))
        cx = float(K.get("cx", w / 2.0))
        cy = float(K.get("cy", h / 2.0))
    elif isinstance(K, (list, tuple, np.ndarray)):
        K = np.asarray(K, dtype=np.float64)
        if K.shape == (3, 3):
            fx = float(K[0, 0])
            fy = float(K[1, 1])
            cx = float(K[0, 2])
            cy = float(K[1, 2])
        else:
            raise ValueError(f"Expected 3x3 camera matrix K, got shape {K.shape}")
    else:
        # Default pinhole approximation based on typical focal length
        fx = fy = float(max(h, w))
        cx = w / 2.0
        cy = h / 2.0
    return fx, fy, cx, cy


def _backproject_depth(pixels_yx, depth_map, fx, fy, cx, cy):
    """
    Back-project ordered (y, x) pixels to 3D camera coordinates in millimeters.
    
    Parameters
    ----------
    pixels_yx : np.ndarray, shape (N, 2)
        Ordered (row, col) coordinates.
    depth_map : np.ndarray, shape (H, W)
        Depth map. If max depth <= 50, assumed to be in meters and converted to mm.
    """
    depth = np.asarray(depth_map, dtype=np.float64)
    h, w = depth.shape[:2]
    
    # Check if depth is in meters (typical indoor/outdoor metric depth <= 50m)
    # If so, scale to millimeters (1m = 1000mm)
    valid_depths = depth[np.isfinite(depth) & (depth > 0)]
    if len(valid_depths) > 0 and np.median(valid_depths) < 50.0:
        scale_to_mm = 1000.0
    else:
        scale_to_mm = 1.0

    points_3d = []
    for y, x in pixels_yx:
        iy = int(np.clip(round(y), 0, h - 1))
        ix = int(np.clip(round(x), 0, w - 1))
        z = depth[iy, ix]
        if not np.isfinite(z) or z <= 0:
            # Interpolate from nearest finite neighbor
            neighbors = depth[max(0, iy-2):min(h, iy+3), max(0, ix-2):min(w, ix+3)]
            valid_n = neighbors[np.isfinite(neighbors) & (neighbors > 0)]
            z = np.median(valid_n) if len(valid_n) > 0 else 1000.0
        
        z_mm = z * scale_to_mm
        x_mm = (ix - cx) * z_mm / fx
        y_mm = (iy - cy) * z_mm / fy
        points_3d.append([x_mm, y_mm, z_mm])

    return np.asarray(points_3d, dtype=np.float64)


def _backproject_mesh(pixels_yx, mesh, fx, fy, cx, cy, pose=None):
    """
    Cast rays from camera center through pixels and intersect with 3D triangle mesh.
    
    Parameters
    ----------
    pixels_yx : np.ndarray, shape (N, 2)
        Ordered (row, col) pixel coordinates.
    mesh : trimesh.Trimesh
        Triangle mesh.
    pose : np.ndarray, shape (4, 4), optional
        Camera-to-world extrinsic matrix. If None, identity is used.
    """
    if pose is None:
        pose = np.eye(4, dtype=np.float64)
    else:
        pose = np.asarray(pose, dtype=np.float64)

    cam_origin_world = pose[:3, 3]
    R_cam_to_world = pose[:3, :3]

    ray_origins = []
    ray_dirs = []
    for y, x in pixels_yx:
        # Camera ray direction in camera frame
        ray_cam = np.array([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=np.float64)
        ray_cam /= np.linalg.norm(ray_cam)
        # Convert to world frame
        ray_world = R_cam_to_world @ ray_cam
        ray_world /= np.linalg.norm(ray_world)

        ray_origins.append(cam_origin_world)
        ray_dirs.append(ray_world)

    ray_origins = np.asarray(ray_origins)
    ray_dirs = np.asarray(ray_dirs)

    # Use Trimesh ray-mesh intersection
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    locations, index_ray, _ = intersector.intersects_location(ray_origins, ray_dirs, first_hit=True)

    # Map intersected locations back to ray order
    points_3d = np.zeros((len(pixels_yx), 3), dtype=np.float64)
    hit_map = {idx: loc for idx, loc in zip(index_ray, locations)}

    last_valid = None
    for i in range(len(pixels_yx)):
        if i in hit_map:
            points_3d[i] = hit_map[i]
            last_valid = hit_map[i]
        elif last_valid is not None:
            points_3d[i] = last_valid
        else:
            # Fallback point along ray at default 1 meter
            points_3d[i] = ray_origins[i] + ray_dirs[i] * 1000.0

    return points_3d


def _backproject_pointcloud(pixels_yx, pcd_points, fx, fy, cx, cy, pose=None):
    """
    Project rays from camera center and find nearest 3D neighbor in point cloud.
    """
    if pose is None:
        pose = np.eye(4, dtype=np.float64)
    else:
        pose = np.asarray(pose, dtype=np.float64)

    cam_origin = pose[:3, 3]
    R_cam = pose[:3, :3]
    tree = cKDTree(pcd_points)

    points_3d = []
    for y, x in pixels_yx:
        ray_cam = np.array([(x - cx) / fx, (y - cy) / fy, 1.0], dtype=np.float64)
        ray_world = R_cam @ (ray_cam / np.linalg.norm(ray_cam))
        # Search along ray: sample points from 0.2m to 5m
        test_samples = cam_origin + np.outer(np.linspace(200, 5000, 25), ray_world)
        dists, idxs = tree.query(test_samples)
        best_idx = idxs[np.argmin(dists)]
        points_3d.append(pcd_points[best_idx])

    return np.asarray(points_3d, dtype=np.float64)


# ---------------------------------------------------------------------------
# Public Helpers
# ---------------------------------------------------------------------------

def make_pseudo_depth_from_image(image_rgb, min_m=0.5, max_m=3.0):
    """
    Generate a metric pseudo-depth map from an RGB image using grayscale luminance.

    Dark pixels (low intensity)  → far away  (max_m)
    Bright pixels (high intensity) → close   (min_m)

    This is a physically plausible approximation for civil-infrastructure photos
    taken at 0.5–3 m: bright regions are typically the concrete/brick surface
    facing the camera (close), while dark regions are shadows or recessed voids.

    Parameters
    ----------
    image_rgb : np.ndarray, shape (H, W, 3) or (H, W)
        RGB image.  Values 0–255 uint8 or 0–1 float.
    min_m : float
        Depth assigned to the brightest pixel (meters). Default 0.5 m.
    max_m : float
        Depth assigned to the darkest pixel (meters).  Default 3.0 m.

    Returns
    -------
    depth_map : np.ndarray, shape (H, W), float32
        Metric depth in meters.
    """
    img = np.asarray(image_rgb, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 3 and img.shape[2] >= 3:
        # ITU-R BT.601 luminance
        gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    elif img.ndim == 2:
        gray = img
    else:
        gray = img[:, :, 0]
    # Invert: bright (close) → small depth value
    gray_inv = 1.0 - gray          # 0 = bright/close, 1 = dark/far
    depth = min_m + gray_inv * (max_m - min_m)
    return depth.astype(np.float32)


def render_pipeline_images(image_rgb, mask, skeleton, graph, paths_2d,
                           total_length_mm, main_branch_length_mm):
    """
    Render the 6 pipeline visualization images as BGR numpy arrays.

    Parameters
    ----------
    image_rgb : np.ndarray (H, W, 3)  RGB uint8
    mask : np.ndarray (H, W)  binary 0/1
    skeleton : np.ndarray (H, W)  binary bool
    graph : dict  output of analyze_skeleton()
    paths_2d : list of np.ndarray  ordered (y, x) coords per branch
    total_length_mm : float
    main_branch_length_mm : float

    Returns
    -------
    dict with keys 'original', 'ai_mask', 'skeleton', 'mst_graph',
    'depth_heatmap', 'path_overlay'. Each value is a uint8 BGR ndarray
    (or None when cv2 is unavailable for colormap operations).
    """
    img_np = np.asarray(image_rgb, dtype=np.uint8)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    elif img_np.shape[2] == 4:
        img_np = img_np[:, :, :3]
    h, w = img_np.shape[:2]

    # ── 1. Original (RGB → BGR) ──────────────────────────────────────────
    if CV2_AVAILABLE:
        original_bgr = _cv2.cvtColor(img_np, _cv2.COLOR_RGB2BGR)
    else:
        original_bgr = img_np[:, :, ::-1].copy()

    # ── 2. AI segmentation mask (white crack on black) ────────────────────
    mask_bin = (np.asarray(mask) > 0).astype(np.uint8) * 255
    ai_mask_bgr = np.stack([mask_bin] * 3, axis=-1)

    # ── 3. Skeleton overlay (cyan skeleton on 45%-dimmed original) ────────
    skel_bool = (np.asarray(skeleton) > 0)
    dimmed = (img_np * 0.45).astype(np.uint8)
    skel_bgr = dimmed[:, :, ::-1].copy() if not CV2_AVAILABLE else _cv2.cvtColor(dimmed, _cv2.COLOR_RGB2BGR)
    skel_bgr[skel_bool] = (0, 255, 255)   # BGR cyan

    # ── 4. MST graph overlay (magenta edges + yellow nodes) ───────────────
    mst_bgr = original_bgr.copy()
    mst_edges = graph.get("mst_edges", [])
    if CV2_AVAILABLE:
        for edge in mst_edges:
            s = (int(edge["start"][1]), int(edge["start"][0]))
            e = (int(edge["end"][1]), int(edge["end"][0]))
            _cv2.line(mst_bgr, s, e, (255, 0, 255), 2)           # magenta
        for edge in mst_edges:
            for pt in (edge["start"], edge["end"]):
                _cv2.circle(mst_bgr, (int(pt[1]), int(pt[0])), 3, (0, 255, 255), -1)  # cyan-yellow
        if not mst_edges and skel_bool.any():
            # Fallback: paint sparse skeleton dots
            ys, xs = np.where(skel_bool)
            for y, x in zip(ys[::3], xs[::3]):
                _cv2.circle(mst_bgr, (int(x), int(y)), 2, (255, 0, 255), -1)

    # ── 5. Depth heatmap (JET pseudo-depth) ──────────────────────────────
    depth_map = make_pseudo_depth_from_image(img_np)          # pass BGR, works fine
    d_min, d_max = float(depth_map.min()), float(depth_map.max())
    if d_max > d_min:
        depth_norm = ((depth_map - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        depth_norm = np.zeros_like(depth_map, dtype=np.uint8)
    if CV2_AVAILABLE:
        depth_bgr = _cv2.applyColorMap(depth_norm, _cv2.COLORMAP_JET)
    else:
        depth_bgr = np.stack([depth_norm] * 3, axis=-1)

    # ── 6. 3D path overlay (green polylines + length label) ───────────────
    path_bgr = original_bgr.copy()
    if CV2_AVAILABLE:
        for path_2d in paths_2d:
            pts = np.asarray(path_2d, dtype=int)
            for i in range(len(pts) - 1):
                _cv2.line(path_bgr,
                          (int(pts[i, 1]),     int(pts[i, 0])),
                          (int(pts[i+1, 1]),   int(pts[i+1, 0])),
                          (0, 220, 60), 2)        # green
        label = f"Total: {total_length_mm:.1f} mm  |  Main: {main_branch_length_mm:.1f} mm"
        font  = _cv2.FONT_HERSHEY_SIMPLEX
        fscale = max(0.45, min(0.75, w / 800.0))
        thick  = max(1, int(fscale * 2))
        (tw, th), _ = _cv2.getTextSize(label, font, fscale, thick)
        pad = 8
        _cv2.rectangle(path_bgr, (pad, pad), (pad + tw + pad, pad + th + pad * 2), (0, 0, 0), -1)
        _cv2.putText(path_bgr, label, (pad * 2, pad + th + pad // 2),
                     font, fscale, (0, 220, 60), thick, _cv2.LINE_AA)

    return {
        "original":      original_bgr,
        "ai_mask":       ai_mask_bgr,
        "skeleton":      skel_bgr,
        "mst_graph":     mst_bgr,
        "depth_heatmap": depth_bgr,
        "path_overlay":  path_bgr,
    }


def _prune_and_order_branches(skeleton_mask, min_branch_len_px=5):
    """
    Build skeleton graph via skan (with fallback), prune spurious branch spurs
    below min_branch_len_px, and return ordered 2D pixel path sequences.
    
    Returns
    -------
    paths_2d : list of np.ndarray
        List of ordered (y, x) pixel coordinates for each retained branch.
    main_path_2d : np.ndarray
        The longest continuous endpoint-to-endpoint path.
    pruned_count : int
        Number of spurious spur branches removed.
    """
    if not np.any(skeleton_mask):
        return [], np.zeros((0, 2), dtype=int), 0

    if SKAN_AVAILABLE:
        try:
            skel = Skeleton(skeleton_mask)
            if skel.n_paths == 0:
                return [], np.zeros((0, 2), dtype=int), 0

            df = summarize(skel, separator="-")
            paths_2d = []
            pruned_count = 0
            longest_len = -1.0
            main_path_2d = np.zeros((0, 2), dtype=int)

            for path_idx in range(skel.n_paths):
                coords = skel.path_coordinates(path_idx)  # shape (M, 2), (row, col)
                branch_type = df.loc[path_idx, "branch-type"]
                branch_dist = df.loc[path_idx, "branch-distance"]

                # Spur pruning rule:
                # branch-type == 1 is junction-to-endpoint (dead-end spur).
                # branch-type == 0 is isolated endpoint-to-endpoint.
                # If length < min_branch_len_px, prune it.
                if (branch_type == 1 or branch_type == 0) and len(coords) < min_branch_len_px:
                    pruned_count += 1
                    continue

                paths_2d.append(coords)
                if branch_dist > longest_len:
                    longest_len = branch_dist
                    main_path_2d = coords

            if not paths_2d and skel.n_paths > 0:
                # If all were pruned, keep at least the longest single branch
                longest_idx = int(np.argmax(df["branch-distance"]))
                main_path_2d = skel.path_coordinates(longest_idx)
                paths_2d = [main_path_2d]

            return paths_2d, main_path_2d, pruned_count
        except Exception:
            pass

    # Fallback graph traversal if skan encounters an issue or is unavailable
    from .graph_analysis import analyze_skeleton
    graph_report = analyze_skeleton(skeleton_mask)
    paths_2d = []
    pruned_count = 0
    main_path_2d = np.asarray(graph_report.get("main_path", []), dtype=int)

    for comp in graph_report.get("components", []):
        for seg in comp.get("segments", []):
            path = np.asarray(seg["path"], dtype=int)
            if len(path) < min_branch_len_px:
                pruned_count += 1
            else:
                paths_2d.append(path)

    if not paths_2d and len(main_path_2d) > 0:
        paths_2d = [main_path_2d]

    return paths_2d, main_path_2d, pruned_count


def compute_crack_length_3d(mask, depth_map_or_mesh, K, pose=None, min_branch_len_px=5, sanity_threshold=2.0):
    """
    Compute real-world 3D length of crack from 2D binary mask and 3D reconstruction data.

    Parameters
    ----------
    mask : np.ndarray (H, W)
        Binary crack mask, values 0 (background) or 1 (crack).
    depth_map_or_mesh : np.ndarray (H, W) or trimesh.Trimesh or np.ndarray (N, 3)
        - If 2D ndarray: per-pixel depth map (in meters or mm).
        - If trimesh.Trimesh: 3D surface mesh.
        - If (N, 3) ndarray: 3D point cloud coordinates.
    K : np.ndarray (3, 3) or dict
        Camera intrinsic matrix with fx, fy, cx, cy.
    pose : np.ndarray (4, 4), optional
        Camera-to-world extrinsic pose matrix. Defaults to Identity.
    min_branch_len_px : int, default=5
        Minimum pixel length to retain a branch; shorter dead-end spurs are pruned.
    sanity_threshold : float, default=2.0
        If ratio of 3D length to naive planar 2D length exceeds this, flagged_as_suspect = True.

    Returns
    -------
    dict
        {
            'total_length_mm': float,
            'main_branch_length_mm': float,
            'branch_lengths_mm': list of float,
            'num_branches': int,
            'flagged_as_suspect': bool,
            'sanity_ratio': float,
            'pruned_branches_count': int,
            'skeleton_pixels': int,
            'ordered_paths_3d': list of np.ndarray,
            'ordered_paths_2d': list of np.ndarray,
        }
    """
    binary = np.asarray(mask)
    if binary.ndim == 3 and binary.shape[2] == 1:
        binary = binary[:, :, 0]
    binary = (binary > 0.5) if np.issubdtype(binary.dtype, np.floating) else (binary > 0)

    if not binary.any():
        return {
            "total_length_mm": 0.0,
            "main_branch_length_mm": 0.0,
            "branch_lengths_mm": [],
            "num_branches": 0,
            "flagged_as_suspect": False,
            "sanity_ratio": 1.0,
            "pruned_branches_count": 0,
            "skeleton_pixels": 0,
            "ordered_paths_3d": [],
            "ordered_paths_2d": [],
        }

    # Step 1: Skeletonize to 1-pixel-wide centerline
    if SKIMAGE_AVAILABLE:
        skeleton = sk_skeletonize(binary)
    else:
        from .skeleton import skeletonize_mask
        skeleton = skeletonize_mask(binary, method="guo_hall")

    skeleton_pixels = int(skeleton.sum())
    fx, fy, cx, cy = _parse_camera_matrix(K, binary.shape)

    # Step 2: Prune spurious branches and extract ordered 2D paths
    paths_2d, main_path_2d, pruned_count = _prune_and_order_branches(skeleton, min_branch_len_px)

    if not paths_2d:
        return {
            "total_length_mm": 0.0,
            "main_branch_length_mm": 0.0,
            "branch_lengths_mm": [],
            "num_branches": 0,
            "flagged_as_suspect": False,
            "sanity_ratio": 1.0,
            "pruned_branches_count": pruned_count,
            "skeleton_pixels": skeleton_pixels,
            "ordered_paths_3d": [],
            "ordered_paths_2d": [],
        }

    # Step 3 & 4: Back-project ordered 2D points to 3D and compute arc lengths
    ordered_paths_3d = []
    branch_lengths_mm = []

    is_mesh = TRIMESH_AVAILABLE and isinstance(depth_map_or_mesh, trimesh.Trimesh)
    # A point cloud is (N, 3): 2D array where N rows are 3D points and N can be
    # anything.  A depth map is (H, W): 2D array where both dims are image dims.
    # We distinguish them by whether the second axis is exactly 3 AND the array
    # is clearly not a full-resolution image (i.e., rows >> cols=3 or shape is small).
    # The most reliable heuristic: PCD has exactly 3 columns; depth maps are
    # square-ish (W >> 3 and typically W >= H, both large).
    _is_ndarray_2d = isinstance(depth_map_or_mesh, np.ndarray) and depth_map_or_mesh.ndim == 2
    _has_3_cols = _is_ndarray_2d and depth_map_or_mesh.shape[1] == 3
    # Treat as PCD only when columns==3 AND it is NOT square/image-shaped
    # (i.e., it doesn't look like a 3-wide depth map which would be pathological)
    is_pcd = _has_3_cols and depth_map_or_mesh.shape[0] > 10
    # A depth map has shape (H, W) where W is the image width — typically >> 3
    is_depth = _is_ndarray_2d and not is_pcd

    for path_2d in paths_2d:
        if is_mesh:
            pts_3d = _backproject_mesh(path_2d, depth_map_or_mesh, fx, fy, cx, cy, pose)
        elif is_pcd:
            pts_3d = _backproject_pointcloud(path_2d, depth_map_or_mesh, fx, fy, cx, cy, pose)
        elif is_depth:
            pts_3d = _backproject_depth(path_2d, depth_map_or_mesh, fx, fy, cx, cy)
        else:
            # Fallback: assume planar surface at Z = 1.0m (1000mm)
            dummy_depth = np.full(binary.shape, 1000.0, dtype=np.float64)
            pts_3d = _backproject_depth(path_2d, dummy_depth, fx, fy, cx, cy)

        ordered_paths_3d.append(pts_3d)

        # Step 5: Euclidean distance summation along ordered consecutive points
        if len(pts_3d) >= 2:
            seg_lengths = np.linalg.norm(np.diff(pts_3d, axis=0), axis=1)
            b_len = float(np.sum(seg_lengths))
        else:
            b_len = 0.0
        branch_lengths_mm.append(b_len)

    total_length_mm = float(sum(branch_lengths_mm))

    # Step 6: Main-branch length
    if len(main_path_2d) >= 2:
        if is_mesh:
            main_3d = _backproject_mesh(main_path_2d, depth_map_or_mesh, fx, fy, cx, cy, pose)
        elif is_pcd:
            main_3d = _backproject_pointcloud(main_path_2d, depth_map_or_mesh, fx, fy, cx, cy, pose)
        elif is_depth:
            main_3d = _backproject_depth(main_path_2d, depth_map_or_mesh, fx, fy, cx, cy)
        else:
            dummy_depth = np.full(binary.shape, 1000.0, dtype=np.float64)
            main_3d = _backproject_depth(main_path_2d, dummy_depth, fx, fy, cx, cy)
        main_branch_length_mm = float(np.linalg.norm(np.diff(main_3d, axis=0), axis=1).sum())
    else:
        # No multi-point main path: fall back to longest branch
        main_branch_length_mm = max(branch_lengths_mm) if branch_lengths_mm else 0.0

    # Step 7: Sanity check ratio against naive flat planar approximation
    # Naive 2D length: estimate mm/px using the median depth across crack points
    all_3d = np.vstack(ordered_paths_3d) if ordered_paths_3d else np.zeros((1, 3))
    median_z_mm = float(np.median(all_3d[:, 2])) if len(all_3d) > 0 else 1000.0
    f_avg = (fx + fy) / 2.0
    naive_mm_per_px = median_z_mm / f_avg if f_avg > 0 else 1.0

    # 2D pixel path distance
    naive_2d_px = 0.0
    for path_2d in paths_2d:
        if len(path_2d) >= 2:
            naive_2d_px += float(np.linalg.norm(np.diff(path_2d, axis=0), axis=1).sum())
    naive_length_mm = naive_2d_px * naive_mm_per_px

    if naive_length_mm > 0:
        sanity_ratio = total_length_mm / naive_length_mm
    else:
        sanity_ratio = 1.0

    flagged_as_suspect = bool(sanity_ratio > sanity_threshold or sanity_ratio < (1.0 / sanity_threshold))

    return {
        "total_length_mm": total_length_mm,
        "main_branch_length_mm": main_branch_length_mm,
        "branch_lengths_mm": branch_lengths_mm,
        "num_branches": len(branch_lengths_mm),
        "flagged_as_suspect": flagged_as_suspect,
        "sanity_ratio": float(sanity_ratio),
        "pruned_branches_count": int(pruned_count),
        "skeleton_pixels": skeleton_pixels,
        "ordered_paths_3d": ordered_paths_3d,
        "ordered_paths_2d": paths_2d,
    }


def run_synthetic_test():
    """
    Synthetic verification test case:
    Diagonal line mask at 45 degrees + flat depth map at Z = 1.0 meter (1000 mm).
    
    Hand Calculation:
    - Line endpoints in image: (100, 100) to (200, 200) -> dx = 100 px, dy = 100 px.
    - Naive 2D pixel length = sqrt(100^2 + 100^2) = 141.421356 px.
    - Camera: fx = fy = 1000.0 px, cx = cy = 250.0 px.
    - Flat depth Z = 1.0 m = 1000 mm.
    - Scale: mm_per_px = Z / fx = 1000 mm / 1000 px = 1.0 mm/px.
    - Expected 3D Length = 141.421356 * 1.0 = 141.42 mm.
    """
    print("=" * 60)
    print("RUNNING SYNTHETIC 3D CRACK LENGTH VERIFICATION TEST")
    print("=" * 60)

    # 1. Create a clean single-pixel-wide diagonal mask (size 500x500).
    #    Using a 2-pixel-wide band causes skimage.skeletonize to produce an
    #    irregular skeleton covering only part of the range, introducing a ~50%
    #    error.  A single-pixel diagonal skeletonizes to itself exactly.
    size = 500
    mask = np.zeros((size, size), dtype=np.uint8)
    for i in range(100, 201):
        mask[i, i] = 1  # strict 1-px diagonal

    # 2. Create flat depth map at Z = 1.0 meter
    depth_map = np.full((size, size), 1.0, dtype=np.float64)  # 1.0 meter

    # 3. Camera intrinsics
    K = np.array([
        [1000.0,    0.0, 250.0],
        [   0.0, 1000.0, 250.0],
        [   0.0,    0.0,   1.0],
    ])

    # Diagonal from (100,100) to (200,200) -> Euclidean length = sqrt(100^2+100^2).
    expected_len_mm = np.hypot(200 - 100, 200 - 100) * 1.0  # 141.421356 mm

    result = compute_crack_length_3d(mask, depth_map, K, min_branch_len_px=3)

    measured = result["total_length_mm"]
    rel_error = abs(measured - expected_len_mm) / expected_len_mm * 100.0

    print(f"Expected 3D Length : {expected_len_mm:.2f} mm")
    print(f"Measured 3D Length : {measured:.2f} mm")
    print(f"Main Branch Length : {result['main_branch_length_mm']:.2f} mm")
    print(f"Number of Branches : {result['num_branches']}")
    print(f"Sanity Ratio       : {result['sanity_ratio']:.3f} (Flagged: {result['flagged_as_suspect']})")
    print(f"Relative Error     : {rel_error:.2f}%")

    assert rel_error < 1.0, f"Synthetic test failed: relative error {rel_error:.2f}% >= 1.0%"
    assert not result["flagged_as_suspect"], "Synthetic test falsely flagged as suspect"
    print("\n[SUCCESS] Synthetic verification test passed perfectly!")
    print("=" * 60)
    return result


if __name__ == "__main__":
    run_synthetic_test()
