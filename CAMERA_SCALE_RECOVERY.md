# Camera Scale Recovery Branch

This work is on the `camera-scale-recovery` branch. It is intentionally
separate from crack tracing and from `main`.

## Important finding about the current 3D code

`image_3d_heightmap.py` does not reconstruct the camera scene. It uses grayscale
intensity as a normalized visual height:

`height = (gray - min(gray)) / (max(gray) - min(gray)) * height_scale`

The resulting GLB has arbitrary model units. It cannot provide camera distance,
metric depth, surface pose, or mm/px. It remains useful for visualization, but
must not be used for physical crack measurement.

## Current project scale path

The existing integration behaves as follows:

1. `--mm-per-px`: trust a supplied scale.
2. Focal length, sensor width, image width, and distance: use pinhole geometry.
3. Otherwise `autocalibrate.py` reads EXIF, looks up an approximate sensor in
   `CAMERA_DB`, asks a registered depth provider for distance if available, and
   otherwise assumes 1000 mm with low confidence.
4. If camera metadata is unavailable, it falls back to `0.277 mm/px`.

The last two fallbacks are operational defaults, not measured physical scale.
They should be excluded from scientific accuracy claims.

## How pixel-to-real conversion works

For a fronto-parallel planar surface:

`f_x = f_mm * image_width_px / sensor_width_mm`

`mm_per_px = Z_mm / f_x`

Equivalently:

`mm_per_px = (sensor_width_mm / image_width_px) * (Z_mm / f_mm)`

Then `length_mm = length_px * mm_per_px`. This is only a local approximation;
scale changes across a perspective image and a non-planar surface cannot use one
global scalar.

## Methods implemented on this branch

### 1. Known planar reference, recommended baseline

Put a ruler, ArUco board, or checkerboard on the same surface as the crack. If
four marker corners are known, `planar_reference_scale` computes an image-to-
surface homography and maps the crack path into millimetres. This handles
perspective and is stronger than dividing by an average marker side length.

```python
from crack_quantification.camera_scale import planar_reference_scale

result = planar_reference_scale(
    corners_px,          # ordered image corners: TL, TR, BR, BL
    width_mm=200.0,
    height_mm=100.0,
    crack_path_yx=path,
)
print(result["crack_length_mm"])
```

### 2. Calibrated camera plus distance

Use checkerboard calibration to recover `fx`, `fy`, `cx`, `cy`, then obtain a
metric distance from a laser/range sensor, a known target pose, or metric depth.
This yields `Z / fx` and `Z / fy`. EXIF focal length alone is not enough because
distance and sensor dimensions remain uncertain.

### 3. Metric depth / true 3D path

With RGB-D, stereo, LiDAR, or metric multi-view reconstruction, each centreline
pixel is back-projected:

`X = (u-cx)Z/fx`, `Y = (v-cy)Z/fy`, `Z = depth(u,v)`

The real crack length is the sum of 3D segment lengths. Relative monocular depth
must first be anchored to a known distance or object size; otherwise it supplies
shape, not absolute millimetres.

## Running the scale stage independently

The reusable entry points are in [camera_scale.py](crack_quantification/camera_scale.py):

- `intrinsics_from_sensor(...)`
- `calibrate_checkerboard(...)`
- `planar_reference_scale(...)`
- `scale_from_metric_depth(...)`
- `recover_camera_scale(...)`

The intended order is:

`RGB image -> camera calibration/reference/depth -> metric scale -> separate crack-length module`

Do not pass the normalized GLB height values as `depth_map`.

## What is needed for a real run

- Reference method: one image containing the crack and a known planar marker,
  or manually recorded marker corners.
- Camera method: several checkerboard images for intrinsics and a measured
  camera-to-surface distance or metric pose/depth.
- 3D method: calibrated stereo/RGB-D/LiDAR or metric multi-view images. A single
  ordinary RGB image and the current grayscale heightmap are insufficient for
  absolute scale.

The branch currently validates the algorithms on synthetic data. An actual
physical scale result requires one of those metric inputs from the capture setup.