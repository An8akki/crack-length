# Physical Crack-Length Research Protocol

This repository now supports a factorial study from a binary segmentation mask:

`mask -> skeleton -> pixel graph -> path/network metric -> scale -> physical length`

The detector is intentionally outside the experiment. Use
`crack_quantification.research_experiment.evaluate_rows` with rows containing
`mask`, `true_length_mm`, and calibration context. Use
`summarize_results` for MAE, RMSE, MAPE, signed-error standard deviation, and
runtime. Raw rows are suitable for JSON or CSV export.

## Path hypotheses

- `zhang_suen`, `guo_hall`, and `morphological` produce candidate centreline masks.
- `network` sums each undirected 8-connected skeleton edge once. This is the
  appropriate estimate for total branched crack-network length.
- `main_path` is the longest endpoint-to-endpoint geodesic. It is appropriate
  for the dominant crack only, not the total network.
- `mst` is a cycle-free minimum spanning forest. It is useful as a diagnostic
  when thinning creates redundant local cycles, but it is not automatically a
  physical crack-length estimator: Kruskal can remove valid centreline edges.
- `branch_lengths_px` and compressed graph segments expose junction structure
  and support branch-level analysis without double-counting network edges.

## Scale hypotheses and equations

For a planar reference segment of known length $L_r$ and image length $p_r$,

`s = L_r / p_r`, and `L_mm = s * L_px`.

For a pinhole camera with focal length $f$, perpendicular distance $Z`, sensor
width $S_w$, and image width $N_x$,

`s = (S_w / N_x) * (Z / f) = Z / f_x`.

This is only a local scale for a plane viewed frontally. Perspective requires a
planar homography from a four-corner reference (`homography_from_reference`),
and the reference must be coplanar with the crack. EXIF metadata alone cannot
recover absolute scale because distance is still unknown.

Metric depth enables back-projection with

`X = (u-c_x)Z/f_x`, `Y = (v-c_y)Z/f_y`, `Z = Z`,

followed by 3D polyline arc length. Relative monocular depth needs an external
metric anchor; it is not an absolute ruler by itself. Stereo, RGB-D, LiDAR, or
multi-view reconstruction can supply metric depth when their calibration and
registration are known.

## Dataset and ground truth

Create removable, measured crack phantoms or specimens with a traced centreline
whose arc length is measured using a flexible tape, calibrated microscope, or
high-resolution orthographic scan. Record the total network length, dominant
path, each branch, and marker dimensions. Capture repeated images while varying
distance, angle, resolution, blur, lighting, orientation, curvature, branching,
width, material, and planar/non-planar geometry. Keep specimen identity and
capture conditions in a manifest; split by specimen, not by image, to prevent
near-duplicate leakage.

For every valid combination report absolute error, signed error, MAE, RMSE,
MAPE, error standard deviation, runtime, failure rate, and stratified results
by distance, angle, perspective, blur, lighting, and surface type. Include
confidence intervals from specimen-level bootstrap resampling.

## Required visual evidence

Save one reproducible overlay per sample showing the original image, mask,
skeleton, graph edges/MST or selected path, reference marker/homography, and
the final physical length. Save a machine-readable raw-results table and a
summary table. Plot error distributions and error against each capture factor.

## Research interpretation

Skeletonization, graph geodesics, pinhole calibration, homography, depth
back-projection, and photogrammetry are established techniques. The
research-worthy contribution is an evidence-based comparison under controlled
structural-crack conditions, including branch-aware definitions and failure
analysis. Novelty or patentability requires a separate current literature and
prior-art search; this code makes no such claim.