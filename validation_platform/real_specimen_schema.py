"""Standardized research specimen record schema for synthetic and physical civil specimens."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json


@dataclass
class SpecimenRecord:
    """Standardized record for civil infrastructure crack quantification experiments."""
    specimen_id: str                      # Unique ID, e.g. CRK-SYN-0001 or LAB-BEAM-042
    source_type: str                      # 'synthetic' | 'laboratory_specimen' | 'field_inspection'
    surface_type: str                     # 'planar_wall', 'planar_slab', 'inclined_plane', etc.
    is_planar: bool

    # Environmental & Camera Setup
    camera_distance_mm: float
    camera_yaw_deg: float
    camera_pitch_deg: float
    focal_length_mm: float
    image_resolution: str                 # e.g. '1280x720'
    lighting_condition: str

    # Ground Truth Physical Quantities (Analytical / Measured)
    ground_truth_length_mm: float
    ground_truth_mst_mm: float
    ground_truth_dominant_mm: float
    ground_truth_mean_width_mm: float
    ground_truth_max_width_mm: float
    ground_truth_area_mm2: float

    # Estimates from the Three Pipeline Pathways
    estimate_2d_mm: Optional[float] = None
    estimate_planar_mm: Optional[float] = None
    estimate_3d_mm: Optional[float] = None

    # Error Metrics
    error_2d_abs_mm: Optional[float] = None
    error_2d_pct: Optional[float] = None
    error_planar_abs_mm: Optional[float] = None
    error_planar_pct: Optional[float] = None
    error_3d_abs_mm: Optional[float] = None
    error_3d_pct: Optional[float] = None

    # Quality & Diagnostic Indicators
    reprojection_error_px: Optional[float] = None
    perspective_distortion: Optional[str] = None
    planarity_score: float = 90.0
    planarity_label: str = "HIGH"
    recommended_method: str = "PLANAR"
    measurement_reliability: str = "NOT_VALIDATED"
    expected_error_mm: Optional[float] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def create_specimen_record(
    specimen_id: str,
    source_type: str,
    surface_type: str,
    is_planar: bool,
    camera_distance_mm: float,
    camera_yaw_deg: float,
    camera_pitch_deg: float,
    focal_length_mm: float,
    image_resolution: str,
    lighting_condition: str,
    ground_truth: Dict,
    estimates: Optional[Dict] = None,
    diagnostics: Optional[Dict] = None,
) -> SpecimenRecord:
    """Helper to construct a complete validated SpecimenRecord."""
    gt_len = float(ground_truth.get("total_centerline_length_mm") or ground_truth.get("length_mm", 0.0))
    gt_mst = float(ground_truth.get("mst_length_mm", gt_len))
    gt_dom = float(ground_truth.get("dominant_branch_length_mm", gt_len))
    gt_w_mean = float(ground_truth.get("mean_width_mm", 1.0))
    gt_w_max = float(ground_truth.get("max_width_mm", 1.0))
    gt_area = float(ground_truth.get("total_area_mm2") or ground_truth.get("area_mm2", 0.0))

    rec = SpecimenRecord(
        specimen_id=specimen_id,
        source_type=source_type,
        surface_type=surface_type,
        is_planar=is_planar,
        camera_distance_mm=float(camera_distance_mm),
        camera_yaw_deg=float(camera_yaw_deg),
        camera_pitch_deg=float(camera_pitch_deg),
        focal_length_mm=float(focal_length_mm),
        image_resolution=image_resolution,
        lighting_condition=lighting_condition,
        ground_truth_length_mm=gt_len,
        ground_truth_mst_mm=gt_mst,
        ground_truth_dominant_mm=gt_dom,
        ground_truth_mean_width_mm=gt_w_mean,
        ground_truth_max_width_mm=gt_w_max,
        ground_truth_area_mm2=gt_area,
    )

    if estimates:
        e2d = estimates.get("2d")
        ep = estimates.get("planar")
        e3d = estimates.get("3d")

        if e2d is not None and gt_len > 0:
            rec.estimate_2d_mm = round(float(e2d), 2)
            rec.error_2d_abs_mm = round(abs(e2d - gt_len), 2)
            rec.error_2d_pct = round(abs(e2d - gt_len) / gt_len * 100.0, 2)

        if ep is not None and gt_len > 0:
            rec.estimate_planar_mm = round(float(ep), 2)
            rec.error_planar_abs_mm = round(abs(ep - gt_len), 2)
            rec.error_planar_pct = round(abs(ep - gt_len) / gt_len * 100.0, 2)

        if e3d is not None and gt_len > 0:
            rec.estimate_3d_mm = round(float(e3d), 2)
            rec.error_3d_abs_mm = round(abs(e3d - gt_len), 2)
            rec.error_3d_pct = round(abs(e3d - gt_len) / gt_len * 100.0, 2)

    if diagnostics:
        rec.reprojection_error_px = diagnostics.get("reprojection_error_px")
        rec.perspective_distortion = diagnostics.get("perspective_distortion")
        rec.planarity_score = diagnostics.get("planarity_score", 90.0)
        rec.planarity_label = diagnostics.get("planarity_label", "HIGH")
        rec.recommended_method = diagnostics.get("recommended_method", "PLANAR")
        rec.measurement_reliability = diagnostics.get("measurement_reliability", "NOT_VALIDATED")
        rec.expected_error_mm = diagnostics.get("expected_error_mm")

    return rec
