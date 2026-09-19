"""
Calibration Data Schema and State Management for Step 5
Pinnacle Concrete Crack-Depth Project

Provides:
1. CalibrationState enumeration (UNCALIBRATED, CALIBRATED, EXTRAPOLATING_BEYOND_RANGE).
2. CalibrationRecord dataclass with strict field typing and range checks.
3. CalibrationDataset container with validation, CSV loading/saving, and specimen isolation.
4. Specimen-level partitioning (Leave-One-Specimen-Out, Group-K-Fold, Train/Test split).
5. SpecimenLeakageError guard preventing cross-split contamination.
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np
import pandas as pd


class CalibrationState(str, Enum):
    """Explicit operational state of a calibration model or pipeline stage."""
    UNCALIBRATED = "UNCALIBRATED"
    CALIBRATED = "CALIBRATED"
    EXTRAPOLATING_BEYOND_RANGE = "EXTRAPOLATING_BEYOND_RANGE"


class ValidationError(Exception):
    """Raised when calibration dataset fails schema, range, or completeness checks."""
    pass


class SpecimenLeakageError(Exception):
    """Raised when captures from the same physical specimen appear in both train and evaluation splits."""
    pass


# Mandatory schema columns and their expected physical ranges
REQUIRED_CALIBRATION_COLUMNS = [
    "specimen_id",
    "capture_id",
    "true_depth_mean_mm",
    "true_depth_std_mm",
    "true_width_mean_mm",
    "relative_recess_median",
    "relative_recess_mean",
    "relative_recess_p90",
    "crack_width_px",
    "surface_noise_sigma",
    "sensor_standoff_distance_mm",
    "ambient_lux",
    "camera_tilt_deg",
    "rgb_filepath",
    "depth_raw_filepath",
    "crack_mask_filepath",
]

NUMERIC_RANGE_CONSTRAINTS = {
    "true_depth_mean_mm": (0.05, 100.0),       # Valid cracks from 0.05 mm to 100 mm
    "true_depth_std_mm": (0.0, 15.0),          # Non-negative measurement error
    "true_width_mean_mm": (0.05, 50.0),        # Crack aperture 0.05 to 50 mm
    "crack_width_px": (1.0, 2000.0),           # Pixel width >= 1
    "surface_noise_sigma": (0.001, 100.0),     # Positive surface roughness
    "sensor_standoff_distance_mm": (200.0, 3000.0),  # Benchtop standoff 200 mm to 3 m
    "ambient_lux": (10.0, 50000.0),            # Positive illumination
    "camera_tilt_deg": (-45.0, 45.0),          # Angle within +/-45 degrees
}


@dataclass
class CalibrationRecord:
    """Represents a single verified physical calibration capture record."""
    specimen_id: str
    capture_id: str
    true_depth_mean_mm: float
    true_depth_std_mm: float
    true_width_mean_mm: float
    relative_recess_median: float
    relative_recess_mean: float
    relative_recess_p90: float
    crack_width_px: float
    surface_noise_sigma: float
    sensor_standoff_distance_mm: float
    ambient_lux: float
    camera_tilt_deg: float
    rgb_filepath: str
    depth_raw_filepath: str
    crack_mask_filepath: str
    station_x_mm: Optional[float] = None
    station_u_px: Optional[float] = None
    station_v_px: Optional[float] = None

    def __post_init__(self):
        """Perform validation upon instantiation."""
        for field, (min_val, max_val) in NUMERIC_RANGE_CONSTRAINTS.items():
            val = getattr(self, field)
            if not isinstance(val, (int, float, np.number)) or np.isnan(val):
                raise ValidationError(f"Field '{field}' must be a valid non-NaN number, got {val}")
            if not (min_val <= val <= max_val):
                raise ValidationError(
                    f"Field '{field}' value {val} out of valid physical range [{min_val}, {max_val}]"
                )

        # Validate optional station coordinates if supplied
        if self.station_x_mm is not None:
            if not (0.0 <= self.station_x_mm <= 200.0):
                raise ValidationError(f"station_x_mm {self.station_x_mm} out of range [0, 200] mm")
        if self.station_u_px is not None:
            if not (0.0 <= self.station_u_px <= 3840.0):
                raise ValidationError(f"station_u_px {self.station_u_px} out of range [0, 3840] px")
        if self.station_v_px is not None:
            if not (0.0 <= self.station_v_px <= 2160.0):
                raise ValidationError(f"station_v_px {self.station_v_px} out of range [0, 2160] px")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CalibrationDataset:
    """
    Validated container for physical calibration datasets.
    Guarantees structural validity and enforces specimen isolation during splitting.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = self._validate_and_clean(data)

    @classmethod
    def from_records(cls, records: List[CalibrationRecord]) -> "CalibrationDataset":
        """Instantiate dataset from a list of validated CalibrationRecord objects."""
        if not records:
            raise ValidationError("Cannot create CalibrationDataset from an empty list of records.")
        df = pd.DataFrame([r.to_dict() for r in records])
        return cls(df)

    @classmethod
    def from_csv(cls, csv_path: str) -> "CalibrationDataset":
        """Load and validate dataset from a CSV file."""
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise ValidationError(f"Failed to read CSV at '{csv_path}': {str(e)}")
        return cls(df)

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate all required fields, non-null status, and physical constraints."""
        if df.empty:
            raise ValidationError("Calibration dataset is completely empty.")

        # Check for presence of all required columns
        missing_cols = [c for c in REQUIRED_CALIBRATION_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValidationError(f"Dataset is missing required columns: {missing_cols}")

        # Check for null / NaN values
        null_counts = df[REQUIRED_CALIBRATION_COLUMNS].isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0].to_dict()
        if cols_with_nulls:
            raise ValidationError(
                f"Calibration dataset contains missing/NaN values: {cols_with_nulls}. "
                "Physical ground truth cannot be silently imputed."
            )

        # Check range constraints
        for col, (min_val, max_val) in NUMERIC_RANGE_CONSTRAINTS.items():
            violating = df[(df[col] < min_val) | (df[col] > max_val)]
            if len(violating) > 0:
                sample_viol = violating.iloc[0][col]
                raise ValidationError(
                    f"Column '{col}' has {len(violating)} values outside valid range [{min_val}, {max_val}], "
                    f"example: {sample_viol}"
                )

        # Ensure unique capture_id
        if df["capture_id"].duplicated().any():
            dups = df["capture_id"][df["capture_id"].duplicated()].tolist()
            raise ValidationError(f"Duplicate capture_id found in dataset: {dups[:5]}")

        # Return a cleanly formatted copy
        return df.copy().reset_index(drop=True)

    @property
    def num_captures(self) -> int:
        return len(self.data)

    @property
    def num_specimens(self) -> int:
        return self.data["specimen_id"].nunique()

    @property
    def specimen_ids(self) -> List[str]:
        return sorted(self.data["specimen_id"].unique().tolist())

    def get_features_and_targets(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract standardized features X and physical ground-truth target y."""
        features = self.data[[
            "relative_recess_median",
            "relative_recess_mean",
            "relative_recess_p90",
            "crack_width_px",
            "surface_noise_sigma",
            "sensor_standoff_distance_mm",
            "camera_tilt_deg",
        ]].copy()
        targets = self.data["true_depth_mean_mm"].copy()
        return features, targets

    def split_by_specimen(
        self, test_specimen_ids: List[str]
    ) -> Tuple["CalibrationDataset", "CalibrationDataset"]:
        """
        Partition dataset into train and test splits based strictly on specimen_id.
        Guarantees that no physical block is present in both train and test partitions.
        """
        all_specimens = set(self.specimen_ids)
        test_set = set(test_specimen_ids)

        invalid_specimens = test_set - all_specimens
        if invalid_specimens:
            raise ValidationError(
                f"Requested test specimens not found in dataset: {invalid_specimens}"
            )

        if not test_set:
            raise ValidationError("test_specimen_ids cannot be empty.")

        if test_set == all_specimens:
            raise ValidationError("test_specimen_ids cannot encompass all specimens (train would be empty).")

        train_df = self.data[~self.data["specimen_id"].isin(test_set)].copy()
        test_df = self.data[self.data["specimen_id"].isin(test_set)].copy()

        # Enforce specimen isolation assertion
        self.validate_specimen_isolation(train_df, test_df)

        return CalibrationDataset(train_df), CalibrationDataset(test_df)

    def leave_one_specimen_out(self) -> List[Tuple[str, "CalibrationDataset", "CalibrationDataset"]]:
        """
        Generates Leave-One-Specimen-Out (LOSO) cross-validation folds.
        Yields (held_out_specimen_id, train_dataset, test_dataset) for every specimen.
        """
        folds = []
        for held_out in self.specimen_ids:
            train_sub, test_sub = self.split_by_specimen([held_out])
            folds.append((held_out, train_sub, test_sub))
        return folds

    def group_kfold_by_specimen(self, n_splits: int = 3) -> List[Tuple[List[str], "CalibrationDataset", "CalibrationDataset"]]:
        """
        Generates GroupKFold cross-validation folds based on specimen_id.
        Ensures that all captures of any single specimen stay strictly together in either train or test.
        Yields (test_specimen_ids, train_dataset, test_dataset) for each fold.
        """
        all_specs = self.specimen_ids
        if n_splits > len(all_specs):
            raise ValidationError(
                f"n_splits ({n_splits}) cannot exceed total number of unique specimens ({len(all_specs)})."
            )
        if n_splits < 2:
            raise ValidationError("n_splits must be at least 2.")

        # Chunk specimens into n_splits roughly equal groups
        spec_chunks = np.array_split(all_specs, n_splits)
        folds = []
        for chunk in spec_chunks:
            test_ids = chunk.tolist()
            train_sub, test_sub = self.split_by_specimen(test_ids)
            folds.append((test_ids, train_sub, test_sub))
        return folds

    @staticmethod
    def validate_specimen_isolation(train_df: pd.DataFrame, test_df: pd.DataFrame, specimen_col: str = "specimen_id"):
        """
        Specimen Leakage Guard.
        Raises SpecimenLeakageError if any specimen appears in both train and test partitions.
        """
        train_specs = set(train_df[specimen_col].unique())
        test_specs = set(test_df[specimen_col].unique())
        overlap = train_specs.intersection(test_specs)
        if overlap:
            raise SpecimenLeakageError(
                f"CRITICAL LEAKAGE DETECTED: Specimens {overlap} appear in both training and test partitions. "
                "Specimen isolation is violated."
            )


# ---------------------------------------------------------------------------
# PHASE 5.5A: STATION-LEVEL PHYSICAL CALIBRATION SCHEMA
# ---------------------------------------------------------------------------

STATION_GROUND_TRUTH_COLUMNS = [
    "specimen_id",
    "crack_type",
    "station_id",
    "station_x_mm",
    "station_y_mm",
    "station_u_px",
    "station_v_px",
    "physical_ground_truth_mm",
    "rgb_file",
    "raw_depth_file",
    "mask_file",
    "standoff_mm",
    "tilt_deg",
    "capture_id",
    "repeat_id",
    "qc_status",
    "notes",
]

VALID_CRACK_TYPES = {"saw_cut_slot", "tensile_fracture"}
VALID_QC_STATUSES = {"PENDING", "PASS", "FAIL", "PENDING_REVIEW"}


@dataclass
class StationGroundTruthRecord:
    """
    Represents a single station-level physical calibration observation.
    Primary unit of ground-truth representation for Phase 5.5A and beyond.
    """
    specimen_id: str
    crack_type: str
    station_id: str
    station_x_mm: float
    station_y_mm: float
    station_u_px: float
    station_v_px: float
    physical_ground_truth_mm: Optional[float]
    rgb_file: str
    raw_depth_file: str
    mask_file: str
    standoff_mm: float
    tilt_deg: float
    capture_id: str
    repeat_id: int
    qc_status: str
    notes: str = ""

    def __post_init__(self):
        """Validate station-level fields and ranges."""
        if not self.specimen_id.startswith("SPEC-"):
            raise ValidationError(f"Invalid specimen_id format: '{self.specimen_id}'. Must start with 'SPEC-'.")

        if self.crack_type not in VALID_CRACK_TYPES:
            raise ValidationError(f"Invalid crack_type '{self.crack_type}'. Must be one of {VALID_CRACK_TYPES}.")

        if not (0.0 <= self.station_x_mm <= 250.0):
            raise ValidationError(f"station_x_mm {self.station_x_mm} out of bounds [0, 250] mm.")

        if not (-50.0 <= self.station_y_mm <= 50.0):
            raise ValidationError(f"station_y_mm {self.station_y_mm} out of bounds [-50, 50] mm.")

        if not (0.0 <= self.station_u_px <= 3840.0):
            raise ValidationError(f"station_u_px {self.station_u_px} out of bounds [0, 3840] px.")

        if not (0.0 <= self.station_v_px <= 2160.0):
            raise ValidationError(f"station_v_px {self.station_v_px} out of bounds [0, 2160] px.")

        if not (200.0 <= self.standoff_mm <= 3000.0):
            raise ValidationError(f"standoff_mm {self.standoff_mm} out of bounds [200, 3000] mm.")

        if not (-45.0 <= self.tilt_deg <= 45.0):
            raise ValidationError(f"tilt_deg {self.tilt_deg} out of bounds [-45, 45] deg.")

        if self.qc_status not in VALID_QC_STATUSES:
            raise ValidationError(f"Invalid qc_status '{self.qc_status}'. Must be one of {VALID_QC_STATUSES}.")

        # physical_ground_truth_mm may be None or NaN during acquisition preparation
        if self.physical_ground_truth_mm is not None and not np.isnan(self.physical_ground_truth_mm):
            if not (0.05 <= self.physical_ground_truth_mm <= 100.0):
                raise ValidationError(
                    f"physical_ground_truth_mm {self.physical_ground_truth_mm} out of valid range [0.05, 100.0] mm."
                )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StationCalibrationDataset:
    """
    Validated container for station-level physical calibration data.
    Enforces specimen-level isolation to prevent cross-split leakage.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = self._validate_and_clean(data)

    @classmethod
    def from_records(cls, records: List[StationGroundTruthRecord]) -> "StationCalibrationDataset":
        if not records:
            raise ValidationError("Cannot instantiate StationCalibrationDataset from empty records.")
        df = pd.DataFrame([r.to_dict() for r in records])
        return cls(df)

    @classmethod
    def from_csv(cls, csv_path: str) -> "StationCalibrationDataset":
        try:
            df = pd.read_csv(csv_path, comment="#")
        except Exception as e:
            raise ValidationError(f"Failed to read CSV at '{csv_path}': {str(e)}")
        return cls(df)

    def _validate_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValidationError("Station calibration dataset is completely empty.")

        missing_cols = [c for c in STATION_GROUND_TRUTH_COLUMNS if c not in df.columns]
        if missing_cols:
            raise ValidationError(f"Dataset missing required station columns: {missing_cols}")

        return df.copy().reset_index(drop=True)

    @property
    def num_records(self) -> int:
        return len(self.data)

    @property
    def num_specimens(self) -> int:
        return self.data["specimen_id"].nunique()

    @property
    def specimen_ids(self) -> List[str]:
        return sorted(self.data["specimen_id"].unique().tolist())

    def split_by_specimen(
        self, test_specimen_ids: List[str]
    ) -> Tuple["StationCalibrationDataset", "StationCalibrationDataset"]:
        """Partition station dataset strictly by specimen_id without cross-leakage."""
        all_specs = set(self.specimen_ids)
        test_set = set(test_specimen_ids)

        if not test_set:
            raise ValidationError("test_specimen_ids cannot be empty.")
        if test_set == all_specs:
            raise ValidationError("test_specimen_ids cannot encompass all specimens.")

        train_df = self.data[~self.data["specimen_id"].isin(test_set)].copy()
        test_df = self.data[self.data["specimen_id"].isin(test_set)].copy()

        CalibrationDataset.validate_specimen_isolation(train_df, test_df, specimen_col="specimen_id")
        return StationCalibrationDataset(train_df), StationCalibrationDataset(test_df)

