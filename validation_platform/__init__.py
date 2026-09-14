"""Scientific Validation, Error Modeling, and Adaptive Method Selection Platform."""

from .real_specimen_schema import SpecimenRecord, create_specimen_record
from .experiment_runner import (
    run_specimen_trial,
    run_batch_experiments,
    aggregate_experiment_statistics,
)
from .performance_matrix import generate_performance_matrix, identify_failure_modes
from .reliability_model import (
    MeasurementReliabilityModel,
    train_reliability_model,
    predict_measurement_reliability,
)
from .adaptive_selector import (
    recommend_measurement_method,
    AdaptiveMethodRecommendation,
)

__all__ = [
    "SpecimenRecord",
    "create_specimen_record",
    "run_specimen_trial",
    "run_batch_experiments",
    "aggregate_experiment_statistics",
    "generate_performance_matrix",
    "identify_failure_modes",
    "MeasurementReliabilityModel",
    "train_reliability_model",
    "predict_measurement_reliability",
    "recommend_measurement_method",
    "AdaptiveMethodRecommendation",
]
