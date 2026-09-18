"""Leakage-resistant evaluation for neural representations."""

from .baseline import EvaluationResult, evaluate_grouped_probe
from .leakage import assert_no_group_overlap, duplicate_feature_rows
from .synthetic import make_neural_embeddings

__all__ = [
    "EvaluationResult",
    "evaluate_grouped_probe",
    "assert_no_group_overlap",
    "duplicate_feature_rows",
    "make_neural_embeddings",
]
__version__ = "0.1.0"
