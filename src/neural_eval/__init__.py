"""Evaluation infrastructure for brain foundation models."""

from .baseline import EvaluationResult, evaluate_grouped_probe
from .diagnostics import representation_diagnostics
from .foundation import evaluate_probe_suite
from .leakage import assert_no_group_overlap, duplicate_feature_rows
from .synthetic import make_neural_embeddings

__all__ = [
    "EvaluationResult",
    "evaluate_grouped_probe",
    "evaluate_probe_suite",
    "representation_diagnostics",
    "assert_no_group_overlap",
    "duplicate_feature_rows",
    "make_neural_embeddings",
]
__version__ = "0.2.0"
