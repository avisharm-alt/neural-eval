from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .leakage import assert_no_group_overlap, duplicate_feature_rows
from .metrics import binary_metrics, bootstrap_ci
from .splits import grouped_folds


@dataclass
class EvaluationResult:
    folds: pd.DataFrame
    summary: dict[str, float]


def evaluate_grouped_probe(
    X,
    y,
    groups,
    n_splits: int = 5,
    seed: int = 0,
    c: float = 1.0,
) -> EvaluationResult:
    """Evaluate a standardized logistic probe with group-disjoint CV."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)

    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if not (len(X) == len(y) == len(groups)):
        raise ValueError("X, y, and groups must have matching lengths")
    if len(np.unique(y)) != 2:
        raise ValueError("binary labels are required")

    rows = []
    for fold, (train_idx, test_idx) in enumerate(
        grouped_folds(y, groups, n_splits=n_splits, seed=seed), start=1
    ):
        assert_no_group_overlap(groups[train_idx], groups[test_idx])
        duplicated = duplicate_feature_rows(X[train_idx], X[test_idx])

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c,
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        probability = model.predict_proba(X[test_idx])[:, 1]
        metrics = binary_metrics(y[test_idx], probability)

        rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "train_subjects": int(len(np.unique(groups[train_idx]))),
                "test_subjects": int(len(np.unique(groups[test_idx]))),
                "duplicate_feature_rows": int(duplicated),
                **metrics,
            }
        )

    folds = pd.DataFrame(rows)
    summary: dict[str, float] = {}
    for metric in ["balanced_accuracy", "roc_auc", "sensitivity", "specificity"]:
        values = folds[metric].to_numpy(dtype=float)
        lo, hi = bootstrap_ci(values, seed=seed)
        summary[f"{metric}_mean"] = float(np.nanmean(values))
        summary[f"{metric}_ci_low"] = lo
        summary[f"{metric}_ci_high"] = hi

    summary["n_subjects"] = float(len(np.unique(groups)))
    summary["n_samples"] = float(len(y))
    return EvaluationResult(folds=folds, summary=summary)
