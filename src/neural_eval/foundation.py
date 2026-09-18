from __future__ import annotations

import numpy as np
import pandas as pd

from .leakage import assert_no_group_overlap, duplicate_feature_rows
from .metrics import binary_metrics
from .probes import make_probe
from .splits import grouped_folds


def evaluate_probe_suite(
    X,
    y,
    groups,
    n_splits: int = 5,
    seed: int = 0,
) -> pd.DataFrame:
    """Compare linear and MLP probes under subject-held-out cross-validation."""
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
    splits = list(grouped_folds(y, groups, n_splits=n_splits, seed=seed))

    for probe_name in ("linear", "mlp"):
        for fold, (train_idx, test_idx) in enumerate(splits, start=1):
            assert_no_group_overlap(groups[train_idx], groups[test_idx])
            duplicates = duplicate_feature_rows(X[train_idx], X[test_idx])

            model = make_probe(probe_name, seed=seed + fold)
            model.fit(X[train_idx], y[train_idx])
            probability = model.predict_proba(X[test_idx])[:, 1]
            metrics = binary_metrics(y[test_idx], probability)

            rows.append(
                {
                    "probe": probe_name,
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "train_subjects": int(len(np.unique(groups[train_idx]))),
                    "test_subjects": int(len(np.unique(groups[test_idx]))),
                    "duplicate_feature_rows": int(duplicates),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)
