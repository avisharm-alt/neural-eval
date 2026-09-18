from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def grouped_folds(
    y,
    groups,
    n_splits: int = 5,
    seed: int = 0,
):
    """Yield stratified folds with no group appearing on both sides."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("y and groups must have the same length")
    unique_groups = np.unique(groups)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if len(unique_groups) < n_splits:
        raise ValueError("number of unique groups must be >= n_splits")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    dummy_x = np.zeros((len(y), 1))
    yield from splitter.split(dummy_x, y, groups)
