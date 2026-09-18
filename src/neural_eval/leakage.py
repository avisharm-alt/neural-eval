from __future__ import annotations

import hashlib

import numpy as np


def assert_no_group_overlap(train_groups, test_groups) -> None:
    """Raise if any biological unit occurs in both train and test."""
    train = set(np.asarray(train_groups).tolist())
    test = set(np.asarray(test_groups).tolist())
    overlap = train & test
    if overlap:
        preview = sorted(map(str, overlap))[:10]
        raise AssertionError(f"Group leakage detected: {preview}")


def duplicate_feature_rows(x_train, x_test, decimals: int = 8) -> int:
    """Count exact/near-exact feature rows copied across train and test."""
    x_train = np.asarray(x_train, dtype=float)
    x_test = np.asarray(x_test, dtype=float)
    if x_train.ndim != 2 or x_test.ndim != 2:
        raise ValueError("feature arrays must be 2D")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("train/test feature dimensions differ")

    def digest_rows(x):
        rounded = np.round(x, decimals=decimals)
        return {
            hashlib.sha256(row.tobytes()).digest()
            for row in np.ascontiguousarray(rounded)
        }

    return len(digest_rows(x_train) & digest_rows(x_test))
