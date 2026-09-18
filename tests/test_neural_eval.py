import numpy as np
import pytest

from neural_eval.baseline import evaluate_grouped_probe
from neural_eval.leakage import assert_no_group_overlap, duplicate_feature_rows
from neural_eval.synthetic import make_neural_embeddings


def test_group_overlap_is_rejected():
    with pytest.raises(AssertionError):
        assert_no_group_overlap(["s1", "s2"], ["s2", "s3"])


def test_duplicate_rows_are_detected():
    train = np.array([[1.0, 2.0], [3.0, 4.0]])
    test = np.array([[3.0, 4.0], [9.0, 9.0]])
    assert duplicate_feature_rows(train, test) == 1


def test_grouped_probe_runs_and_is_disjoint():
    X, y, groups = make_neural_embeddings(
        n_subjects=20, trials_per_subject=6, n_features=12, seed=4
    )
    result = evaluate_grouped_probe(X, y, groups, n_splits=4, seed=4)
    assert len(result.folds) == 4
    assert (result.folds["duplicate_feature_rows"] == 0).all()
    assert 0.0 <= result.summary["balanced_accuracy_mean"] <= 1.0
