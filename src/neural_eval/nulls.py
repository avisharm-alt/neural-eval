from __future__ import annotations

import numpy as np

from .baseline import evaluate_grouped_probe


def subject_label_permutation_test(
    X,
    y,
    groups,
    observed_score: float | None = None,
    n_permutations: int = 100,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[str, object]:
    """Permutation null that shuffles labels at the subject level."""
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)

    subject_label = {}
    for group in unique_groups:
        labels = np.unique(y[groups == group])
        if len(labels) != 1:
            raise ValueError("each group must have exactly one label for this null")
        subject_label[group] = int(labels[0])

    if observed_score is None:
        observed_score = evaluate_grouped_probe(
            X, y, groups, n_splits=n_splits, seed=seed
        ).summary["roc_auc_mean"]

    rng = np.random.default_rng(seed)
    original = np.array([subject_label[g] for g in unique_groups])
    null_scores = []

    for i in range(n_permutations):
        shuffled = rng.permutation(original)
        lookup = dict(zip(unique_groups, shuffled))
        y_perm = np.array([lookup[g] for g in groups], dtype=int)
        result = evaluate_grouped_probe(
            X, y_perm, groups, n_splits=n_splits, seed=seed + i + 1
        )
        null_scores.append(result.summary["roc_auc_mean"])

    null_scores = np.asarray(null_scores, dtype=float)
    p_value = (1 + np.sum(null_scores >= observed_score)) / (n_permutations + 1)
    return {
        "observed": float(observed_score),
        "null_mean": float(np.mean(null_scores)),
        "p_value": float(p_value),
        "null_scores": null_scores.tolist(),
    }
