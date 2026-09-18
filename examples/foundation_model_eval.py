from neural_eval import (
    evaluate_probe_suite,
    make_neural_embeddings,
    representation_diagnostics,
)

X, y, subjects = make_neural_embeddings(
    n_subjects=40,
    trials_per_subject=20,
    n_features=128,
    seed=11,
)

print(representation_diagnostics(X, seed=11))
print(evaluate_probe_suite(X, y, subjects, n_splits=5, seed=11))
