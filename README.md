# neural-eval

A leakage-resistant evaluation toolkit for EEG and neural-representation models.

The core question is simple: **does a model generalize to new people, or did the evaluation accidentally let subject identity leak across train and test?**

This repository provides subject-held-out baselines, leakage diagnostics, confidence intervals, and permutation controls for trial-level neural data and learned embeddings.

## Why this matters

EEG, MEG, fMRI, and multimodal foundation-model papers often report trial-level performance on datasets where many observations come from the same participant. Randomly splitting rows can put the same subject in both train and test sets, producing optimistic estimates that do not answer the clinical generalization question.

`neural-eval` makes the split contract explicit.

## Features

- Subject-disjoint train/test validation
- Stratified group cross-validation
- Hard leakage assertions
- Exact-feature duplicate checks across folds
- Logistic-regression probe baseline
- Balanced accuracy, ROC AUC, sensitivity, and specificity
- Bootstrap confidence intervals
- Subject-label permutation testing
- Synthetic EEG-like embeddings with controllable subject signal
- CLI that writes fold-level and aggregate results

## Install

```bash
git clone https://github.com/avisharm-alt/neural-eval
cd neural-eval
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quickstart

Run a fully reproducible synthetic benchmark:

```bash
neural-eval demo --subjects 40 --trials 30 --features 64 --seed 11 --out results/demo
```

Evaluate a CSV of embeddings:

```bash
neural-eval evaluate embeddings.csv \
  --label diagnosis \
  --group subject_id \
  --feature-prefix emb_ \
  --splits 5 \
  --out results/model
```

## Expected tabular contract

Each row is one observation/trial.

```text
subject_id,diagnosis,emb_0,emb_1,...,emb_127
s001,0,...
s001,0,...
s002,1,...
```

The grouping column is never used as a model feature. It is used only to enforce subject-disjoint evaluation.

## Python API

```python
from neural_eval.synthetic import make_neural_embeddings
from neural_eval.baseline import evaluate_grouped_probe

X, y, groups = make_neural_embeddings(seed=11)
result = evaluate_grouped_probe(X, y, groups, n_splits=5, seed=11)

print(result.summary)
print(result.folds)
```

## Evaluation philosophy

1. **Split by biological unit, not row.**
2. **Fit preprocessing inside each training fold.**
3. **Report uncertainty, not only a point estimate.**
4. **Run a null control.**
5. **Treat leakage checks as assertions that can fail the run.**

This is intentionally model-agnostic: use it on handcrafted EEG features, encoder embeddings, transformer representations, or multimodal features.

## License

MIT
