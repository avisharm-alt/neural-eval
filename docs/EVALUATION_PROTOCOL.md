# Evaluation protocol

## Unit of generalization

The default biological unit is the subject. Trials from one subject must never be split across training and test folds.

## Probe

The reference baseline is logistic regression after feature standardization. Both steps live inside a scikit-learn pipeline so scaling statistics are estimated from training data only.

## Cross-validation

Stratified group K-fold cross-validation attempts to balance the binary target while preserving subject-disjoint folds. The code asserts that the train/test subject sets do not overlap.

## Leakage audit

For every fold the benchmark records exact or near-exact feature rows shared across train and test. A non-zero count is a warning that preprocessing or caching may have duplicated observations.

## Metrics

The benchmark reports balanced accuracy, ROC AUC, sensitivity, and specificity per fold. Aggregate results include bootstrap confidence intervals across folds.

## Null model

The permutation test shuffles labels at the subject level rather than the row level, preserving within-subject dependence while destroying the target association.

## Intended use

Use this package to evaluate frozen EEG features or learned neural representations. It is not a clinical validation framework and does not substitute for external cohort validation.
