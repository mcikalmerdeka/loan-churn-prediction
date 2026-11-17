# Model Evaluation Caveat: SMOTE Applied Before Cross-Validation

## Summary

The original project reported **85.88% cross-validation recall**. That figure is not a valid generalization estimate. The notebook applied SMOTE to the entire training set *before* creating cross-validation folds, so synthetic rows derived from the same applicant could appear in both the training and validation folds.

On an untouched 25% holdout, the deployed K-Nearest Neighbours model scores **36.76% recall** and **0.440 ROC-AUC**.

This document records the issue, the corrected numbers, and the recommended evaluation pattern. It is the reference for the project's evaluation methodology.

## What the original notebook did

```python
# SMOTE on the whole training set
X_train_resampled, y_train_resampled = SMOTE(
    sampling_strategy=1.0, random_state=42, k_neighbors=5
).fit_resample(X_train, y_train)

# Cross-validation on the already-resampled data
metrics = eval_classification(
    model=knn,
    X_train=X_train_resampled,
    y_train=y_train_resampled,
    X_test=X_test,
    y_test=y_test,
    validate_confusion_matrix=True,
)
```

`eval_classification` creates its folds from whatever data it receives. Because the data was resampled first, each fold is drawn from a mixture of real and synthetic rows.

## Why this leaks

1. **Synthetic neighbours cross folds.** A SMOTE row is a convex combination of real applicants. Its source rows can land in the validation fold while the synthetic row is in the training fold, or vice versa. The model is then partly validated on information it has already seen.
2. **The validation distribution changes.** SMOTE produces roughly 50/50 classes, while the real target rate is 19.55% default. Recall measured on a balanced, partly synthetic fold is not deployment recall.
3. **Model selection is affected.** Hyperparameters chosen by the leaky CV score are selected using an optimistic estimate.

## Reported metrics

### Original leaky CV

| Metric | Value |
|---|---:|
| Recall | 85.88% ± 0.23 pp |
| ROC-AUC | 80.11% ± 0.29 pp |

These are the figures that should **not** be cited as generalization performance.

### Untouched 25% holdout

Split: `train_test_split(test_size=0.25, stratify=y, random_state=42)`; 10,798 test rows.

| Metric | Value |
|---|---:|
| Accuracy | 53.70% |
| Precision | 17.47% |
| Recall | 36.76% |
| F1 | 23.69% |
| ROC-AUC | 43.99% |

Confusion matrix (rows = actual, columns = predicted):

```text
[[5022, 3665],
 [1335,  776]]
```

- TN = 5,022
- FP = 3,665
- FN = 1,335
- TP = 776

### Leak-free CV

SMOTE is fitted inside each training fold; validation folds stay imbalanced. Same KNN hyperparameters, 5 folds × 3 repeats:

| Metric | Value |
|---|---:|
| Recall | 28.80% ± 0.97 pp |
| ROC-AUC | 45.06% ± 0.49 pp |
| Accuracy | 59.79% |

The leak-free CV is consistent with the holdout. The leaky CV is the outlier.

## Correct evaluation pattern

```python
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

model = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("knn", KNeighborsClassifier(
        algorithm="kd_tree",
        leaf_size=25,
        n_neighbors=19,
        p=1,
        weights="distance",
    )),
])

metrics = eval_classification(
    model=model,
    X_train=X_train,   # original, imbalanced
    y_train=y_train,
    X_test=X_test,     # untouched
    y_test=y_test,
    validate_confusion_matrix=True,
)
```

The pipeline applies SMOTE only during `fit`, and only to the training portion of each fold. Validation folds and the test set are never resampled.

For the final deployed model, fitting SMOTE once on the full training set and then fitting the bare estimator is acceptable, because the test set remains untouched. The mistake is using that globally resampled set to estimate cross-validation performance.

## Takeaways

- 85.88% is a leaky CV result. Do not report it as a generalization estimate.
- The model's test-set ROC-AUC is approximately 0.44, which means it does not rank defaulters reliably on this split.
- The limiting factor is the available signal in the features, not the choice of KNN or the amount of tuning.
- The Streamlit app reports test-set metrics computed live. The notebook contains both the original diagnostic cells and the leak-free evaluation section.

## Reproduction

- Notebook: **Honest Evaluation (leak-free)** section.
- App metrics: `loan_prediction/analytics.py::model_metrics`.
- Original experiment: `notebook.ipynb`.
