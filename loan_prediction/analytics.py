"""Model and dataset analytics for the loan prediction app.

Pure functions with no Streamlit dependency so they stay testable and reusable
by the app views.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from utils import calculate_feature_importance

from .pipeline import RANDOM_STATE, TEST_SIZE
from .schema import MODEL_FEATURES, OUTCOME_LABELS, TARGET_COLUMN

# The figure previously advertised by the README and model card. It came from
# cross-validation after SMOTE had already been applied to the whole training
# set, so synthetic rows leaked across folds. It is kept only for the
# side-by-side comparison in the Insights tab.
LEAKY_CV_REFERENCE = {
    "recall": 0.8588,
    "recall_std": 0.0023,
    "roc_auc": 0.8011,
    "roc_auc_std": 0.0029,
    "source": "CV on SMOTE-resampled training data (leaky)",
}

# Produced by notebook.ipynb after moving SMOTE inside each CV fold:
# 5 folds x 3 repeats, same KNN hyperparameters. Kept as a reference because
# the app does not depend on imbalanced-learn and cannot recompute it live.
LEAK_FREE_CV_REFERENCE = {
    "recall": 0.28795,
    "recall_std": 0.00965,
    "roc_auc": 0.45065,
    "roc_auc_std": 0.00486,
    "source": "Leak-free CV, SMOTE inside folds (notebook)",
}

WEAK_SIGNAL_AUC = 0.55


def outcome_label(prediction: int) -> str:
    return OUTCOME_LABELS[int(prediction)]


def predict_frame(model, model_input: pd.DataFrame):
    """Return class predictions and positive-class probabilities."""
    probabilities = model.predict_proba(model_input)[:, 1]
    predictions = model.predict(model_input)
    return predictions, probabilities


def confidence_label(default_share: float) -> str:
    """How far a prediction sits from a coin flip."""
    confidence = max(float(default_share), 1.0 - float(default_share))
    if confidence >= 0.9:
        return "High"
    if confidence >= 0.7:
        return "Moderate"
    return "Low"


def classification_metrics(y_true, predictions, probabilities) -> Dict[str, Any]:
    """Score labels and probabilities once; shared by holdout and batch."""
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": np.array([[tn, fp], [fn, tp]]),
    }


def model_metrics(model, pipeline, reference: pd.DataFrame) -> Dict[str, Any]:
    """Evaluate the persisted model on the notebook's held-out 25% split.

    The split reproduces ``train_test_split(..., test_size=0.25,
    stratify=y, random_state=42)`` from the notebook. This is real, untouched
    data; unlike the notebook's CV on SMOTE-resampled rows, it is not inflated
    by synthetic neighbours.
    """
    features = reference[MODEL_FEATURES]
    target = reference[TARGET_COLUMN]

    _X_train, X_test, _y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    scaled_test = pipeline.scale_matrix(X_test)
    predictions, probabilities = predict_frame(model, scaled_test)
    metrics = classification_metrics(y_test, predictions, probabilities)

    return {
        "n_test": int(len(y_test)),
        **metrics,
        "predictions": predictions,
        "probabilities": probabilities,
        "y_test": y_test.to_numpy(),
    }


def target_balance(reference: pd.DataFrame) -> pd.DataFrame:
    """Applicant counts per target class."""
    counts = reference[TARGET_COLUMN].map(OUTCOME_LABELS).value_counts()
    return counts.rename_axis("Outcome").reset_index(name="Applicants")


def default_rate_by_group(
    engineered_training: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """Default rate per derived group (Profession_Group, State_Group, ...)."""
    if group_column not in engineered_training.columns:
        raise ValueError(f"Unknown group column: {group_column}")

    grouped = (
        engineered_training.groupby(group_column, dropna=False)[TARGET_COLUMN]
        .agg(customers="size", defaults="sum")
        .reset_index()
    )
    grouped["default_rate"] = grouped["defaults"] / grouped["customers"]
    return grouped.sort_values("default_rate", ascending=False).reset_index(drop=True)


def top_categories(training: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    """Most common raw categories, used for the dataset overview."""
    counts = training[column].value_counts().head(n)
    return counts.rename_axis(column).reset_index(name="Applicants")


def numeric_correlation(training: pd.DataFrame) -> pd.DataFrame:
    """Spearman correlation across the raw numeric profile columns."""
    numeric_columns = [
        column
        for column in ("Income", "Age", "Experience", "Current_House_Years")
        if column in training.columns
    ]
    columns = numeric_columns + [TARGET_COLUMN]
    return training[columns].corr(method="spearman")


def feature_relevance(reference: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Statistical feature relevance for the encoded model matrix.

    Mutual information is model-agnostic; it describes the signal available in
    the features, not the KNN's fitted weights. Values near zero confirm the
    weak-signal finding.
    """
    features = reference[MODEL_FEATURES]
    target = reference[TARGET_COLUMN]

    scores = calculate_feature_importance(
        X=features,
        y=target,
        task="classification",
        method="mutual_info",
        k="all",
    )
    scores = scores.rename(columns={"Feature": "feature", "Score": "score"})
    scores = scores.sort_values("score", ascending=False).head(top_n)
    return scores.reset_index(drop=True)
