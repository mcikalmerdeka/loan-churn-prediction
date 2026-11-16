"""Model and dataset insights view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from loan_prediction.analytics import (
    LEAK_FREE_CV_REFERENCE,
    LEAKY_CV_REFERENCE,
    WEAK_SIGNAL_AUC,
    default_rate_by_group,
    feature_relevance,
    numeric_correlation,
    target_balance,
)
from loan_prediction.schema import NUMERIC_PROFILE_COLUMNS
from loan_prediction.ui import (
    confusion_chart,
    correlation_chart,
    default_rate_chart,
    distribution_chart,
    feature_importance_chart,
    target_chart,
)


@st.cache_data(show_spinner=False)
def _cached_relevance(reference: pd.DataFrame) -> pd.DataFrame:
    return feature_relevance(reference)


def render_insights_tab(
    reference: pd.DataFrame,
    training: pd.DataFrame,
    training_groups: pd.DataFrame,
    metrics: dict,
) -> None:
    st.markdown(
        "How the deployed model performs on untouched rows, what the features "
        "actually contain, and why the previously reported recall was inflated."
    )

    st.warning(
        "**The 85.88% recall previously reported is not a holdout result.** It "
        "came from cross-validation after SMOTE had been applied to the whole "
        "training set, so synthetic neighbours leaked across folds. The honest "
        "numbers are below."
    )

    comparison = pd.DataFrame(
        [
            {
                "Evaluation": "Leaky CV (previously reported)",
                "Recall (%)": LEAKY_CV_REFERENCE["recall"] * 100,
                "ROC-AUC": LEAKY_CV_REFERENCE["roc_auc"],
            },
            {
                "Evaluation": "Honest 25% holdout (this app)",
                "Recall (%)": metrics["recall"] * 100,
                "ROC-AUC": metrics["roc_auc"],
            },
            {
                "Evaluation": "Leak-free CV, SMOTE inside folds (notebook)",
                "Recall (%)": LEAK_FREE_CV_REFERENCE["recall"] * 100,
                "ROC-AUC": LEAK_FREE_CV_REFERENCE["roc_auc"],
            },
        ]
    )
    st.dataframe(
        comparison,
        hide_index=True,
        column_config={
            "Recall (%)": st.column_config.NumberColumn("Recall", format="%.2f%%"),
            "ROC-AUC": st.column_config.NumberColumn("ROC-AUC", format="%.3f"),
        },
    )
    st.caption(
        f"Holdout: {metrics['n_test']:,} applicants, stratified 25% split, "
        "random_state=42. The two CV rows are notebook reference values; the "
        "holdout is recomputed by the app."
    )

    with st.container(horizontal=True):
        st.metric("Accuracy", f"{metrics['accuracy']:.1%}", border=True)
        st.metric("Precision", f"{metrics['precision']:.1%}", border=True)
        st.metric("Recall", f"{metrics['recall']:.1%}", border=True)
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}", border=True)

    if metrics["roc_auc"] < WEAK_SIGNAL_AUC:
        st.error(
            f"ROC-AUC is {metrics['roc_auc']:.2f}. A score near 0.5 means the "
            "features carry no reliable ranking signal for this target. The app "
            "is a pipeline demonstration, not a production credit model."
        )

    left, right = st.columns([1, 1.3], gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Confusion matrix · honest holdout**")
            st.altair_chart(
                confusion_chart(metrics["confusion_matrix"]), width="stretch"
            )
            st.caption("Rows are actual outcomes; columns are predictions.")

    with right:
        with st.container(border=True):
            st.markdown("**Feature relevance**")
            relevance = _cached_relevance(reference)
            st.altair_chart(feature_importance_chart(relevance), width="stretch")
            st.caption(
                "Mutual information between each encoded feature and the target. "
                "Scores near zero confirm the weak-signal finding; this is a "
                "property of the data, not the KNN's fitted weights."
            )

    st.subheader("Dataset overview")
    st.caption(
        f"Raw training rows after de-duplication: {len(training):,}. "
        f"Reference frame used by the model: {len(reference):,}."
    )

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            st.markdown("**Target balance**")
            st.altair_chart(
                target_chart(target_balance(reference)), width="stretch"
            )
            st.caption("The source data is imbalanced: most applicants did not default.")

    with right:
        with st.container(border=True):
            st.markdown("**Default rate by profession group**")
            rates = default_rate_by_group(training_groups, "Prof")
            st.altair_chart(
                default_rate_chart(rates, "Prof"), width="stretch"
            )
            st.caption("Descriptive only. These rates are not causal effects.")

    feature = st.segmented_control(
        "Feature distribution",
        NUMERIC_PROFILE_COLUMNS,
        default=NUMERIC_PROFILE_COLUMNS[0],
    )
    if feature:
        with st.container(border=True):
            st.markdown(f"**Distribution · {feature}**")
            st.altair_chart(
                distribution_chart(training, feature), width="stretch"
            )

    with st.container(border=True):
        st.markdown("**Correlation with the target**")
        st.altair_chart(
            correlation_chart(numeric_correlation(training)), width="stretch"
        )
        st.caption(
            "Spearman correlation across the raw numeric fields and the target. "
            "Correlation is not causation."
        )
