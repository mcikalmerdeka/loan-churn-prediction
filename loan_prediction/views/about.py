"""About, model card, and data dictionary view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from loan_prediction.schema import (
    DERIVED_BY_MODEL,
    TARGET_COLUMN,
    USED_BY_MODEL,
    get_feature_definitions,
)
from utils import check_data_information


def _feature_table() -> pd.DataFrame:
    rows = []
    for key, meta in get_feature_definitions().items():
        if key == TARGET_COLUMN:
            status = "Target"
        elif key in DERIVED_BY_MODEL:
            status = "Derived"
        elif key in USED_BY_MODEL:
            status = "Yes"
        else:
            status = "No"

        rows.append(
            {
                "Feature": key.replace("_", " "),
                "Description": meta["description"],
                "Type": f"{meta['data_type']} · {meta['specific_type']}",
                "Used by model": status,
            }
        )
    return pd.DataFrame(rows)


def render_about_tab(
    reference: pd.DataFrame,
    training: pd.DataFrame,
    metrics: dict,
) -> None:
    st.markdown(
        "A lending company wants to reduce loan defaults by replacing manual "
        "credit checks with a repeatable model. This app scores individual "
        "applicants and batches, then shows exactly how much signal the model "
        "actually has."
    )

    st.markdown("**Business metrics from the original analysis**")
    with st.container(horizontal=True):
        st.metric("Default rate", "19.55%", "before modelling", border=True)
        st.metric("Manual approval time", "hours to days", None, border=True)
        st.metric("Model approval time", "seconds", None, border=True)
    st.caption(
        "The default-rate figure is the observed class balance in the source "
        "data. The app does not claim the model reduces it; the honest holdout "
        "below shows why."
    )

    with st.container(border=True):
        st.markdown("**Model card**")
        st.markdown(
            f"- **Algorithm** · K-Nearest Neighbours "
            f"(k=19, Manhattan distance, distance weights)\n"
            f"- **Target** · {TARGET_COLUMN} (1 = Default, 0 = Not Default)\n"
            f"- **Training data** · {len(training):,} unique applicants; the "
            f"notebook used a stratified 75% training split\n"
            f"- **Features** · 8 raw inputs expanded into 20 encoded features\n"
            f"- **Artifact** · `models/tuned_knn_model.joblib`\n"
            f"- **Honest holdout** · accuracy {metrics['accuracy']:.1%}, "
            f"recall {metrics['recall']:.1%}, "
            f"ROC-AUC {metrics['roc_auc']:.3f} on {metrics['n_test']:,} rows\n"
            f"- **Reference frame** · {len(reference):,} encoded rows in "
            f"`data/df_model_rewrite.csv`"
        )

    st.error(
        "**Performance caveat:** the previously advertised 85.88% recall came "
        "from cross-validation on SMOTE-resampled training data. Synthetic "
        "neighbours leaked across folds. The honest holdout is shown above and "
        "is close to random."
    )

    st.subheader("Data dictionary")
    st.caption(
        "Only eight raw fields influence the score. Marital status, house "
        "ownership, and current-job tenure are present in the source data but "
        "unused by the deployed model."
    )
    st.dataframe(
        _feature_table(),
        hide_index=True,
        column_config={
            "Description": st.column_config.TextColumn(width="large"),
            "Used by model": st.column_config.TextColumn(width="small"),
        },
    )

    with st.expander("Training data preview"):
        st.dataframe(training.head(10), hide_index=True)
        st.markdown("**Data information**")
        st.dataframe(
            check_data_information(training, training.columns),
            hide_index=True,
        )

    st.subheader("Limitations")
    st.markdown(
        "- **Weak signal.** On untouched holdout rows the model's ROC-AUC is "
        f"{metrics['roc_auc']:.2f}; a value near 0.5 means it cannot rank "
        "defaulters reliably.\n"
        "- **Leakage-inflated history.** The 85.88% recall figure came from "
        "CV after SMOTE; it is not a generalization estimate.\n"
        "- **Uncalibrated score.** The displayed vote share is the KNN's "
        "neighbour vote, not a calibrated default probability. Training on a "
        "balanced SMOTE sample shifts the prior away from the real 19.55%.\n"
        "- **Unused fields.** Marital status, house ownership, and current-job "
        "tenure are collected but do not affect the prediction.\n"
        "- **Not production-ready.** Predictions support exploration and "
        "pipeline testing, not real lending decisions."
    )

    st.subheader("Author")
    st.markdown(
        "Developed by **Muhammad Cikal Merdeka** · Data Analyst / Data Scientist\n\n"
        "[Email](mailto:mcikalmerdeka@gmail.com) · "
        "[GitHub](https://github.com/mcikalmerdeka) · "
        "[LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)"
    )

    st.caption(
        "Light and dark themes are available from the app menu "
        "(⋮ → Settings → Theme)."
    )
