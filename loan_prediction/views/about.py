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
        "applicants and batches, and summarizes the model and the data behind it."
    )

    st.markdown("**Business metrics from the original analysis**")
    with st.container(horizontal=True):
        st.metric("Default rate", "19.55%", "before modelling", border=True)
        st.metric("Manual approval time", "hours to days", None, border=True)
        st.metric("Model approval time", "seconds", None, border=True)
    st.caption(
        "The default rate is the observed class balance in the source data. "
        "Approval-time figures describe the intended workflow, not a measured "
        "production impact."
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
            f"- **Test-set evaluation** · accuracy {metrics['accuracy']:.1%}, "
            f"recall {metrics['recall']:.1%}, "
            f"ROC-AUC {metrics['roc_auc']:.3f} on {metrics['n_test']:,} "
            f"held-out rows\n"
            f"- **Reference frame** · {len(reference):,} encoded rows in "
            f"`data/df_model_rewrite.csv`"
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
        "- **Dataset scope.** The model is trained on historical applicant "
        "data; patterns can change over time.\n"
        "- **Unused fields.** Marital status, house ownership, and current-job "
        "tenure are collected but do not affect the prediction.\n"
        "- **Version compatibility.** The saved artifact was serialized with "
        "scikit-learn 1.4 and is loaded under 1.8; re-running the notebook "
        "re-saves it under the current version.\n"
        "- **Intended use.** This app is for demonstration and educational "
        "use, not a substitute for a credit policy."
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
