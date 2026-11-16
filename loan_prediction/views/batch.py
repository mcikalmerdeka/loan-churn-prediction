"""Batch scoring view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from loan_prediction.analytics import (
    WEAK_SIGNAL_AUC,
    classification_metrics,
    confidence_label,
    outcome_label,
    predict_frame,
)
from loan_prediction.config import BATCH_EXAMPLE_PATH
from loan_prediction.pipeline import initial_data_transform
from loan_prediction.schema import (
    CITY_GROUP_MAP,
    PROFESSION_TO_GROUP,
    REQUIRED_INPUT_COLUMNS,
    STATE_ZONE_MAP,
    TARGET_COLUMN,
)

BASE_DISPLAY_COLUMNS = [
    "Source row",
    "Income",
    "Age",
    "Experience",
    "Current_House_Years",
    "Car_Ownership",
    "Profession",
    "State",
    "City",
]

PREDICTION_COLUMNS = [
    "Predicted outcome",
    "Default vote share (%)",
    "Confidence",
]


def _drop_unnamed(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove index columns produced by ``to_csv(index=True)`` exports."""
    return frame.loc[
        :, [column for column in frame.columns if not str(column).startswith("Unnamed:")]
    ]


def _unknown_values(cleaned: pd.DataFrame) -> list[str]:
    warnings: list[str] = []

    unknown_professions = sorted(
        set(cleaned["Profession"].dropna()) - set(PROFESSION_TO_GROUP)
    )
    if unknown_professions:
        warnings.append(
            "Professions mapped to the miscellaneous group: "
            + ", ".join(map(str, unknown_professions[:8]))
            + ("…" if len(unknown_professions) > 8 else "")
        )

    fallback_states = sorted(
        set(cleaned["State"].dropna()) - set(STATE_ZONE_MAP)
    )
    if fallback_states:
        warnings.append(
            "States using the notebook's default west zone: "
            + ", ".join(map(str, fallback_states[:8]))
            + ("…" if len(fallback_states) > 8 else "")
        )

    fallback_cities = sorted(set(cleaned["City"].dropna()) - set(CITY_GROUP_MAP))
    if fallback_cities:
        warnings.append(
            "Cities using the notebook's default rural group: "
            + ", ".join(map(str, fallback_cities[:8]))
            + ("…" if len(fallback_cities) > 8 else "")
        )

    return warnings


def _read_upload(uploaded) -> pd.DataFrame | None:
    try:
        return _drop_unnamed(pd.read_csv(uploaded).reset_index(drop=True))
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return None


def _render_results(
    cleaned: pd.DataFrame,
    steps: dict,
    predictions,
    probabilities,
) -> None:
    results = cleaned.loc[steps["scaled"].index].copy()
    results.insert(0, "Source row", results.index + 1)
    results["Predicted outcome"] = [outcome_label(value) for value in predictions]
    results["Default vote share"] = probabilities
    results["Default vote share (%)"] = probabilities * 100
    results["Confidence"] = [confidence_label(value) for value in probabilities]

    target_numeric = None
    if TARGET_COLUMN in results.columns:
        target_numeric = pd.to_numeric(results[TARGET_COLUMN], errors="coerce")
        results["Actual outcome"] = target_numeric.map(
            {0: "Not Default", 1: "Default"}
        )
        if results["Actual outcome"].isna().all():
            results["Actual outcome"] = results[TARGET_COLUMN].map(
                {
                    "No Default": "Not Default",
                    "Not Default": "Not Default",
                    "Default": "Default",
                    "No": "Not Default",
                    "Yes": "Default",
                }
            )

    with st.container(horizontal=True):
        st.metric("Applicants scored", f"{len(results):,}", border=True)
        st.metric(
            "Predicted defaults",
            f"{int(predictions.sum()):,}",
            border=True,
        )
        st.metric(
            "Average vote share",
            f"{probabilities.mean():.1%}",
            border=True,
        )

    if target_numeric is not None:
        valid = target_numeric.notna()
        if valid.any():
            mask = valid.to_numpy()
            batch_metrics = classification_metrics(
                target_numeric[valid].astype(int),
                predictions[mask],
                probabilities[mask],
            )
            st.markdown("**Scores against the uploaded labels**")
            with st.container(horizontal=True):
                st.metric("Accuracy", f"{batch_metrics['accuracy']:.1%}", border=True)
                st.metric("Recall", f"{batch_metrics['recall']:.1%}", border=True)
                st.metric("ROC-AUC", f"{batch_metrics['roc_auc']:.3f}", border=True)
            st.caption(
                "The target column is only used for this comparison; the model "
                "never sees it as an input."
            )

    display_columns = BASE_DISPLAY_COLUMNS.copy()
    if "Actual outcome" in results.columns:
        display_columns.append("Actual outcome")
    display_columns.extend(PREDICTION_COLUMNS)

    st.dataframe(
        results[display_columns],
        hide_index=True,
        column_config={
            "Source row": st.column_config.NumberColumn("Row", format="%d"),
            "Income": st.column_config.NumberColumn("Income", format="%d"),
            "Default vote share (%)": st.column_config.ProgressColumn(
                "Default vote share",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

    st.download_button(
        "Download predictions as CSV",
        data=results[display_columns].to_csv(index=False).encode("utf-8"),
        file_name="loan_predictions.csv",
        mime="text/csv",
        icon=":material/download:",
    )

    with st.expander("Preprocessing trace"):
        trace_tabs = st.tabs(["Cleaned input", "Engineered features", "Model input"])
        with trace_tabs[0]:
            st.dataframe(cleaned.head(20), hide_index=True)
        with trace_tabs[1]:
            engineered_columns = [
                column
                for column in (
                    "Age",
                    "Experience",
                    "Current_House_Years",
                    "Car_Ownership",
                    "Generation",
                    "Experience_Age_Ratio",
                    "Prof",
                    "State",
                    "City",
                )
                if column in steps["engineered"].columns
            ]
            st.dataframe(
                steps["engineered"][engineered_columns].head(20),
                hide_index=True,
            )
        with trace_tabs[2]:
            st.dataframe(steps["scaled"].head(20), hide_index=True)


def render_batch_tab(model, pipeline, training: pd.DataFrame, metrics: dict) -> None:
    st.markdown(
        "Upload an applicant CSV and score every row in one pass. Results "
        "include a predicted outcome, the KNN neighbour vote share, and a "
        "download. The target column is optional."
    )

    if metrics["roc_auc"] < WEAK_SIGNAL_AUC:
        st.warning(
            "The deployed KNN scores below random on real holdout rows. Treat "
            "batch results as a pipeline demonstration, not lending decisions."
        )

    source = st.segmented_control(
        "Data source",
        ["Upload CSV", "Example data"],
        default="Upload CSV",
        key="batch_source",
    )

    raw: pd.DataFrame | None = None
    if source == "Upload CSV":
        uploaded = st.file_uploader(
            "Applicant CSV",
            type="csv",
            help="Same columns as the training data. Extra columns are ignored.",
        )
        if uploaded is not None:
            raw = _read_upload(uploaded)
    elif source == "Example data":
        raw = _drop_unnamed(pd.read_csv(BATCH_EXAMPLE_PATH).reset_index(drop=True))
        st.caption("Using the five-row example from data/batch_example.csv.")

    if raw is None:
        with st.expander("Expected columns"):
            st.markdown(
                "- " + "\n- ".join(REQUIRED_INPUT_COLUMNS) + "\n\n"
                "Marital status, house ownership, and current-job tenure are "
                "accepted but not used by the current model. `Id` and the "
                "target column are optional."
            )
        return

    if raw.empty:
        st.error("The uploaded file has no rows.")
        return

    try:
        cleaned = initial_data_transform(raw)
    except Exception as exc:
        st.error(f"Could not parse the input: {exc}")
        return

    missing = [
        column for column in REQUIRED_INPUT_COLUMNS if column not in cleaned.columns
    ]
    if missing:
        st.error("Missing required columns: " + ", ".join(missing))
        return

    for message in _unknown_values(cleaned):
        st.warning(message)

    try:
        steps = pipeline.transform_steps(cleaned)
        predictions, probabilities = predict_frame(model, steps["scaled"])
    except Exception as exc:
        st.error(f"Could not score the batch: {exc}")
        return

    _render_results(cleaned, steps, predictions, probabilities)
