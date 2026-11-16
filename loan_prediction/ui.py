"""Shared presentation helpers for the loan prediction app."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ACCENT = "#0F766E"
POSITIVE = "#15803D"
NEGATIVE = "#B91C1C"
WARNING = "#B45309"
INK = "#1C1917"
MUTED = "#78716C"


def fmt_percent(value: float) -> str:
    """Format a proportion without rounding extreme values to 0% or 100%."""
    if value >= 0.9995:
        return ">99.9%"
    if value <= 0.0005:
        return "<0.1%"
    return f"{value:.1%}"


def vote_share_progress(default_share: float) -> None:
    """Show the KNN's neighbour vote share, clearly labelled."""
    st.progress(
        min(max(float(default_share), 0.0), 1.0),
        text=f"Default vote share · {fmt_percent(default_share)}",
    )


def confusion_chart(matrix: np.ndarray) -> alt.Chart:
    """2x2 confusion matrix with applicant counts."""
    rows = ["Actual · Not Default", "Actual · Default"]
    columns = ["Predicted · Not Default", "Predicted · Default"]
    frame = (
        pd.DataFrame(matrix, index=rows, columns=columns)
        .rename_axis("actual")
        .reset_index()
        .melt(id_vars="actual", var_name="predicted", value_name="applicants")
    )

    base = alt.Chart(frame).mark_rect(cornerRadius=4).encode(
        x=alt.X("predicted:N", title=None, sort=columns),
        y=alt.Y("actual:N", title=None, sort=rows),
        color=alt.Color(
            "applicants:Q",
            scale=alt.Scale(range=["#F0FDFA", "#99F6E4", "#5EEAD4"]),
            legend=None,
        ),
        tooltip=[
            alt.Tooltip("actual:N", title="Actual"),
            alt.Tooltip("predicted:N", title="Predicted"),
            alt.Tooltip("applicants:Q", title="Applicants"),
        ],
    )
    text = base.mark_text(fontSize=20, fontWeight="bold", color="#134E4A").encode(
        text="applicants:Q"
    )
    return (base + text).properties(height=220)


def feature_importance_chart(table: pd.DataFrame) -> alt.Chart:
    """Mutual-information scores for the encoded model features."""
    order = table["feature"].tolist()
    return (
        alt.Chart(table)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("score:Q", title="Mutual information"),
            y=alt.Y("feature:N", title=None, sort=order),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("score:Q", title="Score", format=".4f"),
            ],
        )
        .properties(height=alt.Step(24))
    )


def target_chart(balance: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(balance)
        .mark_bar(cornerRadius=3, color=ACCENT)
        .encode(
            x=alt.X("Outcome:N", title=None),
            y=alt.Y("Applicants:Q", title="Applicants"),
            tooltip=["Outcome:N", "Applicants:Q"],
        )
        .properties(height=240)
    )


def default_rate_chart(rates: pd.DataFrame, group_column: str) -> alt.Chart:
    """Default rate per category, with volume in the tooltip."""
    return (
        alt.Chart(rates)
        .mark_bar(cornerRadius=3, color=WARNING)
        .encode(
            x=alt.X("default_rate:Q", title="Default rate", axis=alt.Axis(format="%")),
            y=alt.Y(f"{group_column}:N", title=None, sort="-x"),
            tooltip=[
                alt.Tooltip(f"{group_column}:N", title="Group"),
                alt.Tooltip("customers:Q", title="Applicants"),
                alt.Tooltip("default_rate:Q", title="Default rate", format=".2%"),
            ],
        )
        .properties(height=alt.Step(28))
    )


def distribution_chart(training: pd.DataFrame, column: str) -> alt.Chart:
    """Histogram of one raw numeric field."""
    return (
        alt.Chart(training)
        .mark_bar(color=ACCENT, opacity=0.85, cornerRadius=2)
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=28), title=column),
            y=alt.Y("count():Q", title="Applicants"),
            tooltip=[alt.Tooltip("count():Q", title="Applicants")],
        )
        .properties(height=260)
    )


def category_chart(counts: pd.DataFrame, column: str) -> alt.Chart:
    """Most common raw categories."""
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadius=3, color=ACCENT, opacity=0.85)
        .encode(
            x=alt.X("Applicants:Q", title="Applicants"),
            y=alt.Y(f"{column}:N", title=None, sort="-x"),
            tooltip=[alt.Tooltip(f"{column}:N", title="Category"), "Applicants:Q"],
        )
        .properties(height=300)
    )


def correlation_chart(correlation: pd.DataFrame) -> alt.Chart:
    """Spearman correlation heatmap."""
    order = correlation.columns.tolist()
    frame = (
        correlation.rename_axis("row")
        .reset_index()
        .melt(id_vars="row", var_name="column", value_name="rho")
    )

    base = alt.Chart(frame).mark_rect().encode(
        x=alt.X("column:N", title=None, sort=order),
        y=alt.Y("row:N", title=None, sort=order),
        color=alt.Color(
            "rho:Q",
            scale=alt.Scale(scheme="redblue", domain=[-1, 1], reverse=True),
            title="Spearman ρ",
        ),
        tooltip=[
            alt.Tooltip("row:N", title="Feature"),
            alt.Tooltip("column:N", title="Against"),
            alt.Tooltip("rho:Q", title="ρ", format="+.2f"),
        ],
    )
    text = base.mark_text(fontSize=10).encode(
        text=alt.Text("rho:Q", format=".2f"),
        color=alt.condition(
            "abs(datum.rho) > 0.55", alt.value("#FFFFFF"), alt.value(INK)
        ),
    )
    return (base + text).properties(height=alt.Step(42))
