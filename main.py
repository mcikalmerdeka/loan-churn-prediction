"""Loan Default Risk Analyzer — Streamlit entry point."""

from __future__ import annotations

import warnings

import joblib
import pandas as pd
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

from loan_prediction.analytics import model_metrics
from loan_prediction.config import MODEL_PATH, REFERENCE_DATA_PATH, TRAINING_DATA_PATH
from loan_prediction.pipeline import LoanPipeline, initial_data_transform
from loan_prediction.views import (
    render_about_tab,
    render_batch_tab,
    render_insights_tab,
    render_predict_tab,
)

# The artifact was pickled with scikit-learn 1.4.0 while the app runs 1.8.0.
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

st.set_page_config(
    page_title="Loan Default Risk Analyzer",
    page_icon=":material/account_balance:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_training() -> pd.DataFrame:
    """Load, clean, and de-duplicate the raw training data once per session."""
    raw = pd.read_csv(TRAINING_DATA_PATH)
    cleaned = initial_data_transform(raw)
    return cleaned.drop_duplicates().reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_artifacts():
    """Load the model, reference frame, fitted pipeline, and test-set metrics."""
    model = joblib.load(MODEL_PATH)
    reference = pd.read_csv(REFERENCE_DATA_PATH)
    training = load_training()
    pipeline = LoanPipeline.from_reference(REFERENCE_DATA_PATH)
    metrics = model_metrics(model, pipeline, reference)
    training_groups = pipeline.engineer(training)
    return model, reference, training, training_groups, pipeline, metrics


if not MODEL_PATH.exists():
    st.error(f"Model artifact not found at {MODEL_PATH}.")
    st.stop()

model, reference, training, training_groups, pipeline, metrics = load_artifacts()

st.title("Loan Default Risk Analyzer")
st.caption(
    "Score loan applicants, inspect the preprocessing pipeline, and review the "
    "deployed K-Nearest Neighbours model."
)

with st.container(horizontal=True):
    st.badge(
        "K-Nearest Neighbours",
        icon=":material/model_training:",
        color="blue",
    )
    st.badge(
        "20 model features",
        icon=":material/tune:",
        color="gray",
    )
    st.badge(
        f"{len(training):,} unique applicants",
        icon=":material/database:",
        color="gray",
    )

tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["Predict", "Batch", "Insights", "About"]
)

with tab_predict:
    render_predict_tab(model, pipeline, training)

with tab_batch:
    render_batch_tab(model, pipeline)

with tab_insights:
    render_insights_tab(reference, training, training_groups, metrics)

with tab_about:
    render_about_tab(reference, training, metrics)
