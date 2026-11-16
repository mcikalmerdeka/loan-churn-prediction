"""Individual applicant prediction view."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from loan_prediction.analytics import (
    WEAK_SIGNAL_AUC,
    confidence_label,
    outcome_label,
    predict_frame,
)
from loan_prediction.schema import CAR_OWNERSHIP_VALUES
from loan_prediction.ui import fmt_percent, vote_share_progress

INCOME_KEY = "predict_income"
AGE_KEY = "predict_age"
EXPERIENCE_KEY = "predict_experience"
HOUSE_YEARS_KEY = "predict_house_years"
CAR_KEY = "predict_car"
PROFESSION_KEY = "predict_profession"
STATE_KEY = "predict_state"
CITY_KEY = "predict_city"
RESULT_KEY = "predict_result"

EXAMPLE_NOT_DEFAULT = {
    "Income": 8_877_323.0,
    "Age": 60.0,
    "Experience": 12.0,
    "Current_House_Years": 12.0,
    "Car_Ownership": "yes",
    "Profession": "software developer",
    "State": "tamil nadu",
    "City": "rewa",
}

EXAMPLE_DEFAULT = {
    "Income": 3_234_134.0,
    "Age": 25.0,
    "Experience": 2.0,
    "Current_House_Years": 10.0,
    "Car_Ownership": "no",
    "Profession": "police officer",
    "State": "madhya pradesh",
    "City": "rewa",
}


def _load_example(example: dict) -> None:
    st.session_state[INCOME_KEY] = example["Income"]
    st.session_state[AGE_KEY] = example["Age"]
    st.session_state[EXPERIENCE_KEY] = example["Experience"]
    st.session_state[HOUSE_YEARS_KEY] = example["Current_House_Years"]
    st.session_state[CAR_KEY] = example["Car_Ownership"]
    st.session_state[PROFESSION_KEY] = example["Profession"]
    st.session_state[STATE_KEY] = example["State"]
    st.session_state[CITY_KEY] = example["City"]
    st.session_state.pop(RESULT_KEY, None)


def _bounds(data: pd.DataFrame, column: str) -> tuple[float, float, float]:
    series = data[column]
    return float(series.min()), float(series.max()), float(series.mean())


def _options(data: pd.DataFrame, column: str, preferred: str) -> tuple[list[str], int]:
    options = sorted(data[column].dropna().unique().tolist())
    index = options.index(preferred) if preferred in options else 0
    return options, index


def _render_result(result: dict, metrics: dict) -> None:
    st.markdown("### Prediction")

    with st.container(horizontal=True):
        st.metric(
            "Predicted outcome",
            outcome_label(result["prediction"]),
            border=True,
        )
        st.metric(
            "Default vote share",
            fmt_percent(result["default_share"]),
            border=True,
        )
        st.metric(
            "Confidence",
            confidence_label(result["default_share"]),
            border=True,
        )

    vote_share_progress(result["default_share"])
    st.caption(
        "Vote share is the share of the 19 nearest neighbours in the Default "
        "class. It is not calibrated to the real default rate because the "
        "model was trained on a balanced SMOTE sample."
    )

    if metrics["roc_auc"] < WEAK_SIGNAL_AUC:
        st.error(
            f"This model's honest holdout ROC-AUC is {metrics['roc_auc']:.2f} "
            "(below random). Treat the result as a demo, not a lending decision."
        )

    inputs = result["inputs"]
    st.caption(
        f"Scored {inputs['Profession']} · {inputs['City']}, {inputs['State']} · "
        f"age {inputs['Age']:.0f} · income {inputs['Income']:,.0f}"
    )

    with st.expander("Model input (20 features)"):
        st.dataframe(result["model_input"], hide_index=True)


def render_predict_tab(model, pipeline, training: pd.DataFrame, metrics: dict) -> None:
    st.markdown(
        "Score one applicant from the fields that actually influence the model. "
        "Marital status, house ownership, and current-job tenure are collected "
        "in the source data but are not used by the deployed KNN."
    )

    if metrics["roc_auc"] < WEAK_SIGNAL_AUC:
        st.warning(
            "**Model warning:** the deployed KNN scores below random on real "
            f"holdout rows (ROC-AUC {metrics['roc_auc']:.2f}). This app is a "
            "demonstration of the pipeline, not a production credit model."
        )

    with st.container(horizontal=True):
        st.button(
            "Load Not Default example",
            icon=":material/check_circle:",
            type="tertiary",
            on_click=_load_example,
            args=(EXAMPLE_NOT_DEFAULT,),
            key="load_not_default_example",
        )
        st.button(
            "Load Default example",
            icon=":material/cancel:",
            type="tertiary",
            on_click=_load_example,
            args=(EXAMPLE_DEFAULT,),
            key="load_default_example",
        )

    income_min, income_max, income_mean = _bounds(training, "Income")
    age_min, age_max, age_mean = _bounds(training, "Age")
    experience_min, experience_max, experience_mean = _bounds(
        training, "Experience"
    )
    house_min, house_max, house_mean = _bounds(training, "Current_House_Years")

    professions, profession_index = _options(
        training, "Profession", "software developer"
    )
    states, state_index = _options(training, "State", "tamil nadu")
    cities, city_index = _options(training, "City", "rewa")

    with st.form("predict_form", border=True):
        st.markdown("**Applicant profile**")
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown("**Financial**")
            income = st.number_input(
                "Annual income",
                min_value=income_min,
                max_value=income_max,
                value=float(round(income_mean / 100_000) * 100_000),
                step=100_000.0,
                format="%.0f",
                help="Annual income reported by the applicant.",
                key=INCOME_KEY,
            )
            house_years = st.number_input(
                "Years in current residence",
                min_value=house_min,
                max_value=house_max,
                value=float(round(house_mean)),
                step=1.0,
                format="%.0f",
                help="How long the applicant has lived at the current address.",
                key=HOUSE_YEARS_KEY,
            )
            car_ownership = st.selectbox(
                "Car ownership",
                CAR_OWNERSHIP_VALUES,
                format_func=str.title,
                help="Whether the applicant owns a vehicle.",
                key=CAR_KEY,
            )
            profession = st.selectbox(
                "Profession",
                professions,
                index=profession_index,
                format_func=str.title,
                help="Occupation recorded in the training data.",
                key=PROFESSION_KEY,
            )

        with right:
            st.markdown("**Demographic**")
            age = st.number_input(
                "Age",
                min_value=age_min,
                max_value=age_max,
                value=float(round(age_mean)),
                step=1.0,
                format="%.0f",
                help="Applicant age in years.",
                key=AGE_KEY,
            )
            experience = st.number_input(
                "Total professional experience (years)",
                min_value=experience_min,
                max_value=experience_max,
                value=float(round(experience_mean)),
                step=1.0,
                format="%.0f",
                help="Total years of professional experience.",
                key=EXPERIENCE_KEY,
            )
            state = st.selectbox(
                "State",
                states,
                index=state_index,
                format_func=str.title,
                help="State of residence.",
                key=STATE_KEY,
            )
            city = st.selectbox(
                "City",
                cities,
                index=city_index,
                format_func=str.title,
                help="City of residence.",
                key=CITY_KEY,
            )

        submitted = st.form_submit_button(
            "Predict loan status",
            type="primary",
            icon=":material/bolt:",
            key="predict_submit",
        )

    if submitted:
        raw = pd.DataFrame(
            [
                {
                    "Income": float(income),
                    "Age": float(age),
                    "Experience": float(experience),
                    "Current_House_Years": float(house_years),
                    "Car_Ownership": car_ownership,
                    "Profession": profession,
                    "State": state,
                    "City": city,
                }
            ]
        )
        try:
            model_input = pipeline.transform(raw)
            predictions, probabilities = predict_frame(model, model_input)
        except Exception as exc:  # surface the real reason instead of a blank page
            st.error(f"Could not score this applicant: {exc}")
        else:
            st.session_state[RESULT_KEY] = {
                "prediction": int(predictions[0]),
                "default_share": float(probabilities[0]),
                "model_input": model_input,
                "inputs": raw.iloc[0].to_dict(),
            }

    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.caption("Submit the form to score an applicant.")
        return

    _render_result(result, metrics)
