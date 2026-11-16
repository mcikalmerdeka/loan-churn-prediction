"""Project-specific glue for the loan prediction app.

``utils/`` is the generic data-science library; everything that is specific to
this dataset and the persisted KNN lives here. The adapter fits its state from
``data/df_model_rewrite.csv``, the same reference frame the notebook exported,
so there is no extra artifact to keep in sync with the deployed model.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from utils import feature_encoding, feature_scaling

from .config import REFERENCE_DATA_PATH
from .schema import (
    CAR_OWNERSHIP_VALUES,
    CITY_GROUP_MAP,
    CITY_GROUPS,
    COLUMN_RENAMES,
    DEFAULT_CITY_GROUP,
    DEFAULT_STATE_ZONE,
    GENERATION_CATEGORIES,
    ID_COLUMN,
    MODEL_FEATURES,
    PROFESSION_GROUPS,
    PROFESSION_TO_GROUP,
    REQUIRED_INPUT_COLUMNS,
    STATE_ZONE_MAP,
    STATE_ZONES,
    TARGET_COLUMN,
)

# The notebook scales Income with MinMax and Experience_Age_Ratio with
# log1p followed by StandardScaler. Fitting these on the notebook's 75%
# training split reproduces models/feature_scalers.joblib exactly.
SCALING_CONFIG = {
    "minmax": {"columns": ["Income"]},
    "standard": {"columns": ["Experience_Age_Ratio"], "apply_log": True},
}

TEST_SIZE = 0.25
RANDOM_STATE = 42


def initial_data_transform(data: pd.DataFrame) -> pd.DataFrame:
    """Normalise raw headers and categorical text.

    Accepts the Kaggle header (``CITY``, ``Married/Single``, ...), the
    canonical app header, and the ``Marita_Status`` typo from
    ``batch_example.csv``.

    The notebook's ``delhi city`` -> ``new delhi`` replacement happened after
    ``df_encoding`` was copied, so the deployed model was trained with
    ``delhi city`` mapped to the rural group. We mirror training here; making
    both names metro would require retraining the artifact.
    """
    data = data.copy().rename(columns=COLUMN_RENAMES)

    # A file may contain both the raw and canonical spelling of a column.
    data = data.loc[:, ~data.columns.duplicated()]

    if ID_COLUMN in data.columns:
        data = data.drop(columns=ID_COLUMN)

    for column in ("Profession", "State", "City"):
        if column not in data.columns:
            continue
        cleaned = data[column].astype("string")
        cleaned = cleaned.str.replace(r"\[\d+\]", "", regex=True)
        cleaned = cleaned.str.replace("_", " ", regex=False)
        cleaned = cleaned.str.replace(",", " ", regex=False)
        cleaned = cleaned.str.replace(r"\s+", " ", regex=True)
        data[column] = cleaned.str.strip().str.lower()

    if "Car_Ownership" in data.columns:
        data["Car_Ownership"] = (
            data["Car_Ownership"].astype("string").str.strip().str.lower()
        )

    return data


def assign_generation(age: float) -> str:
    """Map an age to the notebook's generation buckets."""
    if age <= 27:
        return "Generation Z"
    if age <= 43:
        return "Generation Millenials"
    if age < 59:
        return "Generation X"
    if age < 69:
        return "Boomers II"
    if age <= 78:
        return "Boomers I"
    return "Other"


def _encoder_fit_frame() -> pd.DataFrame:
    """Frame containing every category combination the encoders must know.

    Fitting on the raw training CSV would require a 20 MB read just to build
    the pipeline. A Cartesian product of the documented categories is smaller,
    deterministic, and covers every value the form can produce.
    """
    rows = (
        {
            "Prof": profession,
            "State": state,
            "City": city,
            "Car_Ownership": car,
            "Generation": generation,
            # Passthrough columns must be present when the encoders are fitted;
            # otherwise feature_encoding records only the encoded columns and
            # drops these on transform. The scalers are fitted separately, so
            # the placeholder values never affect them.
            "Income": 1.0,
            "Current_House_Years": 1.0,
            "Experience_Age_Ratio": 1.0,
        }
        for profession, state, city, car, generation in product(
            PROFESSION_GROUPS,
            STATE_ZONES,
            CITY_GROUPS,
            CAR_OWNERSHIP_VALUES,
            GENERATION_CATEGORIES,
        )
    )
    return pd.DataFrame(rows)


@dataclass
class LoanPipeline:
    """Turns raw applicant rows into the exact matrix the KNN expects."""

    encoders: Dict[str, Any]
    scalers: Dict[str, Any]
    feature_columns: List[str]

    @classmethod
    def from_reference(
        cls,
        reference_path: Union[str, Path] = REFERENCE_DATA_PATH,
    ) -> "LoanPipeline":
        """Fit encoders and scalers from the repository's reference frame."""
        reference = pd.read_csv(reference_path)

        missing = [column for column in MODEL_FEATURES if column not in reference.columns]
        if missing:
            raise ValueError(
                "Reference frame is missing model features: " + ", ".join(missing)
            )

        features = reference[MODEL_FEATURES]
        target = reference[TARGET_COLUMN]

        X_train, _X_test, _y_train, _y_test = train_test_split(
            features,
            target,
            test_size=TEST_SIZE,
            stratify=target,
            random_state=RANDOM_STATE,
        )

        _, scalers = feature_scaling(X_train, scaling_config=SCALING_CONFIG)

        fit_frame = _encoder_fit_frame()
        _, encoders = feature_encoding(
            fit_frame,
            ordinal_columns=["Generation"],
            ordinal_categories={"Generation": GENERATION_CATEGORIES},
            nominal_columns=["Prof", "State", "City"],
            binary_columns=["Car_Ownership"],
            drop_first=True,
            handle_unknown="ignore",
        )

        return cls(
            encoders=encoders,
            scalers=scalers,
            feature_columns=list(MODEL_FEATURES),
        )

    def engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate raw inputs and derive the model's engineered features."""
        missing = [
            column for column in REQUIRED_INPUT_COLUMNS if column not in data.columns
        ]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

        null_columns = [
            column
            for column in REQUIRED_INPUT_COLUMNS
            if data[column].isna().any()
        ]
        if null_columns:
            raise ValueError(
                "Missing values in required columns: " + ", ".join(null_columns)
            )

        engineered = data.copy()

        numeric_columns = ["Income", "Age", "Experience", "Current_House_Years"]
        for column in numeric_columns:
            engineered[column] = pd.to_numeric(engineered[column], errors="raise")

        if (engineered["Age"] <= 0).any():
            raise ValueError("Age must be greater than zero")
        if (engineered["Income"] < 0).any():
            raise ValueError("Income cannot be negative")
        if (engineered["Experience"] < 0).any():
            raise ValueError("Experience cannot be negative")
        if (engineered["Current_House_Years"] < 0).any():
            raise ValueError("Current_House_Years cannot be negative")

        invalid_car = sorted(
            set(engineered["Car_Ownership"].dropna()) - set(CAR_OWNERSHIP_VALUES)
        )
        if invalid_car:
            raise ValueError(
                "Car_Ownership must be 'no' or 'yes'; found: "
                + ", ".join(map(str, invalid_car))
            )

        engineered["Generation"] = engineered["Age"].apply(assign_generation)
        engineered["Experience_Age_Ratio"] = (
            engineered["Experience"] / engineered["Age"]
        )

        engineered["Prof"] = (
            engineered["Profession"]
            .str.strip()
            .str.lower()
            .map(PROFESSION_TO_GROUP)
            # Unknown occupations fall back to the catch-all group. The form
            # only offers known values; this protects batch uploads.
            .fillna("miscellaneous")
        )
        engineered["State"] = (
            engineered["State"]
            .str.strip()
            .str.lower()
            .map(STATE_ZONE_MAP)
            .fillna(DEFAULT_STATE_ZONE)
        )
        engineered["City"] = (
            engineered["City"]
            .str.strip()
            .str.lower()
            .map(CITY_GROUP_MAP)
            .fillna(DEFAULT_CITY_GROUP)
        )

        return engineered

    def encode(self, data: pd.DataFrame) -> pd.DataFrame:
        """Encode categoricals and align to the model's input columns."""
        encoded, _ = feature_encoding(
            data,
            ordinal_columns=["Generation"],
            ordinal_categories={"Generation": GENERATION_CATEGORIES},
            nominal_columns=["Prof", "State", "City"],
            binary_columns=["Car_Ownership"],
            drop_first=True,
            handle_unknown="ignore",
            encoders=self.encoders,
        )
        return encoded.reindex(columns=self.feature_columns, fill_value=0)

    def scale(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply the notebook's two fitted scalers."""
        scaled, _ = feature_scaling(
            data,
            scaling_config=SCALING_CONFIG,
            scaler=self.scalers,
        )
        return scaled

    def transform_steps(self, raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Run the pipeline and return each stage for the batch preview."""
        cleaned = initial_data_transform(raw)
        engineered = self.engineer(cleaned)
        encoded = self.encode(engineered)
        scaled = self.scale(encoded)
        return {
            "cleaned": cleaned,
            "engineered": engineered,
            "encoded": encoded,
            "scaled": scaled,
        }

    def transform(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Run clean -> engineer -> encode -> scale in one call."""
        return self.transform_steps(raw)["scaled"]

    def scale_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Scale an already-encoded frame, such as the notebook reference."""
        missing = [
            column for column in self.feature_columns if column not in frame.columns
        ]
        if missing:
            raise ValueError("Missing model features: " + ", ".join(missing))
        return self.scale(frame[self.feature_columns])
