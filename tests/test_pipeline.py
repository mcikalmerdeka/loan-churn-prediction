"""Regression tests for the loan preprocessing pipeline.

The deployed KNN is a bare estimator: every preprocessing step has to be
reproduced exactly. These tests pin the behaviour that matters for parity:

* column renames / text cleaning
* the 20 model features, in the model's order
* scaler values from the notebook's training split
* actionable failures for malformed input
* the two examples advertised in the app
"""

from __future__ import annotations

import math
import unittest
import warnings

import joblib
import numpy as np
import pandas as pd

from loan_prediction.config import BATCH_EXAMPLE_PATH, MODEL_PATH, REFERENCE_DATA_PATH
from loan_prediction.pipeline import LoanPipeline, initial_data_transform
from loan_prediction.schema import MODEL_FEATURES


class InitialTransformTests(unittest.TestCase):
    def test_renames_cleans_and_lowercases_categorical_values(self) -> None:
        raw = pd.DataFrame(
            {
                "CITY": ["Delhi_City"],
                "STATE": ["Tamil_Nadu"],
                "Married/Single": ["single"],
                "Profession": ["software_developer[2]"],
                "Car_Ownership": [" Yes "],
                "Income": [1_000_000],
                "Age": [30],
                "Experience": [5],
                "CURRENT_HOUSE_YRS": [2],
            }
        )

        transformed = initial_data_transform(raw)

        self.assertIn("City", transformed.columns)
        self.assertIn("State", transformed.columns)
        self.assertIn("Marital_Status", transformed.columns)
        self.assertIn("Current_House_Years", transformed.columns)
        self.assertEqual(transformed["City"].iloc[0], "delhi city")
        self.assertEqual(transformed["State"].iloc[0], "tamil nadu")
        self.assertEqual(transformed["Profession"].iloc[0], "software developer")
        self.assertEqual(transformed["Car_Ownership"].iloc[0], "yes")

    def test_accepts_batch_example_typo_and_drops_id(self) -> None:
        raw = pd.DataFrame(
            {
                "Id": [1],
                "Marita_Status": ["single"],
                "CITY": ["rewa"],
                "STATE": ["madhya pradesh"],
                "Profession": ["police officer"],
                "Car_Ownership": ["no"],
                "Income": [3_000_000],
                "Age": [25],
                "Experience": [2],
                "CURRENT_HOUSE_YRS": [10],
            }
        )

        transformed = initial_data_transform(raw)

        self.assertNotIn("Id", transformed.columns)
        self.assertIn("Marital_Status", transformed.columns)


class LoanPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.pipeline = LoanPipeline.from_reference(REFERENCE_DATA_PATH)
        cls.batch = pd.read_csv(BATCH_EXAMPLE_PATH)

    def test_transform_returns_model_feature_matrix(self) -> None:
        transformed = self.pipeline.transform(self.batch)

        self.assertEqual(list(transformed.columns), MODEL_FEATURES)
        self.assertEqual(len(transformed), len(self.batch))
        self.assertFalse(transformed.isna().any().any())
        values = transformed.to_numpy().ravel()
        self.assertTrue(all(math.isfinite(value) for value in values))

    def test_passthrough_features_keep_their_values(self) -> None:
        raw = self.batch.iloc[[0]]
        transformed = self.pipeline.transform(raw)

        expected_income = self.pipeline.scalers["minmax"].transform(
            pd.DataFrame([[raw["Income"].iloc[0]]], columns=["Income"])
        )[0, 0]
        ratio = raw["Experience"].iloc[0] / raw["Age"].iloc[0]
        expected_ratio = self.pipeline.scalers["standard"].transform(
            pd.DataFrame(
                [[np.log1p(ratio)]], columns=["Experience_Age_Ratio"]
            )
        )[0, 0]

        self.assertAlmostEqual(
            float(transformed["Income"].iloc[0]), expected_income, places=6
        )
        self.assertAlmostEqual(
            float(transformed["Experience_Age_Ratio"].iloc[0]),
            expected_ratio,
            places=6,
        )
        self.assertAlmostEqual(
            float(transformed["Current_House_Years"].iloc[0]),
            float(raw["Current_House_Years"].iloc[0]),
            places=6,
        )

    def test_scalers_match_notebook_training_split(self) -> None:
        minmax = self.pipeline.scalers["minmax"]
        standard = self.pipeline.scalers["standard"]

        self.assertAlmostEqual(float(minmax.data_min_[0]), 10_675.0, places=2)
        self.assertAlmostEqual(float(minmax.data_max_[0]), 9_999_938.0, places=2)
        self.assertAlmostEqual(float(standard.mean_[0]), 0.19741652, places=8)
        self.assertAlmostEqual(float(standard.scale_[0]), 0.13617595, places=8)

    def test_missing_required_column_is_actionable(self) -> None:
        with self.assertRaisesRegex(ValueError, "Income"):
            self.pipeline.transform(self.batch.drop(columns=["Income"]))

    def test_unknown_profession_falls_back_to_miscellaneous(self) -> None:
        row = self.batch.iloc[[0]].copy()
        row["Profession"] = "astronaut"

        transformed = self.pipeline.transform(row)

        prof_columns = [column for column in MODEL_FEATURES if column.startswith("Prof_")]
        other_professions = [
            column for column in prof_columns if column != "Prof_miscellaneous"
        ]
        self.assertTrue((transformed[other_professions].iloc[0] == 0).all())
        self.assertEqual(float(transformed["Prof_miscellaneous"].iloc[0]), 1.0)

    def test_advertised_examples_keep_their_labels(self) -> None:
        warnings.filterwarnings("ignore")
        model = joblib.load(MODEL_PATH)
        examples = {
            0: {
                "Income": 8_877_323,
                "Age": 60,
                "Experience": 12,
                "Car_Ownership": "yes",
                "Profession": "software developer",
                "City": "rewa",
                "State": "tamil nadu",
                "Current_House_Years": 12,
            },
            1: {
                "Income": 3_234_134,
                "Age": 25,
                "Experience": 2,
                "Car_Ownership": "no",
                "Profession": "police officer",
                "City": "rewa",
                "State": "madhya pradesh",
                "Current_House_Years": 10,
            },
        }

        for expected, row in examples.items():
            with self.subTest(expected=expected):
                transformed = self.pipeline.transform(pd.DataFrame([row]))
                self.assertEqual(int(model.predict(transformed)[0]), expected)


if __name__ == "__main__":
    unittest.main()
