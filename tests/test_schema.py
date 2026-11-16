"""Import and schema regression tests for the Streamlit package."""

from __future__ import annotations

import unittest

import pandas as pd

from loan_prediction.config import REFERENCE_DATA_PATH
from loan_prediction.schema import (
    CITY_GROUP_MAP,
    MODEL_FEATURES,
    REQUIRED_INPUT_COLUMNS,
    TARGET_COLUMN,
    get_feature_definitions,
)


class SchemaTests(unittest.TestCase):
    def test_model_feature_order_matches_reference_frame(self) -> None:
        reference = pd.read_csv(REFERENCE_DATA_PATH)
        expected = [
            column for column in reference.columns if column != TARGET_COLUMN
        ]

        self.assertEqual(MODEL_FEATURES, expected)
        self.assertEqual(len(MODEL_FEATURES), 20)

    def test_city_group_keeps_urban_priority(self) -> None:
        # Bhopal and Meerut appear in both the urban and suburban notebook
        # lists; the notebook checks urban first, so the model expects urban.
        self.assertEqual(CITY_GROUP_MAP["bhopal"], "urban")
        self.assertEqual(CITY_GROUP_MAP["meerut"], "urban")
        self.assertEqual(CITY_GROUP_MAP["satna"], "suburban")

    def test_every_required_input_has_a_definition(self) -> None:
        definitions = get_feature_definitions()

        for column in REQUIRED_INPUT_COLUMNS:
            with self.subTest(column=column):
                self.assertIn(column, definitions)


class ImportTests(unittest.TestCase):
    def test_package_modules_import(self) -> None:
        import loan_prediction.analytics  # noqa: F401
        import loan_prediction.config  # noqa: F401
        import loan_prediction.pipeline  # noqa: F401
        import loan_prediction.schema  # noqa: F401
        import loan_prediction.ui  # noqa: F401
        import loan_prediction.views  # noqa: F401


if __name__ == "__main__":
    unittest.main()
