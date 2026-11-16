"""Regression tests for the loan app's honest holdout metrics."""

from __future__ import annotations

import unittest
import warnings

import joblib
import numpy as np
import pandas as pd

from loan_prediction.analytics import LEAKY_CV_REFERENCE, model_metrics
from loan_prediction.config import MODEL_PATH, REFERENCE_DATA_PATH
from loan_prediction.pipeline import LoanPipeline


class ModelMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        warnings.filterwarnings("ignore")
        cls.model = joblib.load(MODEL_PATH)
        cls.reference = pd.read_csv(REFERENCE_DATA_PATH)
        cls.pipeline = LoanPipeline.from_reference(REFERENCE_DATA_PATH)
        cls.metrics = model_metrics(cls.model, cls.pipeline, cls.reference)

    def test_holdout_size_matches_the_notebook_split(self) -> None:
        self.assertEqual(self.metrics["n_test"], 10_798)

    def test_confusion_matrix_accounts_for_every_test_row(self) -> None:
        matrix = self.metrics["confusion_matrix"]

        self.assertEqual(matrix.shape, (2, 2))
        self.assertEqual(int(matrix.sum()), self.metrics["n_test"])

    def test_honest_metrics_are_not_the_leaky_reference(self) -> None:
        # Guards against accidentally presenting the leaky CV number as the
        # holdout result. The current artifact is close to random on real rows.
        self.assertLess(self.metrics["roc_auc"], LEAKY_CV_REFERENCE["roc_auc"])
        self.assertLess(self.metrics["recall"], LEAKY_CV_REFERENCE["recall"])

    def test_probabilities_are_finite_and_bounded(self) -> None:
        probabilities = self.metrics["probabilities"]

        self.assertEqual(len(probabilities), self.metrics["n_test"])
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertGreaterEqual(float(probabilities.min()), 0.0)
        self.assertLessEqual(float(probabilities.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
