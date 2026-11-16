"""Structural regression tests for the upgraded notebook."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebook.ipynb"


def _cell_text(cell: dict) -> str:
    return "".join(cell["source"])


class NotebookMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = cls.notebook["cells"]
        cls.text = "\n".join(_cell_text(cell) for cell in cls.cells)

    def test_code_cells_compile(self) -> None:
        failures = []
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] != "code":
                continue
            source = "\n".join(
                line
                for line in _cell_text(cell).splitlines()
                if not line.lstrip().startswith(("%", "!", "?"))
            )
            if not source.strip():
                continue
            try:
                compile(source, f"cell-{index}", "exec")
            except SyntaxError as exc:
                failures.append((index, exc.msg, exc.lineno))

        self.assertEqual(failures, [])

    def test_duplicated_helpers_are_removed(self) -> None:
        duplicated = [
            "def check_data_information",
            "def describe_numerical_combined",
            "def describe_categorical_combined",
            "def plot_dynamic_hisplots_kdeplots",
            "def identify_distribution_types",
            "def plot_dynamic_boxplots_violinplots",
            "def outlier_detection",
            "def plot_dynamic_countplot",
            "def analyze_categorical_relationships",
            "def calculate_vif",
            "def calculate_feature_importance",
            "def eval_classification",
            "def tune_model",
            "def scale_features",
        ]
        for name in duplicated:
            with self.subTest(name=name):
                self.assertNotIn(name, self.text)

    def test_honest_evaluation_section_exists(self) -> None:
        self.assertIn("## Honest Evaluation (leak-free)", self.text)
        self.assertIn("SMOTE is fitted inside each CV fold", self.text)
        self.assertIn("Honest holdout (deployed KNN)", self.text)

    def test_evaluation_calls_do_not_emit_extra_curves(self) -> None:
        for index, cell in enumerate(self.cells):
            if cell["cell_type"] != "code":
                continue
            source = _cell_text(cell)
            if "eval_classification(" not in source:
                continue
            with self.subTest(cell=index):
                self.assertIn("plot_roc_curve=False", source)
                self.assertIn("plot_pr_curve=False", source)


if __name__ == "__main__":
    unittest.main()
