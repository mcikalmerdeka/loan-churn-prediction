"""Regression tests for the shared visualization helpers."""

from __future__ import annotations

import unittest
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from utils import plot_dynamic_hisplots_kdeplots

warnings.filterwarnings(
    "ignore", message="FigureCanvasAgg is non-interactive", category=UserWarning
)


class DynamicPlotColumnTests(unittest.TestCase):
    def tearDown(self) -> None:
        plt.close("all")

    def test_kdeplot_accepts_pandas_index(self) -> None:
        # df.columns is a pandas Index, not a list. It must not be evaluated
        # in a boolean context.
        frame = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                "b": [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            }
        )

        plot_dynamic_hisplots_kdeplots(
            frame,
            col_series=frame.columns,
            plot_type="kdeplot",
            ncols=2,
            figsize=(4, 3),
        )


if __name__ == "__main__":
    unittest.main()
