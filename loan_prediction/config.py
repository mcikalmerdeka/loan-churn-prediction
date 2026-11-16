"""Filesystem paths used by the loan prediction app.

Keeping these in one module avoids path arithmetic in individual views and
keeps the app working regardless of the current working directory.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

TRAINING_DATA_PATH = DATA_DIR / "Training Data.csv"
REFERENCE_DATA_PATH = DATA_DIR / "df_model_rewrite.csv"
BATCH_EXAMPLE_PATH = DATA_DIR / "batch_example.csv"
MODEL_PATH = MODELS_DIR / "tuned_knn_model.joblib"
