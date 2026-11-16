# Loan Prediction Based on Customer Behavior

![Project Header](assets/Project%20Header.jpg)

A machine learning solution for automated credit risk assessment using customer behavior and demographic data.

> **Model integrity note (2026):** the original project reported 85.88% test recall. That number came from cross-validation after SMOTE had been applied to the whole training set, so synthetic neighbours leaked across folds. On a real, untouched holdout the deployed KNN scores near random. The honest numbers and the corrected evaluation are documented below and in the app's **Insights** tab.

## Project Overview

End-to-end data science project that analyzes customer demographics, income, assets, and employment data to predict loan default risk. The repository now follows the modular layout used by the `clicked_ads/` reference project:

- `utils/` is the generic, reusable data-science library (EDA, preprocessing, feature selection, evaluation, tuning).
- `loan_prediction/` holds every dataset-specific rule: schema, preprocessing pipeline, analytics, charts, and Streamlit views.
- `main.py` is a thin entry point that loads the persisted model and renders four tabs.
- `notebook.ipynb` keeps the analysis narrative and uses `utils` instead of duplicating helpers.

## Key Results

| Evaluation | Recall | ROC-AUC |
|---|---:|---:|
| Leaky CV (SMOTE before folds) — **do not report** | 85.88% | 0.801 |
| Honest 25% holdout (deployed KNN) | 36.76% | 0.440 |
| Honest 5-fold CV, same KNN, no resampling | 1.18% | 0.453 |
| Honest 5×3 CV, SMOTE inside folds, same KNN | 28.80% ± 0.97 | 0.451 ± 0.005 |
| Honest 5-fold CV, balanced logistic regression | 47.66% | 0.508 |

The honest holdout is a single stratified 25% split (`random_state=42`). The leak-free CV row applies SMOTE inside each training fold, never before it. All honest estimates sit near ROC-AUC 0.5, which means the available features carry little usable signal for `Risk_Flag`. This is a property of the dataset, not a tuning failure.

- **Model algorithm:** K-Nearest Neighbours (k=19, Manhattan distance, distance weights)
- **Dataset size:** 252,000 loan applications; 43,190 unique rows after de-duplication; 32,392 in the notebook's training split
- **Primary goal:** high recall to minimise false negatives (approving high-risk borrowers). The deployed artifact does not achieve that goal reliably.

## Project Structure

```
├── data/
│   ├── Training Data.csv
│   ├── df_model_rewrite.csv      # encoded reference frame used by the model
│   └── batch_example.csv
├── docs/
├── loan_prediction/              # application package
│   ├── config.py                 # filesystem paths
│   ├── schema.py                 # column names, mappings, feature definitions
│   ├── pipeline.py               # raw row -> 20-feature model matrix
│   ├── analytics.py              # honest holdout metrics and dataset analytics
│   ├── ui.py                     # Altair chart helpers
│   └── views/                    # Predict, Batch, Insights, About
├── models/
│   ├── tuned_knn_model.joblib
│   └── feature_scalers.joblib    # reference copy of the notebook scalers
├── reference_revamp/             # separate clicked-ads reference implementation
├── tasks/                        # original bootcamp stage submissions
├── tests/                        # pipeline, analytics, and schema regression tests
├── utils/                        # generic reusable data-science library
├── .streamlit/config.toml        # shared theme
├── main.py                       # Streamlit entry point
├── notebook.ipynb                # EDA, preprocessing, modelling, evaluation
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.14+
- uv (recommended) or pip

### Installation

```bash
git clone https://github.com/mcikalmerdeka/Loan-Prediction-Based-on-Costumer-Behaviour.git
cd Loan-Prediction-Based-on-Costumer-Behaviour

python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
# or
uv sync
```

### Run the App

```bash
streamlit run main.py
```

Access the app at `http://localhost:8501`.

### Run the Tests

```bash
python -m unittest discover -s tests -v
```

## App Features

- **Predict** — score one applicant from the eight fields that actually affect the model. Marital status, house ownership, and current-job tenure are not requested because the deployed model ignores them.
- **Batch** — upload a CSV or use the five-row example, score every row, inspect the preprocessing trace, and download predictions. If the file includes `Risk_Flag`, the app also reports accuracy, recall, and ROC-AUC for that batch.
- **Insights** — honest holdout metrics, confusion matrix, feature relevance, target balance, default rates by profession group, distributions, and correlation. The leaky 85.88% figure is shown side by side with the honest holdout and labelled.
- **About** — purpose, model card, data dictionary, limitations, training-data preview, and data information.

## Preprocessing Pipeline

The persisted KNN is a bare estimator, so the app reproduces the notebook's preprocessing exactly:

1. Rename raw headers (`CITY`, `Married/Single`, `CURRENT_JOB_YRS`, ...).
2. Clean and lower-case `Profession`, `City`, and `State`.
3. Derive `Generation`, `Experience_Age_Ratio`, and the profession / state / city groups.
4. Encode `Generation` ordinally, `Car_Ownership` as binary, and the groups as one-hot (`drop_first=True`).
5. Scale `Income` with MinMax and `Experience_Age_Ratio` with log1p + StandardScaler.

The scalers are fitted at startup from the reference frame's training split; the encoders are fitted from the documented categories in `loan_prediction/schema.py`. The scaler values match `models/feature_scalers.joblib` exactly.

## Technical Stack

- Python 3.14+
- pandas, numpy
- scikit-learn (KNN, preprocessing, evaluation)
- category-encoders
- Streamlit (four-tab app)
- Altair (charts, bundled with Streamlit)
- joblib (model serialization)
- matplotlib / seaborn (notebook)

The notebook additionally uses TensorFlow, SHAP, and imbalanced-learn for the experiment and leak-free evaluation. Those are notebook-only extras and are not required to run the Streamlit app.

## Business Problem

A lending company needed to improve credit risk assessment and reduce financial losses from loan defaults. The manual process was slow and produced inconsistent decisions.

### Business Metrics

- **Default rate (%) [MAIN]:** the share of applicants who fail to repay. The source data has a 19.55% default rate.
- **Approval time [SECONDARY]:** the time taken to process an application. The app demonstrates seconds-level scoring, but this is a pipeline capability, not a measured production impact.

### Corrected Impact Statement

The original write-up claimed the model reduced the default rate to 7.06% and improved approval efficiency by 99.44%. The 7.06% figure came from the leaky CV confusion matrix and is not a valid model benefit. The efficiency figure is an illustrative assumption about process time, not a measurement.

## Known Limitations

- **Weak signal:** honest ROC-AUC is close to 0.5 on real holdout rows.
- **Leakage history:** SMOTE-before-CV inflated the reported recall; the notebook now labels those cells as diagnostics.
- **Uncalibrated score:** the displayed vote share is the KNN neighbour vote, not a calibrated default probability. Training on a balanced SMOTE sample shifts the prior away from the real 19.55%.
- **Unused fields:** `Marital_Status`, `House_Ownership`, and `Current_Job_Years` are collected but ignored.
- **Version mismatch:** the artifact was pickled with scikit-learn 1.4.0 and is loaded under 1.8.0. The app suppresses the compatibility warning; re-running the notebook re-saves the artifact under the current version.
- **Not production-ready:** predictions support exploration and pipeline testing, not real lending decisions.

## Try the Live App

Link to the deployed app: [Loan Prediction Based on Customer Behavior](https://loan-churn-prediction-y5rm4xjsqyommkzjtjjejy.streamlit.app/)

## Author

**Muhammad Cikal Merdeka** | Data Analyst / Data Scientist

- [GitHub](https://github.com/mcikalmerdeka)
- [LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)
- [Email](mailto:mcikalmerdeka@gmail.com)
