# Loan Prediction Based on Customer Behavior

![Project Header](assets/Project%20Header.jpg)

A machine learning solution for automated credit risk assessment using customer behavior and demographic data.

## Project Overview

End-to-end data science project that analyzes customer demographics, income, assets, and employment data to predict loan default risk. The repository follows the modular layout used by the `clicked_ads/` reference project:

- `utils/` is the generic, reusable data-science library (EDA, preprocessing, feature selection, evaluation, tuning).
- `loan_prediction/` holds every dataset-specific rule: schema, preprocessing pipeline, analytics, charts, and Streamlit views.
- `main.py` is a thin entry point that loads the persisted model and renders four tabs.
- `notebook.ipynb` keeps the analysis narrative and uses `utils` instead of duplicating helpers.

## Model Summary

- **Algorithm:** K-Nearest Neighbours (k=19, Manhattan distance, distance weights)
- **Dataset size:** 252,000 loan applications; 43,190 unique rows after de-duplication; 32,392 in the training split
- **Features:** 8 raw inputs expanded into 20 encoded features
- **Target:** `Risk_Flag` (1 = Default, 0 = Not Default)

### Test-Set Metrics

| Metric | Value |
|---|---:|
| Accuracy | 53.70% |
| Precision | 17.47% |
| Recall | 36.76% |
| F1 | 23.69% |
| ROC-AUC | 0.440 |

Metrics are computed on a stratified 25% test split (10,798 applicants, `random_state=42`). See [docs/model-evaluation-caveat.md](docs/model-evaluation-caveat.md) for the evaluation methodology and a documented cross-validation caveat.

## Project Structure

```
├── data/
│   ├── Training Data.csv
│   ├── df_model_rewrite.csv      # encoded reference frame used by the model
│   └── batch_example.csv
├── docs/
│   └── model-evaluation-caveat.md
├── loan_prediction/              # application package
│   ├── config.py                 # filesystem paths
│   ├── schema.py                 # column names, mappings, feature definitions
│   ├── pipeline.py               # raw row -> 20-feature model matrix
│   ├── analytics.py              # test-set metrics and dataset analytics
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

- **Predict** — score one applicant from the eight fields that affect the model. Marital status, house ownership, and current-job tenure are not requested because the deployed model ignores them.
- **Batch** — upload a CSV or use the five-row example, score every row, inspect the preprocessing trace, and download predictions. If the file includes `Risk_Flag`, the app also reports accuracy, recall, and ROC-AUC for that batch.
- **Insights** — test-set metrics, confusion matrix, feature relevance, target balance, default rates by profession group, distributions, and correlation.
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

The notebook additionally uses imbalanced-learn for its evaluation section. It is a notebook-only extra and is not required to run the Streamlit app.

## Business Problem

A lending company needed to improve credit risk assessment and reduce financial losses from loan defaults. The manual process was slow and produced inconsistent decisions.

### Business Metrics

- **Default rate (%) [MAIN]:** the share of applicants who fail to repay. The source data has a 19.55% default rate.
- **Approval time [SECONDARY]:** the time taken to process an application. The app demonstrates seconds-level scoring, but this is a pipeline capability, not a measured production impact.

The app is a demonstration of the scoring workflow. It does not claim a measured reduction in the default rate.

## Known Limitations

- **Dataset scope:** the model is trained on historical applicant data; patterns can change over time.
- **Unused fields:** `Marital_Status`, `House_Ownership`, and `Current_Job_Years` are collected but ignored by the model.
- **Version mismatch:** the artifact was pickled with scikit-learn 1.4.0 and is loaded under 1.8.0. The app suppresses the compatibility warning; re-running the notebook re-saves the artifact under the current version.
- **Intended use:** predictions support exploration and pipeline testing, not real lending decisions.

## Try the Live App

Link to the deployed app: [Loan Prediction Based on Customer Behavior](https://loan-churn-prediction-y5rm4xjsqyommkzjtjjejy.streamlit.app/)

## Author

**Muhammad Cikal Merdeka** | Data Analyst / Data Scientist

- [GitHub](https://github.com/mcikalmerdeka)
- [LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)
- [Email](mailto:mcikalmerdeka@gmail.com)
