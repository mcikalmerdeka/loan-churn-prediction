# Loan Prediction Based on Customer Behavior

![Project Header](https://raw.githubusercontent.com/mcikalmerdeka/Loan-Prediction-Based-on-Costumer-Behaviour/refs/heads/main/Assets/Project%20Header.jpg)

A machine learning solution for automated credit risk assessment to predict loan default probability based on customer behavior and demographic data.

**This is the repository for the final project of the Rakamin Data Science Bootcamp Batch 39 that I attended. We are required to go through the end-to-end process of a data science project analyzing factors influencing the default (failure to pay) of a lending company. The original progress and result of each stage can be seen in the "Original Indonesian Version" folder, while what is displayed here is the result of the conversion from Indonesian to English, accompanied by further improvements made after the completion of the final project.**

## Project Overview

End-to-end data science project that analyzes customer demographics, income, assets, and employment data to predict loan default risk. Includes comprehensive EDA, preprocessing pipelines, K-Nearest Neighbors (KNN) classification model, and an interactive Streamlit dashboard for real-time loan risk prediction.

## Key Results

- **Model Algorithm**: K-Nearest Neighbors (KNN) with hyperparameter tuning
- **Performance Metrics**:
  - Training Recall: 97.97% ± 0.06
  - Testing Recall: 85.88% ± 0.23
- **Dataset Size**: 252,000 loan applications (32,000+ after preprocessing)
- **Primary Goal**: High recall to minimize false negatives (approving high-risk borrowers)

## Members of Group 3 (**Dackers**)

1. Muhammad Cikal Merdeka (Leader)
2. Maulana Rifan Haditama
3. Maulana Ibrahim
4. Maria Meidiana Siahaan
5. Revita Rahmadini
6. Nugraha Eddy Wijayanto
7. Mochamad Ali Mustofa

## Project Structure

```
├── analysis/               # Jupyter notebooks for EDA and model training
├── Assets/                 # Project images and media
├── data/                   # Raw and processed datasets
│   ├── Training Data.csv
│   ├── df_model_rewrite.csv
│   └── batch_example.csv
├── models/                 # Trained model artifacts
│   ├── tuned_knn_model.joblib
│   └── fitted_scalers.joblib
├── Original Indonesian Version/  # Original bootcamp deliverables
├── scripts/                # Legacy scripts folder
├── utils/                  # Reusable preprocessing and ML functions
│   ├── preprocessing.py
│   └── feature_definitions.py
├── main.py                 # Streamlit application
├── notebook.ipynb          # EDA and model training notebook
├── pyproject.toml          # Project dependencies (uv/pip)
├── requirements.txt        # Pip-compatible dependencies
└── README.md              # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.8+
- uv (recommended) or pip

### Installation

```bash
# Clone repository
git clone https://github.com/mcikalmerdeka/Loan-Prediction-Based-on-Costumer-Behaviour.git
cd Loan-Prediction-Based-on-Costumer-Behaviour

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies (using pip)
pip install -r requirements.txt

# Or using uv (faster alternative)
uv sync
```

### Run the App

```bash
streamlit run main.py
```

Access the app at `http://localhost:8501`

## Features

- **Data Input Options**: Manual individual entry or batch CSV upload
- **Interactive Preprocessing**: Step-by-step data transformation with visual feedback
  - Data type conversion
  - Missing value handling
  - Outlier detection and filtering
  - Feature engineering (age groups, experience ratios, location groupings)
  - Categorical encoding (ordinal + one-hot)
  - Feature scaling (MinMax + StandardScaler)
- **Model Performance Display**: View model metrics and training information
- **Real-time Predictions**: Instant loan default risk assessment
- **Batch Processing**: Efficient processing of multiple loan applications
- **Example Data**: Built-in test cases for exploring model behavior
- **Data Dictionary**: Comprehensive feature explanations and definitions

## Technical Stack

- Python 3.8+
- scikit-learn (KNN classifier, preprocessing, model tuning)
- pandas, numpy (data processing)
- matplotlib, seaborn (visualization)
- Streamlit (web application)
- joblib (model serialization)
- scipy (statistical functions)

## Business Problem

A lending company needed to improve their credit risk assessment process to reduce financial losses from loan defaults. The manual verification procedure resulted in numerous inaccurate assessments, approving loans to borrowers who ultimately failed to repay. Additionally, the manual process consumed considerable time, making it inefficient for scaling operations.

### Problem Statement

How can we develop an automated system that predicts the creditworthiness of prospective borrowers to reduce the selection of high-risk customers and improve the efficiency of the loan approval process?

### Business Metrics

- **Default Rate (%)** [MAIN]: The percentage of customers who fail to repay their loans. Calculated as: (Number of Loan Defaults / Total Number of Customers) × 100. Lower default rates indicate more effective risk assessment.

- **Approval Time** [SECONDARY]: The time taken to process loan applications. Streamlining this process reduces operational costs and improves customer satisfaction.

### Goals

- **Primary**: Enhanced credit risk evaluation through machine learning implementation with high recall to minimize false negatives
- **Secondary**: Increased efficiency in credit risk assessment for faster processing times

### Objectives

1. Build a machine learning model that predicts credit risk with high recall rate to reduce financial losses from inaccurate assessments
2. Provide automated credit risk assessment decisions in minimal time
3. Deliver actionable insights for loan officers to support decision-making

## Project Background

This project originated from the final project assignment of the Rakamin Data Science Bootcamp Batch 39. The **`Original Indonesian Version`** folder contains the original bootcamp deliverables including assignment instructions, presentations, and team submissions. The project has been further developed with additional improvements based on panelist feedback, work experience, and best practices in MLOps and software engineering.

## Recent Updates

- **Refactored project structure** with optimized data loading and caching
- **Performance optimizations** including:
  - Cached column metadata to avoid loading large datasets repeatedly
  - Batch prediction support (predict all rows at once instead of looping)
  - Pre-fitted scalers saved to disk for faster inference
  - Reduced memory footprint and startup time
- **Updated dependencies** to essential packages only
- **Improved error handling** and user feedback in the Streamlit interface

## Try the Live App

Link to the deployed app: [Loan Prediction Based on Customer Behavior](https://loan-churn-prediction-y5rm4xjsqyommkzjtjjejy.streamlit.app/)

## Author

**Muhammad Cikal Merdeka** | Data Analyst/Data Scientist

- [GitHub](https://github.com/mcikalmerdeka)
- [LinkedIn](https://www.linkedin.com/in/mcikalmerdeka)
- [Email](mailto:mcikalmerdeka@gmail.com)
