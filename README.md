# Fraud Detection & Transaction Risk Analysis

A machine-learning project for detecting fraudulent transactions and translating model predictions into interpretable risk tiers.

## Project Overview

This project explores transaction-level fraud detection using machine learning, data analysis, model comparison, and explainable AI (SHAP).

### Key objectives

- Analyze transaction and fraud patterns
- Handle class imbalance
- Compare machine-learning models
- Evaluate fraud detection using precision-recall focused metrics
- Explain model predictions using SHAP
- Convert predictions into practical risk tiers
- Provide a lightweight application for model-based prediction

## Project Highlights

- Exploratory data analysis with transaction and fraud visualizations
- Class-imbalance analysis
- Correlation and feature analysis
- Model comparison
- Precision-recall evaluation
- SHAP-based model interpretability
- Risk-tier analysis
- Flask-based prediction application

## Repository Structure

```text
fraud-detection-ml/
│
├── app.py
├── analysis.ipynb
├── model.pkl
├── requirements.txt
├── summary.pdf
├── model_comparison.png
├── shap_summary.png
│
└── charts/
    ├── class_imbalance.png
    ├── correlation_heatmap.png
    ├── fraud_by_hour.png
    ├── fraud_rate_by_hour.png
    ├── missing_values.png
    ├── precision_recall_curve.png
    ├── risk_tiers.png
    ├── risk_tier_donut.png
    ├── scatter_amt_vs_hour.png
    ├── shap_beeswarm.png
    ├── shap_global_summary.png
    ├── shap_waterfall.png
    ├── tier_hourly.png
    ├── transaction_amt.png
    └── transaction_amt_distribution.png
```

## Dataset

The original transaction and identity datasets are intentionally **not stored in this GitHub repository** because of their large size.

The repository contains the analysis notebook, generated visualizations, trained model artifact, and documentation needed to understand the workflow.

> Place the original dataset locally according to the paths expected by `analysis.ipynb` before rerunning the complete training workflow.

## Workflow

```text
Raw Transaction Data
        ↓
Data Cleaning & Preparation
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Class-Imbalance Analysis
        ↓
Model Training
        ↓
Model Comparison
        ↓
Fraud Prediction
        ↓
SHAP Explainability
        ↓
Risk-Tier Classification
        ↓
Flask Prediction App
```

## Explainable AI

SHAP (SHapley Additive exPlanations) is used to understand which features contribute to model predictions.

The repository includes:

- Global feature-importance visualizations
- SHAP beeswarm analysis
- SHAP waterfall explanations
- Model-level interpretability outputs

## Risk Tiers

Predicted transaction risk is translated into understandable risk categories so that model output can be interpreted beyond a simple fraud/non-fraud label.

The visualizations in `charts/` show how transaction risk varies across the analyzed data.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python app.py
```

Then open the local URL shown by Flask in your browser.

## Analysis Notebook

Open:

```text
analysis.ipynb
```

The notebook contains the analysis and modelling workflow used to produce the project's results and visualizations.

## Project Outputs

The repository includes:

- Model comparison results
- Fraud distribution analysis
- Transaction amount analysis
- Time-based fraud analysis
- Precision-recall visualization
- Risk-tier visualizations
- SHAP explainability plots
- Project summary PDF

## Important Note

The trained model is included as a portfolio artifact. For production deployment, the complete preprocessing pipeline, model versioning, input validation, monitoring, and secure model-serving architecture should be maintained together.

## Author

** veerabhadra chary**

B.Tech — Computer Science 

### Skills Demonstrated

`Python` `Machine Learning` `Pandas` `NumPy` `Scikit-learn` `SHAP` `Data Analysis` `Data Visualization` `Flask` `Git` `GitHub`

---

⭐ If you find this project useful, consider starring the repository.
"# Fraud-Detection-Analysis" 
