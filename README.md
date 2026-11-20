# Insurance Claim Prediction: ML Pipeline with MLOps

An end-to-end machine learning project for **SafeDrive Insurance Ltd.** that predicts whether a policyholder will file a claim, from policy, vehicle and safety features. It goes beyond modelling and covers hyperparameter tuning, automated feature generation, experiment tracking, model versioning, drift monitoring and explainability.

## Problem
SafeDrive's claim payouts are rising, and manual underwriting is inconsistent. The goal is to find the factors linked to claim risk and build a model that flags higher-risk policies.

## Data
58,592 policies × 44 columns. Only **6.4%** of policies have a claim (about a 15:1 class imbalance), so accuracy is misleading and the project focuses on **ROC-AUC, recall and F1**.

## Results (test set, 11,719 policies)

| Model | Features | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 53 | 0.585 | 0.548 | 0.080 | 0.140 |
| Optuna-tuned Decision Tree | 53 | 0.646 | 0.653 | 0.089 | 0.157 |
| + Featuretools features | 57 | 0.638 | 0.663 | 0.089 | 0.156 |
| **Final: tuned Decision Tree (top 20 features)** | **20** | **0.637** | **0.683** | **0.087** | **0.154** |

- **Final model:** 5-fold CV ROC-AUC **0.641 ± 0.012**. It uses about 65% fewer features than the full model with almost no loss in ROC-AUC (0.6382 to 0.6366).
- **Honest takeaway:** claim risk is hard to predict from this data. Individual features correlate only weakly with claims (best |r| = 0.08). The model catches about 68% of claims, but precision stays low (8.7% vs a 6.4% base rate), so it's best used to **prioritise policies for review**, not to make automatic decisions.
- **Threshold trade-off:** lowering the decision threshold to 0.3 raises recall to 96% but precision falls to 7.2%, so the threshold should follow how costly missed claims are compared with extra reviews.

## What Drives Claims
- **Area cluster:** claim rates range from 3.9% (C17) to 10.7% (C18), a gap of about 2.8×. C14, C8, C3 and C2 are reliably above average. C18 and C22 have high rates but too few policies to be firm, so they go on a watch-list.
- **Speed alert:** 6.4% claim rate with it vs 4.1% without.
- **Policy tenure** is the strongest single numeric correlate, and **age of car** is one of the top features in the final model.
- SHAP and permutation importance agree that area cluster combined with policy tenure (a Featuretools feature) is the top predictor.

## Pipeline

| Notebook | What it does |
|---|---|
| `01_data_preprocessing_eda_feature_engineering` | Cleaning, EDA, redundant-feature removal, outlier capping, encoding, new features (`High_Risk_Driver_Flag`, `safety_equipment_count`, `Power_to_Weight_Ratio`), VIF-based pruning. Saves `Cleaned_df.csv`. |
| `02_ml_model_building_and_validation` | Stratified split, logistic regression baseline with assumption checks, **Optuna**-tuned decision tree, 5-fold CV, **MLflow** tracking, model versioning with **joblib** plus JSON metadata, **Evidently** drift report, **SHAP**. |
| `03_automl_model_comparison_and_versioning` | **Featuretools** deep feature synthesis (2,809 candidates, 4 retained after screening against the target), re-tuned tree, top-20 feature reduction, MLflow comparison log, versioning, drift report, SHAP. |

Key preprocessing steps: parsed power and torque text into numbers, dropped redundant columns (`make`/`model`, `is_power_steering`, duplicated safety flags), median/mode imputation, IQR outlier capping, one-hot encoding, `StandardScaler` fitted on training data only. Feature generation excludes the target to avoid leakage.

**Monitoring:** Evidently found **no data drift** between train and test (0 of 53 features, and 0 of 20 in the final model).

## Tech Stack
Python · pandas · scikit-learn · statsmodels · Optuna · MLflow · Featuretools · Evidently · SHAP · joblib · matplotlib · seaborn

## Run It
```bash
git clone https://github.com/tanmayanju16616-web/insurance-claim-prediction-mlops.git
cd insurance-claim-prediction-mlops
pip install -r requirements.txt
jupyter notebook
```
Run the notebooks in order (01, 02, 03). The raw dataset is not included: put it in the project folder as `dataset.csv` before running notebook 01. Notebooks 02 and 03 read `Cleaned_df.csv`, which notebook 01 creates.

## Author
**Anju Kanwar**
