# Customer Churn Prediction Pipeline

**Objective**  
Build an end‑to‑end ML pipeline that predicts customer churn and deploys a real‑time FastAPI service with MLOps best practices.

## Project Structure
├── data/               # Raw / cleaned / processed datasets
├── src/                # Reusable Python modules
│   ├── data/           # Ingestion & cleaning scripts
│   ├── features/       # Feature engineering pipelines
│   ├── models/         # Training / evaluation / inference
│   └── api/            # FastAPI application
├── notebooks/          # Exploratory analyses
├── tests/              # Unit tests (pytest)
├── reports/            # Generated reports & model cards
├── figures/            # Saved visualizations
└── env.yml             # Conda environment
## Quick Start
```bash
conda env create -f env.yml
conda activate churn-env
jupyter lab

| Date        | Model                | ROC‑AUC ± std | PR‑AUC ± std | Run ID |
|-------------|----------------------|---------------|--------------|--------|
| 2025‑07‑30 | Logistic Regression | 0.848 ± 0.011 | 0.661 ± 0.015 | `logreg_l2_balanced` |
#### Day 12 – Tree Boosting & Tuning
- Baseline XGBoost (5‑fold) → ROC‑AUC ≈ 0.88, PR‑AUC ≈ 0.71  
- Optuna (30 trials) → Best PR‑AUC ≈ 0.74  
- Best model saved at `models/xgb_optuna_best.pkl`  
- All trials logged in MLflow experiment **“xgb_optuna”**

### Current Production Model (Day 20)

| Model | ROC-AUC | PR-AUC | File |
|-------|:------:|:------:|------|
| **XGBoost (Optuna)** | 0.872 | 0.712 | `models/final/model.pkl` |
MD
