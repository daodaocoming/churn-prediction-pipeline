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
