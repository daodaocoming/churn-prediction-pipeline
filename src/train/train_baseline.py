"""
train_baseline.py – Day 11 Baseline (frozen)

• Loads data/processed/X_train.pkl and y_train.pkl
• Performs 5‑fold StratifiedKFold CV → records ROC‑AUC & PR‑AUC
• Retrains on the full training set → saves models/logreg.pkl
• Logs all hyper‑parameters, metrics, and the model artifact to the
  MLflow experiment “baseline_logreg”
"""

from pathlib import Path
import joblib, numpy as np, mlflow, mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

# ─────────────────────────────── Paths ────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]      # src/train/../..
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
MODEL_DIR    = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

X = joblib.load(DATA_DIR / "X_train.pkl")
y = joblib.load(DATA_DIR / "y_train.pkl")

# ─────────────────────────────── Cross‑validation ─────────────────────
clf = LogisticRegression(
    penalty="l2",
    C=1.0,
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
scores  = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_estimator=False)

roc_mean, roc_std = scores["test_roc_auc"].mean(), scores["test_roc_auc"].std(ddof=0)
pr_mean,  pr_std  = scores["test_pr_auc"].mean(),  scores["test_pr_auc"].std(ddof=0)

print(f"CV ROC‑AUC : {roc_mean:.3f} ± {roc_std:.3f}")
print(f"CV PR‑AUC  : {pr_mean :.3f} ± {pr_std :.3f}")

# ─────────────────────────────── MLflow Logging ───────────────────────
mlflow.set_experiment("baseline_logreg")

with mlflow.start_run(run_name="logreg_l2_balanced"):
    mlflow.log_params(clf.get_params())
    mlflow.log_metrics({
        "roc_auc_mean": roc_mean,
        "roc_auc_std":  roc_std,
        "pr_auc_mean":  pr_mean,
        "pr_auc_std":   pr_std,
    })

    # Train on the full dataset & save the model
    clf.fit(X, y)
    model_path = MODEL_DIR / "logreg.pkl"
    joblib.dump(clf, model_path)
    mlflow.log_artifact(model_path)

# For CLI feedback
print("✅ Model saved →", model_path.relative_to(PROJECT_ROOT))
