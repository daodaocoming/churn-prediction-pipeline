"""
train_lgbm_baseline.py – Day 19
* 5-fold CV ROC/PR-AUC on LightGBMClassifier (sklearn API)
* Retrain on full data → save models/lgbm_baseline.txt
* Log to MLflow experiment “baseline_lgbm”
"""

from pathlib import Path
import joblib, mlflow
from sklearn.model_selection import StratifiedKFold, cross_validate
from lightgbm import LGBMClassifier

# ─── Paths ───────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[2]
DATA  = ROOT / "data" / "processed"
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)

X = joblib.load(DATA / "X_train.pkl")
y = joblib.load(DATA / "y_train.pkl")

# ─── Model ───────────────────────────────────────────
clf = LGBMClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=-1,              # -1 = no limit
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary",
    random_state=42,
    n_jobs=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
scores  = cross_validate(clf, X, y, cv=cv, scoring=scoring, return_estimator=False)

roc_mu, roc_sd = scores["test_roc_auc"].mean(), scores["test_roc_auc"].std(ddof=0)
pr_mu , pr_sd  = scores["test_pr_auc"].mean() , scores["test_pr_auc"].std(ddof=0)

print(f"CV ROC-AUC : {roc_mu:.3f} ± {roc_sd:.3f}")
print(f"CV PR-AUC  : {pr_mu :.3f} ± {pr_sd :.3f}")

# ─── MLflow ──────────────────────────────────────────
mlflow.set_experiment("baseline_lgbm")
with mlflow.start_run(run_name="lgbm_default"):
    mlflow.log_params(clf.get_params())
    mlflow.log_metrics({
        "roc_auc_mean": roc_mu, "roc_auc_std": roc_sd,
        "pr_auc_mean" : pr_mu , "pr_auc_std" : pr_sd,
    })

    clf.fit(X, y)
    model_path = MODELS / "lgbm_baseline.txt"
    clf.booster_.save_model(model_path)
    mlflow.log_artifact(model_path)

print("✅ saved →", model_path.relative_to(ROOT))
