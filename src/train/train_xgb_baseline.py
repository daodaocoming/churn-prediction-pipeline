from pathlib import Path
import joblib, mlflow, numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from xgboost import XGBClassifier

# ─── 路径 ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

X = joblib.load(DATA / "X_train.pkl")
y = joblib.load(DATA / "y_train.pkl")

# ─── 模型 & CV ────────────────────────────────────────────────────────
clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision"}
res = cross_validate(clf, X, y, cv=cv, scoring=scoring, n_jobs=-1)

roc_mu, roc_sd = res["test_roc_auc"].mean(), res["test_roc_auc"].std(ddof=0)
pr_mu , pr_sd  = res["test_pr_auc"].mean() , res["test_pr_auc"].std(ddof=0)
print(f"CV‑ROC AUC : {roc_mu:.3f} ± {roc_sd:.3f}")
print(f"CV‑PR  AUC : {pr_mu :.3f} ± {pr_sd :.3f}")

# ─── MLflow ──────────────────────────────────────────────────────────
mlflow.set_experiment("baseline_xgb")
with mlflow.start_run(run_name="xgb_default"):
    mlflow.log_params(clf.get_params())
    mlflow.log_metrics({
        "roc_auc_mean": roc_mu, "roc_auc_std": roc_sd,
        "pr_auc_mean" : pr_mu , "pr_auc_std" : pr_sd,
    })

    clf.fit(X, y)
    model_path = MODEL_DIR / "xgb_baseline.pkl"
    joblib.dump(clf, model_path)
    mlflow.log_artifact(model_path)

print("✅ saved →", model_path.relative_to(ROOT))
