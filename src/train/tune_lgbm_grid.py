"""
tune_lgbm_optuna.py – Day 19
────────────────────────────
• 60 Optuna trials to tune LightGBMClassifier
• Optimises 5-fold CV PR-AUC (Average Precision)
• Logs every trial to MLflow experiment “lgbm_optuna”
• Saves the best booster → models/lgbm_optuna_best.txt
"""

from pathlib import Path
import joblib, optuna, mlflow
from sklearn.model_selection import StratifiedKFold, cross_val_score
from lightgbm import LGBMClassifier

# ─── Paths ───────────────────────────────────────────────────────────
ROOT  = Path(__file__).resolve().parents[2]
DATA  = ROOT / "data" / "processed"
MODEL = ROOT / "models"; MODEL.mkdir(exist_ok=True)
DB    = ROOT / "optuna_lgbm.db"            # separate DB from XGB if you like

X = joblib.load(DATA / "X_train.pkl")
y = joblib.load(DATA / "y_train.pkl")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ─── Objective ───────────────────────────────────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        # core
        "n_estimators": trial.suggest_int("n_estimators", 300, 900, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 10),  # -1 = no limit
        # tree complexity
        "num_leaves": trial.suggest_int("num_leaves", 31, 255, step=32),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 40, step=5),
        # regularisation
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
        # subsampling
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        # fixed
        "objective": "binary",
        "metric": "average_precision",     # LightGBM’s AP
        "random_state": 42,
        "n_jobs": -1,
    }

    clf = LGBMClassifier(**params)
    pr_auc = cross_val_score(
        clf, X, y, cv=cv, scoring="average_precision", n_jobs=-1
    ).mean()

    # 额外算一下 ROC-AUC，存在 user_attr 里
    roc_auc = cross_val_score(
        clf, X, y, cv=cv, scoring="roc_auc", n_jobs=-1
    ).mean()
    trial.set_user_attr("roc_auc", roc_auc)

    return pr_auc  # We maximise PR-AUC

# ─── Optuna Study ────────────────────────────────────────────────────
study = optuna.create_study(
    study_name="lgbm_pr_auc",
    direction="maximize",
    storage=f"sqlite:///{DB}",
    load_if_exists=True,
)

# ─── MLflow callback ────────────────────────────────────────────────
mlflow.set_experiment("lgbm_optuna")

def mlflow_cb(study: optuna.Study, trial: optuna.Trial):
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params(trial.params)
        mlflow.log_metrics({
            "pr_auc": trial.value,
            "roc_auc": trial.user_attrs["roc_auc"],
        })
        mlflow.set_tags({"optuna_study": study.study_name})

# ─── Run optimisation ───────────────────────────────────────────────
study.optimize(objective, n_trials=60, callbacks=[mlflow_cb], show_progress_bar=True)

best = study.best_trial
print(f"🎯 Best PR-AUC : {best.value:.4f}")
print("Best params :", best.params)

# ─── Train best model on full data & save ───────────────────────────
best_params = best.params | {
    "objective": "binary",
    "metric": "average_precision",
    "random_state": 42,
    "n_jobs": -1,
}
best_model = LGBMClassifier(**best_params).fit(X, y)
out_path = MODEL / "lgbm_optuna_best.txt"
best_model.booster_.save_model(out_path)
print("✅ Best model saved →", out_path.relative_to(ROOT))

# log artefact to MLflow
with mlflow.start_run(run_name="best_model"):
    mlflow.log_params(best_params)
    mlflow.log_metric("pr_auc_full", best.value)
    mlflow.log_artifact(out_path)
