"""
tune_xgb_optuna.py – Day 12  (方案 A：持久化 Optuna Study)
---------------------------------------------------------
* 30 trials → 优化 n_estimators, max_depth, learning_rate, subsample, colsample_bytree
* 目标：5-fold PR-AUC 均值最大
* 每个 trial 结果写入同一份 sqlite DB (optuna.db) & MLflow experiment “xgb_optuna”
"""

from pathlib import Path
import joblib, optuna, mlflow
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# ─────────────────────────── 路径 ────────────────────────────
ROOT     = Path(__file__).resolve().parents[2]       # 项目根
DATA_DIR = ROOT / "data" / "processed"
DB_PATH  = ROOT / "optuna.db"                        # 唯一持久化文件

# ─────────────────────────── 数据 ────────────────────────────
X = joblib.load(DATA_DIR / "X_train.pkl")
y = joblib.load(DATA_DIR / "y_train.pkl")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ────────────────────────── 目标函数 ──────────────────────────
def objective(trial: optuna.Trial) -> float:
    params = {
        "n_estimators":      trial.suggest_int(   "n_estimators", 200, 800, step=100),
        "max_depth":         trial.suggest_int(   "max_depth",    3,   10),
        "learning_rate":     trial.suggest_float( "learning_rate", 1e-3, 0.3, log=True),
        "subsample":         trial.suggest_float( "subsample",    0.5, 1.0),
        "colsample_bytree":  trial.suggest_float( "colsample_bytree", 0.5, 1.0),
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
    }

    clf = XGBClassifier(**params)

    pr_auc = cross_val_score(clf, X, y, cv=cv,
                             scoring="average_precision", n_jobs=-1).mean()
    roc_auc = cross_val_score(clf, X, y, cv=cv,
                              scoring="roc_auc", n_jobs=-1).mean()

    trial.set_user_attr("roc_auc", roc_auc)
    return pr_auc

# ──────────────────────── Optuna Study ───────────────────────
study = optuna.create_study(
    study_name="xgb_pr_auc",
    direction="maximize",
    storage=f"sqlite:///{DB_PATH}",   # ★ 持久化到同一 DB
    load_if_exists=True
)

# ──────────────────────── MLflow 设置 ────────────────────────
mlflow.set_experiment("xgb_optuna")

def mlflow_callback(study: optuna.Study, trial: optuna.Trial):
    with mlflow.start_run(run_name=f"trial_{trial.number}"):
        mlflow.log_params(trial.params)
        mlflow.log_metric("pr_auc", trial.value)
        mlflow.log_metric("roc_auc", trial.user_attrs["roc_auc"])
        mlflow.set_tags({"optuna_study": study.study_name})

# ─────────────────────────── 开始优化 ─────────────────────────
study.optimize(
    objective,
    n_trials=30,
    callbacks=[mlflow_callback],
    show_progress_bar=True
)

# ─────────────────────────── 输出最佳 ─────────────────────────
best = study.best_trial
print(f"🎯 Best PR-AUC : {best.value:.4f}")
print("Best params :", best.params)
