import joblib, optuna
from xgboost import XGBClassifier
from pathlib import Path

ROOT = Path.cwd().parents[0] / "churn-prediction-pipeline"  
X = joblib.load(ROOT/"data/processed/X_train.pkl")
y = joblib.load(ROOT/"data/processed/y_train.pkl")

study = optuna.load_study(study_name="xgb_pr_auc", storage="sqlite:///optuna.db")   

best_params = study.best_trial.params
best_params.update({"objective": "binary:logistic", "eval_metric": "aucpr", "random_state": 42, "n_jobs": -1})

best_clf = XGBClassifier(**best_params).fit(X, y)
joblib.dump(best_clf, ROOT/"models/xgb_optuna_best.pkl")
print("✅ xgb_optuna_best.pkl saved")


