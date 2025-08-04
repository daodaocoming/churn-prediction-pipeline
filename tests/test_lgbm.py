# tests/test_lgbm.py
import joblib, numpy as np, lightgbm as lgb
from pathlib import Path

MODEL_DIR = Path("models")
DATA_DIR  = Path("data/processed")

def _load_lgb_model(path: Path):
    """
    Load either a pickled LGBMClassifier *or* a raw booster .txt file.
    Returns an object with a .predict(X) ndarray output.
    """
    try:        # try pickle first
        obj = joblib.load(path)
        # pickled wrapper has predict_proba, use [:,1]
        return lambda X: obj.predict_proba(X)[:, 1]
    except Exception:
        # fall back to raw booster
        booster = lgb.Booster(model_file=str(path))
        return lambda X: booster.predict(X)

def test_lgbm_proba_shape_no_nan():
    import joblib, numpy as np
    X = joblib.load(DATA_DIR / "X_train.pkl")

    # prefer tuned model, else baseline
    model_path = (
        MODEL_DIR / "lgbm_optuna_best.txt"
        if (MODEL_DIR / "lgbm_optuna_best.txt").exists()
        else MODEL_DIR / "lgbm_baseline.txt"
    )

    proba_fn = _load_lgb_model(model_path)
    proba = proba_fn(X)

    assert proba.shape[0] == X.shape[0]
    assert not np.isnan(proba).any()

