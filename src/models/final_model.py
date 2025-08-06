from pathlib import Path
import joblib, lightgbm as lgbm

_ROOT = Path(__file__).resolve().parents[2]
_PATH = _ROOT / "models" / "final" / "model.pkl"

def load():
    if _PATH.suffix == ".pkl":
        return joblib.load(_PATH)
    if _PATH.suffix == ".txt":
        return lgbm.Booster(model_file=str(_PATH))
    raise ValueError(f"Unsupported model format: {_PATH}")
