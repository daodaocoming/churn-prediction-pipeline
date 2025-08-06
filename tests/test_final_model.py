import numpy as np, joblib
from src.models.final_model import load

def test_final_model_predict_shape():
    X = joblib.load("data/processed/X_train.pkl")[:50]
    model = load()
    proba = (model.predict_proba(X)[:,1]
             if hasattr(model, "predict_proba")
             else model.predict(X))
    assert proba.shape == (50,)
    assert np.isfinite(proba).all()
