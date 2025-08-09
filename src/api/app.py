from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, numpy as np, pandas as pd, pathlib
from shap import TreeExplainer

from src.api.schemas import TelcoInput 

# ---- load artifacts once
ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = joblib.load(ROOT / "models" / "final" / "model.pkl")
PIPE  = joblib.load(ROOT / "models" / "feature_pipeline_v2.pkl")
FEATURE_NAMES = PIPE.named_steps["pre"].get_feature_names_out()

explainer = TreeExplainer(MODEL)

app = FastAPI(title="Churn Predictor API", version="0.1")

class PredictRequest(BaseModel):
    customer_id: str
    data: TelcoInput

class PredictResponse(BaseModel):
    customer_id: str
    churn_probability: float
    top_features: list[str]

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        df = pd.DataFrame([req.data.model_dump()])
        X = PIPE.transform(df)
        proba = float(MODEL.predict_proba(X)[:, 1])

        shap_val = explainer.shap_values(X, check_additivity=False)[0]  # 1-sample
        top_idx = np.argsort(np.abs(shap_val))[::-1][:3]
        top_feats = [str(FEATURE_NAMES[i]) for i in top_idx]

        return {
            "customer_id": req.customer_id,
            "churn_probability": proba,
            "top_features": top_feats,
        }
    except Exception as e:
        # helpful 400 instead of 500 on schema/column issues
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")
