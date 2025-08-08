from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
import joblib, numpy as np, pandas as pd, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODEL = joblib.load(ROOT/"models/final/model.pkl")
PIPE  = joblib.load(ROOT/"models/feature_pipeline_v2.pkl")

class TelcoSample(BaseModel):
    customer_id: str
    data: dict  

app = FastAPI(title="Churn Predictor API", version="0.1")

@app.post("/predict")
def predict(sample: TelcoSample):
    df = pd.DataFrame([sample.data])
    X = PIPE.transform(df)
    proba = float(MODEL.predict_proba(X)[:,1])
    return {"customer_id": sample.customer_id, "churn_probability": proba}
