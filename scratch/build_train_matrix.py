# scratch/build_train_matrix.py 
import pathlib as pl, joblib, pandas as pd, numpy as np
from src.features import feature_lists as fl
from src.features.feature_pipeline import build_preprocessor

# --- paths --------------------------------------------------------------
PIPE_PATH   = "models/feature_pipeline_v2.pkl"   
DATA_CLEAN  = "data/clean/telco_clean.parquet"
OUT_DIR     = pl.Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- read data ----------------------------------------------------------
df = pd.read_parquet(DATA_CLEAN)
y  = df["Churn"].map({"Yes": 1, "No": 0}).values


X_raw = df[fl.numeric_features + fl.categorical_low_card]

# --- load pipeline & fit‑transform --------------------------------------
pipe, _ = build_preprocessor()          # fresh pipeline in current env
X_trans = pipe.fit_transform(X_raw, y)  # fit on full data, then transform

# --- persist ------------------------------------------------------------
joblib.dump(X_trans, OUT_DIR / "X_train.pkl")
joblib.dump(y,       OUT_DIR / "y_train.pkl")
joblib.dump(pipe,    "models/feature_pipeline_v2_fitted.pkl")   
print("✅ Saved matrices to", OUT_DIR)

