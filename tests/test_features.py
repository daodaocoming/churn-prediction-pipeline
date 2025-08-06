import joblib, numpy as np, pandas as pd
from src.features.feature_pipeline import build_preprocessor
from src.features import feature_lists as fl

def test_pipeline_shape_and_nan():
    df = pd.read_parquet("data/clean/telco_clean.parquet")
    X_raw = df[fl.numeric_features + fl.categorical_low_card + ["Contract"]]
    y = df["Churn"].map({"Yes": 1, "No": 0}).values
    pipe, _ = build_preprocessor()
    X_t = pipe.fit_transform(X_raw, y)
    assert X_t.shape[0] == len(y)
    assert not np.isnan(X_t).any()
