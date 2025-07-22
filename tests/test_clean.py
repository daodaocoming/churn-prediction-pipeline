from pathlib import Path
import pandas as pd
import numpy as np
import importlib

import src.data.clean_data as cln

RAW  = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
CLEAN= Path("data/clean/telco_clean.parquet")

if not CLEAN.exists():
    cln.main.callback(input=str(RAW), output=str(CLEAN))

df_raw   = pd.read_csv(RAW)
df_clean = pd.read_parquet(CLEAN)

def test_row_count():
    assert len(df_raw) == len(df_clean), "Row count changed!"

def test_no_missing():
    assert df_clean.isna().sum().sum() == 0, "Missing values remain!"

def test_column_types():
    expected_bool   = ["SeniorCitizen"]
    expected_float  = ["TotalCharges"]
    for col in expected_bool:
        assert df_clean[col].dtype == "bool"
    for col in expected_float:
        assert np.issubdtype(df_clean[col].dtype, np.floating)

def test_outlier_clipping():
    num_cols = df_clean.select_dtypes(include="number").columns
    for col in num_cols:
        q1, q3 = df_clean[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        assert df_clean[col].between(low, high).all(), f"Outliers remain in {col}"
