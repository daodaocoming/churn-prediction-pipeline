"""
feature_pipeline.py
Builds a ColumnTransformer:
  • numeric   → StandardScaler
  • low‑card  → OneHotEncoder(drop='first')
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ---------- feature lists ----------
from . import feature_lists as fl          # numeric_features, categorical_low_card


def build_preprocessor(
    num_cols: List[str] = fl.numeric_features,
    cat_cols: List[str] = fl.categorical_low_card,
) -> Tuple[ColumnTransformer, List[str]]:
    """Return (fitted ColumnTransformer, feature_names list)."""

    numeric_t = Pipeline([("scaler", StandardScaler())])
    cat_t = Pipeline([("ohe", OneHotEncoder(drop="first",
                                            sparse_output=False,
                                            handle_unknown="ignore"))])

    pre = ColumnTransformer(
        [("num", numeric_t, num_cols),
         ("cat", cat_t,   cat_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    dummy_num = pd.DataFrame({c: [0] for c in num_cols})
    dummy_cat = pd.concat([
        pd.DataFrame({c: ["__A__"] for c in cat_cols}),
        pd.DataFrame({c: ["__B__"] for c in cat_cols})
    ])

    dummy = pd.concat([dummy_num, dummy_cat], axis=1)
    pre.fit(dummy)

    feature_names = (
        num_cols +
        pre.named_transformers_["cat"].named_steps["ohe"]
           .get_feature_names_out(cat_cols).tolist()
    )
    return pre, feature_names


def save_pipeline(
    pipeline: ColumnTransformer,
    path: str | Path = "models/feature_pipeline.pkl",
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)

