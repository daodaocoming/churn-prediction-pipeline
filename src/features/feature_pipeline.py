"""
feature_pipeline.py · Day10
Numeric → StandardScaler
AvgMonthlySpend → KBinsDiscretizer(4, quantile) → one‑hot
Low‑card Cats (+ TenureBucket, Is_MonthToMonth) → OneHotEncoder(drop='first')
Pipeline = AddDerivedFeatures  ➜  ColumnTransformer
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import (KBinsDiscretizer, OneHotEncoder,
                                   StandardScaler)

from . import feature_lists as fl
from .transformers import AddDerivedFeatures


def build_preprocessor(
    num_cols: List[str] = fl.numeric_features,
    cat_cols: List[str] = fl.categorical_low_card,
) -> Tuple[SkPipeline, List[str]]:
    """Return `(full_pipeline, feature_names)` ready for fit/transform."""

    # ─────────────────────────────── transformers ──────────────────────────────
    numeric_t = SkPipeline([("scaler", StandardScaler())])

    bins_t = SkPipeline([
        (
            "kbins",
            KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile"),
        )
    ])

    cat_t = SkPipeline(
        [
            (
                "ohe",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
            )
        ]
    )

    # ────────────────────────────── column lists ───────────────────────────────
    bin_cols = ["AvgMonthlySpend"]                       # 派生后再分箱
    extra_cat = ["TenureBucket", "Is_MonthToMonth"]     # 新增派生类别列
    all_cat_cols = cat_cols + extra_cat

    pre = ColumnTransformer(
        [
            ("num", numeric_t, num_cols),
            ("kbins", bins_t, bin_cols),
            ("cat", cat_t, all_cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # ──────────────────────── wrap with derived‑feature adder ───────────────────
    pipe = SkPipeline([("derive", AddDerivedFeatures()), ("pre", pre)])

    # ───────────────────────────── dummy fit for names ─────────────────────────
    dummy_num = pd.DataFrame({c: [0] for c in num_cols + bin_cols})
    dummy_cat = pd.DataFrame({c: ["__A__"] for c in all_cat_cols})
    dummy = pd.concat([dummy_num, dummy_cat], axis=1)

    pipe.fit(dummy)
    feature_names = pipe.named_steps["pre"].get_feature_names_out().tolist()

    return pipe, feature_names


def save_pipeline(pipeline: SkPipeline, path: str | Path = "models/feature_pipeline_v2.pkl") -> None:
    """Persist pipeline via joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


