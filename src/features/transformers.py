"""
transformers.py
---------------
Reusable custom transformers for feature engineering.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class AddDerivedFeatures(BaseEstimator, TransformerMixin):
    """
    Add AvgMonthlySpend, Is_MonthToMonth, TenureBucket columns.
    * Assumes raw columns already exist in X.
    * Returns **full DataFrame** with new columns appended.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self                          # no state

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        contract_col = X["Contract"]
        if isinstance(contract_col, pd.DataFrame):
            contract_col = contract_col.iloc[:, 0]

        X["Is_MonthToMonth"] = (contract_col == "Month-to-month").astype("category")

        
        X["AvgMonthlySpend"] = X["TotalCharges"] / (X["tenure"] + 1)

        bins = [0, 6, 12, 24, np.inf]
        labels = ["0-6", "6-12", "12-24", "24+"]
        X["TenureBucket"] = pd.cut(X["tenure"], bins=bins, labels=labels, right=False)

        X = X.loc[:, ~X.columns.duplicated()]

        return X
