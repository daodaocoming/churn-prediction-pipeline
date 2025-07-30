# scratch/build_feature_lists.py
"""
Generates three types of feature lists:
  ① numeric_features
  ② categorical_low_card  (n_unique ≤ 15)
  ③ categorical_high_card (n_unique > 15)
Output saved to: src/features/feature_lists.py
"""

from pathlib import Path
import pandas as pd

# ------------ 1. Load cleaned data ------------
DATA_PATH = Path("data/clean/telco_clean.parquet")
df = pd.read_parquet(DATA_PATH)
print("Loaded:", df.shape)

# ------------ 2. Define helper function ------------
DROP_COLS = {"customerID","Churn"}  # List of unique ID / leakage columns, can be extended
MAX_LOW_CARD = 15           # Threshold for low cardinality, can be adjusted

def classify_column(col: pd.Series) -> str:
    """Returns category label: numeric / low / high / drop"""
    name = col.name
    if name in DROP_COLS:
        return "drop"
    if pd.api.types.is_numeric_dtype(col):
        return "num"
    # For other types, classify based on cardinality
    n_unique = col.nunique(dropna=True)
    return "low" if n_unique <= MAX_LOW_CARD else "high"


# ------------ 3. Classify columns one by one ------------
numeric_features, low_card, high_card = [], [], []
for c in df.columns:
    tag = classify_column(df[c])
    if tag == "num":
        numeric_features.append(c)
    elif tag == "low":
        low_card.append(c)
    elif tag == "high":
        high_card.append(c)

print("Numeric features :", numeric_features)
print("Low-cardinality :", low_card)
print("High-cardinality:", high_card)

# ------------ 4. Write to feature_lists.py ------------
OUT_PATH = Path("src/features/feature_lists.py")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

template = f'''"""
Mixed manually/automatically generated feature lists (Day 9 – Step 2)
------------------------------------------------
numeric_features      : Continuous/discrete numeric columns
categorical_low_card  : Cardinality ≤ {MAX_LOW_CARD}
categorical_high_card : Cardinality > {MAX_LOW_CARD}
"""

numeric_features = {numeric_features}

categorical_low_card = {low_card}

categorical_high_card = {high_card}
'''

OUT_PATH.write_text(template, encoding="utf-8")
print(f"✅  Saved feature lists → {OUT_PATH}")
