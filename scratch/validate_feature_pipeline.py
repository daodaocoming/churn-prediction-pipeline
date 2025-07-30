from pathlib import Path
import pandas as pd

from src.features.feature_pipeline import build_preprocessor, save_pipeline
import src.features.feature_lists as fl

df = pd.read_parquet("data/clean/telco_clean.parquet")
X = df[fl.numeric_features + fl.categorical_low_card]
y = df["Churn"].map({"Yes": 1, "No": 0})

pipe, _ = build_preprocessor()     

X_trans = pipe.fit_transform(X, y)
feat_names = pipe.get_feature_names_out()

print("✅ shape:", X_trans.shape)
print("✅ n_features (=len(feat_names)):", len(feat_names))
assert X_trans.shape[1] == len(feat_names)

save_pipeline(pipe)
print("✅ pipeline saved to models/feature_pipeline.pkl")

