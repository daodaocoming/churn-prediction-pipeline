"""
clean_data.py · Telco Churn data‑cleaning script
------------------------------------------------
1. Read data/raw/*.csv
2. Impute missing values:
     • numerical  → column mean
     • categorical → "Unknown"
3. Cast column types:
     • TotalCharges → float
     • SeniorCitizen → bool
     • every Yes/No column → category
4. Save cleaned data to data/clean/telco_clean.parquet

Usage (from project root):

    python -m src.data.clean_data \
        --input  data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
        --output data/clean/telco_clean.parquet
"""
from __future__ import annotations

import click
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["TotalCharges"] = (
        pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True),
                      errors="coerce")
    )

    df["SeniorCitizen"] = df["SeniorCitizen"].astype("bool")
    return df

def impute(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(exclude="number").columns
  
    for col in num_cols:
        if df[col].isna().any():
            mean_val = df[col].mean()
            df[col].fillna(mean_val, inplace=True)

    for col in cat_cols:
        if df[col].isna().any():
            df[col].fillna("Unknown", inplace=True)
    return df

YES_NO = {"Yes", "No"}

def cast_yes_no(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        uniq = set(df[col].dropna().unique())
        if uniq <= YES_NO:
            df[col] = df[col].astype("category")
    return df

@click.command()
@click.option("--input",  "-i", type=click.Path(exists=True, dir_okay=False),
              default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
@click.option("--output", "-o", type=click.Path(dir_okay=False),
              default="data/clean/telco_clean.parquet")
def main(input: str, output: str) -> None:
    inp  = Path(input).resolve()
    outp = Path(output).resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    def rel(p: Path) -> Path:
        try:
            return p.relative_to(Path.cwd())
        except ValueError:
            return p

    print(f"📥  Reading {rel(inp)}")
    df = load_raw(inp)

    print("🧹  Imputing missing values …")
    df = impute(df)

    print("🔖  Casting Yes/No columns …")
    df = cast_yes_no(df)

    print(f"💾  Writing {rel(outp)}")
    df.to_parquet(outp, index=False)
    print("✅  Done. Clean shape:", df.shape)

if __name__ == "__main__":
    main()
