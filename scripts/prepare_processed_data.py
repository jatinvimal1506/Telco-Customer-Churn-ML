import os
import sys
import pandas as pd

# make src importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

def find_raw_file():
    candidates = [
        os.path.join(project_root, "data", "raw", "telco_churn.csv"),
        os.path.join(project_root, "data", "raw", "Telco-Customer-Churn.csv"),
        "data/raw/telco_churn.csv",
        "data/raw/Telco-Customer-Churn.csv"
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("Raw data file not found in data/raw/")

RAW = find_raw_file()
OUT = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")

print(f"📖 Reading raw data from: {RAW}")
df = pd.read_csv(RAW)

# Preprocess
df = preprocess_data(df, target_col="Churn")

# Target to 0/1
if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

assert df["Churn"].isna().sum() == 0, "Churn has NaNs after preprocess"
assert set(df["Churn"].unique()) <= {0, 1}, "Churn not 0/1 after preprocess"

# Features
df_processed = build_features(df, target_col="Churn")

# Save
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)
print(f"✅ Processed dataset saved to {OUT} | Shape: {df_processed.shape}")

