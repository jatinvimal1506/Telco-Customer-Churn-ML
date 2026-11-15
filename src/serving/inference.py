"""
INFERENCE PIPELINE - Production ML Model Serving with Feature Consistency
=========================================================================

This module provides the core inference functionality for the Telco Churn prediction model.
It ensures that serving-time feature transformations exactly match training-time transformations,
which is CRITICAL for model accuracy in production.

Key Responsibilities:
1. Load MLflow-logged model and feature metadata from training
2. Apply identical feature transformations as used during training
3. Ensure correct feature ordering for model input
4. Convert model predictions to user-friendly output
"""

import os
import glob
import json

os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import pandas as pd
import mlflow
import mlflow.pyfunc

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# === MODEL RESOLUTION ===
def _find_model_and_features():
    """Locate the model path and feature columns from artifacts or MLflow."""
    candidate_model_dirs = []
    
    if "MODEL_DIR" in os.environ and os.path.exists(os.environ["MODEL_DIR"]):
        candidate_model_dirs.append(os.environ["MODEL_DIR"])
        
    candidate_model_dirs.extend([
        "/app/model",
        os.path.join(PROJECT_ROOT, "artifacts", "model"),
        os.path.join(PROJECT_ROOT, "src", "serving", "model", "3b1a41221fc44548aed629fa42b762e0", "artifacts", "model"),
        os.path.join(PROJECT_ROOT, "src", "serving", "model", "2ac205f95a264d49b964ab362fe5f4e6", "artifacts", "model"),
    ])
    
    # Also search mlruns and serving/model globs
    candidate_model_dirs.extend(glob.glob(os.path.join(PROJECT_ROOT, "mlruns", "*", "*", "artifacts", "model")))
    candidate_model_dirs.extend(glob.glob(os.path.join(PROJECT_ROOT, "src", "serving", "model", "*", "artifacts", "model")))

    model_dir = None
    loaded_model = None

    for cand in candidate_model_dirs:
        if os.path.exists(cand):
            try:
                loaded_model = mlflow.pyfunc.load_model(cand)
                model_dir = cand
                print(f"✅ Model loaded successfully from {model_dir}")
                break
            except Exception as e:
                continue

    if loaded_model is None:
        raise RuntimeError(f"Could not load MLflow model from any candidate path: {candidate_model_dirs}")

    # === FEATURE COLUMNS RESOLUTION ===
    feature_cols = None
    candidate_feature_files = []
    if model_dir:
        candidate_feature_files.extend([
            os.path.join(model_dir, "feature_columns.txt"),
            os.path.join(model_dir, "..", "feature_columns.txt"),
            os.path.join(os.path.dirname(model_dir), "feature_columns.txt"),
        ])
    candidate_feature_files.extend([
        os.path.join(PROJECT_ROOT, "artifacts", "feature_columns.txt"),
        os.path.join(PROJECT_ROOT, "artifacts", "feature_columns.json"),
        os.path.join(PROJECT_ROOT, "src", "serving", "model", "3b1a41221fc44548aed629fa42b762e0", "artifacts", "feature_columns.txt"),
    ])
    candidate_feature_files.extend(glob.glob(os.path.join(PROJECT_ROOT, "mlruns", "*", "*", "artifacts", "feature_columns.txt")))
    candidate_feature_files.extend(glob.glob(os.path.join(PROJECT_ROOT, "src", "serving", "model", "*", "artifacts", "feature_columns.txt")))

    for fpath in candidate_feature_files:
        if os.path.exists(fpath):
            try:
                if fpath.endswith(".json"):
                    with open(fpath, "r") as f:
                        feature_cols = json.load(f)
                else:
                    with open(fpath, "r") as f:
                        feature_cols = [ln.strip() for ln in f if ln.strip()]
                if feature_cols:
                    print(f"✅ Loaded {len(feature_cols)} feature columns from {fpath}")
                    break
            except Exception:
                continue

    if not feature_cols:
        raise RuntimeError("Failed to load feature columns for inference.")

    return loaded_model, model_dir, feature_cols

model, MODEL_DIR, FEATURE_COLS = _find_model_and_features()

# === FEATURE TRANSFORMATION CONSTANTS ===
BINARY_MAP = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": {"No": 0, "Yes": 1},
    "Dependents": {"No": 0, "Yes": 1},
    "PhoneService": {"No": 0, "Yes": 1},
    "PaperlessBilling": {"No": 0, "Yes": 1},
}

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

def _serve_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply identical feature transformations as used during model training.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Drop id if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 1. Numeric Type Coercion
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
        elif c == "SeniorCitizen":
            df["SeniorCitizen"] = 0

    # 2. Binary Feature Encoding
    for c, mapping in BINARY_MAP.items():
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.strip()
                .map(mapping)
                .astype("Int64")
                .fillna(0)
                .astype(int)
            )

    # 3. One-Hot Encoding for Remaining Categorical Features
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns]
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    # 4. Boolean to Integer Conversion
    bool_cols = df.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # 5. Feature Alignment with Training Schema
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)

    return df

def predict_proba(input_dict: dict) -> float:
    """
    Returns probability of churn for the input customer data.
    """
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)
    
    # Try predict_proba via pyfunc or underlying model
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_enc)[0, 1]
    else:
        # Check underlying model object
        underlying = getattr(model, "_model_impl", None)
        if underlying and hasattr(underlying, "predict_proba"):
            prob = underlying.predict_proba(df_enc)[0, 1]
        else:
            preds = model.predict(df_enc)
            prob = float(preds[0])
    return float(prob)

def predict(input_dict: dict) -> str:
    """
    Main prediction function for customer churn inference.

    Returns:
        "Likely to churn" or "Not likely to churn"
    """
    df = pd.DataFrame([input_dict])
    df_enc = _serve_transform(df)

    try:
        preds = model.predict(df_enc)
        if hasattr(preds, "tolist"):
            preds = preds.tolist()

        if isinstance(preds, (list, tuple)) and len(preds) == 1:
            result = preds[0]
        else:
            result = preds
    except Exception as e:
        raise Exception(f"Model prediction failed: {e}")

    if result == 1 or (isinstance(result, (float, int)) and result >= 0.35):
        return "Likely to churn"
    else:
        return "Not likely to churn"
