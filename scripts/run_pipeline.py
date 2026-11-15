#!/usr/bin/env python3
"""
Telco Customer Churn ML Training Pipeline
Runs sequentially: load -> validate -> preprocess -> feature engineering -> train -> evaluate -> log artifacts
"""

import os
import sys

# Ensure MLflow allows file-based tracking in MLflow 3.x
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

import time
import json
import joblib
import argparse
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# Fix import path for local modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Local modules
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features
from src.utils.validate_data import validate_telco_data

def resolve_input_file(user_input: str) -> str:
    """Resolve input path, checking common filenames if default doesn't exist."""
    if os.path.exists(user_input):
        return user_input
    
    # Try relative to project root
    candidate = os.path.join(project_root, user_input)
    if os.path.exists(candidate):
        return candidate

    # Try common alternatives
    for alt in [
        os.path.join(project_root, "data", "raw", "telco_churn.csv"),
        os.path.join(project_root, "data", "raw", "Telco-Customer-Churn.csv"),
        os.path.join("data", "raw", "telco_churn.csv"),
        os.path.join("data", "raw", "Telco-Customer-Churn.csv"),
    ]:
        if os.path.exists(alt):
            return alt

    raise FileNotFoundError(f"Input data file '{user_input}' could not be found.")

def main(args):
    """
    Main training pipeline orchestrator.
    """
    input_file = resolve_input_file(args.input)
    print(f"Using input dataset: {input_file}")

    # === MLflow Setup ===
    mlruns_path = args.mlflow_uri or f"file:///{os.path.abspath(os.path.join(project_root, 'mlruns')).replace(os.sep, '/')}"
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"Started MLflow run ID: {run_id}")

        # Log params
        mlflow.log_param("model", "xgboost")
        mlflow.log_param("threshold", args.threshold)
        mlflow.log_param("test_size", args.test_size)

        # === STAGE 1: Data Loading & Validation ===
        print("Loading data...")
        df = load_data(input_file)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

        print("Validating data quality with Great Expectations...")
        is_valid, failed = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass", int(is_valid))

        if not is_valid:
            mlflow.log_text(json.dumps(failed, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"Data quality check failed. Issues: {failed}")
        else:
            print("Data validation passed. Logged to MLflow.")

        # === STAGE 2: Data Preprocessing ===
        print("Preprocessing data...")
        target = args.target
        df = preprocess_data(df, target_col=target)

        processed_path = os.path.join(project_root, "data", "processed", "telco_churn_processed.csv")
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        df.to_csv(processed_path, index=False)
        print(f"Processed dataset saved to {processed_path} | Shape: {df.shape}")

        # === STAGE 3: Feature Engineering ===
        print(" Building features...")
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in data")

        df_enc = build_features(df, target_col=target)

        for c in df_enc.select_dtypes(include=["bool"]).columns:
            df_enc[c] = df_enc[c].astype(int)
        print(f"Feature engineering completed: {df_enc.shape[1]} features")

        # Save Feature Metadata for Serving Consistency
        artifacts_dir = os.path.join(project_root, "artifacts")
        os.makedirs(artifacts_dir, exist_ok=True)

        feature_cols = list(df_enc.drop(columns=[target]).columns)

        with open(os.path.join(artifacts_dir, "feature_columns.json"), "w") as f:
            json.dump(feature_cols, f, indent=2)

        with open(os.path.join(artifacts_dir, "feature_columns.txt"), "w") as f:
            f.write("\n".join(feature_cols))

        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")

        preprocessing_artifact = {
            "feature_columns": feature_cols,
            "target": target
        }
        joblib.dump(preprocessing_artifact, os.path.join(artifacts_dir, "preprocessing.pkl"))
        mlflow.log_artifact(os.path.join(artifacts_dir, "preprocessing.pkl"))
        print(f"Saved {len(feature_cols)} feature columns for serving consistency")

        # === STAGE 4: Train/Test Split ===
        print("Splitting data...")
        X = df_enc.drop(columns=[target])
        y = df_enc[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=args.test_size,
            stratify=y,
            random_state=42
        )
        print(f"Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f" Class imbalance ratio: {scale_pos_weight:.2f} (applied to positive class)")

        # === STAGE 5: Model Training ===
        print(" Training XGBoost model...")
        model = XGBClassifier(
            n_estimators=301,
            learning_rate=0.034,
            max_depth=7,
            subsample=0.95,
            colsample_bytree=0.98,
            n_jobs=-1,
            random_state=42,
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight
        )

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("train_time", train_time)
        print(f"Model trained in {train_time:.2f} seconds")

        # === STAGE 6: Model Evaluation ===
        print("Evaluating model performance...")
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= args.threshold).astype(int)
        pred_time = time.time() - t1
        mlflow.log_metric("pred_time", pred_time)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        print(f"Model Performance:")
        print(f"Precision: {precision:.3f} | Recall: {recall:.3f}")
        print(f"F1 Score: {f1:.3f} | ROC AUC: {roc_auc:.3f}")

        # === STAGE 7: Model Serialization and Logging ===
        print("Saving model to MLflow and local artifacts...")
        mlflow.xgboost.log_model(model, artifact_path="model")

        # Also save model artifacts directly to artifacts/model for immediate serving
        local_model_dir = os.path.join(artifacts_dir, "model")
        if os.path.exists(local_model_dir):
            import shutil
            shutil.rmtree(local_model_dir)
        mlflow.xgboost.save_model(model, path=local_model_dir)

        # Copy feature_columns.txt directly into artifacts/model as well for Docker/convenience
        with open(os.path.join(local_model_dir, "feature_columns.txt"), "w") as f:
            f.write("\n".join(feature_cols))

        print(f"Model saved to MLflow and {local_model_dir}")

        print(f"\nPerformance Summary:")
        print(f"   Training time: {train_time:.2f}s")
        print(f"   Inference time: {pred_time:.4f}s")
        print(f"   Samples per second: {len(X_test)/pred_time:.0f}")

        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))

        return model

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run churn pipeline with XGBoost + MLflow")
    p.add_argument("--input", type=str, default="data/raw/telco_churn.csv",
                   help="path to CSV (e.g., data/raw/telco_churn.csv)")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn")
    p.add_argument("--mlflow_uri", type=str, default=None,
                   help="override MLflow tracking URI, else uses project_root/mlruns")

    args = p.parse_args()
    main(args)
