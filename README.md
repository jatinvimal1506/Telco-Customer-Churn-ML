# Telco Customer Churn - End-to-End MLOps Pipeline

Full machine-learning solution for predicting customer churn in a telecom setting—from automated data validation and feature engineering to XGBoost modeling, experiment tracking with MLflow, and serving via FastAPI & Gradio.

---

## 🚀 Quick Start

### 1. Environment Setup
Activate the virtual environment:
```bash
# Windows
.\venv\Scripts\activate

# Linux / Mac
source venv/bin/activate
```

Install dependencies if needed:
```bash
pip install -r requirements.txt
```

---

## 🛠️ Pipeline Execution

### Complete ML Training Pipeline
Runs sequentially: Data Loading → Great Expectations Validation → Data Cleaning → Feature Engineering → XGBoost Training → Evaluation → Artifact & Model Logging.
```bash
python scripts/run_pipeline.py --input data/raw/telco_churn.csv --target Churn
```

### Prepare Processed Data
Creates the processed dataset directly:
```bash
python scripts/prepare_processed_data.py
```

---

## 🧪 Pipeline Testing

```bash
# Test Phase 1: Data loading, preprocessing & feature engineering
python scripts/test_pipeline_phase1_data_features.py

# Test Phase 2: XGBoost modeling & hyperparameter optimization (Optuna)
python scripts/test_pipeline_phase2_modeling.py

# Test Serving: FastAPI endpoint & in-process TestClient
python scripts/test_fastapi.py
```

---

## 🌐 Local Web App & API Serving

Start the unified FastAPI REST API + Gradio interactive UI:
```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```
- **REST API Health Check**: `GET http://localhost:8000/`
- **Predict API Endpoint**: `POST http://localhost:8000/predict`
- **Interactive Web UI**: `http://localhost:8000/ui`
- **Interactive Swagger Docs**: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t telco-churn-app .

# Run container
docker run -p 8000:8000 telco-churn-app
```

---

## 📁 Repository Structure

```text
telco-churn/
├── data/
│   ├── raw/
│   │   └── telco_churn.csv              # Raw input dataset
│   └── processed/
│       └── telco_churn_processed.csv    # Engineered features dataset
├── notebooks/
│   └── eda01.ipynb                      # Exploratory Data Analysis & experimentation
├── scripts/
│   ├── run_pipeline.py                  # End-to-end training & MLflow logging pipeline
│   ├── prepare_processed_data.py        # Generates processed features file
│   ├── test_pipeline_phase1_data_features.py
│   ├── test_pipeline_phase2_modeling.py
│   └── test_fastapi.py
├── src/
│   ├── data/
│   │   ├── load_data.py                 # Safe data loading
│   │   └── preprocess.py                # Data cleaning & type coercion
│   ├── features/
│   │   └── build_features.py            # Binary & one-hot feature encoding
│   ├── models/
│   │   ├── train.py                     # XGBoost training & metric logging
│   │   ├── evaluate.py                  # Evaluation report & confusion matrix
│   │   └── tune.py                      # Optuna hyperparameter tuning
│   ├── serving/
│   │   ├── inference.py                 # Serving-time transformation & predictions
│   │   └── model/                       # Serialized models and feature schemas
│   ├── utils/
│   │   ├── utils.py                     # Logging configuration
│   │   └── validate_data.py             # Great Expectations data validation suite
│   └── app/
│       ├── main.py                      # FastAPI + mounted Gradio web UI
│       └── app.py                       # Alternative lightweight app entry point
├── artifacts/                           # Saved feature lists and models
├── mlruns/                              # MLflow tracking backend
├── dockerfile                           # Container definition
├── requirements.txt                     # Production requirements
└── requirements-dev.txt                 # Dev requirements
```

