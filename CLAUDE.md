# CLAUDE.md

This file provides guidance when working with code in this repository.

## Commands

### Training Pipeline
```bash
# Run the complete ML training pipeline
python scripts/run_pipeline.py --input data/raw/telco_churn.csv --target Churn

# Prepare processed data only
python scripts/prepare_processed_data.py
```

### Testing
```bash
# Test data processing and feature engineering
python scripts/test_pipeline_phase1_data_features.py

# Test model training and evaluation
python scripts/test_pipeline_phase2_modeling.py

# Test FastAPI endpoints
python scripts/test_fastapi.py
```

### Local Development
```bash
# Run the FastAPI + Gradio application locally
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build and run the containerized application
docker build -t telco-churn-app .
docker run -p 8000:8000 telco-churn-app
```

