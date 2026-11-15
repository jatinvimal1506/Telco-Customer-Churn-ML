import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

sample_data = {
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 350.75
}

def test_via_testclient():
    from fastapi.testclient import TestClient
    from src.app.main import app

    client = TestClient(app)
    # Test health check
    res_root = client.get("/")
    print("Health check status:", res_root.status_code, "body:", res_root.json())
    assert res_root.status_code == 200

    # Test predict
    res_pred = client.post("/predict", json=sample_data)
    print("Predict status:", res_pred.status_code, "body:", res_pred.json())
    assert res_pred.status_code == 200
    assert "prediction" in res_pred.json()
    print("✅ In-process FastAPI TestClient test PASSED!")

def test_via_http():
    import requests
    url = "http://127.0.0.1:8000/predict"
    try:
        response = requests.post(url, json=sample_data, timeout=3)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print(f"Server not running at {url} ({e}). Running in-process TestClient instead...")
        test_via_testclient()

if __name__ == "__main__":
    test_via_http()

