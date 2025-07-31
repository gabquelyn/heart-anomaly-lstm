from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow import keras
from typing import List, Dict, Any
import numpy as np
import json
import os

app = FastAPI()

# Load the model
try:
    model = keras.models.load_model("models/heart_anomaly_medical_lstm_176k.keras")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# Load saved metrics from JSON file
metrics = {}
metrics_path = "models/medical_metrics_176k.json"
if os.path.exists(metrics_path):
    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
        print("✅ Metrics loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load metrics: {e}")


class PatientInfo(BaseModel):
    patient_id: str
    age: int
    gender: str

class PredictionRequest(BaseModel):
    data: List[List[float]]
    patient_info: PatientInfo
    device_id: str

@app.get("/health")
def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "Model ready",
        "metrics": metrics or "Metrics file not found or failed to load"
    }

@app.post("/predict")
def predict(data: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_array = np.array(data.data).reshape(1, 100, 5)  # Assuming 100 timesteps and 5 features
        prediction = model.predict(input_array)

        result = {
            "prediction": {
                "risk_level": "high" if prediction[0][0] > 0.5 else "low",
                "anomaly_probability": float(prediction[0][0]),
                "urgency": "Immediate" if prediction[0][0] > 0.75 else "Monitor"
            },
            "clinical_assessment": {
                "interpretation": "Possible cardiac anomaly detected. Recommend further evaluation." if prediction[0][0] > 0.5 else "Normal cardiac activity."
            },
            "model_info": {
                "model_type": "LSTM",
                "tensorflow_version": keras.__version__
            }
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
