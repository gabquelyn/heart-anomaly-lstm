import requests
import random
import math

API_URL = "http://localhost:8000"

def generate_normal_data():
    return [
        [
            random.uniform(-0.25, 0.25) + math.sin(i * 0.1),  # ECG
            random.uniform(75, 85),                           # HR
            random.uniform(97, 100),                          # SpO2
            random.uniform(120, 130),                         # Systolic BP
            random.uniform(80, 85),                           # Diastolic BP
        ]
        for i in range(100)
    ]

def generate_abnormal_data():
    return [
        [
            random.uniform(-0.75, 0.75) + math.sin(i * 0.2),  # ECG
            random.uniform(150, 180),                         # HR
            random.uniform(88, 96),                           # SpO2
            random.uniform(180, 200),                         # Systolic BP
            random.uniform(110, 125),                         # Diastolic BP
        ]
        for i in range(100)
    ]

def test_health():
    print("\n🔍 Testing health check...")
    res = requests.get(f"{API_URL}/health")
    res.raise_for_status()
    data = res.json()
    print("✅ Health Check:")
    print(f"   Status: {data.get('status')}")
    print(f"   Metrics: {data.get('metrics')}")

def test_prediction(data, label):
    print(f"\n📡 Testing {label} cardiac prediction...")
    payload = {
        "data": data,
        "patient_info": {
            "patient_id": f"DOCKER_TEST_{label.upper()}",
            "age": 65 if label == "normal" else 70,
            "gender": "M" if label == "normal" else "F",
        },
        "device_id": f"DOCKER_ECG_{label.upper()}",
    }

    res = requests.post(f"{API_URL}/predict", json=payload)
    res.raise_for_status()
    result = res.json()

    pred = result["prediction"]
    print(f"✅ {label.capitalize()} Data Prediction:")
    print(f"   Risk Level: {pred.get('risk_level')}")
    print(f"   Probability: {round(pred.get('anomaly_probability', 0) * 100, 1)}%")
    print(f"   Urgency: {pred.get('urgency')}")
    print(f"   Clinical: {result.get('clinical_assessment', {}).get('interpretation')}")
    print(f"   Model Type: {result.get('model_info', {}).get('model_type')}")
    print(f"   TensorFlow: {result.get('model_info', {}).get('tensorflow_version')}")

if __name__ == "__main__":
    print("🐳 TESTING TENSORFLOW DOCKER API")
    print("=" * 50)

    try:
        test_health()
        test_prediction(generate_normal_data(), "normal")
        test_prediction(generate_abnormal_data(), "abnormal")
        print("\n🎉 Docker TensorFlow API tests completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("💡 Make sure the Docker container is running:")
        print("   docker-compose up -d")
