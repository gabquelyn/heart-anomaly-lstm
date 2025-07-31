#!/bin/bash

echo "🎓 Training Heart Anomaly Model in Docker"
echo "========================================="

MODEL_PATH="models/heart_anomaly_medical_lstm_176k.h5"
TFJS_OUTPUT_DIR="models/tfjs_model"

# Check if training data exists
if [ ! -f "data/processed/real_medical_sequences.npy" ]; then
    echo "📥 Downloading real medical data first..."
    docker run --rm \
        -v $(pwd)/data:/app/data \
        heart-anomaly-training:latest \
        python3 scripts/download_complete_real_data.py
fi

# ✅ Skip training if model already exists
if [ -f "$MODEL_PATH" ]; then
    echo "🛑 Model already exists at $MODEL_PATH. Skipping training."
else
    echo "🧠 Training TensorFlow model..."
    docker run --rm \
        -v $(pwd)/models:/app/models \
        -v $(pwd)/data:/app/data \
        heart-anomaly-training:latest \
        python3 scripts/train_model_real_medical_data.py

    echo "✅ Model training completed!"
fi

# # 🔁 Convert to TensorFlow.js if not already converted
# if [ ! -d "$TFJS_OUTPUT_DIR" ]; then
#     echo "🔁 Converting .keras model to TensorFlow.js format..."
#     docker run --rm \
#         -v $(pwd)/models:/app/models \
#         heart-anomaly-training:latest \
#         python3 scripts/convert_to_keras.py
# else
#     echo "⚠️  TFJS model already exists at $TFJS_OUTPUT_DIR. Skipping conversion."
# fi
