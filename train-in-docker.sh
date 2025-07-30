#!/bin/bash

echo "🎓 Training Heart Anomaly Model in Docker"
echo "========================================="

# Check if training data exists
if [ ! -f "data/processed/real_medical_sequences.npy" ]; then
    echo "📥 Downloading real medical data first..."
    docker run --rm \
        -v $(pwd)/data:/app/data \
        heart-anomaly-training:latest \
        python3 scripts/download_complete_real_data.py
fi

# Train the model
echo "🧠 Training TensorFlow model..."
docker run --rm \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    heart-anomaly-training:latest \
    python3 scripts/train_model_real_medical_data.py

echo "✅ Model training completed!"
echo "📁 Check the models/ directory for trained model files"
