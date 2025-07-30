#!/bin/bash

echo "🐳 Building Heart Anomaly Detection Docker Images"
echo "=================================================="

# Build main application image
echo "📦 Building main application image..."
docker build -t heart-anomaly-api:latest .

# Build training image
echo "📦 Building training image..."
docker build -f Dockerfile.training -t heart-anomaly-training:latest .

echo "✅ Docker images built successfully!"
echo ""
echo "Available images:"
docker images | grep heart-anomaly

echo ""
echo "🚀 To run the application:"
echo "   docker-compose up -d"
echo ""
echo "🎓 To train models:"
echo "   docker run -v \$(pwd)/models:/app/models -v \$(pwd)/data:/app/data heart-anomaly-training:latest"
