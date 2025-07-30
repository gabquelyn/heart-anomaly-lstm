#!/bin/bash

echo "🐳 Starting Heart Anomaly Detection System"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Build images if they don't exist
if [[ "$(docker images -q heart-anomaly-api:latest 2> /dev/null)" == "" ]]; then
    echo "📦 Building Docker images..."
    ./docker-build.sh
fi

# Start the services
echo "🚀 Starting services with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Show logs
echo "📋 Recent logs:"
docker-compose logs --tail=20 heart-anomaly-api

echo ""
echo "✅ Heart Anomaly Detection System is running!"
echo "🔗 API: http://localhost:3000"
echo "🔗 Health: http://localhost:3000/health"
echo "🔗 Model Info: http://localhost:3000/model/info"
echo ""
echo "📊 To view logs: docker-compose logs -f heart-anomaly-api"
echo "🛑 To stop: docker-compose down"
