# Use official Python image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System dependencies (optional: curl for healthcheck)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements_real_data.txt .
RUN pip install --no-cache-dir -r requirements_real_data.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/processed

# Copy any existing data
COPY data/ ./data/

# Expose FastAPI port
EXPOSE 8000

# Health check (adjust if you add a real endpoint)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
