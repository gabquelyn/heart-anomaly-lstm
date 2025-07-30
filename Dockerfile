# Use official Node.js runtime as base image
FROM node:18-slim

# Set working directory
WORKDIR /app

# Install Python and system dependencies for TensorFlow
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package.json ./
COPY requirements_real_data.txt ./

# Install Node.js dependencies
RUN npm install

# Install Python dependencies
RUN pip3 install -r requirements_real_data.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p models data/processed

# Expose ports
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:3000/health || exit 1

# Default command
CMD ["node", "server-medical-data.js"]
