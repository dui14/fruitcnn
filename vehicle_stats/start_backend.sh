#!/bin/bash

echo "Starting Vehicle Statistics Backend..."

# Navigate to backend directory
cd "$(dirname "$0")/backend"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Create necessary directories
mkdir -p uploads outputs

# Start FastAPI server
echo "Starting FastAPI server on http://localhost:8000"
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
