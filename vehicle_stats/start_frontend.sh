#!/bin/bash

echo "Starting Vehicle Statistics Frontend..."

# Navigate to frontend directory
cd "$(dirname "$0")/frontend"

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Start React development server
echo "Starting React app on http://localhost:3000"
npm start
