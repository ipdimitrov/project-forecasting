#!/bin/bash

echo "Starting Forecasting API locally..."

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Error: .env file not found!"
    echo "Please create a .env file with your configuration."
    echo "You can copy .env.example and modify it:"
    echo "   cp .env.example .env"
    echo "   nano .env  # or use your preferred editor"
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Python 3 not found. Please install Python 3.12+ and try again."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Creating one..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment"
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import fastapi, uvicorn" &> /dev/null; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies"
        exit 1
    fi
fi

# Change to project directory
cd "$(dirname "$0")/.." || { echo "Failed to change to project directory"; exit 1; }

echo "Starting API server..."
echo "Loading configuration from .env file..."
echo "API will be available at: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"

# Start the API
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload