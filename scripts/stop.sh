#!/bin/bash

echo "Stopping Forecasting API..."

# Find and kill uvicorn processes
PIDS=$(pgrep -f "uvicorn.*backend.main:app" 2>/dev/null)

if [ -z "$PIDS" ]; then
    echo "No running API processes found"
    exit 0
fi

echo "Found running API processes: $PIDS"
echo "Stopping processes..."

# Try graceful shutdown first
kill $PIDS 2>/dev/null

# Wait a moment for graceful shutdown
sleep 2

# Check if processes are still running
REMAINING=$(pgrep -f "uvicorn.*backend.main:app" 2>/dev/null)

if [ ! -z "$REMAINING" ]; then
    echo "Processes still running, forcing shutdown..."
    kill -9 $REMAINING 2>/dev/null
fi

# Verify all processes are stopped
FINAL_CHECK=$(pgrep -f "uvicorn.*backend.main:app" 2>/dev/null)

if [ -z "$FINAL_CHECK" ]; then
    echo "All API processes stopped successfully"
else
    echo "Warning: Some processes may still be running"
fi

echo ""
echo "To restart the API:"
echo "   ./scripts/run.sh"