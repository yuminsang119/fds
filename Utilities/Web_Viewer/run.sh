#!/bin/bash
# FDS Web Viewer - Launch Script
# Usage: ./run.sh [port]

PORT=${1:-8080}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  FDS Web Viewer"
echo "  Fire Dynamics Simulator Visualization"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required"
    exit 1
fi

# Install dependencies
echo "Installing dependencies..."
pip install -q -r "$SCRIPT_DIR/requirements.txt" 2>/dev/null || \
    pip3 install -q -r "$SCRIPT_DIR/requirements.txt"

echo ""
echo "Starting server on port $PORT..."
echo "Open http://localhost:$PORT in your browser"
echo ""
echo "  1. Click 'Load Demo' to load sample fire simulation"
echo "  2. Use mouse to rotate/zoom the 3D view"
echo "  3. Press Play to animate the simulation"
echo ""
echo "Press Ctrl+C to stop"
echo "--------------------------------------------"

cd "$SCRIPT_DIR/backend"
python3 -m uvicorn server:app --host 0.0.0.0 --port "$PORT" --reload
