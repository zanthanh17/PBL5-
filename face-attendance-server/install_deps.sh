#!/bin/bash
# Install dependencies for Face Attendance Server

set -e

echo "Installing Python dependencies..."

# Install to user site-packages (since venv has issues)
pip3 install --user -r requirements.txt

echo ""
echo "✓ Dependencies installed successfully!"
echo ""
echo "To run the server:"
echo "  uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

