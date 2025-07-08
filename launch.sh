#!/bin/bash
# DataViz Launch Script for macOS/Linux

echo "Starting DataViz - Intelligent EDA Web App..."
echo

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install/update requirements
echo "Installing/updating packages..."
pip install -r requirements.txt

# Launch the application
echo
echo "Launching DataViz application..."
echo "Open your browser and navigate to: http://localhost:8501"
echo "Press Ctrl+C to stop the application."
echo

streamlit run app.py
