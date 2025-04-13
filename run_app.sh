#!/bin/bash
# This script activates the virtual environment and runs the Streamlit app

# Path to the virtual environment
VENV_PATH="./venv"

# Activate the virtual environment
source "$VENV_PATH/bin/activate"

# Run the Streamlit app
streamlit run home.py

# Keep terminal open
echo "Press any key to close..."
read -n 1