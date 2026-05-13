#!/usr/bin/env bash
set -euo pipefail

# Always run from project root (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Prefer .venv, fallback to venv
if [ -d ".venv" ]; then
  VENV_PATH=".venv"
elif [ -d "venv" ]; then
  VENV_PATH="venv"
else
  echo "Virtual environment not found (.venv or venv)."
  echo "Create it first, e.g. python3 -m venv .venv"
  exit 1
fi

source "$VENV_PATH/bin/activate"

# Ensure dependencies (including shap) are installed in this exact venv.
python -m pip install -r requirements.txt

# Use the same interpreter for streamlit and installed packages.
python -m streamlit run home.py --server.address 127.0.0.1 --server.port 8501