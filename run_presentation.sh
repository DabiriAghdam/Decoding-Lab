#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8000}"
export PORT
export PYTHONWARNINGS=ignore
if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

eval "$(conda shell.bash hook)"
conda activate presentation
python3 /Users/ahda/Desktop/Presentation/server.py
