#!/bin/bash

set -e

if [ ! -d ".venv" ]; then
  echo "❌ .venv not found. Run: make install or setup.sh"
  exit 1
fi

source .venv/bin/activate

echo "🚀 Running training..."
python scripts/train.py "$@"  # Pass through any command line arguments 