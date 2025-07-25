#!/usr/bin/env bash

set -euo pipefail

# Create virtual environment if not already present
if [ ! -d ".venv" ]; then
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Upgrade packaging tools
pip install --upgrade pip setuptools wheel

# Install package in editable mode with dependencies
pip install -e .

echo "Done. Activate the environment with:\n  source .venv/bin/activate"


