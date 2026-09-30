#!/usr/bin/env bash
# ShareGuard setup for macOS / Linux (CPU path). Tested on Linux x86-64, Python 3.13.
set -euo pipefail
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m shareguard.app.main check
python -m pytest -q
