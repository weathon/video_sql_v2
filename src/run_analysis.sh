#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
exec .venv_baselines/bin/python -u -m src.analysis runs/mmlifelong-week/collection/questions/*/batch.json --output runs/analysis
