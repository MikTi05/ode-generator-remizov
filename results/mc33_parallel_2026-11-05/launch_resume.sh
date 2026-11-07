#!/bin/sh
set -eu
PYTHONDONTWRITEBYTECODE=1 caffeinate -i .venv/bin/python -B diagnostics/mc33_parallel.py run \
  --out results/mc33_parallel_2026-11-05/production \
  --workers 8 --resume
