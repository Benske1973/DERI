#!/usr/bin/env bash
set -euo pipefail

python3 init_db.py
python3 run_bot.py

