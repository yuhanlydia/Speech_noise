#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:-configs/experiment/mvp_daa_declared.yaml}
python -m sar.daa_smoke --config "$CONFIG"
