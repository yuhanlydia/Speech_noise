#!/usr/bin/env bash
set -euo pipefail
python -m sar.gpu_smoke --config "${1:-configs/experiment/mvp_qacr_eval.yaml}" --evaluate
