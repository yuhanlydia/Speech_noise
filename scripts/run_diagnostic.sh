#!/usr/bin/env bash
set -euo pipefail
python -m sar.gpu_smoke --config configs/experiment/mvp_diagnostic.yaml --evaluate "$@"
