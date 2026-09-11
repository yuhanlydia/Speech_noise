#!/usr/bin/env bash
set -euo pipefail
python -m sar.validity_smoke \
  --config configs/experiment/mvp_capability_gate.yaml \
  --source-manifest data/mvp/source.jsonl \
  --eligible-manifest data/mvp/pairs_eligible.jsonl
