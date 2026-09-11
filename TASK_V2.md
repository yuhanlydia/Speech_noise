# TASK V2 — RTX 3090 Decisive Run

Use branch `feature/daa-mvp`. Do not reuse old generated data or old result summaries.

## 0. Sync and install

```bash
git fetch origin
git checkout feature/daa-mvp
git pull --ff-only origin feature/daa-mvp

source .venv/bin/activate
pip install -e '.[dev,data,gpu]'
pytest -q
```

## 1. Rebuild V2 data

```bash
rm -rf data/mvp results/mvp_* results/data_audit reports/run_v2
python scripts/prepare_public_mvp.py \
  --output-dir data/mvp \
  --num-pairs 128 \
  --protocol dev \
  --event-source esc50 \
  --seed 0
```

Old data is invalid for V2 because the prompt policy changed.

## 2. Audit data

```bash
python scripts/audit_mvp_data.py \
  --source-manifest data/mvp/source.jsonl \
  --pairs-manifest data/mvp/pairs.jsonl
```

Stop immediately unless `ok=true` and `num_errors=0`.

## 3. Capability gate

```bash
bash scripts/run_capability_gate.sh
cat results/mvp_capability_gate/summary.json
```

Decision:

- eligible < 32: STOP. Do not interpret this dataset/backbone pairing.
- eligible 32–63: exploratory only.
- eligible >= 64: proceed.

## 4. Mixed Base on eligible pairs

```bash
bash scripts/run_diagnostic.sh
cat results/mvp_diagnostic/summary.json
```

STOP the project if:

```text
PairSwitchAcc >= 0.80
AND PairSwitchAcc >= min(IgnoreAcc, UseAcc) - 0.05
```

## 5. Oracle-KV — mandatory before self-declared DAA

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_oracle.yaml
cat results/mvp_daa_oracle/summary.json
```

Compute:

```text
oracle_gain = oracle_pair_switch - base_pair_switch
rescue_fraction = oracle_gain / (1 - base_pair_switch)
```

STOP the DAA/KV line if `oracle_gain < 0.05` OR `rescue_fraction < 0.15`.
Do not run more complex routing to rescue a failed Oracle gate.

## 6. Only if Oracle passes

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_prompt_only.yaml
bash scripts/run_daa.sh configs/experiment/mvp_daa_fixed.yaml
```

Compare:

```text
Fixed DAA PairSwitchAcc - Prompt-only PairSwitchAcc
```

If gain < 0.03, the evidence for a causal KV contribution is weak.

If Oracle passes but fixed DAA SelectionSwitchAcc < 0.50, run at most one 7B NF4 replication. If that also fails, stop the zero-shot selector.

## 7. Optional only after fixed DAA succeeds

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_declared.yaml
```

This is a segmentation ablation, not the primary method.

## 8. Export results so ChatGPT can inspect them

```bash
python scripts/export_run_v2.py

git add reports/run_v2
git commit -m "results: add validity-gated DAA V2 run"
git push origin feature/daa-mvp
```

Do not commit generated WAVs or `results/`.

## Required report back

Report these fields verbatim:

```text
HEAD
GPU
pytest result
data audit ok/errors
capability target_only_acc
event_only_acc
eligible_pairs / total
Base IgnoreAcc / UseAcc / SAR / PairSwitchAcc
Oracle PairSwitchAcc / oracle_gain / rescue_fraction
Prompt-only PairSwitchAcc (if run)
Fixed DAA PairSwitchAcc / SelectionSwitchAcc / ignore_target_coverage_mean (if run)
Declared DAA results (only if run)
any OOM / protocol failure stages
```
