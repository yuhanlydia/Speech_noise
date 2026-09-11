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
rm -rf data/mvp results/mvp_* results/data_audit results/daa_identity reports/run_v2
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

## 4. Real-Qwen DAA hook identity sanity

Before interpreting any masked result, verify the custom attention hook is an identity when every audio token remains allowed:

```bash
python scripts/check_daa_identity.py
cat results/daa_identity/summary.json
```

Required:

```text
ok = true
predictions_match = true
max_abs_logprob_diff <= 1e-4
```

If this fails, STOP. Treat it as a code / Transformers-integration problem, not a research result.

## 5. Mixed Base on eligible pairs

```bash
bash scripts/run_diagnostic.sh
cat results/mvp_diagnostic/summary.json
```

STOP the whole project if:

```text
PairSwitchAcc >= 0.80
AND PairSwitchAcc >= min(IgnoreAcc, UseAcc) - 0.05
```

## 6. Oracle location controls — mandatory before self-declared DAA

First test whether simply telling the model the ground-truth acoustic region helps, without modifying KV:

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_oracle_prompt_only.yaml
cat results/mvp_daa_oracle_prompt_only/summary.json
```

Then apply the exact same oracle address table and focus declaration with the runtime KV mask:

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_oracle.yaml
cat results/mvp_daa_oracle/summary.json
```

Compute:

```text
oracle_total_gain = Oracle-KV PairSwitchAcc - Base PairSwitchAcc
oracle_kv_gain = Oracle-KV PairSwitchAcc - Oracle-Prompt-only PairSwitchAcc
failure_rescue_fraction = oracle_total_gain / (1 - Base PairSwitchAcc)
```

Decision:

- if `oracle_total_gain < 0.05` OR `failure_rescue_fraction < 0.15`: STOP the whole DAA/focusing line;
- if Oracle-Prompt rescues but `oracle_kv_gain < 0.03`: location prompting helps but the KV-mask claim is weak; do not build a paper around KV control;
- if `oracle_kv_gain >= 0.03`: the runtime KV intervention has a plausible causal effect; proceed.

## 7. Self-declared fixed-block controls

Only run if the Oracle stage justifies continuing:

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_prompt_only.yaml
bash scripts/run_daa.sh configs/experiment/mvp_daa_fixed.yaml
```

Compare:

```text
self_kv_gain = Fixed-DAA PairSwitchAcc - Prompt-only PairSwitchAcc
```

If `self_kv_gain < 0.03`, evidence for the self-declared KV intervention is weak.

If Oracle-KV works but fixed DAA `SelectionSwitchAcc < 0.50`, run at most one Qwen2.5-Omni-7B NF4 replication. If 7B also fails, stop the zero-shot declarative selector.

## 8. Optional only after fixed DAA succeeds

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_declared.yaml
```

This is a segmentation ablation, not the primary method.

## 9. Export results so ChatGPT can inspect them

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
hook identity ok / max_abs_logprob_diff
Base IgnoreAcc / UseAcc / SAR / PairSwitchAcc
Oracle-Prompt-only PairSwitchAcc
Oracle-KV PairSwitchAcc / oracle_total_gain / oracle_kv_gain / rescue_fraction
Self Prompt-only PairSwitchAcc (if run)
Fixed DAA PairSwitchAcc / SelectionSwitchAcc / ignore_target_coverage_mean (if run)
Declared DAA results (only if run)
any OOM / protocol failure stages
```
