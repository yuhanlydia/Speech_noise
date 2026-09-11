# 480-pair V2 expansion: Base stopping result reproduced

The expanded run completed Data audit, Capability, real Hook identity, and
Mixed Base on the RTX A4000 16GB. **Base again meets the protocol stop rule.**
Oracle and self-declared stages were therefore not run.

Capability retained **238/480 pairs**. Base correctly answered both queries for
**213/238 = 89.50%**, above both 80% and
`min(IgnoreAcc, UseAcc) - 0.05 = 87.0168%`.

| Base metric | Original 63 eligible pairs | Expanded 238 eligible pairs | Newly added 175 eligible pairs |
|---|---:|---:|---:|
| IgnoreAcc | 92.06% | 92.02% | 92.00% |
| UseAcc | 96.83% | 97.48% | 97.71% |
| SAR | 94.38% | 94.67% | 94.77% |
| PairSwitchAcc | 88.89% | 89.50% | 89.71% |

The expanded data contain the original 128 pairs plus 352 new pairs. All 128
repeated capability predictions and all 126 repeated Base predictions match
the original run. Repeated Base option log-probabilities also match exactly.
The 175 newly eligible pairs independently contribute 157 both-correct cases;
they are a descriptive subset, not an additional gate or a held-out study.

This is a **post-hoc development-set stability check**, not an independent
confirmation study. Exceeding the numerical 64-eligible threshold does not
change that design limitation. The stable result supports stopping this V2
setup under its predefined Base rule; it makes no claim about untested
Oracle efficacy, causal KV-mask effects, or all possible audio settings.

## Required report fields

```text
HEAD: 55a2c74cee225486d120ee4cf42e4454c6447d6c (runtime code; expanded configs are committed alongside this report)
GPU: NVIDIA RTX A4000, 16376 MiB
pytest result: 84 passed (see pytest.txt for elapsed time)
data audit ok/errors: true / 0; 480 pairs
capability target_only_acc: 0.4979166666666667 (239/480)
event_only_acc: 0.9958333333333333 (478/480)
eligible_pairs / total: 238 / 480
hook identity ok / max_abs_logprob_diff: true / 0.0; predictions_match=true; tolerance=1e-4
Base IgnoreAcc / UseAcc / SAR / PairSwitchAcc: 0.9201680672268907 / 0.9747899159663865 / 0.9466917587434086 / 0.8949579831932774
Oracle-Prompt-only PairSwitchAcc: NOT RUN — Base stop gate
Oracle-KV PairSwitchAcc / oracle_total_gain / oracle_kv_gain / rescue_fraction: NOT RUN / N/A / N/A / N/A — Base stop gate
Self Prompt-only PairSwitchAcc (if run): NOT RUN — Base stop gate
Fixed DAA PairSwitchAcc / SelectionSwitchAcc / ignore_target_coverage_mean (if run): NOT RUN / N/A / N/A — Base stop gate
Declared DAA results (only if run): NOT RUN — Base stop gate
any OOM / protocol failure stages: 0 OOM; 0 runtime errors in this expansion; no scientific stop was bypassed
```

GPU memory sampled every 2 seconds peaked at **6,207 MiB (6.06 GiB)**. This
describes the completed stages; it is not a memory guarantee for unrun DAA or
generation stages. The original run's environment and rotary API fixes were
already present; this expansion changed no runtime source code.

## Reproduction

See [PROTOCOL.md](PROTOCOL.md) for the expansion's scope and stopping rules,
and [environment.json](environment.json) plus
[requirements-frozen.txt](requirements-frozen.txt) for provenance. From the
repository root, with the recorded dependencies and system packages installed:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1  # only when the recorded model/datasets are cached

python scripts/prepare_public_mvp.py --output-dir data/mvp_expanded_480 \
  --num-pairs 480 --protocol dev --event-source esc50 --seed 0
python scripts/audit_mvp_data.py \
  --source-manifest data/mvp_expanded_480/source.jsonl \
  --pairs-manifest data/mvp_expanded_480/pairs.jsonl \
  --output results/v2_expanded_480/data_audit/summary.json
# Continue only if the audit is ok with zero errors.
python -m sar.validity_smoke \
  --config configs/experiment/v2_expanded_480/mvp_capability_gate.yaml \
  --source-manifest data/mvp_expanded_480/source.jsonl \
  --eligible-manifest data/mvp_expanded_480/pairs_eligible.jsonl
# Check the eligible-count gate before continuing.
python scripts/check_daa_identity.py \
  --config configs/experiment/v2_expanded_480/mvp_daa_fixed.yaml \
  --output results/v2_expanded_480/daa_identity/summary.json
# Continue only if identity passes at the original 1e-4 tolerance.
python -m sar.gpu_smoke \
  --config configs/experiment/v2_expanded_480/mvp_diagnostic.yaml --evaluate
# Evaluate the Base stop rule here. This run stopped.
python scripts/export_run_v2.py --results-root results/v2_expanded_480 \
  --output-dir reports/run_v2_expanded_480
```

Use fresh output paths for a new run. The original `reports/run_v2`, original
data, and original results were preserved. Configurations for downstream
conditions are supplied for reproducibility, but their presence does not mean
those conditions were run.

## Evidence

- [summary.json](summary.json) / [SUMMARY.md](SUMMARY.md): repository exporter
  output; unrun stages are null / not run.
- [decision.json](decision.json): exact evaluated stop conditions and skip reasons.
- [comparison.json](comparison.json): full-set and new-subset metrics, plus
  prediction/log-probability reproducibility checks.
- [pair_outcomes.csv](pair_outcomes.csv): all 480 pairs; blank Base cells denote
  ineligible pairs that were not scored in Base.
- [validated_counts.json](validated_counts.json): independently verified counts.
- [data_audit.json](data_audit.json): audit summary and hash of the full local
  per-pair audit. Full details remain under `results/v2_expanded_480/data_audit`.

The 476 Base rows were verified to match the eligible manifest exactly, with
matching within-pair waveform hashes and finite option scores. WAVs and the
`results/` directory are not committed.
