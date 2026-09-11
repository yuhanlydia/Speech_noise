# V2 run: stop at the Base gate (exploratory)

Executed on NVIDIA RTX A4000 16GB, as explicitly requested in place of the
RTX 3090 named in the task. The 3B NF4 configuration completed capability,
real hook identity, and Mixed Base without OOM. GPU memory sampled every
2 seconds peaked at 4,941 MiB (about 4.83 GiB); this is a sampled high-water
mark, not an allocator-level guarantee for unrun DAA/generation stages.

**Protocol decision: STOP at TASK_V2.md step 5.** Base PairSwitchAcc is
56/63 = 0.888889, satisfying both `>= 0.80` and
`>= min(IgnoreAcc, UseAcc) - 0.05 = 0.870635`.
Oracle and self-declared experiments were therefore not run. No claim about
Oracle efficacy or a causal KV-mask effect can be made from this run.

**Evidence tier: exploratory only.** Capability produced 63 eligible pairs,
one below the 64-pair formal threshold. We retained the original seed, data,
model and eligibility rule. This result does not establish that relevance
switching failures never occur; it says this specific V2 setup meets its
predefined Base stopping rule.

## Required fields

```text
HEAD: c06d9fb5672a8edc3c8dc6fa402e654420416a26 (code plus verified compatibility fix)
Starting HEAD: 4aebdf1e44dbbe3ac490800726f69710b72037a8
GPU: NVIDIA RTX A4000, 16376 MiB
pytest result: 84 passed in 4.83s
data audit ok/errors: true / 0 (128 fresh pairs)
capability target_only_acc: 0.4921875 (63/128)
event_only_acc: 0.9921875 (127/128)
eligible_pairs / total: 63 / 128; exploratory only
hook identity ok / max_abs_logprob_diff: true / 0.0; predictions_match=true
Base IgnoreAcc / UseAcc / SAR / PairSwitchAcc: 0.9206349206349206 / 0.9682539682539683 / 0.943844204348406 / 0.8888888888888888
Oracle-Prompt-only PairSwitchAcc: NOT RUN — Base stop gate
Oracle-KV PairSwitchAcc / oracle_total_gain / oracle_kv_gain / rescue_fraction: NOT RUN / N/A / N/A / N/A — Base stop gate
Self Prompt-only PairSwitchAcc (if run): NOT RUN — Base stop gate
Fixed DAA PairSwitchAcc / SelectionSwitchAcc / ignore_target_coverage_mean (if run): NOT RUN / N/A / N/A — Base stop gate
Declared DAA results (only if run): NOT RUN — Base stop gate
any OOM / protocol failure stages: 0 OOM; preparation and hook integration errors resolved as described below; no failed scientific gate was bypassed
```

All 126 Base rows correspond exactly to the 63 eligible pairs. Paired waveform
hashes match and every recorded Base option log-probability is finite.
Correct counts are 58 ignore, 61 use, and 56 both. See
[pair_outcomes.csv](pair_outcomes.csv) for all 128 pairs; blank Base cells mean
the pair was ineligible and was not evaluated in Base.

## Environment and resolved execution errors

- Fresh checkout used the original 128-pair `dev`, ESC-50, seed 0 recipe;
  no old data or summaries were reused.
- Direct `python scripts/prepare_public_mvp.py` initially could not import
  `scripts`. Running with `PYTHONPATH` set to the repository root resolved it;
  data generation code was unchanged.
- The original missing-dependency unit test attempted a real model load on a
  GPU-equipped environment. It now explicitly simulates absent Transformers.
- Qwen processor required additional `torchvision`, `Pillow`, and `librosa`
  packages. Triton required `build-essential` and `python3-dev`. Both errors
  occurred before capability produced results and were resolved before the
  complete capability run. `espeak-ng` and `ffmpeg` were also installed.
- Transformers 5.17.0 moved multimodal frequency composition into its rotary
  embedding module and removed `apply_multimodal_rotary_pos_emb`. The hook now
  selects the upstream API explicitly without catching/retrying rotary errors.
  Native attention comparisons with full and selective audio masks pass.
  The original real-model identity check was rerun unchanged and passed with
  **0.0** difference at the required **1e-4** tolerance.
- Scientific configuration, scorer fail-fast behavior, datasets, prompts,
  eligibility and stopping thresholds were not changed. The only runtime
  source change is the DAA rotary API compatibility fix.

Exact package versions, model revision, manifest hashes, and environment
settings are in [environment.json](environment.json) and
[requirements-frozen.txt](requirements-frozen.txt). For reproduction, use a
fresh checkout of the code HEAD above and follow `TASK_V2.md` with:

```bash
source .venv/bin/activate
pip install -e '.[dev,data,gpu]'
pip install torchvision Pillow librosa
export PYTHONPATH="$PWD"
export OMP_NUM_THREADS=4
```

Install the system packages listed above first. The frozen requirements record
the exact versions used rather than promising equivalence to future releases.
Follow every gate, including the Base stop rule; the remaining stages are not
an unattended command queue.

## Exported evidence

- [summary.json](summary.json) and [SUMMARY.md](SUMMARY.md): output of the
  repository's `scripts/export_run_v2.py`; unrun stages remain `null` / `not run`.
- [decision.json](decision.json): evaluated thresholds and downstream skip reasons.
- [data_audit.json](data_audit.json): complete fresh-data audit.
- [validated_counts.json](validated_counts.json): counts independently checked
  against the per-pair results.
- [pytest.txt](pytest.txt): final test output.

Generated WAVs and the `results/` directory remain untracked. Full execution
logs and predictions remain locally under `results/`.
