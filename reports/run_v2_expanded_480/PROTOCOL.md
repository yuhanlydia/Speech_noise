# Expanded V2 development-set stability check

This is a post-hoc extension of the original 128-pair V2 run, requested after
that run met its Base stopping rule. It is not an independent confirmation
experiment, and it does not replace reports/run_v2.

- Source code: 55a2c74cee225486d120ee4cf42e4454c6447d6c.
- Hardware: NVIDIA RTX A4000 16GB.
- Model: Qwen/Qwen2.5-Omni-3B, NF4, float16, thinker only, eager attention.
- Data: 480 pairs; dev protocol; ESC-50; seed 0; all other preparation defaults.
- 480 is the available development ESC-50 candidate count without reusing event
  rows. The 913 eligible MMLU development questions are sufficient.
- All WAVs are regenerated and all model scores recomputed. The original 128
  waveforms are reproduced exactly; 352 pair waveforms are new to this run.
- Keep every eligibility, identity, Base, Oracle and self-routing threshold from
  TASK_V2.md. A scientific stop remains a stop; debugging may repair integration
  faults but may not change a gate to force downstream experimentation.
- Data: data/mvp_expanded_480. Results: results/v2_expanded_480.
- Configs: configs/experiment/v2_expanded_480. Only manifest/output paths differ
  from the original experiment configs. All 7 configs were loaded and compared.
- PYTHONPATH is the repository root, OMP_NUM_THREADS=4, HF_HUB_OFFLINE=1.
  Cached model and dataset revisions are the same as the original run.

Run order: fresh data audit; isolated capability; real all-audio hook identity;
Mixed Base on eligible pairs; then Oracle-Prompt-only, Oracle-KV, and self
conditions only if their preceding gates allow continuation. Export with:

```bash
python scripts/export_run_v2.py \
  --results-root results/v2_expanded_480 \
  --output-dir reports/run_v2_expanded_480
```

Any comparison on the 352 new pairs is descriptive; the gate for this run uses
the full expanded eligible set. This expansion was chosen after observing the
128-pair result, so meeting the numerical eligible-count threshold does not make
this a preregistered, held-out confirmation study.
