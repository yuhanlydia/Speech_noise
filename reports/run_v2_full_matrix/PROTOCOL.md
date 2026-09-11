# Full V2 comparison matrix — performance stopping rules disabled

This run follows the user's explicit instruction to complete all experimental
conditions without stopping on accuracy, gain, rescue fraction, or selection
thresholds. It supersedes the performance stopping rules of TASK_V2.md for this
run only. Historical stopped-run reports remain unchanged.

## Scope

- Same validated 480-pair dev dataset, seed 0, ESC-50 source.
- RTX A4000 16GB; Qwen2.5-Omni 3B and one 7B NF4 replication.
- For each model: isolated capability, real all-audio hook identity, Mixed Base,
  Oracle-Prompt-only, Oracle-KV, Self Prompt-only, Self Fixed-DAA, and
  model-declared segmentation.
- Performance metrics are measurements, never permission to run the next stage.
  No condition is omitted because its gain or accuracy is low or negative.
- Data integrity and hook identity remain implementation checks. Investigate
  and fix code/environment faults before interpreting affected measurements.
- Keep model-specific capability-eligible manifests for the matched control
  comparisons. Report the eligible counts and common-eligible cross-model
  subset; eligibility-count thresholds do not stop execution.
- Keep original prompts, generation limits, masks, and correctness/selection
  definitions. The 80% retained-target rule describes selection quality; it
  does not decide whether later experiments run.
- Reuse the already verified 3B capability/identity/Base artifacts from the
  480-pair expansion. Record provenance rather than presenting them as new runs.
- All five previously unrun 3B DAA conditions are newly evaluated. All 7B
  conditions are evaluated in this full-matrix run.
- Positive, neutral, negative, and malformed-declaration outcomes are reported.
  Parsing failures remain visible and are distinguished from runtime bugs.

## Reuse and comparison

Configs under configs/experiment/v2_full_matrix/{3b,7b} share the same structure.
Model ID, model-specific manifest path, and output directory differ. The 7B
Self/declared configs also enable the validated declaration decoding optimization
described below; other generation limits and prompts are shared.
A new model can reuse these controls with its own capability manifest and an
explicitly verified attention adapter. Do not assume that two models' raw
eligible-cohort metrics compare the same cases; include shared-cohort metrics.

Data and frozen dependencies are inherited from the preceding expansion.
New results are under results/v2_full_matrix; prior reports/data are preserved.
This remains a post-hoc development-set mechanism experiment, not a held-out
confirmation or a general cocktail-party robustness benchmark.

## Execution

Use the environment and system dependencies recorded in reports/run_v2 and
reports/run_v2_expanded_480. Activate .venv, set PYTHONPATH to the repository
root and OMP_NUM_THREADS=4. HF_HUB_OFFLINE=1 is used only after the necessary
model/data revision has been downloaded.

For new model results, invoke sar.validity_smoke with that model's capability
config, data/mvp_expanded_480/source.jsonl and its eligible output path. Invoke
scripts/check_daa_identity.py with its fixed-DAA config and model-specific
identity output. Run sar.gpu_smoke --evaluate with the diagnostic config.
Run sar.daa_smoke with each of the five DAA configs. The config list itself
is the complete work list; there are no performance-conditioned branches.

For matched Oracle controls, verify that address tables and declarations agree.
For matched self controls, compare the actual generated declarations as well
as final answer accuracy. Report any mismatches rather than attributing every
observed difference to KV alone.

Two independent 3B conditions may run concurrently when memory permits. The
7B model initially runs alone. Hardware/runtime errors may lead to a serial
retry, documented as an execution adjustment, not a change to the experiment.

### Recorded declaration reuse

`python -m sar.daa_replay` provides a matched fixed-block control: reuse the
same model's recorded Self Prompt-only declarations and recompute every valid
answer with the KV mask enabled. It accepts only model-selected fixed blocks.
Source config, loaded manifest records, and result bytes must match the producer
`run_metadata.json`; output may not overwrite source artifacts. Configs may
differ only in the mask flag and output directory. No cross-model selection
transfer is allowed. Non-finite scores and reasoning exceptions fail loudly.

Declaration failures remain failures in both controls. Inspect their error
messages before replay to distinguish malformed output from runtime faults.
The original independent 3B Self runs were already in progress when producer
receipts were added; do not retrospectively label them receipt-producing runs.
Check their recorded declaration agreement and fresh replay predictions before
using this optimization for the 7B fixed condition. Historical 3B DAA outputs
do not contain log-probability vectors, so this check can compare predictions,
not independently recorded score-vector equality.

Example after the 7B Self Prompt-only stage and source-error audit:

```bash
python -m sar.daa_replay \
  --config configs/experiment/v2_full_matrix/7b/mvp_daa_fixed.yaml \
  --source-config configs/experiment/v2_full_matrix/7b/mvp_daa_prompt_only.yaml \
  --source-results results/v2_full_matrix/7b/mvp_daa_prompt_only/results.jsonl
```

### Optional declaration decoding optimization

`method.daa.stop_on_complete_declaration` defaults to false. When enabled, greedy
decoding ends after the first complete declaration recognized by the existing
parser. The stopping criterion examines only newly generated tokens, never the
prompt's example tags. It does not change the prompt, token ceiling, parsed
selection, or treatment of invalid content. Incomplete declarations still run
until the original generation limit or model EOS. Raw text loses trailing
continuations, so validate parsed outcomes and prefix agreement rather than
full raw-text equality across this optimization.

The already-running independent 3B Self and declared stages retain their
original decoding. Enable the optimization for subsequent 7B Self/declared
stages only after the real 3B prefix/selection checks and parser tests pass.
It is an execution optimization, not a new acoustic selection method.
