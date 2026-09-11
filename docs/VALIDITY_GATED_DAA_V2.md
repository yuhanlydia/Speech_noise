# Validity-Gated DAA V2 — Decisive Experiment Protocol

This protocol replaces the original ungated MVP for scientific decision-making.
The old run must **not** be used to accept or reject the research hypothesis because it mixed base capability failure with relevance-switch failure and leaked relevance through the query wording.

## Hypothesis

For the same waveform `x = target + event`, acoustic relevance should change with the query:

- target query: retain target speech and avoid the event;
- event query: select the event.

The key question is not whether query-conditioned audio selection exists in general. The key falsifiable claim is whether a Speech LM shows **same-waveform relevance reversal failure** and whether explicit declarative focusing can causally recover it.

## Mandatory fresh data

Prompts changed in V2. Delete any old generated MVP before running:

```bash
rm -rf data/mvp results/mvp_* results/data_audit reports/run_v2
python scripts/prepare_public_mvp.py \
  --output-dir data/mvp \
  --num-pairs 128 \
  --protocol dev \
  --event-source esc50 \
  --seed 0
```

Do **not** reuse an old `pairs.jsonl` generated before V2.

## Gate 0 — Data integrity

```bash
python scripts/audit_mvp_data.py \
  --source-manifest data/mvp/source.jsonl \
  --pairs-manifest data/mvp/pairs.jsonl
```

Required: `ok=true`, zero errors.

The audit recomputes waveform hashes and checks sample rates, offsets, source spans, event durations, finite samples, and use/ignore waveform identity.

If this fails: fix data. Do not run a model.

## Gate 1 — Isolated capability eligibility

```bash
bash scripts/run_capability_gate.sh
```

For each pair, evaluate:

1. target speech alone + target/MMLU query;
2. event audio alone + event query.

A pair is eligible iff both are correct:

```text
eligible = target_only_correct AND event_only_correct
```

Output:

```text
results/mvp_capability_gate/summary.json
data/mvp/pairs_eligible.jsonl
```

Decision:

- `<32` eligible pairs out of 128: **invalid setup**. Do not interpret relevance switching. Use easier items / a stronger backbone.
- `32–63`: diagnostic only; do not claim a robust phenomenon.
- `>=64`: proceed.

## Gate 2 — Does a relevance-switch gap actually exist?

All remaining experiments use **only** `pairs_eligible.jsonl`.

```bash
bash scripts/run_diagnostic.sh
```

Primary metrics:

- IgnoreAcc
- UseAcc
- SAR
- PairSwitchAcc

Stop the whole project if the eligible mixed-audio Base already has strong pairwise performance. Operational stop rule:

```text
PairSwitchAcc >= 0.80
AND
PairSwitchAcc >= min(IgnoreAcc, UseAcc) - 0.05
```

That means there is no practically large relevance-switch gap to solve.

## Gate 3 — Oracle-KV causal upper bound

Before testing model-declared focus, ask whether **perfect focus** can rescue the mixed failure.

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_oracle.yaml
```

Oracle uses ground-truth temporal regions from the generated source metadata:

- ignore query: pre-event target region;
- use query: exact event region.

This is the decisive test of the KV-focus mechanism.

Compute:

```text
oracle_gain = Oracle PairSwitchAcc - Base PairSwitchAcc
failure_rescue_fraction = oracle_gain / (1 - Base PairSwitchAcc)
```

Stop the DAA/KV-routing line if either condition holds:

```text
oracle_gain < 0.05
OR
failure_rescue_fraction < 0.15
```

If perfect ground-truth focus cannot help, do not spend time training a selector or improving semantic segmentation.

## Gate 4 — Separate explicit reasoning from actual KV intervention

Only run if Oracle passes.

### 4A. Prompt-only

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_prompt_only.yaml
```

The model declares the same fixed blocks and receives the same `<focus_audio ...>` prompt, but no KV mask is applied.

### 4B. Self-declared fixed-block DAA

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_fixed.yaml
```

The model chooses fixed temporal block IDs and the runtime applies the pre-softmax audio-KV mask.

The causal KV claim is weak if:

```text
Fixed-DAA PairSwitchAcc - Prompt-only PairSwitchAcc < 0.03
```

If Oracle works but self-declared `SelectionSwitchAcc < 0.50`, one 7B NF4 replication is reasonable. If 7B also fails, stop the zero-shot declarative-routing method.

## Secondary ablation — model-declared segmentation

Do not run this before the fixed-block gates above.

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_declared.yaml
```

This tests whether the Speech LM can also create a semantic temporal address space. Failure here does not invalidate fixed-block DAA; it only shows that timestamp/segmentation declaration is unreliable.

## Correct selection definition

For the use query:

```text
ground-truth event midpoint must fall inside a selected block
```

For the ignore query:

```text
event must be avoided
AND
>= 80% of the pre-event target region must be retained
```

This prevents a model from receiving credit for simply dropping most of the audio.

## Export results for review

`results/` is intentionally git-ignored. After the run:

```bash
python scripts/export_run_v2.py

git add reports/run_v2
git commit -m "results: add validity-gated DAA V2 run"
git push
```

This makes the actual stage summaries visible on GitHub.

## Decision tree

```text
Data audit fails
  -> FIX DATA

Capability eligible < 32
  -> INVALID SETUP; DO NOT CLAIM GAP

Eligible Base has no pairwise gap
  -> STOP PROJECT

Base has gap, Oracle-KV does not rescue
  -> STOP DAA/KV LINE

Oracle rescues, Prompt-only ~= Fixed DAA
  -> KV intervention adds little; DAA method claim is weak

Oracle rescues, Fixed DAA > Prompt-only, selection switches correctly
  -> CONTINUE: strong mechanism candidate

Fixed DAA works; declared segmentation fails
  -> keep fixed address space; discard semantic segmentation component
```

## Current scope

The first V2 dataset is deliberately time-separated MMLU speech + ESC-50 event. It is a mechanism test for query-relative acoustic evidence selection, **not** yet a cocktail-party/noise robustness benchmark. Only after V2 passes should the project add partial overlap and competing human speech.
