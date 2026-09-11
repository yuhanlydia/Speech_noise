# Validity-Gated DAA V2 — Decisive Experiment Protocol

This protocol replaces the original ungated MVP for scientific decision-making. The old run must **not** be used to accept or reject the research hypothesis because it mixed base capability failure with relevance-switch failure and leaked relevance through the query wording.

## Hypothesis

For the same waveform `x = target + event`, acoustic relevance should change with the query:

- target query: retain target speech and avoid the event;
- event query: select the event.

The key claim is not generic query-conditioned audio selection. The falsifiable claim is whether a Speech LM shows **same-waveform relevance reversal failure** and whether explicit declarative focusing changes the model's functional use of acoustic evidence.

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

Required: `ok=true`, zero errors. The audit recomputes waveform hashes and checks sample rates, offsets, source spans, event durations, finite samples, and use/ignore waveform identity.

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

- `<32` eligible pairs out of 128: **invalid setup**. Do not interpret relevance switching.
- `32–63`: diagnostic only.
- `>=64`: proceed.

## Gate 2 — Does a relevance-switch gap actually exist?

All remaining experiments use **only** `pairs_eligible.jsonl`.

```bash
bash scripts/run_diagnostic.sh
```

Primary metrics: IgnoreAcc, UseAcc, SAR, PairSwitchAcc.

Stop the whole project if:

```text
PairSwitchAcc >= 0.80
AND
PairSwitchAcc >= min(IgnoreAcc, UseAcc) - 0.05
```

Then there is no practically large relevance-switch gap to solve.

## Gate 3 — Oracle location and Oracle-KV controls

Before testing model-declared focus, separate **knowing where to listen** from the actual KV intervention.

### Gate 3A — Oracle-Prompt-only

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_oracle_prompt_only.yaml
```

The ground-truth pre-event/event temporal address table and correct focus declaration are given to the model, but no KV mask is applied.

This measures the explicit-location effect.

### Gate 3B — Oracle-KV

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_oracle.yaml
```

The textual address/focus information is matched to Gate 3A, and the runtime additionally masks all unselected audio keys before softmax.

Define:

```text
oracle_total_gain = Oracle-KV PairSwitchAcc - Base PairSwitchAcc
oracle_kv_gain = Oracle-KV PairSwitchAcc - Oracle-Prompt-only PairSwitchAcc
failure_rescue_fraction = oracle_total_gain / (1 - Base PairSwitchAcc)
```

Decision:

```text
oracle_total_gain < 0.05
OR failure_rescue_fraction < 0.15
    -> STOP the DAA/focusing line

Oracle-Prompt rescues but oracle_kv_gain < 0.03
    -> explicit localization helps, but the KV-control claim is weak

oracle_kv_gain >= 0.03
    -> actual KV masking has a plausible causal contribution
```

If perfect ground-truth focus cannot help, do not spend time training a selector or improving semantic segmentation.

## Gate 4 — Self-declared fixed-block controls

Only run if the oracle stage justifies continuing.

### Gate 4A — Self Prompt-only

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_prompt_only.yaml
```

The model declares fixed temporal block IDs. The final answer pass receives the same block table and focus declaration as the KV condition, but the runtime does not mask KV.

### Gate 4B — Self-declared fixed-block DAA

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_fixed.yaml
```

The model chooses the same kind of fixed block IDs and the runtime applies the pre-softmax audio-KV mask.

Define:

```text
self_kv_gain = Fixed-DAA PairSwitchAcc - Self-Prompt-only PairSwitchAcc
```

The causal self-routing claim is weak if `self_kv_gain < 0.03`.

If Oracle-KV works but self-declared `SelectionSwitchAcc < 0.50`, one Qwen2.5-Omni-7B NF4 replication is reasonable. If 7B also fails, stop the zero-shot declarative-routing method.

## Secondary ablation — model-declared segmentation

Do not run this before the fixed-block gates above.

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_declared.yaml
```

This tests whether the Speech LM can also create a semantic temporal address space. Failure here does not invalidate fixed-block DAA; it only shows timestamp/segmentation declaration is unreliable.

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
git push origin feature/daa-mvp
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

Base has gap, Oracle focus does not rescue
  -> STOP DAA / FOCUSING LINE

Oracle-Prompt rescues, Oracle-KV adds <3pp
  -> localization/prompt effect; weak KV claim

Oracle-KV works, self Prompt-only ~= Fixed DAA
  -> zero-shot KV self-routing claim is weak

Oracle-KV works and Fixed DAA beats Self Prompt-only with correct relevance reversal
  -> CONTINUE: strong mechanism candidate

Fixed DAA works; declared segmentation fails
  -> keep fixed address space; discard semantic segmentation component
```

## Current scope

The first V2 dataset is deliberately time-separated MMLU speech + ESC-50 event. It is a mechanism test for query-relative acoustic evidence selection, **not** yet a cocktail-party/noise robustness benchmark. Only after V2 passes should the project add partial overlap and competing human speech.
