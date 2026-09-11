# Speech_noise

## Selective Acoustic Relevance / Validity-Gated Declarative Acoustic Attention

This repository asks one falsifiable question:

> **For the exact same acoustic scene, can a Speech LM change which acoustic evidence is functionally used when only the query changes?**

The hypothesis is relational relevance:

```text
R = R(acoustic evidence, query)
```

rather than treating a sound as globally “noise” or globally “evidence”.

## Important: the original MVP is superseded

The first MVP mixed three effects:

1. whether the model could solve the spoken MMLU item at all;
2. whether it could recognize the ESC-50 event at all;
3. whether it could switch acoustic relevance in the mixed waveform.

It also used query wording that explicitly said `ignore` / `irrelevant`, which leaked the intended relevance relation.

**Do not use the old run to accept or reject the research hypothesis.**

The current protocol is **Validity-Gated DAA V2**. It proves both component capabilities first, measures the mixed-audio gap second, then separates explicit location prompting from the causal effect of an actual audio-KV mask.

Canonical execution instructions:

```text
TASK_V2.md
docs/VALIDITY_GATED_DAA_V2.md
```

---

# 1. Method

## DAA — Declarative Acoustic Attention

The Speech LM is given an addressable acoustic context and explicitly declares which block IDs it needs for the current query:

```text
<focus_audio blocks="B2,B3">
```

The runtime can then mask unselected audio keys **before attention softmax** for query/answer consumer tokens. The attention distribution is therefore renormalized over the selected audio KV plus normal text/local context.

The primary V2 condition uses **fixed temporal blocks**. Model-generated timestamp segmentation is a separate capability and must not be conflated with relevance selection. Model-declared semantic segmentation remains a secondary ablation.

## QACR baseline

The previous learned soft router remains available:

```text
r_j(q) = sigmoid((W_q e_q)^T (W_a h_j) / sqrt(d_r))
```

QACR is not the primary V2 method because query-conditioned audio routing by itself is no longer a sufficiently distinctive contribution.

---

# 2. Backbone and hardware

Default:

```text
Qwen/Qwen2.5-Omni-3B
Thinker only
4-bit NF4
eager attention
```

The code is designed for a 16 GB A4000 and a 24 GB RTX 3090.

Install:

```bash
python -m venv .venv
source .venv/bin/activate
sudo apt-get update && sudo apt-get install -y espeak-ng
pip install -e '.[dev,data,gpu]'
pytest -q
python -m compileall src scripts
```

---

# 3. Public V2 data

Primary development data:

- reasoning content: `cais/mmlu`, split `validation`;
- target speech: local eSpeak-NG synthesis of the MMLU question and four choices;
- acoustic event: `ashraq/esc50`, folds 1/2/3, restricted to distinctive classes;
- confirmation: MMLU `test` + ESC-50 folds 4/5;
- optional second-stage stress: LibriSpeech `validation.clean` / `test.clean`.

The first mechanism test is intentionally time-separated:

```text
spoken MMLU -> short gap -> ESC-50 event
```

This is **not yet a cocktail-party/noise benchmark**. It first tests whether query-relative acoustic evidence selection exists at all.

### V2 query policy

The query itself defines relevance. We do not tell the model what to ignore.

Target query:

```text
Which option correctly answers the spoken multiple-choice question?
Answer with A, B, C, or D only.
```

Event query:

```text
Which of the following sounds can be heard in the recording?
A. ... B. ... C. ... D. ...
Answer with A, B, C, or D only.
```

No `ignore`, `irrelevant`, or relevance-label leakage is used.

---

# 4. Run V2 — in this order

## Step 0 — regenerate fresh data

**Delete the old MVP. Old pairs contain obsolete prompts.**

```bash
rm -rf data/mvp results/mvp_* results/data_audit reports/run_v2

python scripts/prepare_public_mvp.py \
  --output-dir data/mvp \
  --num-pairs 128 \
  --protocol dev \
  --event-source esc50 \
  --seed 0
```

## Step 1 — data audit

```bash
python scripts/audit_mvp_data.py \
  --source-manifest data/mvp/source.jsonl \
  --pairs-manifest data/mvp/pairs.jsonl
```

Required:

```text
ok = true
num_errors = 0
```

The audit rereads the actual WAVs and verifies sample rates, offsets, event spans, finite samples, use/ignore waveform identity, and recomputed SHA-256 hashes.

## Step 2 — isolated capability gate

```bash
bash scripts/run_capability_gate.sh
```

Each pair is first tested on:

```text
target WAV only + target/MMLU query
event WAV only  + event query
```

A pair is retained only when both are correct:

```text
eligible = target_only_correct AND event_only_correct
```

Outputs:

```text
results/mvp_capability_gate/summary.json
data/mvp/pairs_eligible.jsonl
```

Decision:

- `<32 / 128` eligible: invalid setup; do not interpret a relevance gap;
- `32–63`: exploratory only;
- `>=64`: proceed.

All later configs use `pairs_eligible.jsonl`.

## Step 3 — mixed Base

```bash
bash scripts/run_diagnostic.sh
```

Primary metrics:

- `IgnoreAcc`
- `UseAcc`
- `SAR`
- `PairSwitchAcc`

Stop the whole project if:

```text
PairSwitchAcc >= 0.80
AND
PairSwitchAcc >= min(IgnoreAcc, UseAcc) - 0.05
```

Then there is no practically large relevance-switch gap to solve.

## Step 4 — oracle location controls

These are mandatory before spending time on self-declared selection.

### 4A. Oracle-Prompt-only

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_oracle_prompt_only.yaml
```

Ground-truth target/event temporal regions are shown in the address table and focus declaration, but **no KV mask is applied**.

This measures whether merely telling the model where the relevant evidence is located helps.

### 4B. Oracle-KV

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_oracle.yaml
```

The prompt/address information is matched to 4A, but now the runtime also masks unselected audio KV before softmax.

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

Oracle-Prompt helps but oracle_kv_gain < 0.03
    -> location prompting helps, but the KV-control claim is weak

oracle_kv_gain >= 0.03
    -> actual KV masking has a plausible causal contribution
```

## Step 5 — self-declared fixed-block controls

Only run if the oracle stage justifies continuing.

### 5A. Self Prompt-only

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_prompt_only.yaml
```

The model declares fixed block IDs; the final prompt contains the same address table and focus tag, but no KV mask.

### 5B. Fixed DAA with KV mask

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_fixed.yaml
```

Correct selection requires:

```text
use: event midpoint is inside a selected block
ignore: event is avoided AND >=80% of the pre-event target region is retained
```

Define:

```text
self_kv_gain = Fixed-DAA PairSwitchAcc - Self-Prompt-only PairSwitchAcc
```

If `self_kv_gain < 0.03`, the self-declared KV intervention is not a strong method claim.

If Oracle-KV works but zero-shot `SelectionSwitchAcc < 0.50`, run at most one Qwen2.5-Omni-7B NF4 replication. If 7B also fails, stop the zero-shot declarative selector.

## Step 6 — declared segmentation ablation

Only after fixed DAA succeeds:

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_declared.yaml
```

Failure here means semantic/timestamp declaration is unreliable. It does not refute fixed-address DAA.

---

# 5. Causal comparison table

| Condition | Focus source | KV mask | Purpose |
|---|---|---:|---|
| Base | none | no | establish eligible mixed gap |
| Oracle-Prompt | ground truth | no | explicit-location upper bound |
| Oracle-KV | ground truth | yes | causal KV-mask upper bound |
| Self Prompt-only | model | no | explicit self-selection control |
| Fixed DAA | model | yes | primary zero-shot DAA |
| Declared DAA | model + model segmentation | yes | segmentation ablation |

Critical differences:

```text
Base -> Oracle-Prompt       : does knowing where to listen help?
Oracle-Prompt -> Oracle-KV  : does KV masking itself help?
Self Prompt -> Fixed DAA    : does self-declared KV control help?
```

---

# 6. Make results visible on GitHub

Generated audio and `results/` are intentionally ignored. After running:

```bash
python scripts/export_run_v2.py

git add reports/run_v2
git commit -m "results: add validity-gated DAA V2 run"
git push origin feature/daa-mvp
```

Then the actual metrics are visible under:

```text
reports/run_v2/summary.json
reports/run_v2/SUMMARY.md
```

---

# 7. Stop rules

We do **not** keep adding modules until something works.

```text
Data audit fails
  -> FIX DATA

Insufficient isolated capability
  -> INVALID BENCHMARK/BACKBONE PAIRING

Eligible Base has no mixed pairwise gap
  -> STOP PROJECT

Gap exists but Oracle focus does not rescue
  -> STOP DAA / FOCUSING LINE

Oracle-Prompt rescues but Oracle-KV adds <3pp
  -> explicit localization effect; weak KV claim

Oracle-KV works but self selector fails on 3B and one 7B check
  -> STOP ZERO-SHOT DAA SELECTOR

Oracle-KV and self-declared KV both beat their prompt-only controls
  -> CONTINUE; strong mechanism candidate
```

Only after V2 passes should the project move to partial overlap, competing human speech, multiple backbones, and external speech/audio benchmarks.

---

# Code map

```text
src/sar/data/public_mvp.py      public data construction
src/sar/data/audit.py           waveform / metadata integrity audit
src/sar/validity.py             isolated capability eligibility gate
src/sar/validity_smoke.py       capability GPU launcher
src/sar/data/blocks.py          acoustic address space
src/sar/methods/daa.py          focus protocol / attention semantics
src/sar/models/daa_hook.py      pre-softmax Qwen audio-KV masking
src/sar/models/qwen_omni_daa.py prompt-only + masked DAA wrapper
src/sar/daa_pipeline.py         oracle/model focus + validity metrics
src/sar/daa_smoke.py            DAA evaluator
src/sar/report.py               tracked V2 report export
scripts/prepare_public_mvp.py
scripts/audit_mvp_data.py
scripts/run_capability_gate.sh
scripts/run_diagnostic.sh
scripts/run_daa.sh
scripts/export_run_v2.py
```

QACR remains under `src/sar/methods/qacr.py` as a learned-routing baseline, but it is not part of the decisive V2 gate sequence.
