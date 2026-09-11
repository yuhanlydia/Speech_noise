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

The current protocol is **Validity-Gated DAA V2**. It first proves that both component capabilities exist, then tests the mixed waveform, then asks whether perfect ground-truth KV focusing can causally rescue the failure.

Full protocol:

```text
docs/VALIDITY_GATED_DAA_V2.md
```

---

# 1. Method

## DAA — Declarative Acoustic Attention

The Speech LM is given an addressable acoustic context and explicitly declares which block IDs it needs for the current query:

```text
<focus_audio blocks="B2,B3">
```

The runtime then masks unselected audio keys **before attention softmax** for the query/answer consumer tokens. Therefore the attention distribution is renormalized over the selected audio KV plus normal text/local context.

The primary V2 condition uses **fixed temporal blocks**. This is deliberate: model-generated timestamp segmentation is a separate capability and must not be conflated with relevance selection.

Model-declared semantic segmentation remains a secondary ablation.

## QACR baseline

The previous learned soft router remains in the repository:

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

The code is designed for a 16 GB A4000 and also runs on a 24 GB RTX 3090.

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

Primary development dataset:

- reasoning content: `cais/mmlu`, `validation`;
- target speech: local eSpeak-NG synthesis of the MMLU question and four choices;
- acoustic event: `ashraq/esc50`, folds 1/2/3, restricted to distinctive classes;
- confirmation data: MMLU `test` + ESC-50 folds 4/5;
- optional second-stage stress: LibriSpeech `validation.clean` / `test.clean`.

The first mechanism test is time-separated:

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

This creates:

```text
data/mvp/source.jsonl
data/mvp/pairs.jsonl
data/mvp/sources/target/*.wav
data/mvp/sources/esc50/*.wav
data/mvp/audio/*.wav
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

The audit rereads the actual WAVs and checks sample rates, offsets, event spans, finite samples, use/ignore waveform identity, and recomputed SHA-256 hashes.

If this fails, fix data before using a GPU result.

## Step 2 — isolated capability gate

```bash
bash scripts/run_capability_gate.sh
```

Each pair is tested on:

```text
target WAV only + target/MMLU query
event WAV only  + event query
```

A pair is retained only when both answers are correct:

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

All later configs read `pairs_eligible.jsonl`.

## Step 3 — mixed Base

```bash
bash scripts/run_diagnostic.sh
```

Metrics:

- `IgnoreAcc`
- `UseAcc`
- `SAR`
- `PairSwitchAcc`

Stop the whole project when the eligible mixed Base is already strong:

```text
PairSwitchAcc >= 0.80
AND
PairSwitchAcc >= min(IgnoreAcc, UseAcc) - 0.05
```

Then there is no large relevance-switch failure worth solving.

## Step 4 — Oracle-KV rescue: decisive mechanism gate

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_oracle.yaml
```

Oracle uses the known generated temporal regions:

```text
ignore query -> pre-event target region
use query    -> exact event region
```

Define:

```text
oracle_gain = Oracle PairSwitchAcc - Base PairSwitchAcc
failure_rescue_fraction = oracle_gain / (1 - Base PairSwitchAcc)
```

**Stop the DAA/KV-routing line** if:

```text
oracle_gain < 0.05
OR
failure_rescue_fraction < 0.15
```

If perfect focus cannot help, selector training or better segmentation is not the answer.

## Step 5 — Prompt-only control

Only run if Oracle passes:

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_prompt_only.yaml
```

The model emits/receives the same focus declaration but the runtime does **not** mask KV.

This isolates gains caused by explicit focus prompting from gains caused by the actual attention intervention.

## Step 6 — self-declared fixed-block DAA

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_fixed.yaml
```

Correct selection requires:

```text
use: event midpoint is inside a selected block
ignore: event is avoided AND >=80% of the pre-event target region is retained
```

The KV intervention is not compelling if:

```text
Fixed-DAA PairSwitchAcc - Prompt-only PairSwitchAcc < 0.03
```

If Oracle works but zero-shot `SelectionSwitchAcc < 0.50`, one Qwen2.5-Omni-7B NF4 replication is reasonable. If 7B also fails, stop the zero-shot declarative selector.

## Step 7 — declared segmentation ablation

Only after fixed DAA:

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_declared.yaml
```

Failure here means semantic/timestamp declaration is unreliable. It does not by itself refute fixed-address DAA.

---

# 5. Causal comparison table

The V2 paper claim must be supported by all of these conditions:

| Condition | Focus source | KV mask | Purpose |
|---|---|---:|---|
| Base | none | no | mixed relevance-switch gap |
| Oracle-KV | ground truth | yes | causal upper bound / line-kill test |
| Prompt-only | model | no | explicit-reasoning control |
| Fixed DAA | model | yes | primary zero-shot method |
| Declared DAA | model + model segmentation | yes | harder segmentation ablation |

The critical comparisons are:

```text
Base -> Oracle-KV
Prompt-only -> Fixed DAA
```

---

# 6. Make results visible on GitHub

`results/` and generated audio are intentionally ignored. After the experiment:

```bash
python scripts/export_run_v2.py

git add reports/run_v2
git commit -m "results: add validity-gated DAA V2 run"
git push
```

Then the actual metrics are available under:

```text
reports/run_v2/summary.json
reports/run_v2/SUMMARY.md
```

---

# 7. Research stopping rule

We do **not** keep adding modules until something works.

```text
Data invalid
  -> fix data

Insufficient isolated capability
  -> invalid benchmark/backbone pairing

No mixed gap among eligible pairs
  -> STOP PROJECT

Gap exists but Oracle-KV cannot rescue
  -> STOP DAA / KV-routing line

Oracle rescues but Fixed DAA ~= Prompt-only
  -> explicit prompt effect, weak KV-method claim

Oracle rescues and Fixed DAA > Prompt-only with correct relevance reversal
  -> CONTINUE; strong mechanism candidate
```

Only after this sequence passes should the project move to partial overlap, competing human speech, multiple backbones, and external speech/audio benchmarks.

---

# Code map

```text
src/sar/data/public_mvp.py      public data construction
src/sar/data/audit.py           waveform / metadata integrity audit
src/sar/validity.py             isolated capability eligibility gate
src/sar/validity_smoke.py       capability GPU launcher
src/sar/data/blocks.py          acoustic block address space
src/sar/methods/daa.py          focus protocol and attention semantics
src/sar/models/daa_hook.py      pre-softmax Qwen audio-KV masking
src/sar/models/qwen_omni_daa.py Base / prompt-only / masked DAA wrapper
src/sar/daa_pipeline.py         oracle/model focus + selection metrics
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
