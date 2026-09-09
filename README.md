# Speech_noise

**Selective Acoustic Relevance and Declarative Acoustic Attention for Speech / Audio Language Models**

This repository studies a narrow question:

> **Does a Speech LM know which part of the exact same acoustic scene should matter for the current query?**

The controlled benchmark uses **same-waveform relevance-switch pairs**. One waveform is paired with two questions:

- `q_ignore`: a particular event/source should not influence the answer;
- `q_use`: that exact same event/source is necessary evidence.

Only the query changes. The central hypothesis is therefore

```text
R = R(acoustic event, query)
```

rather than treating a sound class as globally “noise” or globally “evidence”.

## Current primary method: DAA

The primary training-free experiment is **Declarative Acoustic Attention (DAA)**, inspired by the idea that an LM can explicitly declare which context region it needs.

DAA uses three passes:

```text
GLOBAL LISTEN
    -> declare addressable acoustic blocks once per waveform
FOCUS
    -> for each query, declare the block(s) needed
REASON
    -> runtime exposes only selected audio KV to query/answer tokens
```

Example global declaration:

```text
<audio_blocks>
B1|0.00|1.40|target speaker
B2|1.40|2.10|dog bark
B3|2.10|3.50|traffic
</audio_blocks>
```

For “What animal is audible?” the model may emit:

```text
<focus_audio blocks="B2">
```

For “What number did the speaker say?” on the **same waveform**, it should select the speech block instead.

The final pass does not merely prompt the model to focus. The Qwen Thinker attention hook maps selected time spans to audio placeholder tokens and masks unselected audio keys **before softmax**, so attention is renormalized over the declared acoustic context plus text/local tokens.

### Important protocol property

Block segmentation is generated **once per waveform pair** and reused for both `use` and `ignore` queries. Thus the experiment tests query-dependent focus, not query-dependent re-segmentation.

### Two block conditions

- `declared`: the Speech LM itself proposes semantic/time-local blocks;
- `fixed`: deterministic temporal blocks isolate selection quality from segmentation quality.

There is **no silent fallback**. An invalid declaration is recorded as a protocol failure and counts as an incorrect example.

## QACR baseline

The previous primary method, **QACR — Query-Conditioned Audio Contribution Routing**, remains implemented as a learned-routing baseline. QACR learns a soft token gate

```text
r_j(q) = sigmoid((W_q e_q)^T (W_a h_j) / sqrt(d_r))
```

and scales audio value contributions after softmax. DAA is deliberately different: it uses the model's explicit declaration and performs hard context exclusion before softmax.

## Why both methods exist

Earlier experiments showed that a representation direction can predict failure without being safe to suppress. The current research therefore compares two mechanisms:

1. **hidden learned relevance** — QACR;
2. **explicit self-declared relevance** — DAA.

The same-waveform benchmark decides whether either mechanism actually improves relevance switching.

## Backbone and hardware

Primary backbone:

```text
Qwen/Qwen2.5-Omni-3B
```

Target setup for a single RTX A4000 16 GB:

- Thinker only;
- Talker disabled;
- 4-bit NF4;
- text/canonical MCQ scoring;
- eager attention for exact DAA/QACR hooks.

For DAA, `layers: []` means focus is enforced on **all Thinker self-attention layers**. A restricted layer list can be used for mechanistic ablations.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
python -m compileall src scripts
```

GPU dependencies:

```bash
pip install -e '.[dev,gpu]'
```

## Data: same-waveform relevance-switch pairs

Prepare the controlled source manifest following:

```text
examples/source_manifest.example.jsonl
```

Then build mixed waveforms and paired records:

```bash
python scripts/build_mvp_manifest.py /path/to/source.jsonl data/mvp
```

Each `use` / `ignore` pair must share the exact same waveform path and SHA-256 hash.

For the first DAA experiment, prefer **time-separable events** so a temporal block can isolate the event. Overlapping speakers are not treated as source-separated merely because they share a time span.

## Gate A: base relevance-switch diagnostic

```bash
bash scripts/run_diagnostic.sh
```

Primary task metrics:

- `IgnoreAcc`
- `UseAcc`
- `SAR` — harmonic mean of the two
- `PairSwitchAcc` — both queries correct for the same waveform

If the base model already has high PairSwitchAcc close to both one-sided accuracies, the research direction should stop.

## DAA dry-run

```bash
python -m sar.daa_smoke \
  --config configs/experiment/mvp_daa_declared.yaml \
  --dry-run
```

## DAA with model-declared event blocks

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_declared.yaml
```

## DAA fixed-block control

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_fixed.yaml
```

The fixed-block condition answers a critical diagnostic question:

> If model-declared DAA fails, is the failure caused by bad acoustic segmentation or by bad query-dependent selection?

## DAA outputs

The evaluator writes standard result rows plus:

- selected block ids;
- raw block declaration;
- raw focus declaration;
- whether the labeled event was selected;
- protocol failure stage.

Summary metrics include:

- protocol completion rate;
- use-event selection rate;
- ignore-event avoidance rate;
- selection switch accuracy;
- reasoning accuracy given correct use selection;
- reasoning accuracy given correct ignore avoidance;
- IgnoreAcc / UseAcc / SAR / PairSwitchAcc.

This allows two failures to be separated:

```text
selection failure: model chose the wrong acoustic evidence
reasoning failure: model chose the right evidence but still answered incorrectly
```

## QACR experiments

Existing QACR scripts are preserved:

```bash
bash scripts/train_qacr.sh
bash scripts/eval_relevance_switch.sh
```

QACR is not removed or rewritten by the DAA branch.

## External benchmark adapters

Evaluation adapter contracts remain available for:

- RSA-Bench — irrelevant acoustic context / robustness;
- MMSU — acoustic evidence retention;
- SH-Bench — speaker/policy-selective evidence use;
- VoxSafeBench — context-dependent safety evidence.

The controlled same-waveform experiment must pass before scaling to these benchmarks.

## Go / No-Go

### Gate A — phenomenon

Base model must show a meaningful same-waveform relevance-switch weakness.

### Gate B — declarative selection

DAA must achieve useful protocol coverage and the selected event must reverse appropriately between `q_use` and `q_ignore`. Compare declared vs fixed blocks.

### Gate C — functional focused reasoning

DAA must improve SAR / PairSwitchAcc, or at minimum demonstrate that correct block selection isolates the remaining problem to downstream reasoning.

If Gate B fails, do not add DAA training simply to rescue the result. If Gate C fails despite correct focus, investigate the reasoning interface rather than adding larger routers.

## Code layout

```text
src/sar/data/blocks.py          acoustic block parsing and time->token mapping
src/sar/methods/daa.py          scan/focus protocol and pure attention semantics
src/sar/models/daa_hook.py      Qwen pre-softmax focused-attention hook
src/sar/models/qwen_omni_daa.py Qwen global/focus/reason wrapper
src/sar/daa_pipeline.py         same-waveform DAA pipeline + mechanism metrics
src/sar/daa_smoke.py            A4000 evaluator / dry-run
src/sar/methods/qacr.py         previous learned routing baseline
```

Design and implementation notes:

```text
docs/superpowers/specs/2026-09-09-declarative-acoustic-attention-design.md
docs/superpowers/plans/2026-09-09-daa-mvp.md
```

## Verification boundary

The DAA core protocol, block mapping, attention semantics, controller/cache behavior, pair pipeline, strict config, and evaluator have CPU tests. A real Qwen2.5-Omni CUDA run is still required before claiming any empirical DAA improvement or Gate A/B/C result.
