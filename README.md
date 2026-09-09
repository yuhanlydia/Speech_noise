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

There is **no silent fallback**. An invalid declaration is recorded as a protocol failure and counts as an incorrect example. Declared blocks must also cover the full waveform rather than silently omitting inconvenient audio.

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

Target setup for a single RTX A4000 16 GB or RTX 3090 24 GB:

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
sudo apt-get update && sudo apt-get install -y espeak-ng
pip install -e '.[dev,data,gpu]'
pytest -q
python -m compileall src scripts
```

The `data` extra installs Hugging Face `datasets` / `huggingface_hub`. `espeak-ng` is used only to synthesize the controlled MMLU target speech; no cloud TTS key is required.

# Public one-command data preparation

You do **not** need to provide private WAV files for the first experiment.

The default public MVP uses:

1. **Target reasoning content:** `cais/mmlu`, config `all`.
2. **Target speech:** the MMLU question + four choices synthesized locally with eSpeak-NG.
3. **Acoustic event:** `ashraq/esc50`, restricted to acoustically distinctive classes such as dog, siren, rain, footsteps, alarm, car horn, train, airplane, etc.
4. **Same-waveform pair:** the spoken MMLU question is followed by the event. The exact same resulting waveform is used for both queries.

The two queries are:

```text
q_ignore: answer the spoken MMLU question; the later event is irrelevant
q_use:    identify the later acoustic event; ignore the MMLU question
```

Thus the same event must switch from **nuisance** to **evidence** solely because the query changed.

## Development protocol

Development sources are deliberately separated from confirmation sources:

```text
DEV
  MMLU:       validation
  ESC-50:     folds 1,2,3
  LibriSpeech validation.clean

CONFIRM
  MMLU:       test
  ESC-50:     folds 4,5
  LibriSpeech test.clean
```

Do not run `--protocol confirm` until the DAA settings and analysis protocol are frozen.

## Build the default environmental-event MVP

```bash
python scripts/prepare_public_mvp.py \
  --output-dir data/mvp \
  --num-pairs 128 \
  --protocol dev \
  --event-source esc50 \
  --seed 0
```

This command downloads the public source datasets, synthesizes/normalizes the WAV files, writes:

```text
data/mvp/source.jsonl
data/mvp/public_mvp_metadata.json
data/mvp/pairs.jsonl
data/mvp/sources/...
data/mvp/audio/...
```

and therefore removes the previous `data/mvp/pairs.jsonl` blocker.

The generated event is time-separable from the target speech in this first mechanism test. This is intentional: DAA must first demonstrate that it can select an addressable block before we move to overlapping-source stress tests.

## Real-human competing-speech stress

The same preparation script also supports public LibriSpeech:

```bash
python scripts/prepare_public_mvp.py \
  --output-dir data/mvp_librispeech \
  --num-pairs 128 \
  --protocol dev \
  --event-source librispeech \
  --seed 0
```

Here the later event is a **real human LibriSpeech utterance** and `q_use` asks which phrase the background speaker said. LibriSpeech is the second-stage stress condition; run ESC-50 first.

The TARS synthesized spoken-MMLU corpus can be used later as an additional replication, but it is not a default dependency because the public-data access path is heavier and can require Hugging Face account acceptance. The default MVP is intentionally runnable from `cais/mmlu + ESC-50` without that blocker.

## Manual data path (optional)

If you already have your own target/event WAVs, the original manual builder remains supported. Prepare a source manifest following:

```text
examples/source_manifest.example.jsonl
```

then run:

```bash
python scripts/build_mvp_manifest.py /path/to/source.jsonl data/mvp
```

Each `use` / `ignore` pair must share the exact same waveform path and SHA-256 hash.

For the first DAA experiment, prefer **time-separable events** so a temporal block can isolate the event. Overlapping speakers are not treated as source-separated merely because they share a time span.

# Experiment sequence

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

## DAA fixed-block control first

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_fixed.yaml
```

Run fixed blocks before model-declared blocks. This isolates query-dependent selection from acoustic segmentation quality.

## DAA with model-declared event blocks

```bash
bash scripts/run_daa.sh configs/experiment/mvp_daa_declared.yaml
```

The declared condition answers the harder question: can the Speech LM itself construct a useful acoustic address space before selecting query-relevant evidence?

## DAA outputs

The evaluator writes standard result rows plus:

- selected block ids;
- raw block declaration;
- raw focus declaration;
- whether the labeled event midpoint was selected;
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
src/sar/data/public_mvp.py     public MMLU/ESC-50/LibriSpeech preparation
scripts/prepare_public_mvp.py  one-command source + pairs builder
src/sar/data/blocks.py         acoustic block parsing and time->token mapping
src/sar/methods/daa.py         scan/focus protocol and pure attention semantics
src/sar/models/daa_hook.py     Qwen pre-softmax focused-attention hook
src/sar/models/qwen_omni_daa.py Qwen global/focus/reason wrapper
src/sar/daa_pipeline.py        same-waveform DAA pipeline + mechanism metrics
src/sar/daa_smoke.py           A4000/3090 evaluator / dry-run
src/sar/methods/qacr.py        previous learned routing baseline
```

Design and implementation notes:

```text
docs/superpowers/specs/2026-09-09-declarative-acoustic-attention-design.md
docs/superpowers/plans/2026-09-09-daa-mvp.md
```
