# Speech_noise

**Selective Acoustic Relevance for Speech / Audio Language Models**

This repository tests a narrow, falsifiable hypothesis:

> **Acoustic relevance is relational rather than intrinsic.** The same acoustic source/event can be nuisance for one query and necessary evidence for another.

The MVP therefore uses **same-waveform relevance-switch pairs**. A waveform is constructed once, hashed once, and paired with two questions:

- `q_ignore`: the event/source should not affect the answer;
- `q_use`: the same event/source is required evidence.

Only the query changes. This prevents a model from solving the benchmark by treating a sound class, SNR, speaker, or waveform as globally “noise” or globally “evidence”.

## Research question

For an acoustic source/event `d` and query `q`, we test

```text
R = R(d, q)
```

rather than the common implicit simplification

```text
R = R(d).
```

The primary method is **QACR — Query-Conditioned Audio Contribution Routing**. QACR does not overwrite absolute K/V states. It gates the post-softmax contribution of audio key/value positions to Thinker self-attention:

```text
o_i' = sum_{j in text} a_ij V_j + sum_{j in audio} r_j(q) a_ij V_j

r_j(q) = sigmoid((W_q e_q)^T (W_a h_j) / sqrt(d_r)).
```

Non-audio contributions are unchanged. With `r=1`, the routed attention exactly reduces to base eager attention.

## Why this repo exists

Prior experiments on noise-induced KV drift showed a useful warning: a representation direction can be predictive of failure without being safe to suppress. QACR therefore changes the **contribution of acoustic evidence conditional on the query**, rather than projecting an absolute global KV subspace.

The MVP is deliberately small. If the same-waveform relevance-switch phenomenon or the routing mechanism fails, the project stops rather than adding larger gates, RL, or extra modules.

## Repository layout

```text
configs/
  data/                     data manifest config
  experiment/               base / QACR experiment configs
  method/                   QACR and matched routing controls
  model/                    Qwen2.5-Omni-3B NF4 config
examples/
  source_manifest.example.jsonl
scripts/
  build_mvp_manifest.py     deterministic mixer + paired manifest builder
  run_diagnostic.sh         base relevance-switch evaluation
  train_qacr.sh             router-only supervised MVP training
  eval_relevance_switch.sh  QACR evaluation
src/sar/
  data/                     schemas, mixing, benchmark adapter contracts
  models/                   Qwen2.5-Omni wrapper + exact eager-attention hook
  methods/                  QACR and matched routing baselines
  config.py                 strict YAML schema
  metrics.py                IgnoreAcc / UseAcc / SAR / PairSwitchAcc
  train.py                  router objectives
  gpu_smoke.py              dry-run, training, evaluation launcher
tests/                      CPU-safe TDD suite
```

## Primary backbone

The default model is:

```text
Qwen/Qwen2.5-Omni-3B
```

The constrained setup is intended for a single 16 GB GPU such as an RTX A4000:

- Thinker-only text generation/scoring;
- Talker disabled;
- 4-bit NF4 loading;
- backbone frozen for QACR training;
- only the tiny `W_q` / `W_a` router is optimized;
- one-forward A/B/C/D next-token scoring for canonical single-token MCQ options;
- eager attention is required for exact post-softmax audio-contribution routing.

QACR is slower than FlashAttention because exact contribution routing needs the eager attention probabilities. The point of the MVP is mechanism validation, not serving throughput.

## Installation

CPU tests and manifest construction:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
```

GPU experiments:

```bash
pip install -e '.[dev,gpu]'
```

The GPU extra installs `transformers`, `accelerate`, and `bitsandbytes`. The Qwen wrapper uses `Qwen2_5OmniThinkerForConditionalGeneration` and `Qwen2_5OmniProcessor` lazily, so importing the repository does not require those packages.

## 1. Build controlled same-waveform pairs

Prepare a JSONL source manifest following `examples/source_manifest.example.jsonl`. Each row points to a target waveform and one event/source stem. No benchmark audio is vendored into this repository.

Important fields:

```json
{
  "pair_id": "dog-0001",
  "target_wav": "/data/targets/math_0001.wav",
  "event_wav": "/data/events/dog_0001.wav",
  "event_type": "dog_bark",
  "offset_samples": 48000,
  "snr_db": 5.0,
  "source_mask_valid": true,
  "q_ignore": "...",
  "a_ignore": "C",
  "options_ignore": ["A", "B", "C", "D"],
  "q_use": "...",
  "a_use": "B",
  "options_use": ["A", "B", "C", "D"]
}
```

Build the mixture and paired manifest:

```bash
python scripts/build_mvp_manifest.py /path/to/source.jsonl data/mvp
```

The builder writes one mixed waveform per pair and two records referencing the **same absolute path and SHA-256 waveform hash**. Pair validation rejects hash mismatches.

### `source_mask_valid`

`source_mask_valid=true` means the source/event can be approximated by a temporal interval, so the builder's start/end time can be mapped to an audio-token span for auxiliary routing supervision.

This is **not source separation**. Do not set it to true for overlapping competing speech when the same time interval also contains target speech. Overlapping-speaker experiments should initially use QA supervision only, or a real source mask from a benchmark/separation system.

## 2. Validate experiment wiring without loading a model

```bash
python -m sar.gpu_smoke \
  --config configs/experiment/mvp_diagnostic.yaml \
  --dry-run
```

This checks config strictness, paired records, hashes, counts, and output paths without importing GPU-only model dependencies.

## 3. Gate A — base same-waveform diagnostic

```bash
bash scripts/run_diagnostic.sh
```

This evaluates the base Thinker on both queries for every waveform pair and writes:

```text
results/mvp_diagnostic/results.jsonl
results/mvp_diagnostic/summary.json
```

The core metrics are:

- **IgnoreAcc** — accuracy when the event/source should not matter;
- **UseAcc** — accuracy when that same event/source is evidence;
- **SAR** — harmonic mean of IgnoreAcc and UseAcc;
- **PairSwitchAcc** — fraction of waveforms where *both* paired queries are correct.

A model that always ignores background audio can score high on IgnoreAcc but low on UseAcc. A model that indiscriminately uses every sound can show the opposite pattern. SAR and PairSwitchAcc require both behaviors.

## 4. Train QACR

```bash
bash scripts/train_qacr.sh
```

The default config freezes the entire Qwen Thinker and optimizes only QACR at layer 0. For canonical A/B/C/D options, training uses a single forward pass and CE on next-token option logits.

When `source_mask_valid=true`, the auxiliary switch objective encourages the event token span toward:

```text
q_use    -> gate ~= 1
q_ignore -> gate ~= 0
```

while an identity penalty protects other audio tokens toward gate 1. If no valid temporal source mask exists, the auxiliary switch/identity terms are skipped for that record and QA loss remains active.

Checkpoint:

```text
artifacts/qacr/router.pt
```

The checkpoint contains only the small router state and metadata, not Qwen weights.

## 5. Evaluate QACR

```bash
bash scripts/eval_relevance_switch.sh
```

In addition to task metrics, QACR evaluation reports event-gate means and the use-minus-ignore event gate gap when valid temporal event masks are available.

## Matched baselines

The repository includes common routing controls under `src/sar/methods/`:

- `StaticAudioGate` — query-independent audio strength;
- `LayerRouter` — query-conditioned layer weights;
- `HeadRouter` — query-conditioned head weights;
- `FixedKVSubspace` — prior fixed-subspace mechanistic control;
- `OracleMaskRouter` — upper bound when a source relevance mask is available;
- `QACRRouter` — token/source-level query-conditioned contribution routing.

The controlled tensor implementations and parameter-count interfaces are tested now. The first GPU launcher intentionally wires only Base and QACR; layer/head GPU baselines are activated after Gate A/B justify scaling, per the preregistered stopping logic in the design spec.

## External benchmark roles

Evaluation-only adapter contracts are provided for:

- **RSA-Bench** — primarily the “ignore irrelevant acoustic context” side;
- **MMSU** — acoustic evidence retention / “use” tasks;
- **SH-Bench** — speaker/policy-selective use;
- **VoxSafeBench** — context-dependent acoustic safety evidence.

Adapters require locally prepared/exported `pairs.jsonl` manifests. They never auto-download benchmark audio. The controlled same-waveform diagnostic is run before external scaling.

## Go / No-Go criteria

The approved design is in `docs/superpowers/specs/2026-09-02-selective-acoustic-relevance-design.md`.

- **Gate A — phenomenon:** base must show a meaningful relevance-switch weakness; one-sided success should not trivially imply high PairSwitchAcc.
- **Gate B — mechanism:** QACR gates must change with query relevance on the exact same waveform and outperform query-independent controls at relevance separation.
- **Gate C — functional benefit:** QACR must improve SAR/PairSwitchAcc without materially degrading clean behavior; development target is <= 1 percentage point clean drop.

If A fails, stop. If B fails, do not scale. If C fails, do not add complexity merely to rescue the validation result.

## Testing

All core mechanics are CPU-testable:

```bash
pytest -q
python -m compileall src scripts
```

Tests cover same-waveform hash invariants, deterministic mixing, hand-computed metrics, canonical option scoring helpers, exact QACR identity at gate=1, text-contribution isolation, query-dependent gate switching, cached-key gate padding, baseline interfaces, local benchmark adapters, config strictness, training objectives, and dry-run experiment validation.

## Current verification boundary

The repository code is designed so CPU verification does **not** require downloading Qwen weights. A real Qwen GPU run must still be executed in the target CUDA environment before reporting any empirical QACR result. The code should not be cited as having passed Gate A/B/C until those GPU experiments have actually run.
