# Selective Acoustic Relevance — Design Specification

Date: 2026-09-02
Status: Approved design
Repository: `yuhanlydia/Speech_noise`

## 1. Research question

The repository tests one falsifiable hypothesis:

> Acoustic relevance is relational, not intrinsic: the same acoustic source/event should be used or ignored depending on the current query.

For a fixed waveform `x` containing a target utterance plus acoustic source/event `d`, create paired queries:

- `q_ignore`: `d` should not influence the answer.
- `q_use`: `d` is necessary evidence for the answer.

The waveform must be byte-identical for the two queries. Query identity is the intended causal change.

## 2. Gap and hidden assumption

Many robustness, enhancement, routing, and representation approaches implicitly bind relevance to a modality, source class, layer, head, or global acoustic strength. We test the stronger relational formulation:

`R = R(d, q)` rather than `R = R(d)`.

A dog bark, bystander utterance, alarm, or other acoustic event is neither globally noise nor globally evidence.

## 3. Experimental protocol

Each same-waveform pair contains:

- one waveform and SHA-256 hash;
- one annotated source/event span or source stem where available;
- `q_ignore`, `y_ignore`;
- `q_use`, `y_use`;
- source/event metadata, seed, SNR/TIR, offset, sample rate;
- optional source/time mask.

MVP source families:

1. Environmental events: dog bark, siren, rain, knock, footsteps, car horn, baby cry, music.
2. Competing speech: target speaker plus irrelevant bystander speech.
3. Conflicting speech: target speaker plus competing answer/instruction.

Primary development data is controlled and generated from local source manifests. The repository does not vendor benchmark audio or copyrighted corpora.

Evaluation-only adapters are reserved for RSA-Bench, MMSU, SH-Bench, and VoxSafeBench.

## 4. Models

Primary backbone: `Qwen/Qwen2.5-Omni-3B`.

Default constrained configuration:

- 4-bit NF4 where supported;
- Thinker enabled;
- Talker disabled;
- text answer scoring only;
- audio-prefill positions tracked separately from text prompt and generated tokens.

The wrapper exposes input preparation, canonical answer scoring, audio token positions, and hookable attention contribution tensors.

Secondary adapter interfaces are reserved for Qwen2.5-Omni-7B and Phi-4-Multimodal.

## 5. Core method: QACR

**Query-Conditioned Audio Contribution Routing (QACR)** does not overwrite absolute K/V states. It gates the contribution of audio positions to a reasoning token.

Base attention output:

`o_i = sum_{j in T} a_ij V_j + sum_{j in A} a_ij V_j`

QACR:

`o'_i = sum_{j in T} a_ij V_j + sum_{j in A} r_j(q) a_ij V_j`

with

`r_j(q) = sigmoid((W_q e_q)^T (W_a h_j) / sqrt(d_r))`.

`T` denotes non-audio positions, `A` audio-prefill positions, `e_q` a query-only representation, and `h_j` the audio token representation.

### Required invariants

1. Text contribution is never gated.
2. Gate=1 exactly recovers base audio contribution up to numerical tolerance.
3. The same waveform may produce different gates under different queries.
4. Routing applies only to configured Thinker layers.
5. Talker and generated-token state remain untouched.

### MVP objective

`L = L_QA + lambda_switch * L_switch + lambda_identity * L_identity`

- `L_QA`: paired answer CE/canonical option likelihood.
- `L_switch`: optional relevance supervision when masks are known.
- `L_identity`: preserves clean/no-event behavior.

The supervised MVP is intentionally minimal. Unsupervised counterfactual relevance targets are deferred until the mechanism validates.

## 6. Baselines

All baselines use one evaluation API and matched data:

1. Base backbone.
2. Static Audio Gate — query-independent scalar/per-layer gate.
3. Query-Conditioned Layer Routing.
4. Query-Conditioned Head Routing.
5. Fixed KV Subspace — mechanistic control from the prior line of work.
6. Oracle Source Mask — upper-bound routing when masks exist.
7. QACR — token/source-level query-conditioned contribution routing.

Trainable parameter counts are reported.

## 7. Metrics

Primary:

- `IgnoreAcc`.
- `UseAcc`.
- `SAR = 2 * IgnoreAcc * UseAcc / (IgnoreAcc + UseAcc)`.
- `PairSwitchAcc`: fraction of waveform pairs where both use and ignore queries are correct.

Routing diagnostics where masks exist:

- relevant-source mean gate;
- irrelevant-source mean gate;
- gate-separation AUC;
- same-waveform query gate-flip magnitude.

Preservation:

- clean accuracy change;
- no-event identity deviation;
- output KL against base when routing should be unnecessary;
- robust accuracy on non-switch controls.

## 8. Go / No-Go gates

### Gate A — phenomenon

Base must show a meaningful relevance-switch gap: good one-sided accuracy must not trivially imply good `PairSwitchAcc`.

### Gate B — mechanism

On the same waveform, QACR gates must change with query/source relevance and separate relevance better than static/query-independent controls.

### Gate C — functional benefit

QACR must improve `SAR` and `PairSwitchAcc` over Base/static routing without materially degrading clean accuracy. Development target: clean drop <= 1 percentage point.

If Gate A fails, stop. If Gate B fails, do not scale. If Gate C fails, do not add complex modules merely to rescue validation.

## 9. Repository architecture

```text
Speech_noise/
├── README.md
├── pyproject.toml
├── configs/{model,data,method,experiment}/
├── scripts/
├── src/sar/
│   ├── data/{schema,mixing,relevance_pairs}.py
│   ├── data/adapters/{rsa_bench,mmsu,sh_bench,voxsafebench}.py
│   ├── models/{base,qwen_omni}.py
│   ├── methods/{base,qacr,static_gate,layer_router,head_router,fixed_kv_subspace,oracle_mask}.py
│   ├── metrics.py
│   ├── config.py
│   ├── train.py
│   └── eval.py
└── tests/
```

## 10. Data flow and reproducibility

1. Read local source manifests.
2. Deterministically mix target and event/source stems.
3. Persist seed, source IDs, offsets, SNR/TIR, sample rate, waveform hash.
4. Create use/ignore records pointing to the exact same waveform hash.
5. Model wrapper identifies text/query and audio-prefill positions.
6. Method computes routing or leaves model untouched.
7. Canonical scorer evaluates options/task output.
8. Evaluator groups by pair and writes JSONL plus summary JSON.

Manifest validation rejects pairs whose hashes differ. Missing benchmark roots raise actionable errors. GPU OOM guidance may be printed, but configs are never silently changed. Results record model id, quantization, seed, config, and git commit when available.

## 11. Test-first strategy

CPU unit tests precede GPU integration:

- gate=1 identity;
- text-contribution isolation;
- same audio/different query changes gates;
- only audio-prefill positions are routable;
- hand-computed pair metrics;
- byte-identical pair waveform/hash;
- config smoke without GPU/model weights;
- tiny synthetic attention integration with gradients.

## 12. MVP execution order

1. Unit tests.
2. Build controlled same-waveform manifest.
3. Evaluate Base for Gate A.
4. Train/evaluate Static Gate and QACR.
5. If Gate B/C pass, add layer/head baselines.
6. Only then run external adapters.

## 13. Out of scope

Talker quality, RL/GRPO, self-distillation, source-slot architectures, learned speech separation, large-scale benchmark training, and broad privacy/safety claims are extensions only after the relevance-switch mechanism validates.
