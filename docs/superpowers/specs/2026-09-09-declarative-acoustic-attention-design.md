# Declarative Acoustic Attention (DAA) Design

## Goal

Test whether a Speech LM can **declare which acoustic region it needs for the current query and then reason using only that region's audio KV**, rather than relying on a learned global relevance gate.

The motivating phenomenon remains Selective Acoustic Relevance:

```text
same waveform + same event + different query -> different functional relevance
```

For the same event `d`, relevance is relational:

```text
R = R(d, q)
```

not intrinsic to the sound class.

## Why DAA replaces QACR as the primary intervention

QACR learns a hidden query-conditioned contribution gate. Recent work on declarative attention suggests a stronger hypothesis: an off-the-shelf LM may already know which context region it needs and can expose that choice explicitly. DAA therefore makes acoustic relevance an **addressable model declaration** instead of an external learned router.

QACR stays in the repository as a learned-routing baseline.

## Three-phase protocol

### Phase 1 — Global acoustic scan

The model receives the full waveform once, without the downstream query, and emits a strict block declaration:

```text
<audio_blocks>
B1|0.00|1.35|target speaker
B2|1.35|2.10|dog bark
B3|2.10|3.70|traffic
</audio_blocks>
```

The declaration is generated **once per waveform pair** and reused for both `q_use` and `q_ignore`. This prevents the segmentation itself from changing with query relevance.

Two segmentation conditions are required:

1. `declared`: model-generated event/time blocks;
2. `fixed`: deterministic temporal blocks, isolating focus selection from segmentation quality.

There is no silent fallback from declared to fixed blocks. Parse failure is an experimental failure and is reported.

### Phase 2 — Declarative focus

Given the same waveform, the addressable block table, and the current query, the model must emit only:

```text
<focus_audio blocks="B2">
```

or a bounded multi-block declaration such as:

```text
<focus_audio blocks="B1,B3">
```

The model does not answer during this phase.

### Phase 3 — KV-focused reasoning

The final reasoning pass uses the original waveform and query plus the explicit focus declaration. The inference hook maps selected time spans to Qwen audio placeholder tokens and masks **unselected audio keys before softmax**.

Text/local keys remain available. The mask is only consumed by positions after the audio context (query and generated answer tokens). Audio-token computation itself is not globally zeroed.

For a text/reasoning consumer token `i`:

```text
A'_ij = -inf   if j is an unselected audio key
A'_ij = A_ij   otherwise
```

followed by the ordinary softmax and value aggregation.

This is intentionally different from QACR's post-softmax contribution scaling: DAA implements actual context exclusion and attention renormalization.

## Backbone and resource constraints

Primary backbone:

```text
Qwen/Qwen2.5-Omni-3B
```

Target hardware: single 16 GB RTX A4000.

- Thinker only;
- Talker disabled;
- 4-bit NF4;
- eager attention required for exact block masking;
- no DAA training in the first experiment;
- `layers=[]` means apply focus to all Thinker self-attention layers.

## Data

Use the existing same-waveform pair manifest. Each waveform has exactly two queries:

- `ignore`: event/source should not affect the answer;
- `use`: the same event/source is necessary evidence.

Controlled MVP should prefer time-separable events so temporal blocks can isolate the relevant event. Overlapping speakers are explicitly out of scope unless a source mask/separation system is supplied.

## Baselines

1. Base full-audio attention;
2. QACR learned relevance routing;
3. DAA with fixed temporal blocks;
4. DAA with model-declared blocks;
5. random block selection (next baseline to wire if Gate A is strong);
6. oracle event block (upper bound when valid source timing exists).

## Primary metrics

Task metrics:

- IgnoreAcc;
- UseAcc;
- SAR harmonic mean;
- PairSwitchAcc.

Protocol/mechanism metrics:

- protocol completion rate;
- selection switch accuracy;
- use-event selection rate;
- ignore-event avoidance rate;
- reasoning accuracy given correct use selection;
- reasoning accuracy given correct ignore avoidance;
- failure-stage counts (`block_declaration`, `focus_use`, `focus_ignore`, focused reasoning).

## Falsifiable decomposition

DAA distinguishes two failure types:

### Selection failure

The event required by the query is not selected, or an irrelevant event is selected.

### Reasoning failure

The declared focus is correct but the final answer is still wrong.

This distinction is a central scientific output, not only a debugging statistic.

## Go / No-Go

### Gate A — relevance-switch phenomenon

Base full-audio model must show a meaningful same-waveform switching weakness. If PairSwitchAcc is already close to the one-sided accuracies, stop the project.

### Gate B — declarative selection

DAA must produce valid declarations at useful coverage and selection must switch with query relevance. Fixed-block DAA should separate focus quality from segmentation quality.

### Gate C — functional focus

Focused KV reasoning must improve SAR or PairSwitchAcc relative to Base without causing a material loss on the opposite side of the switch.

If Gate B fails, do not add training. If Gate C fails even with correct selection, the problem is downstream reasoning rather than evidence location.

## Current scope exclusions

No RL, no self-distillation, no source-slot model, no learned separation, no free-form source diarization, and no benchmark scaling until the controlled DAA diagnostic passes its gates.
