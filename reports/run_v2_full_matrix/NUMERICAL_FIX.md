# 7B numerical failure and repair

The initial 7B FP16 attempt is invalid for analysis. The identity check returned
NaN, despite both argmax predictions being A. A real-model probe established:

- All four option log-probabilities were NaN for both the unmodified reference
  and all-audio-allowed hook on the same mixed waveform.
- The isolated speech score vector was also all NaN, yet the old argmax selected
  A and could incorrectly admit the pair into the eligible cohort.
- The first observed non-finite text-layer output was at layer index 27.
- Every parameter was on cuda:0; this was not CPU offloading or an OOM.
- The same NF4 model and input in BF16 produced finite reference/hook vectors
  `[-0.314453125, -3.75, -1.5625, -4.0]`, with exact identity.

Discard the initial 179-pair FP16 cohort. Its raw artifacts are preserved under
`results/v2_full_matrix/invalid_attempts/7b_fp16`, outside the valid result tree.
Rebuild capability and every 7B condition with explicit `model.torch_dtype:
bfloat16`. BF16 is supported by this RTX A4000 and retains 16-bit compute storage.
The 3B default remains FP16; audit its existing score outputs before retaining
those results. Cross-model comparisons therefore compare these documented
model/precision configurations, not model size in isolation.

## Guardrails in the evaluator

Canonical option scores reject NaN and positive/negative infinity with a
`NonFiniteScoreError`. This error propagates through capability, Base, DAA,
replay, and identity; it cannot invoke Base fallback or become an ordinary
protocol-failure row. Identity also checks vectors explicitly before comparing.

Both free-form generation entry points reject NaN, positive infinity, and
rows without any finite candidate. Individual suppressed generation tokens may
still be negative infinity, as required by normal decoding constraints. Invalid
model-generated declarations remain ordinary parser failures, distinguishable
from numerical failures. Attention hooks are restored even on these errors.

New capability and DAA outputs retain option log-probability vectors so later
numerical audits do not require reconstructing every answer pass.
