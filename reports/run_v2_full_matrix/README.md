# Full DAA V2 matrix — execution in progress

Performance-based stopping gates have been disabled under the user's explicit
instruction. This report directory is an active full-matrix run, not a claim
that every stage has finished. Historical gate-stopped reports are preserved.

Completed 3B results on the same 238 eligible pairs from 480 audited dev pairs:

| Condition | PairSwitchAcc |
|---|---:|
| Mixed Base (verified reuse) | 89.50% |
| Oracle Prompt-only | 82.35% |
| Oracle KV | 79.41% |
| Self Prompt-only | 76.05% |
| Self Fixed DAA | 8.40% |

The independent Self controls produced identical declarations on all 476 rows;
24 fresh replay predictions matched the independently evaluated Fixed DAA run.
Both Self controls had zero runtime/protocol failures. SelectionSwitchAcc was
0%; mean retained target coverage was 17.20%. These outcomes do not stop the
remaining experiments.

3B model-declared segmentation is running. The 7B capability, identity, Base,
Oracle controls, Self controls, and segmentation ablation are queued. Final
exports, shared-eligible comparisons, paired rescue/harm counts, and complete
hardware accounting will replace this progress snapshot after all stages finish.

Validation: 131 tests passed; declaration stopping preserved 24 real 3B focus
prefixes/selections and 4 real scan parse outcomes. See PROTOCOL.md for the
reproducible config paths, decoder optimization, declaration replay, and scope.

7B update: the first FP16 identity check exposed non-finite reference scores.
That attempt and its 179-pair cohort are invalid and archived. BF16 passed the
same real-model probe; the capability cohort and all 7B stages will be rebuilt.
See NUMERICAL_FIX.md. The full experiment matrix remains in progress.
