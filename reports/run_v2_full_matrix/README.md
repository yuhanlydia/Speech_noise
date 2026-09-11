# Full DAA V2 matrix — 3B complete, 7B in progress

Progress snapshot: 2026-09-11T05:23:06.579804+00:00. Implementation commit: `59c47be`.

The user explicitly removed all performance-based stopping gates. Every condition is being evaluated regardless of accuracy, gain, rescue fraction, or selection quality. This is a progress report, not a claim that the entire matrix has finished.

The same audited 480 dev pairs yield 238 eligible 3B pairs and 271 eligible 7B pairs. The 7B run uses BF16 after the numerical repair described below.

| Condition | 3B PairSwitchAcc | 7B PairSwitchAcc |
|---|---:|---:|
| Mixed Base | 89.50% | 92.25% |
| Oracle Prompt-only | 82.35% | 87.08% |
| Oracle KV | 79.41% | pending |
| Self Prompt-only | 76.05% | pending |
| Self Fixed DAA | 8.40% | pending |
| Declared segmentation | 0.00% (declarations failed) | pending |

The 3B segmentation ablation attempted all 238 pairs: 202 lacked a complete block tag and 36 contained invalid block lines. No pair reached its answer pass. Both 3B Self fixed-block controls completed without protocol failures, with identical declarations on all 476 rows, SelectionSwitchAcc 0%, and mean retained target coverage 17.20%.

On the 184 pairs eligible for both models, Base succeeds on 170/184 for 3B and 172/184 for 7B. Final paired rescue/harm counts and all-condition shared-cohort comparisons will be added after 7B finishes. Do not compare different eligible cohorts as a pure model-size effect.

Numerical validation: all 960 isolated 3B score vectors and 1,904 recomputed DAA answer vectors were finite and preserved the original predictions; all 476 stored Base vectors were finite. Both models passed the real all-audio identity check with zero score difference. Guarded generation reproduced 24 sampled 3B focus prefixes/selections and four sampled scan outcomes; this was a sampled generation audit, not full regeneration of every historical declaration. 131 tests passed.

The initial 7B FP16 attempt produced NaN scores and its 179-pair cohort is invalid. Those artifacts are archived outside the valid result tree. The valid BF16 capability result is speech-only 56.67%, event-only 99.58%, eligible 271/480. See [NUMERICAL_FIX.md](NUMERICAL_FIX.md).

See [3B stage summaries](3b/SUMMARY.md), [protocol and reusable commands](PROTOCOL.md), [full 3B score audit](3b_numerical_audit.json), and [guarded generation sample audit](3b_guarded_generation_audit.json). This remains a post-hoc development-set mechanism experiment, not a held-out confirmation.
