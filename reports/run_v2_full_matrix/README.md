# Full DAA V2 matrix — complete

Final export: 2026-09-11T06:16:39.690263+00:00. Implementation commit: `0ab5f644caff1b601f8f6385bd2a3871de0e1c86`.

All six comparison conditions were attempted for every model-eligible pair. No stage stopped because of low accuracy, gain, rescue fraction, or selection quality. Data and numerical correctness checks remained mandatory debugging checks.

The 7B declared-segmentation value below includes the documented endpoint repair: the original full run scored 9/271 (3.32%); 42 overflow candidates were regenerated with unchanged prompts, and 31 newly valid pairs were re-evaluated. The combined repair result is 18/271 (6.64%). Original failed raw text was unavailable, so this is a subset-regenerated repair, not pure re-parsing. See [endpoint repair and provenance](ENDPOINT_REPAIR.md) and [captured declarations](7b_endpoint_repair.json).

The audited 480-pair dev dataset yields 238 eligible 3B pairs and 271 eligible 7B pairs; 184 pairs are eligible for both. PairSwitchAcc requires both ignore and use answers to be correct. Malformed declarations count as unsuccessful pairs and are reported separately.

| Condition | 3B PairSwitchAcc (238 pairs) | 7B PairSwitchAcc (271 pairs) |
|---|---:|---:|
| Mixed Base | 89.50% | 92.25% |
| Oracle Prompt-only | 82.35% | 87.08% |
| Oracle KV | 79.41% | 84.87% |
| Self Prompt-only | 76.05% | 78.60% |
| Self Fixed DAA | 8.40% | 39.48% |
| Declared segmentation | 0.00% | 6.64% |

## Paired changes and selection

| Model | Oracle Prompt vs Base: rescued / harmed | Oracle KV vs Base: rescued / harmed | Fixed KV vs Self Prompt: rescued / harmed |
|---|---:|---:|---:|
| 3B | 7 / 24 | 8 / 32 | 2 / 163 |
| 7B | 8 / 22 | 8 / 28 | 0 / 106 |

Rescued pairs show local positive effects; harmed pairs and net gains are included to avoid selecting only favorable outcomes. The Oracle controls share identical block tables and selections. The Self fixed controls also share their recorded declarations; 3B was independently generated, while 7B uses validated declaration replay.

- 3B Self: 0 declaration-failure pairs; retained target coverage averages 17.20% over 238 parsed pairs. SelectionSwitchAcc is 0.00% over all eligible pairs. Fixed DAA has 0 failure pairs.
- 3B declared segmentation: 0 pairs reached answer evaluation; 238 failed declarations. Failure messages are enumerated in comparison.json.
- 7B Self: 9 declaration-failure pairs; retained target coverage averages 67.21% over 262 parsed pairs. SelectionSwitchAcc is 0.37% over all eligible pairs. Fixed DAA has 9 failure pairs.
- 7B declared segmentation: 53 pairs reached answer evaluation; 218 failed declarations. Failure messages are enumerated in comparison.json.

## Shared eligible cohort

This comparison uses the same 184 pairs. It still does not isolate model size alone: 3B uses FP16 compute and 7B BF16 compute, both with NF4 weights.

| Condition | 3B PairSwitchAcc | 7B PairSwitchAcc |
|---|---:|---:|
| Mixed Base | 92.39% | 93.48% |
| Oracle Prompt-only | 84.78% | 90.76% |
| Oracle KV | 84.24% | 87.50% |
| Self Prompt-only | 80.43% | 79.89% |
| Self Fixed DAA | 10.87% | 37.50% |
| Declared segmentation | 0.00% | 5.98% |

## Correctness, execution, and reuse

The actual GPU was an RTX A4000 16GB. Sampled aggregate peak usage was 15,409 MiB (15.05 GiB), including concurrent stages; no OOM was observed in valid stages. All GPU jobs have finished.

The initial 7B FP16 attempt produced NaN scores and its 179-pair cohort is invalid. It is archived outside the valid result tree. BF16 repaired the numerical issue; the entire 7B capability stage and all comparisons use the rebuilt 271-pair cohort. See [NUMERICAL_FIX.md](NUMERICAL_FIX.md).

Both models passed real all-audio hook identity with zero score difference on the recorded sanity case. All 960 isolated 3B score vectors and 1,904 recomputed DAA answer vectors were finite and preserved the original predictions; all 476 stored Base vectors were finite. Guarded generation reproduced 24 sampled focus prefixes/selections and four sampled scan outcomes. This is a sampled generation audit, not full regeneration of every historical declaration. The 7B replay identity check independently compares declarations, predictions, and score vectors; see its JSON artifact. 139 tests passed.

The reusable paths include model-specific configs, producer-validated declaration replay, optional stopping after complete declarations, fail-fast numerical checks, and stage report export. Original prompts, generation ceilings, and selection definitions are retained; the 80% retained-target threshold measures selection quality and never stops subsequent experiments.

This is a post-hoc development-set mechanism experiment, not a held-out confirmation. The 480 pairs include the earlier 128 cases. Own-cohort scores must not be interpreted as pure cross-model improvements. All selection rates expose their denominators, and declaration failures must not be confused with valid-answer accuracy.

See [3B summaries](3b/SUMMARY.md), [7B summaries](7b/SUMMARY.md), [all metrics and failure messages](comparison.json), [per-pair outcomes](pair_outcomes.csv), [protocol and reusable commands](PROTOCOL.md), [environment and revisions](environment.json), and [artifact hashes](artifact_hashes.json).
