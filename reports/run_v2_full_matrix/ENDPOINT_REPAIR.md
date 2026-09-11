# Declared-segmentation endpoint precision repair

The scan prompt displays waveform duration to two decimal places, while the
original parser rejects endpoints more than 1 microsecond past the actual
waveform. A 10.126-second waveform is displayed as 10.13 seconds; repeating
that displayed endpoint therefore failed validation. This is a prompt/parser
contract mismatch, separate from the performance stopping rules.

The repair preserves the prompt and generation settings. It maps only the last
block's endpoint to the exact waveform end when all of the following hold:

- The endpoint violates the existing 1-microsecond limit.
- It exactly equals the duration displayed by the scan prompt.
- The display rounds upward by at most 0.005 seconds (plus floating-point epsilon).
- The last block starts before the true waveform end.

Every original partition check then still runs. Arbitrary or intermediate
boundary overflows, overlapping spans, missing coverage, and malformed tags
are not accepted by this rule. Previously accepted tables remain unchanged.
Eight regression cases cover the repair and these exclusions; 139 tests pass.

The original complete 7B ablation had 22 answer-evaluated pairs, 249 declaration
failures, and PairSwitchAcc 9/271 (3.32%). Among the failures, 42 reported an
endpoint overflow. Their original raw generations were not saved, so a new
same-prompt generation is required to diagnose actual rounding effects. Matching
the old error string on regeneration is not proof that the original raw text
was identical. A repair audit records both parser outcomes on each newly
captured raw declaration; any changed answer is freshly computed.

The original run is archived separately. The final report identifies the
subset-regenerated repair and preserves the original all-pair result. All 238
3B ablation failures were missing tags or invalid lines, which occur before
this endpoint rule and are unaffected. All originally successful 7B tables are
also checked for exact equality under the repaired parser.

## Observed repair outcome

All 42 candidates reproduced their historical old-parser error message on the newly generated text. The repaired parser accepted 31; their answers were freshly evaluated. All 31 reached answer evaluation, increasing completed pairs from 22 to 53, and pair successes from 9 to 18. The repaired combined result is 18/271 (6.64%); 218 pairs remain declaration failures. The other 11 candidates remain unsuccessful. Their new parser errors are recorded alongside their preserved historical failure rows.

The original 22 successful tables are exactly unchanged, and all 238 3B missing-tag/invalid-line failures occur before the repaired rule. See [per-case evidence](7b_endpoint_repair.json), [original summary](7b_declared_original_summary.json), and [original producer receipt](7b_declared_original_run_metadata.json).
