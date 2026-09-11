# Validity-Gated DAA V2 Run

## capability
```json
{
  "num_pairs": 480,
  "target_only_acc": 0.5666666666666667,
  "event_only_acc": 0.9958333333333333,
  "eligible_pairs": 271,
  "eligibility_rate": 0.5645833333333333
}
```

## hook_identity
```json
{
  "ok": true,
  "predictions_match": true,
  "max_abs_logprob_diff": 0.0,
  "tolerance": 0.0001,
  "pair_id": "dev-esc50-0001",
  "role": "ignore",
  "reference_prediction": "A",
  "hook_prediction": "A"
}
```

## base
```json
{
  "ignore_acc": 0.940959409594096,
  "use_acc": 0.977859778597786,
  "sar": 0.9590547828555209,
  "pair_switch_acc": 0.922509225092251,
  "num_pairs": 271.0
}
```

## oracle_prompt_only
```json
{
  "ignore_acc": 0.8856088560885609,
  "use_acc": 0.977859778597786,
  "sar": 0.9294508786672026,
  "pair_switch_acc": 0.8708487084870848,
  "num_pairs": 271.0,
  "selection_switch_acc": 1.0,
  "selection_labeled_pairs": 271.0,
  "use_event_selection_rate": 1.0,
  "ignore_event_avoid_rate": 1.0,
  "ignore_target_coverage_mean": 1.0,
  "ignore_valid_selection_rate": 1.0,
  "reasoning_acc_given_use_selection": 0.977859778597786,
  "reasoning_acc_given_ignore_avoidance": 0.8856088560885609,
  "reasoning_acc_given_valid_ignore_selection": 0.8856088560885609,
  "protocol_failure_pairs": 0.0,
  "protocol_completion_rate": 1.0,
  "protocol_failure_stages": {},
  "focus_source": "oracle",
  "apply_kv_mask": false,
  "block_strategy": "fixed"
}
```

## oracle_kv
```json
{
  "ignore_acc": 0.8634686346863468,
  "use_acc": 0.981549815498155,
  "sar": 0.9187306273062732,
  "pair_switch_acc": 0.8487084870848709,
  "num_pairs": 271.0,
  "selection_switch_acc": 1.0,
  "selection_labeled_pairs": 271.0,
  "use_event_selection_rate": 1.0,
  "ignore_event_avoid_rate": 1.0,
  "ignore_target_coverage_mean": 1.0,
  "ignore_valid_selection_rate": 1.0,
  "reasoning_acc_given_use_selection": 0.981549815498155,
  "reasoning_acc_given_ignore_avoidance": 0.8634686346863468,
  "reasoning_acc_given_valid_ignore_selection": 0.8634686346863468,
  "protocol_failure_pairs": 0.0,
  "protocol_completion_rate": 1.0,
  "protocol_failure_stages": {},
  "focus_source": "oracle",
  "apply_kv_mask": true,
  "block_strategy": "fixed"
}
```

## prompt_only
```json
{
  "ignore_acc": 0.8265682656826568,
  "use_acc": 0.915129151291513,
  "sar": 0.8685971605478767,
  "pair_switch_acc": 0.7859778597785978,
  "num_pairs": 271.0,
  "selection_switch_acc": 0.003816793893129771,
  "selection_labeled_pairs": 262.0,
  "use_event_selection_rate": 0.29389312977099236,
  "ignore_event_avoid_rate": 0.5801526717557252,
  "ignore_target_coverage_mean": 0.6720695589202704,
  "ignore_valid_selection_rate": 0.04198473282442748,
  "reasoning_acc_given_use_selection": 0.974025974025974,
  "reasoning_acc_given_ignore_avoidance": 0.8421052631578947,
  "reasoning_acc_given_valid_ignore_selection": 0.8181818181818182,
  "protocol_failure_pairs": 9.0,
  "protocol_completion_rate": 0.966789667896679,
  "protocol_failure_stages": {
    "focus_ignore": 5,
    "focus_use": 4
  },
  "focus_source": "model",
  "apply_kv_mask": false,
  "block_strategy": "fixed"
}
```

## fixed_daa
```json
{
  "ignore_acc": 0.5867158671586716,
  "use_acc": 0.6309963099630996,
  "sar": 0.6080509896008052,
  "pair_switch_acc": 0.3948339483394834,
  "num_pairs": 271.0,
  "selection_switch_acc": 0.003816793893129771,
  "selection_labeled_pairs": 262.0,
  "use_event_selection_rate": 0.29389312977099236,
  "ignore_event_avoid_rate": 0.5801526717557252,
  "ignore_target_coverage_mean": 0.6720695589202704,
  "ignore_valid_selection_rate": 0.04198473282442748,
  "reasoning_acc_given_use_selection": 0.961038961038961,
  "reasoning_acc_given_ignore_avoidance": 0.4144736842105263,
  "reasoning_acc_given_valid_ignore_selection": 0.9090909090909091,
  "protocol_failure_pairs": 9.0,
  "protocol_completion_rate": 0.966789667896679,
  "protocol_failure_stages": {
    "focus_ignore": 5,
    "focus_use": 4
  },
  "focus_source": "model",
  "apply_kv_mask": true,
  "block_strategy": "fixed",
  "selection_replayed": true,
  "selection_source_results": "results/v2_full_matrix/7b/mvp_daa_prompt_only/results.jsonl",
  "selection_source_sha256": "c8c50b24dfe6472772fd89c2f8c2a57f25285e02213f0096fdc235510c10fcd6",
  "manifest_sha256": "4cbab9368dacbd6fba876e3f7054459bd5bd212f3249a1f4d87fe63435c7234f"
}
```

## declared_daa
```json
{
  "ignore_acc": 0.0996309963099631,
  "use_acc": 0.14391143911439114,
  "sar": 0.11774572291177457,
  "pair_switch_acc": 0.06642066420664207,
  "num_pairs": 271.0,
  "selection_switch_acc": 0.05660377358490566,
  "selection_labeled_pairs": 53.0,
  "use_event_selection_rate": 0.6037735849056604,
  "ignore_event_avoid_rate": 0.8301886792452831,
  "ignore_target_coverage_mean": 0.5416568076799159,
  "ignore_valid_selection_rate": 0.11320754716981132,
  "reasoning_acc_given_use_selection": 0.96875,
  "reasoning_acc_given_ignore_avoidance": 0.4318181818181818,
  "reasoning_acc_given_valid_ignore_selection": 0.5,
  "protocol_failure_pairs": 218.0,
  "protocol_completion_rate": 0.19557195571955718,
  "protocol_failure_stages": {
    "block_declaration": 218
  },
  "focus_source": "model",
  "apply_kv_mask": true,
  "block_strategy": "declared"
}
```
