# Validity-Gated DAA V2 Run

## capability
```json
{
  "num_pairs": 480,
  "target_only_acc": 0.4979166666666667,
  "event_only_acc": 0.9958333333333333,
  "eligible_pairs": 238,
  "eligibility_rate": 0.49583333333333335
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
  "ignore_acc": 0.9201680672268907,
  "use_acc": 0.9747899159663865,
  "sar": 0.9466917587434086,
  "pair_switch_acc": 0.8949579831932774,
  "num_pairs": 238.0
}
```

## oracle_prompt_only
```json
{
  "ignore_acc": 0.8571428571428571,
  "use_acc": 0.957983193277311,
  "sar": 0.9047619047619048,
  "pair_switch_acc": 0.8235294117647058,
  "num_pairs": 238.0,
  "selection_switch_acc": 1.0,
  "selection_labeled_pairs": 238.0,
  "use_event_selection_rate": 1.0,
  "ignore_event_avoid_rate": 1.0,
  "ignore_target_coverage_mean": 1.0,
  "ignore_valid_selection_rate": 1.0,
  "reasoning_acc_given_use_selection": 0.957983193277311,
  "reasoning_acc_given_ignore_avoidance": 0.8571428571428571,
  "reasoning_acc_given_valid_ignore_selection": 0.8571428571428571,
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
  "ignore_acc": 0.8235294117647058,
  "use_acc": 0.9663865546218487,
  "sar": 0.8892571112952223,
  "pair_switch_acc": 0.7941176470588235,
  "num_pairs": 238.0,
  "selection_switch_acc": 1.0,
  "selection_labeled_pairs": 238.0,
  "use_event_selection_rate": 1.0,
  "ignore_event_avoid_rate": 1.0,
  "ignore_target_coverage_mean": 1.0,
  "ignore_valid_selection_rate": 1.0,
  "reasoning_acc_given_use_selection": 0.9663865546218487,
  "reasoning_acc_given_ignore_avoidance": 0.8235294117647058,
  "reasoning_acc_given_valid_ignore_selection": 0.8235294117647058,
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
  "ignore_acc": 0.8025210084033614,
  "use_acc": 0.9495798319327731,
  "sar": 0.8698788868065211,
  "pair_switch_acc": 0.7605042016806722,
  "num_pairs": 238.0,
  "selection_switch_acc": 0.0,
  "selection_labeled_pairs": 238.0,
  "use_event_selection_rate": 0.0,
  "ignore_event_avoid_rate": 1.0,
  "ignore_target_coverage_mean": 0.17196396789713156,
  "ignore_valid_selection_rate": 0.0,
  "reasoning_acc_given_use_selection": null,
  "reasoning_acc_given_ignore_avoidance": 0.8025210084033614,
  "reasoning_acc_given_valid_ignore_selection": null,
  "protocol_failure_pairs": 0.0,
  "protocol_completion_rate": 1.0,
  "protocol_failure_stages": {},
  "focus_source": "model",
  "apply_kv_mask": false,
  "block_strategy": "fixed"
}
```

## fixed_daa
```json
{
  "ignore_acc": 0.3277310924369748,
  "use_acc": 0.3403361344537815,
  "sar": 0.33391469795465356,
  "pair_switch_acc": 0.08403361344537816,
  "num_pairs": 238.0,
  "selection_switch_acc": 0.0,
  "selection_labeled_pairs": 238.0,
  "use_event_selection_rate": 0.0,
  "ignore_event_avoid_rate": 1.0,
  "ignore_target_coverage_mean": 0.17196396789713156,
  "ignore_valid_selection_rate": 0.0,
  "reasoning_acc_given_use_selection": null,
  "reasoning_acc_given_ignore_avoidance": 0.3277310924369748,
  "reasoning_acc_given_valid_ignore_selection": null,
  "protocol_failure_pairs": 0.0,
  "protocol_completion_rate": 1.0,
  "protocol_failure_stages": {},
  "focus_source": "model",
  "apply_kv_mask": true,
  "block_strategy": "fixed"
}
```

## declared_daa
```json
{
  "ignore_acc": 0.0,
  "use_acc": 0.0,
  "sar": 0.0,
  "pair_switch_acc": 0.0,
  "num_pairs": 238.0,
  "selection_switch_acc": null,
  "selection_labeled_pairs": 0.0,
  "use_event_selection_rate": null,
  "ignore_event_avoid_rate": null,
  "ignore_target_coverage_mean": null,
  "ignore_valid_selection_rate": null,
  "reasoning_acc_given_use_selection": null,
  "reasoning_acc_given_ignore_avoidance": null,
  "reasoning_acc_given_valid_ignore_selection": null,
  "protocol_failure_pairs": 238.0,
  "protocol_completion_rate": 0.0,
  "protocol_failure_stages": {
    "block_declaration": 238
  },
  "focus_source": "model",
  "apply_kv_mask": true,
  "block_strategy": "declared"
}
```
