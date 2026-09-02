from __future__ import annotations

from collections import defaultdict
from typing import Sequence


def compute_pair_metrics(rows: Sequence[dict]) -> dict[str, float]:
    by_role = {"ignore": [], "use": []}
    by_pair: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        role = row["role"]
        if role not in by_role:
            raise ValueError(f"unknown role: {role}")
        correct = bool(row["correct"])
        by_role[role].append(correct)
        by_pair[str(row["pair_id"])][role] = correct
    if not by_role["ignore"] or not by_role["use"]:
        raise ValueError("both ignore and use rows are required")
    ignore_acc = sum(by_role["ignore"]) / len(by_role["ignore"])
    use_acc = sum(by_role["use"]) / len(by_role["use"])
    sar = 0.0 if (ignore_acc + use_acc) == 0 else 2 * ignore_acc * use_acc / (ignore_acc + use_acc)
    complete = [v for v in by_pair.values() if set(v) == {"ignore", "use"}]
    if not complete:
        raise ValueError("no complete use/ignore pairs")
    pair_switch_acc = sum(v["ignore"] and v["use"] for v in complete) / len(complete)
    return {
        "ignore_acc": float(ignore_acc),
        "use_acc": float(use_acc),
        "sar": float(sar),
        "pair_switch_acc": float(pair_switch_acc),
        "num_pairs": float(len(complete)),
    }
