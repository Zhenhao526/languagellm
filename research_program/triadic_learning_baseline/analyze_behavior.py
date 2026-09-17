"""Analyze saved final actions without loading policies or running inference."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path
import numpy as np

NAMES = ("A", "B", "C")


def decode(who, index):
    if index == 0:
        return None
    if not 1 <= index <= 16:
        raise ValueError("Noncanonical action index")
    v = int(index) - 1
    return v // 4, (v // 2) % 2, [i for i in range(3) if i != who][v % 2]


def accepts(need, material, destination):
    resource, target = divmod(int(need), 3)
    return (material in ((0, 1), (2, 3), (0, 2), (1, 3))[resource]
            and destination in ((0,), (1,), (0, 1))[target])


def score(packed, choices):
    actions = [decode(i, k) for i, k in enumerate(choices)]
    active = [i for i, a in enumerate(actions) if a is not None]
    flags = [False] * 3
    executed = [False] * 3
    if len(active) != 2:
        return f"active_{len(active)}", 0.0, None, flags, executed
    i, j = active
    a, b = actions[i], actions[j]
    if a[:2] != b[:2] or a[2] != j or b[2] != i:
        return "two_unmatched", 0.0, None, flags, executed
    for who in active:
        flags[who] = accepts(packed[who], packed[3 + a[0]], a[1])
        executed[who] = True
    value = sum(flags) / 2
    return f"matched_{value}", value, NAMES[i] + NAMES[j], flags, executed


def pair_compatible(needs, i, j):
    return any(accepts(needs[i], m, d) and accepts(needs[j], m, d)
               for m in range(4) for d in range(2))


def self_test():
    # ABC indices encode complete actions, not filtered plans.
    state = (0, 0, 0, 0, 1, 2, 3, 1, 2, 3)
    assert score(state, (1, 1, 0))[:3] == ("matched_1.0", 1.0, "AB")
    assert score(state, (1, 1, 1))[0] == "active_3"
    assert score(state, (1, 0, 0))[0] == "active_1"
    assert score(state, (0, 0, 0))[0] == "active_0"
    assert score(state, (1, 3, 0))[0] == "two_unmatched"
    assert score((0, 3, 0) + state[3:], (1, 1, 0))[1] == .5
    assert score((3, 3, 0) + state[3:], (1, 1, 0))[1] == 0
    assert score(state, (2, 0, 1))[:3] == ("matched_1.0", 1.0, "AC")
    assert score(state, (0, 2, 2))[:3] == ("matched_1.0", 1.0, "BC")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.self_test:
        print("9 synthetic settlement cases passed; no saved results read")
        return
    if not args.run or not args.out:
        parser.error("--run and --out required")
    original = args.run / "execution/results.json"
    results = json.loads(original.read_text())
    assert results["status"] == "completed" and len(results["seeds"]) == 4
    summary = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Independent formula on all saved final actions; no forward pass or retraining.",
               "source_results_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
               "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "seeds": []}
    for seed in results["seeds"]:
        outputs = {}
        for name, expected in seed["final"].items():
            path = args.run / f"execution/seed_{seed['seed']}/final_{name}.npz"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            assert digest == expected["data_sha256"]
            with np.load(path, allow_pickle=False) as saved:
                states, choices = saved["states"], saved["action_indices"]
                assert len(states) == expected["worlds"]
                assert np.array_equal(saved["action_probabilities"].argmax(-1), choices)
                categories, successful_pairs, matched_pairs = Counter(), Counter(), Counter()
                mismatch_flags, unmet_need_flags = Counter(), Counter()
                incompatible = {NAMES[i]+NAMES[j]: {"worlds": 0, "full_successes": 0}
                                for i, j in combinations(range(3), 2)}
                total = 0.0
                for k, (state, action) in enumerate(zip(states, choices)):
                    category, value, pair, satisfied, executed = score(state, action)
                    assert value == saved["greedy_reward"][k]
                    assert np.array_equal(saved["satisfied"][k], satisfied)
                    assert np.array_equal(saved["executed"][k], executed)
                    categories[category] += 1
                    total += value
                    if pair:
                        matched_pairs[pair] += 1
                    if category == "two_unmatched":
                        decoded = [decode(i, k) for i, k in enumerate(action)]
                        i, j = [i for i, a in enumerate(decoded) if a is not None]
                        a, b = decoded[i], decoded[j]
                        if a[2] != j or b[2] != i:
                            mismatch_flags["partner_not_mutual"] += 1
                        if a[0] != b[0]:
                            mismatch_flags["different_site"] += 1
                        if a[1] != b[1]:
                            mismatch_flags["different_destination"] += 1
                    if pair and value < 1:
                        for i, ok in enumerate(satisfied):
                            if executed[i] and not ok:
                                site, destination, _ = decode(i, action[i])
                                resource, target = divmod(int(state[i]), 3)
                                if state[3 + site] not in ((0, 1), (2, 3), (0, 2), (1, 3))[resource]:
                                    unmet_need_flags["resource_attribute_not_accepted"] += 1
                                if destination not in ((0,), (1,), (0, 1))[target]:
                                    unmet_need_flags["destination_not_accepted"] += 1
                    if value == 1:
                        successful_pairs[pair] += 1
                    for i, j in combinations(range(3), 2):
                        if not pair_compatible(state[:3], i, j):
                            entry = incompatible[NAMES[i]+NAMES[j]]
                            entry["worlds"] += 1
                            entry["full_successes"] += value == 1
                assert total == expected["greedy_reward_sum"]
                assert sum(successful_pairs.values()) == expected["greedy_full_successes"]
                outputs[name] = {"worlds": len(states), "data_sha256": digest,
                    "categories": dict(categories), "successful_pair_counts": dict(successful_pairs),
                    "matched_pair_counts": dict(matched_pairs),
                    "two_unmatched_world_flags_nonexclusive": dict(mismatch_flags),
                    "executed_unsatisfied_agent_flags_nonexclusive": dict(unmet_need_flags),
                    "when_pair_incompatible": incompatible,
                    "all_argmax_rewards_execution_satisfaction_verified": True}
        summary["seeds"].append({"seed": seed["seed"], "partitions": outputs})
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "behavior.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"seeds": len(summary["seeds"]), "worlds_verified": sum(
        p["worlds"] for s in summary["seeds"] for p in s["partitions"].values()), "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
