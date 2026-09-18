"""Independent deterministic replay audit for the open-world expansion study."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from . import design, environment, policy, runner


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _check_stream(row, episode):
    for key, value in runner._stream_hashes(episode).items():
        if row[key] != value:
            raise AssertionError(f"stream mismatch {key}")


def _parent_replay(execution, result):
    seed = int(result["seed"]); architecture = result["architecture"]
    population = result["population"]; community = int(result["community"])
    updates = int(result["updates"])
    run = Path(execution) / "parents" / f"seed_{seed}_{architecture}_{population}_community_{community}"
    rows = _read_rows(run / "training.jsonl")
    assert len(rows) == updates
    communities, _ = runner.load_checkpoint(run / "checkpoint_0000.npz", architecture)
    params = communities[0]
    max_error = 0.0; checkpoints = 0
    for update, row in enumerate(rows, 1):
        episode = design.episode_stream(seed, design.BATCH_SIZE, support="old_world", update=update)
        episode["partner_id"] = np.full(len(episode["goal"]), community * design.PARTNERS_PER_COMMUNITY, dtype=np.int8)
        episode["community_id"] = np.full(len(episode["goal"]), community, dtype=np.int8)
        _check_stream(row, episode)
        tr = environment.rollout([params], None, episode, architecture, population, "hidden")
        grad = runner.parent_gradient(params, episode, tr)
        norm, scale = runner.update_params(params, grad)
        max_error = max(max_error,
                         abs(row["return_mean"] - float(tr["team_return"].mean())),
                         abs(row["gradient_norm"] - norm),
                         abs(row["gradient_clip_scale"] - scale))
        assert row["parameter_sha256"] == policy.parameter_hash(params)
        if update in result["checkpoints"]:
            assert row.get("checkpoint_sha256") == runner.sha(run / f"checkpoint_{update:04d}.npz")
            checkpoints += 1
    assert result["final_parameter_sha256"] == policy.parameter_hash(params)
    assert result["final_checkpoint_sha256"] == runner.sha(run / f"checkpoint_{updates:04d}.npz")
    return len(rows), checkpoints, max_error


def _child_replay(execution, result):
    seed = int(result["seed"]); condition = result["condition"]
    architecture, population, visibility, support = design.parse_child_condition(condition)
    run = Path(execution) / "children" / f"seed_{seed}_{condition}"
    rows = _read_rows(run / "training.jsonl")
    updates = int(result.get("updates", len(rows)))
    assert len(rows) == updates
    communities, fresh = runner.load_checkpoint(run / "checkpoint_0000.npz", architecture)
    assert fresh is not None
    max_error = 0.0; checkpoints = 0
    for update, row in enumerate(rows, 1):
        episode = design.episode_stream(seed, design.BATCH_SIZE, support=support, update=update)
        if support == "old_combo":
            assert not np.any(np.asarray(episode["goal_meaning"]) == design.heldout_combo(seed))
        if support in ("expanded_single_alternating", "expanded_single_pair"):
            assert not np.any(np.asarray(episode["goal_meaning"]) == design.NEW_VALUE * design.VALUES + design.NEW_VALUE)
        _check_stream(row, episode)
        tr = environment.rollout(communities, fresh, episode, architecture, population, visibility)
        grad = runner.child_gradient(fresh, episode, tr, visibility)
        norm, scale = runner.update_params(fresh, grad)
        max_error = max(max_error,
                         abs(row["return_mean"] - float(tr["team_return"].mean())),
                         abs(row["gradient_norm"] - norm),
                         abs(row["gradient_clip_scale"] - scale))
        assert row["parameter_sha256"] == policy.parameter_hash(fresh)
        assert row["community_parameter_sha256"] == [policy.parameter_hash(params) for params in communities]
        if update in result["checkpoints"]:
            assert row.get("checkpoint_sha256") == runner.sha(run / f"checkpoint_{update:04d}.npz")
            checkpoints += 1
    assert result["final_parameter_sha256"] == policy.parameter_hash(fresh)
    assert result["final_community_parameter_sha256"] == [policy.parameter_hash(params) for params in communities]
    assert result["final_checkpoint_sha256"] == runner.sha(run / f"checkpoint_{updates:04d}.npz")
    return len(rows), checkpoints, max_error


def _pair(left, right, fields):
    if left is None or right is None:
        return 0
    aa = {(int(row["seed"]), int(row["update"])): row for row in left}
    bb = {(int(row["seed"]), int(row["update"])): row for row in right}
    keys = sorted(set(aa) & set(bb))
    for key in keys:
        for field in fields:
            if aa[key].get(field) != bb[key].get(field):
                raise AssertionError(f"pair mismatch {field} {key}")
    return len(keys)


def audit(prepared, execution):
    _, cfg = runner.verify(prepared)
    payload = json.loads((Path(execution) / "results.json").read_text())
    parent_rows = child_rows = parent_checkpoints = child_checkpoints = 0
    max_error = 0.0
    child_log = {}
    for result in payload["parents"]:
        rows, checkpoints, error = _parent_replay(execution, result)
        parent_rows += rows; parent_checkpoints += checkpoints; max_error = max(max_error, error)
    for result in payload["children"]:
        rows, checkpoints, error = _child_replay(execution, result)
        child_rows += rows; child_checkpoints += checkpoints; max_error = max(max_error, error)
        child_log[(int(result["seed"]), result["condition"])] = _read_rows(
            Path(execution) / "children" / f"seed_{result['seed']}_{result['condition']}" / "training.jsonl")

    def child(seed, architecture, population, visibility, support):
        return child_log.get((int(seed), f"{architecture}_{population}_{visibility}_{support}"))

    fields = ["scene_sha256", "goal_sha256", "partner_sha256", "role_sha256",
              "message_uniform_sha256", "action_uniform_sha256"]
    architecture_pairs = support_pairs = 0
    for seed in design.SEEDS:
        for support in design.SUPPORTS:
            for left, right in combinations(design.ARCHITECTURES, 2):
                architecture_pairs += _pair(
                    child(seed, left, "aligned", "hidden", support),
                    child(seed, right, "aligned", "hidden", support), fields)
        # The two expanded arms share scenes/goals/uniforms; only the role
        # schedule changes for new meanings, so role_sha256 is excluded.
        comparable = [field for field in fields if field != "role_sha256"]
        for architecture in design.ARCHITECTURES:
            for alternating, pair in (("expanded_alternating", "expanded_pair"),
                                       ("expanded_single_alternating", "expanded_single_pair")):
                support_pairs += _pair(
                    child(seed, architecture, "aligned", "hidden", alternating),
                    child(seed, architecture, "aligned", "hidden", pair), comparable)

    expected_parent = len(design.SEEDS) * len(design.PARENT_CONDITIONS) * design.COMMUNITIES
    expected_child = len(design.SEEDS) * len(design.CHILD_CONDITIONS)
    complete = len(payload["parents"]) == expected_parent and len(payload["children"]) == expected_child
    return {
        "schema": "open_world_expansion_audit_v1",
        "status": "passed" if complete and max_error <= 1e-12 else ("partial" if max_error <= 1e-12 else "failed"),
        "prepared_schema": cfg["schema"],
        "parent_runs": len(payload["parents"]), "child_runs": len(payload["children"]),
        "expected_parent_runs": expected_parent, "expected_child_runs": expected_child,
        "parent_training_log_rows": parent_rows, "child_training_log_rows": child_rows,
        "parent_checkpoints": parent_checkpoints, "child_checkpoints": child_checkpoints,
        "paired_architecture_rows": architecture_pairs, "paired_support_rows": support_pairs,
        "paired_visibility_rows": 0, "paired_population_rows": 0,
        "max_abs_replay_error": float(max_error), "raw_execution_tree": str(execution),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.prepared, args.execution)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
