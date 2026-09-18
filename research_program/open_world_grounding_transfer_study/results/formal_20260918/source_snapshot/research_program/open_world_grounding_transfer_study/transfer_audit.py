"""Replay audit for the frozen-incumbent cultural-transmission stage."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from . import design, environment, policy, runner


def _rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _check_stream(row, episode):
    for key, value in runner._stream_hashes(episode).items():
        if row[key] != value:
            raise AssertionError(f"stream mismatch {key}")


def _replay_one(execution, result):
    seed = int(result["seed"])
    condition = result["condition"]
    architecture, population, visibility, surface_mapping, support = design.parse_transfer_condition(condition)
    run = Path(execution) / "transfers" / f"seed_{seed}_{condition}"
    rows = _rows(run / "training.jsonl")
    updates = int(result["updates"])
    assert len(rows) == updates
    communities, fresh = runner.load_checkpoint(run / "checkpoint_0000.npz", architecture)
    assert fresh is not None
    incumbent_hash = result["incumbent_parameter_sha256"]
    assert [policy.parameter_hash(x) for x in communities] == [incumbent_hash, incumbent_hash]
    max_error = 0.0
    checkpoints = 1 if 0 in result["checkpoints"] else 0
    for update, row in enumerate(rows, 1):
        episode = design.episode_stream(seed, design.BATCH_SIZE, support=support, update=update,
                                        surface_mapping=surface_mapping)
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
        assert row["community_parameter_sha256"] == [incumbent_hash, incumbent_hash]
        assert row["incumbent_parameter_sha256"] == incumbent_hash
        if update in result["checkpoints"]:
            assert row.get("checkpoint_sha256") == runner.sha(run / f"checkpoint_{update:04d}.npz")
            checkpoints += 1
    final_path = run / f"checkpoint_{updates:04d}.npz"
    assert result["final_parameter_sha256"] == policy.parameter_hash(fresh)
    assert result["final_community_parameter_sha256"] == [incumbent_hash, incumbent_hash]
    assert result["final_checkpoint_sha256"] == runner.sha(final_path)
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
                raise AssertionError(f"paired stream mismatch {field} {key}")
    return len(keys)


def audit(prepared, execution):
    _, cfg = runner.verify(prepared)
    payload = json.loads((Path(execution) / "results.json").read_text())
    logs = {}
    rows_total = checkpoints_total = 0
    max_error = 0.0
    for result in payload["transfers"]:
        rows, checkpoints, error = _replay_one(execution, result)
        rows_total += rows
        checkpoints_total += checkpoints
        max_error = max(max_error, error)
        logs[(int(result["seed"]), result["condition"])] = _rows(
            Path(execution) / "transfers" / f"seed_{result['seed']}_{result['condition']}" / "training.jsonl")

    def get(seed, architecture, mapping, support):
        return logs.get((int(seed), f"{architecture}_aligned_hidden_{mapping}_{support}"))

    stream_fields = ["scene_sha256", "fresh_scene_sha256", "goal_sha256", "partner_sha256",
                     "message_uniform_sha256", "action_uniform_sha256"]
    paired_rows = 0
    for seed in design.SEEDS:
        for architecture in design.ARCHITECTURES:
            supports = list(design.TRANSFER_SUPPORTS)
            for mapping in design.OBJECT_SURFACE_MAPPINGS:
                for left, right in combinations(supports, 2):
                    paired_rows += _pair(get(seed, architecture, mapping, left),
                                         get(seed, architecture, mapping, right), stream_fields)
            for support in supports:
                paired_rows += _pair(get(seed, architecture, "identity", support),
                                     get(seed, architecture, "reverse", support),
                                     [field for field in stream_fields if field != "fresh_scene_sha256"])

    expected = len(design.SEEDS) * len(design.TRANSFER_CONDITIONS)
    complete = len(payload["transfers"]) == expected
    return {
        "schema": "open_world_grounding_transfer_audit_v1",
        "status": "passed" if complete and max_error <= 1e-12 else ("partial" if max_error <= 1e-12 else "failed"),
        "prepared_schema": cfg["schema"],
        "transfer_runs": len(payload["transfers"]),
        "expected_transfer_runs": expected,
        "training_log_rows": rows_total,
        "checkpoints": checkpoints_total,
        "paired_stream_rows": paired_rows,
        "max_abs_replay_error": float(max_error),
        "incumbent_frozen": True,
        "raw_execution_tree": str(execution),
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
