"""Independent replay audit for parent and child runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, runner, policy, environment


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare_metric(stored, fresh):
    max_error = 0.0
    for kind, modes in stored.items():
        for mode, values in modes.items():
            if not isinstance(values, dict) or "team_return_mean" not in values:
                continue
            other = fresh[kind][mode]
            for key in ("team_return_mean", "team_return_sd", "positive_episode_rate"):
                x, y = values[key], other[key]
                if x is None or y is None:
                    design.require(x is None and y is None, f"metric null mismatch {kind}/{mode}/{key}")
                else:
                    max_error = max(max_error, abs(float(x) - float(y)))
    return max_error


def _check_checkpoint(path, params, row, update):
    design.require(path.is_file() and row.get("checkpoint_sha256") == runner.sha(path), f"checkpoint receipt mismatch {path}")
    with np.load(path, allow_pickle=False) as data:
        for name in ("sender_logits", "worker_logits"):
            max_error = float(np.max(np.abs(np.asarray(data[name]) - params[name])))
            design.require(max_error < 1e-12, f"checkpoint array mismatch {path}/{name}")


def audit(prepared, execution, out):
    _, cfg = runner.verify(prepared)
    execution = Path(execution)
    payload = json.loads((execution / "results.json").read_text())
    parent_results = payload["parents"]
    child_results = payload["children"]
    max_error = 0.0
    parent_logs = child_logs = parent_checkpoints = child_checkpoints = 0
    parent_by = {}
    child_by = {}

    for result in parent_results:
        seed, form, task = int(result["seed"]), result["form"], result["task"]
        key = (seed, form, task)
        design.require(key not in parent_by, f"duplicate parent {key}")
        parent_by[key] = result
        params = policy.make_policy(seed, form, "joint_history")
        design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), "parent initial hash mismatch")
        rows = load_rows(Path(execution) / "parents" / f"seed_{seed}_{form}_{task}" / "training.jsonl")
        design.require(len(rows) == int(result["updates"]), "parent training row count mismatch")
        run = Path(execution) / "parents" / f"seed_{seed}_{form}_{task}"
        for row in rows:
            update = int(row["update"])
            ep = design.episode_stream(seed, design.BATCH_SIZE, task=task, update=update, support="full")
            tr = environment.rollout(params, ep, form, task, "joint_history", "live", sample=True)
            grad = runner.gradient(params, ep, tr, form, "joint_history", "live", child=False)
            norm = float(np.sqrt(sum(float((v * v).sum()) for v in grad.values())))
            scale = min(1.0, 5.0 / max(norm, 1e-12))
            max_error = max(max_error, abs(float(tr["team_return"].mean()) - float(row["return_mean"])), abs(norm - float(row["gradient_norm"])), abs(scale - float(row["gradient_clip_scale"])))
            for name, value in (("world_sha256", ep["site_type"]), ("goal_sha256", ep["goal"]), ("partner_sha256", ep["partner_id"]), ("message_uniform_sha256", ep["message_uniforms"]), ("action_uniform_sha256", ep["action_uniforms"])):
                design.require(row[name] == design.array_sha(value), f"parent stream mismatch {key}/{name}")
            for name in params:
                params[name] -= design.LEARNING_RATE * scale * grad[name]
            design.require(row["parameter_sha256"] == policy.parameter_hash(params), f"parent parameter replay mismatch {key}/{update}")
            if update in result["checkpoints"]:
                _check_checkpoint(run / f"checkpoint_{update:04d}.npz", params, row, update)
                parent_checkpoints += 1
            parent_logs += 1
        design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"parent final hash mismatch {key}")
        fresh = runner.evaluate(params, seed, form, task, "joint_history", "live", design.heldout_goal(seed))
        max_error = max(max_error, compare_metric(result["final"], fresh))

    for result in child_results:
        seed, condition = int(result["seed"]), result["condition"]
        key = (seed, condition)
        design.require(key not in child_by, f"duplicate child {key}")
        child_by[key] = result
        representation, form, task, support, channel = design.parse_child_condition(condition)
        parent_checkpoint = Path(result["parent_checkpoint"])
        parent = runner.load_checkpoint(parent_checkpoint)
        params = policy.make_policy(seed + 191000, form, representation, sender_logits=parent["sender_logits"])
        design.require(result["parent_parameter_sha256"] == policy.parameter_hash(parent), f"child parent hash mismatch {key}")
        design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), f"child initial hash mismatch {key}")
        run = Path(execution) / "children" / f"seed_{seed}_{condition}"
        rows = load_rows(run / "training.jsonl")
        design.require(len(rows) == int(result["updates"]), "child training row count mismatch")
        heldout = design.heldout_goal(seed)
        for row in rows:
            update = int(row["update"])
            ep = design.episode_stream(seed, design.BATCH_SIZE, task=task, update=update, support=support, heldout=heldout)
            if support == "leave_one_out":
                design.require(not np.any(design.goal_index(ep["goal"]) == heldout), f"heldout leakage {key}/{update}")
            tr = environment.rollout(params, ep, form, task, representation, channel, sample=True, partner_filter=design.TARGET_WORKER)
            grad = runner.gradient(params, ep, tr, form, representation, channel, child=True)
            norm = float(np.sqrt(sum(float((v * v).sum()) for v in grad.values())))
            scale = min(1.0, 5.0 / max(norm, 1e-12))
            active = tr["active"]
            returned = float(tr["team_return"][active].mean()) if active.any() else 0.0
            max_error = max(max_error, abs(returned - float(row["return_mean"])), abs(norm - float(row["gradient_norm"])), abs(scale - float(row["gradient_clip_scale"])))
            for name, value in (("world_sha256", ep["site_type"]), ("goal_sha256", ep["goal"]), ("partner_sha256", ep["partner_id"]), ("message_uniform_sha256", ep["message_uniforms"]), ("action_uniform_sha256", ep["action_uniforms"])):
                design.require(row[name] == design.array_sha(value), f"child stream mismatch {key}/{name}")
            params["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * grad["worker_logits"][design.TARGET_WORKER]
            design.require(row["parameter_sha256"] == policy.parameter_hash(params), f"child parameter replay mismatch {key}/{update}")
            if update in result["checkpoints"]:
                _check_checkpoint(run / f"checkpoint_{update:04d}.npz", params, row, update)
                child_checkpoints += 1
            child_logs += 1
        design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"child final hash mismatch {key}")
        fresh = runner.evaluate(params, seed, form, task, representation, channel, heldout)
        max_error = max(max_error, compare_metric(result["final"], fresh))

    requested_children = {(int(row["seed"]), row["condition"]) for row in child_results}
    expected_child = requested_children
    expected_parent = {(seed, row["form"], row["task"]) for seed, _ in requested_children for row in parent_results if int(row["seed"]) == seed}
    design.require(set(parent_by) == expected_parent, "parent accounting mismatch")
    design.require(set(child_by) == expected_child, "child accounting mismatch")

    paired_rows = 0
    for seed in design.SEEDS:
        for representation in design.REPRESENTATIONS:
            for form in design.FORMS:
                for task in design.TASKS:
                    if representation == "slot_local" and form == "mono9":
                        continue
                    for support in design.SUPPORTS:
                        live_key = (seed, f"{representation}_{form}_{task}_{support}_live")
                        silent_key = (seed, f"{representation}_{form}_{task}_{support}_silent")
                        if live_key not in child_by or silent_key not in child_by:
                            continue
                        live = child_by[live_key]; silent = child_by[silent_key]
                        left = load_rows(Path(execution) / "children" / f"seed_{seed}_{live['condition']}" / "training.jsonl")
                        right = load_rows(Path(execution) / "children" / f"seed_{seed}_{silent['condition']}" / "training.jsonl")
                        for lrow, rrow in zip(left, right):
                            for name in ("world_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                                design.require(lrow[name] == rrow[name], f"paired stream mismatch {live_key}/{name}")
                            paired_rows += 1
    design.require(max_error < 1e-12, f"replay error {max_error}")
    report = {"schema": "ecological_factorization_audit_v1", "status": "passed", "parent_runs": len(parent_by), "child_runs": len(child_by), "parent_training_log_rows": parent_logs, "child_training_log_rows": child_logs, "parent_checkpoints": parent_checkpoints, "child_checkpoints": child_checkpoints, "paired_child_trajectory_rows": paired_rows, "max_abs_replay_error": max_error}
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepared", required=True); parser.add_argument("--execution", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); audit(args.prepared, args.execution, args.out)
