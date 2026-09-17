"""Independent replay and paired-stream audit for heterogeneous grounding."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, runner, policy, environment


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def compare_metric(stored, fresh):
    """Return the largest numerical discrepancy in a stored evaluation."""
    err = 0.0
    for goal_kind, modes in stored.items():
        if not isinstance(modes, dict):
            continue
        for mode, values in modes.items():
            if not isinstance(values, dict) or "team_return_mean" not in values:
                continue
            other = fresh[goal_kind][mode]
            for key in ("team_return_mean", "team_return_sd", "positive_episode_rate"):
                x, y = values[key], other[key]
                if x is None or y is None:
                    design.require(x is None and y is None, f"metric null mismatch {goal_kind}/{mode}/{key}")
                else:
                    err = max(err, abs(float(x) - float(y)))
    return err


def checkpoint_ok(path, params, row):
    design.require(path.is_file() and row.get("checkpoint_sha256") == runner.sha(path), f"checkpoint receipt mismatch {path}")
    with np.load(path, allow_pickle=False) as data:
        for name in ("sender_logits", "worker_logits"):
            design.require(float(np.max(np.abs(np.asarray(data[name]) - params[name]))) < 1e-12, f"checkpoint array mismatch {path}/{name}")


def stream_ok(row, ep):
    names = {
        "site_semantic_sha256": ep["site_semantic"],
        "site_surface_sha256": ep["site_surface"],
        "goal_sha256": ep["goal"],
        "partner_sha256": ep["partner_id"],
        "message_uniform_sha256": ep["message_uniforms"],
        "action_uniform_sha256": ep["action_uniforms"],
    }
    for name, value in names.items():
        design.require(row[name] == design.array_sha(value), f"stream mismatch {name}")


def _parent_key(result):
    return (int(result["seed"]), result["population_mode"], result["form"], result["task"])


def _child_key(result):
    return (int(result["seed"]), result["condition"])


def audit(prepared, execution, out):
    """Replay every update and verify all paired random streams."""
    runner.verify(prepared)
    execution = Path(execution)
    payload = json.loads((execution / "results.json").read_text())
    max_error = 0.0
    parents, children = {}, {}
    parent_logs = child_logs = parent_checkpoints = child_checkpoints = 0

    for result in payload["parents"]:
        key = _parent_key(result)
        design.require(key not in parents, f"duplicate parent {key}")
        parents[key] = result
        seed, population_mode, form, task = key
        params = policy.make_policy(seed, form, "joint_history")
        design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), f"parent initial hash mismatch {key}")
        run = execution / "parents" / f"seed_{seed}_{population_mode}_{form}_{task}"
        log_rows = rows(run / "training.jsonl")
        design.require(len(log_rows) == int(result["updates"]), f"parent log count mismatch {key}")
        for row in log_rows:
            update = int(row["update"])
            ep = design.episode_stream(seed, design.BATCH_SIZE, task=task, support="full", population_mode=population_mode, update=update)
            tr = environment.rollout(params, ep, form, task, "joint_history", "live", sample=True)
            grad = runner.gradient(params, ep, tr, form, "joint_history", "live", child=False)
            norm = float(np.sqrt(sum(float((value * value).sum()) for value in grad.values())))
            scale = min(1.0, 5.0 / max(norm, 1e-12))
            max_error = max(
                max_error,
                abs(float(tr["team_return"].mean()) - float(row["return_mean"])),
                abs(norm - float(row["gradient_norm"])),
                abs(scale - float(row["gradient_clip_scale"])),
            )
            stream_ok(row, ep)
            for name in params:
                params[name] -= design.LEARNING_RATE * scale * grad[name]
            design.require(row["parameter_sha256"] == policy.parameter_hash(params), f"parent parameter mismatch {key}/{update}")
            if update in result["checkpoints"]:
                checkpoint_ok(run / f"checkpoint_{update:04d}.npz", params, row)
                parent_checkpoints += 1
            parent_logs += 1
        design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"parent final mismatch {key}")
        fresh = runner.evaluate(params, seed, population_mode, form, task, "joint_history", "live", design.heldout_goal(seed), None, partner_filter=None)
        max_error = max(max_error, compare_metric(result["final"], fresh))

    for result in payload["children"]:
        key = _child_key(result)
        design.require(key not in children, f"duplicate child {key}")
        children[key] = result
        seed, condition = key
        population_mode, representation, form, task, mapping, support, channel = design.parse_child_condition(condition)
        design.require(result["population_mode"] == population_mode, f"child mode mismatch {key}")
        parent = runner.load_checkpoint(result["parent_checkpoint"])
        params = policy.make_policy(seed + 191000, form, representation, sender_logits=parent["sender_logits"])
        design.require(result["parent_parameter_sha256"] == policy.parameter_hash(parent), f"child parent mismatch {key}")
        design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), f"child initial mismatch {key}")
        run = execution / "children" / f"seed_{seed}_{condition}"
        log_rows = rows(run / "training.jsonl")
        design.require(len(log_rows) == int(result["updates"]), f"child log count mismatch {key}")
        heldout = design.heldout_goal(seed)
        for row in log_rows:
            update = int(row["update"])
            ep = design.episode_stream(seed, design.BATCH_SIZE, task=task, support=support, heldout=heldout, population_mode=population_mode, child_mapping=mapping, update=update)
            if support == "leave_one_out":
                design.require(not np.any(design.goal_index(ep["goal"]) == heldout), f"heldout leakage {key}/{update}")
            tr = environment.rollout(params, ep, form, task, representation, channel, sample=True, partner_filter=design.TARGET_WORKER)
            grad = runner.gradient(params, ep, tr, form, representation, channel, child=True)
            norm = float(np.sqrt(sum(float((value * value).sum()) for value in grad.values())))
            scale = min(1.0, 5.0 / max(norm, 1e-12))
            active = tr["active"]
            returned = float(tr["team_return"][active].mean()) if active.any() else 0.0
            max_error = max(
                max_error,
                abs(returned - float(row["return_mean"])),
                abs(norm - float(row["gradient_norm"])),
                abs(scale - float(row["gradient_clip_scale"])),
            )
            stream_ok(row, ep)
            params["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * grad["worker_logits"][design.TARGET_WORKER]
            design.require(row["parameter_sha256"] == policy.parameter_hash(params), f"child parameter mismatch {key}/{update}")
            if update in result["checkpoints"]:
                checkpoint_ok(run / f"checkpoint_{update:04d}.npz", params, row)
                child_checkpoints += 1
            child_logs += 1
        design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"child final mismatch {key}")
        fresh = runner.evaluate(params, seed, population_mode, form, task, representation, channel, heldout, mapping, partner_filter=design.TARGET_WORKER)
        max_error = max(max_error, compare_metric(result["final"], fresh))

    expected_parents = {
        (seed, child["population_mode"], child["form"], child["task"])
        for (seed, _), child in children.items()
    }
    design.require(set(parents) == expected_parents, "parent accounting mismatch")

    paired_live_silent = paired_mapping = 0
    # Pairing checks use only the conditions actually present in this execution.
    for seed in design.SEEDS:
        for mode in design.POPULATION_MODES:
            for rep in design.REPRESENTATIONS:
                for form in design.FORMS:
                    if rep == "slot_local" and form == "mono4":
                        continue
                    for task in design.TASKS:
                        for support in design.SUPPORTS:
                            for mapping in design.MAPPINGS:
                                live_key = (seed, _condition(mode, rep, form, task, mapping, support, "live"))
                                silent_key = (seed, _condition(mode, rep, form, task, mapping, support, "silent"))
                                live, silent = children.get(live_key), children.get(silent_key)
                                if live and silent:
                                    left_rows = rows(execution / "children" / f"seed_{seed}_{live['condition']}" / "training.jsonl")
                                    right_rows = rows(execution / "children" / f"seed_{seed}_{silent['condition']}" / "training.jsonl")
                                    for left, right in zip(left_rows, right_rows):
                                        for name in ("site_semantic_sha256", "site_surface_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                                            design.require(left[name] == right[name], f"live/silent pair mismatch {name}")
                                        paired_live_silent += 1
                                identity_key = (seed, _condition(mode, rep, form, task, "identity", support, "live"))
                                swap_key = (seed, _condition(mode, rep, form, task, "swap", support, "live"))
                                identity, swap = children.get(identity_key), children.get(swap_key)
                                if identity and swap:
                                    left_rows = rows(execution / "children" / f"seed_{seed}_{identity['condition']}" / "training.jsonl")
                                    right_rows = rows(execution / "children" / f"seed_{seed}_{swap['condition']}" / "training.jsonl")
                                    for left, right in zip(left_rows, right_rows):
                                        for name in ("site_semantic_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                                            design.require(left[name] == right[name], f"mapping pair mismatch {name}")
                                        design.require(left["site_surface_sha256"] != right["site_surface_sha256"], "surface remapping is a no-op")
                                        paired_mapping += 1

    design.require(max_error < 1e-12, f"replay error {max_error}")
    report = {
        "schema": "heterogeneous_grounding_audit_v1",
        "status": "passed",
        "parent_runs": len(parents),
        "child_runs": len(children),
        "parent_training_log_rows": parent_logs,
        "child_training_log_rows": child_logs,
        "parent_checkpoints": parent_checkpoints,
        "child_checkpoints": child_checkpoints,
        "paired_live_silent_rows": paired_live_silent,
        "paired_mapping_rows": paired_mapping,
        "max_abs_replay_error": max_error,
    }
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def _condition(mode, rep, form, task, mapping, support, channel):
    return f"{mode}_{rep}_{form}_{task}_{mapping}_{support}_{channel}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.execution, args.out), ensure_ascii=False))
