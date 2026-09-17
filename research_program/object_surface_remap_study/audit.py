"""Independent replay and paired-stream audit for the remapping study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, runner, policy, environment


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def compare_metric(stored, fresh):
    err = 0.0
    for kind, modes in stored.items():
        if not isinstance(modes, dict):
            continue
        for mode, values in modes.items():
            if not isinstance(values, dict) or "team_return_mean" not in values:
                continue
            other = fresh[kind][mode]
            for key in ("team_return_mean", "team_return_sd", "positive_episode_rate"):
                x, y = values[key], other[key]
                if x is None or y is None:
                    design.require(x is None and y is None, f"metric null mismatch {kind}/{mode}/{key}")
                else:
                    err = max(err, abs(float(x) - float(y)))
    return err


def checkpoint_ok(path, params, row):
    design.require(path.is_file() and row.get("checkpoint_sha256") == runner.sha(path), f"checkpoint receipt mismatch {path}")
    with np.load(path, allow_pickle=False) as data:
        for name in ("sender_logits", "worker_logits"):
            design.require(float(np.max(np.abs(np.asarray(data[name]) - params[name]))) < 1e-12, f"checkpoint array mismatch {path}/{name}")


def stream_ok(row, ep):
    for name, value in {"site_semantic_sha256": ep["site_semantic"], "site_surface_sha256": ep["site_surface"], "goal_sha256": ep["goal"], "partner_sha256": ep["partner_id"], "message_uniform_sha256": ep["message_uniforms"], "action_uniform_sha256": ep["action_uniforms"]}.items():
        design.require(row[name] == design.array_sha(value), f"stream mismatch {name}")


def audit(prepared, execution, out):
    runner.verify(prepared)
    execution = Path(execution); payload = json.loads((execution / "results.json").read_text())
    max_error = 0.0; parents, children = {}, {}; parent_logs = child_logs = parent_checkpoints = child_checkpoints = 0
    for result in payload["parents"]:
        seed, form, task = int(result["seed"]), result["form"], result["task"]; key = (seed, form, task); design.require(key not in parents, f"duplicate parent {key}"); parents[key] = result
        params = policy.make_policy(seed, form, "joint_history"); design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), "parent initial hash mismatch")
        run = execution / "parents" / f"seed_{seed}_{form}_{task}"; log_rows = rows(run / "training.jsonl"); design.require(len(log_rows) == int(result["updates"]), "parent log count mismatch")
        for row in log_rows:
            update = int(row["update"]); ep = design.episode_stream(seed, design.BATCH_SIZE, task=task, support="full", mapping="identity", update=update); tr = environment.rollout(params, ep, form, task, "joint_history", "live", sample=True); grad = runner.gradient(params, ep, tr, form, "joint_history", "live", child=False); norm = float(np.sqrt(sum(float((v * v).sum()) for v in grad.values()))); scale = min(1.0, 5.0 / max(norm, 1e-12)); max_error = max(max_error, abs(float(tr["team_return"].mean()) - float(row["return_mean"])), abs(norm - float(row["gradient_norm"])), abs(scale - float(row["gradient_clip_scale"]))); stream_ok(row, ep)
            for name in params: params[name] -= design.LEARNING_RATE * scale * grad[name]
            design.require(row["parameter_sha256"] == policy.parameter_hash(params), f"parent parameter mismatch {key}/{update}")
            if update in result["checkpoints"]: checkpoint_ok(run / f"checkpoint_{update:04d}.npz", params, row); parent_checkpoints += 1
            parent_logs += 1
        design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"parent final mismatch {key}"); max_error = max(max_error, compare_metric(result["final"], runner.evaluate(params, seed, form, task, "joint_history", "live", design.heldout_goal(seed), "identity")))

    for result in payload["children"]:
        seed, condition = int(result["seed"]), result["condition"]; rep, form, task, mapping, support, channel = design.parse_child_condition(condition); key = (seed, condition); design.require(key not in children, f"duplicate child {key}"); children[key] = result
        parent = runner.load_checkpoint(result["parent_checkpoint"]); params = policy.make_policy(seed + 191000, form, rep, sender_logits=parent["sender_logits"]); design.require(result["parent_parameter_sha256"] == policy.parameter_hash(parent), f"child parent mismatch {key}"); design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), f"child initial mismatch {key}")
        run = execution / "children" / f"seed_{seed}_{condition}"; log_rows = rows(run / "training.jsonl"); design.require(len(log_rows) == int(result["updates"]), "child log count mismatch"); heldout = design.heldout_goal(seed)
        for row in log_rows:
            update = int(row["update"]); ep = design.episode_stream(seed, design.BATCH_SIZE, task=task, support=support, heldout=heldout, mapping=mapping, update=update)
            if support == "leave_one_out": design.require(not np.any(design.goal_index(ep["goal"]) == heldout), f"heldout leakage {key}/{update}")
            tr = environment.rollout(params, ep, form, task, rep, channel, sample=True, partner_filter=design.TARGET_WORKER); grad = runner.gradient(params, ep, tr, form, rep, channel, child=True); norm = float(np.sqrt(sum(float((v * v).sum()) for v in grad.values()))); scale = min(1.0, 5.0 / max(norm, 1e-12)); active = tr["active"]; returned = float(tr["team_return"][active].mean()) if active.any() else 0.0; max_error = max(max_error, abs(returned - float(row["return_mean"])), abs(norm - float(row["gradient_norm"])), abs(scale - float(row["gradient_clip_scale"]))); stream_ok(row, ep)
            params["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * grad["worker_logits"][design.TARGET_WORKER]; design.require(row["parameter_sha256"] == policy.parameter_hash(params), f"child parameter mismatch {key}/{update}")
            if update in result["checkpoints"]: checkpoint_ok(run / f"checkpoint_{update:04d}.npz", params, row); child_checkpoints += 1
            child_logs += 1
        design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"child final mismatch {key}"); max_error = max(max_error, compare_metric(result["final"], runner.evaluate(params, seed, form, task, rep, channel, heldout, mapping)))

    expected_parents = {(seed, row["form"], row["task"]) for seed, _ in children for row in parents.values() if int(row["seed"]) == seed}; design.require(set(parents) == expected_parents, "parent accounting mismatch")
    paired_live_silent = paired_mapping = 0
    for seed in design.SEEDS:
        for rep in design.REPRESENTATIONS:
            for form in design.FORMS:
                if rep == "slot_local" and form == "mono4": continue
                for task in design.TASKS:
                    for support in design.SUPPORTS:
                        for mapping in design.MAPPINGS:
                            live = children.get((seed, _condition(rep, form, task, mapping, support, "live"))); silent = children.get((seed, _condition(rep, form, task, mapping, support, "silent")))
                            if live and silent:
                                for left, right in zip(rows(execution / "children" / f"seed_{seed}_{live['condition']}" / "training.jsonl"), rows(execution / "children" / f"seed_{seed}_{silent['condition']}" / "training.jsonl")):
                                    for name in ("site_semantic_sha256", "site_surface_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"): design.require(left[name] == right[name], f"live/silent pair mismatch {name}")
                                    paired_live_silent += 1
                            identity = children.get((seed, _condition(rep, form, task, "identity", support, "live"))); swap = children.get((seed, _condition(rep, form, task, "swap", support, "live")))
                            if identity and swap:
                                for left, right in zip(rows(execution / "children" / f"seed_{seed}_{identity['condition']}" / "training.jsonl"), rows(execution / "children" / f"seed_{seed}_{swap['condition']}" / "training.jsonl")):
                                    for name in ("site_semantic_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"): design.require(left[name] == right[name], f"mapping pair mismatch {name}")
                                    design.require(left["site_surface_sha256"] != right["site_surface_sha256"], "surface remapping is a no-op"); paired_mapping += 1
    design.require(max_error < 1e-12, f"replay error {max_error}")
    report = {"schema": "object_surface_remap_audit_v1", "status": "passed", "parent_runs": len(parents), "child_runs": len(children), "parent_training_log_rows": parent_logs, "child_training_log_rows": child_logs, "parent_checkpoints": parent_checkpoints, "child_checkpoints": child_checkpoints, "paired_live_silent_rows": paired_live_silent, "paired_mapping_rows": paired_mapping, "max_abs_replay_error": max_error}
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n"); return report


def _condition(rep, form, task, mapping, support, channel):
    return f"{rep}_{form}_{task}_{mapping}_{support}_{channel}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepared", required=True); parser.add_argument("--execution", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); print(json.dumps(audit(args.prepared, args.execution, args.out), ensure_ascii=False))
