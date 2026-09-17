"""Independent replay and paired-stream audit for incumbent transfer."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, environment, policy, runner


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


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
            design.require(
                float(np.max(np.abs(np.asarray(data[name]) - params[name]))) < 1e-12,
                f"checkpoint array mismatch {path}/{name}",
            )


def stream_ok(row, ep):
    expected = {
        "site_semantic_sha256": ep["site_semantic"],
        "site_surface_sha256": ep["site_surface"],
        "goal_sha256": ep["goal"],
        "partner_sha256": ep["partner_id"],
        "message_uniform_sha256": ep["message_uniforms"],
        "action_uniform_sha256": ep["action_uniforms"],
    }
    for name, value in expected.items():
        design.require(row[name] == design.array_sha(value), f"stream mismatch {name}")


def _replay_parent(result, execution):
    seed, form, task = int(result["seed"]), result["form"], result["task"]
    params = policy.make_policy(seed, form, "joint_history")
    design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), "parent initial hash mismatch")
    run = execution / "parents" / f"seed_{seed}_{form}_{task}"
    log_rows = rows(run / "training.jsonl")
    design.require(len(log_rows) == int(result["updates"]), "parent log count mismatch")
    max_error = 0.0
    checkpoints = 0
    for row in log_rows:
        update = int(row["update"])
        ep = design.episode_stream(seed, design.BATCH_SIZE, task=task, mapping="identity", update=update)
        tr = environment.rollout(params, ep, form, task, "joint_history", "live", sample=True)
        grad = runner.gradient(params, ep, tr, form, "joint_history", "live", child=False)
        norm, scale = runner._update_params(params, grad, child=False)
        max_error = max(
            max_error,
            abs(float(tr["team_return"].mean()) - float(row["return_mean"])),
            abs(norm - float(row["gradient_norm"])),
            abs(scale - float(row["gradient_clip_scale"])),
        )
        stream_ok(row, ep)
        design.require(row["parameter_sha256"] == policy.parameter_hash(params), f"parent parameter mismatch {seed}/{update}")
        if update in result["checkpoints"]:
            checkpoint_ok(run / f"checkpoint_{update:04d}.npz", params, row)
            checkpoints += 1
    design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"parent final mismatch {seed}")
    max_error = max(
        max_error,
        compare_metric(
            result["final"],
            runner.evaluate(params, seed, form, task, "joint_history", "live", design.heldout_goal(seed), "identity"),
        ),
    )
    return max_error, len(log_rows), checkpoints


def _replay_child(result, execution):
    seed, condition = int(result["seed"]), result["condition"]
    initialization, mapping, channel = design.parse_child_condition(condition)
    parent = runner.load_checkpoint(result["parent_checkpoint"])
    params = runner.child_initial_params(seed, initialization, parent)
    design.require(result["initialization"] == initialization, f"child initialization mismatch {seed}/{condition}")
    design.require(result["parent_parameter_sha256"] == policy.parameter_hash(parent), f"child parent mismatch {seed}/{condition}")
    design.require(result["initial_parameter_sha256"] == policy.parameter_hash(params), f"child initial mismatch {seed}/{condition}")
    if initialization == "incumbent":
        design.require(
            np.array_equal(params["worker_logits"][design.TARGET_WORKER], parent["worker_logits"][design.TARGET_WORKER]),
            f"incumbent copy mismatch {seed}/{condition}",
        )
    else:
        design.require(
            not np.array_equal(params["worker_logits"][design.TARGET_WORKER], parent["worker_logits"][design.TARGET_WORKER]),
            f"fresh worker unexpectedly equals parent {seed}/{condition}",
        )
    sender_initial = params["sender_logits"].copy()
    run = execution / "children" / f"seed_{seed}_{condition}"
    log_rows = rows(run / "training.jsonl")
    design.require(len(log_rows) == int(result["updates"]), "child log count mismatch")
    heldout = design.heldout_goal(seed)
    max_error = 0.0
    checkpoints = 0
    for row in log_rows:
        update = int(row["update"])
        ep = design.episode_stream(seed, design.BATCH_SIZE, task="factorized", mapping=mapping, update=update)
        tr = environment.rollout(
            params,
            ep,
            "dual2",
            "factorized",
            "joint_history",
            channel,
            sample=True,
            partner_filter=design.TARGET_WORKER,
        )
        grad = runner.gradient(params, ep, tr, "dual2", "joint_history", channel, child=True)
        norm, scale = runner._update_params(params, grad, child=True)
        active = tr["active"]
        returned = float(tr["team_return"][active].mean()) if active.any() else 0.0
        max_error = max(
            max_error,
            abs(returned - float(row["return_mean"])),
            abs(norm - float(row["gradient_norm"])),
            abs(scale - float(row["gradient_clip_scale"])),
        )
        stream_ok(row, ep)
        design.require(row["parameter_sha256"] == policy.parameter_hash(params), f"child parameter mismatch {seed}/{condition}/{update}")
        if update in result["checkpoints"]:
            checkpoint_ok(run / f"checkpoint_{update:04d}.npz", params, row)
            checkpoints += 1
    design.require(np.array_equal(params["sender_logits"], sender_initial), f"sender changed in child {seed}/{condition}")
    design.require(result["final_parameter_sha256"] == policy.parameter_hash(params), f"child final mismatch {seed}/{condition}")
    # The initial endpoint is checked before training; reconstruct it explicitly.
    initial_params = runner.child_initial_params(seed, initialization, parent)
    max_error = max(
        max_error,
        compare_metric(
            result["initial"],
            runner.evaluate(initial_params, seed, "dual2", "factorized", "joint_history", channel, heldout, mapping),
        ),
        compare_metric(
            result["final"],
            runner.evaluate(params, seed, "dual2", "factorized", "joint_history", channel, heldout, mapping),
        ),
    )
    return max_error, len(log_rows), checkpoints


def _pair_streams(execution, children, seeds):
    paired_live_silent = 0
    paired_mapping = 0
    for seed in seeds:
        for initialization in design.INITIALIZATIONS:
            for mapping in design.MAPPINGS:
                live = children[(seed, f"{initialization}_{mapping}_live")]
                silent = children[(seed, f"{initialization}_{mapping}_silent")]
                live_rows = rows(execution / "children" / f"seed_{seed}_{live['condition']}" / "training.jsonl")
                silent_rows = rows(execution / "children" / f"seed_{seed}_{silent['condition']}" / "training.jsonl")
                design.require(len(live_rows) == len(silent_rows), "live/silent length mismatch")
                for left, right in zip(live_rows, silent_rows):
                    for name in ("site_semantic_sha256", "site_surface_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                        design.require(left[name] == right[name], f"live/silent pair mismatch {name}")
                    paired_live_silent += 1
            identity = children[(seed, f"{initialization}_identity_live")]
            swap = children[(seed, f"{initialization}_swap_live")]
            identity_rows = rows(execution / "children" / f"seed_{seed}_{identity['condition']}" / "training.jsonl")
            swap_rows = rows(execution / "children" / f"seed_{seed}_{swap['condition']}" / "training.jsonl")
            design.require(len(identity_rows) == len(swap_rows), "mapping pair length mismatch")
            for left, right in zip(identity_rows, swap_rows):
                for name in ("site_semantic_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                    design.require(left[name] == right[name], f"mapping pair mismatch {name}")
                design.require(left["site_surface_sha256"] != right["site_surface_sha256"], "surface remapping is a no-op")
                paired_mapping += 1
    return paired_live_silent, paired_mapping


def audit(prepared, execution, out):
    runner.verify(prepared)
    execution = Path(execution)
    payload = json.loads((execution / "results.json").read_text())
    parents, children = {}, {}
    max_error = 0.0
    parent_logs = child_logs = parent_checkpoints = child_checkpoints = 0
    for result in payload["parents"]:
        key = (int(result["seed"]), result["form"], result["task"])
        design.require(key not in parents, f"duplicate parent {key}")
        parents[key] = result
        err, logs, checkpoints = _replay_parent(result, execution)
        max_error = max(max_error, err)
        parent_logs += logs
        parent_checkpoints += checkpoints
    for result in payload["children"]:
        key = (int(result["seed"]), result["condition"])
        design.require(key not in children, f"duplicate child {key}")
        children[key] = result
        err, logs, checkpoints = _replay_child(result, execution)
        max_error = max(max_error, err)
        child_logs += logs
        child_checkpoints += checkpoints
    expected_parents = {(seed, "dual2", "factorized") for seed, _ in children}
    design.require(set(parents) == expected_parents, "parent accounting mismatch")
    observed_seeds = sorted({seed for seed, _ in children})
    expected_children = {(seed, condition) for seed in observed_seeds for condition in design.CHILD_CONDITIONS}
    design.require(set(children) == expected_children, "child accounting mismatch")
    paired_live_silent, paired_mapping = _pair_streams(execution, children, observed_seeds)
    design.require(max_error < 1e-12, f"replay error {max_error}")
    report = {
        "schema": "object_surface_transfer_audit_v1",
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
        "sender_frozen_in_children": True,
        "incumbent_copy_checked": True,
    }
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.prepared, args.execution, args.out), ensure_ascii=False))
