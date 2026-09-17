"""Independent replay and pairing audit for held-out composition runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, runner


def close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit(prepared, execution, out):
    plan, cfg = runner.verify(prepared)
    execution = Path(execution)
    payload = json.loads((execution / "results.json").read_text())
    results = payload["results"]
    expected_keys = {(int(seed), condition) for seed in design.SEEDS for condition in design.CONDITIONS}
    expected = len(results)
    design.require(expected > 0, "no runs")
    seen = set()
    max_error = 0.0
    rows_checked = 0
    checkpoints = 0
    pair_rows = 0
    by_pair = {}
    audit_rows = []
    for result in results:
        key = (int(result["seed"]), result["condition"])
        design.require(key in expected_keys, f"invalid run {key}")
        design.require(key not in seen, f"duplicate run {key}")
        seen.add(key)
        support, channel = design.parse_condition(result["condition"])
        seed = int(result["seed"])
        parent_path = runner.parent_path(cfg["parent_root"], seed)
        design.require(result["parent_checkpoint_sha256"] == runner.sha(parent_path), "parent hash mismatch")
        parent = runner.load_checkpoint(parent_path)
        params = runner.policy.clone(parent)
        replacement = runner.base_policy.make_policy(seed + 190000, design.FORM)
        params["worker_logits"][design.TARGET_WORKER] = replacement["worker_logits"][0]
        design.require(result["parent_parameter_sha256"] == runner.base_policy.combined_parameter_hash(parent), "parent parameter hash mismatch")
        design.require(result["initial_parameter_sha256"] == runner.base_policy.combined_parameter_hash(params), "initial parameter hash mismatch")
        log_path = execution / f"seed_{seed}_{result['condition']}" / "training.jsonl"
        logged = load_rows(log_path)
        design.require(len(logged) == int(result["updates"]), "training row count mismatch")
        for row in logged:
            update = int(row["update"])
            episode = design.episode_stream(seed, design.BATCH_SIZE, update=update, support=support, heldout=design.heldout_goal(seed))
            if support == "leave_one_out":
                design.require(not np.any(design.base.goal_index(episode["goal"]) == design.heldout_goal(seed)), "held-out goal leaked into child support")
            trajectory = runner.rollout(params, episode, channel, sample=True)
            beta = design.entropy_coefficient(update)
            grad = runner.gradient(params, episode, trajectory, channel, beta)
            norm = float(np.sqrt(sum(float((value * value).sum()) for value in grad.values())))
            scale = min(1.0, 5.0 / max(norm, 1e-12))
            expected_return = float(trajectory["team_return"][trajectory["active"]].mean())
            max_error = max(max_error, abs(expected_return - float(row["return_mean"])), abs(norm - float(row["gradient_norm"])), abs(scale - float(row["gradient_clip_scale"])))
            design.require(row["world_sha256"] == design.array_sha(episode["site_type"]), "world stream mismatch")
            design.require(row["goal_sha256"] == design.array_sha(episode["goal"]), "goal stream mismatch")
            design.require(row["partner_sha256"] == design.array_sha(episode["partner_id"]), "partner stream mismatch")
            design.require(row["message_uniform_sha256"] == design.array_sha(episode["message_uniforms"]), "message stream mismatch")
            design.require(row["action_uniform_sha256"] == design.array_sha(episode["action_uniforms"]), "action stream mismatch")
            params["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * grad["worker_logits"][design.TARGET_WORKER]
            design.require(row["parameter_sha256"] == runner.base_policy.combined_parameter_hash(params), "parameter replay mismatch")
            if update in result["checkpoints"]:
                checkpoint = execution / f"seed_{seed}_{result['condition']}" / f"checkpoint_{update:04d}.npz"
                design.require(checkpoint.is_file() and row["checkpoint_sha256"] == runner.sha(checkpoint), "checkpoint receipt mismatch")
                with np.load(checkpoint, allow_pickle=False) as data:
                    for key in ("sender_logits_hidden", "sender_logits_visible", "worker_logits"):
                        max_error = max(max_error, float(np.max(np.abs(np.asarray(data[key]) - params[key]))))
                checkpoints += 1
            rows_checked += 1
        design.require(result["final_parameter_sha256"] == runner.base_policy.combined_parameter_hash(params), "final parameter mismatch")
        recomputed = runner.evaluate(params, seed, design.heldout_goal(seed), channel)
        for kind in ("all", "seen", "heldout"):
            for mode in ("natural", "permuted"):
                a = recomputed[kind][mode]["team_return_mean"]
                b = result["final"][kind][mode]["team_return_mean"]
                max_error = max(max_error, abs(float(a) - float(b)))
        audit_rows.append({"seed": seed, "condition": result["condition"], "support": support, "channel": channel, "heldout_goal": design.heldout_goal(seed), "heldout_natural": result["final"]["heldout"]["natural"]["team_return_mean"], "parent_composable": bool(cfg["parent_composable"][str(seed)])})
        by_pair.setdefault((seed, support), {})[channel] = logged
    for (seed, support), arms in by_pair.items():
        design.require(set(arms) == {"live", "silent"}, "missing live/silent pair")
        live, silent = arms["live"], arms["silent"]
        design.require(len(live) == len(silent), "paired log length mismatch")
        for left, right in zip(live, silent):
            for key in ("world_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                design.require(left[key] == right[key], f"paired stream mismatch {seed}/{support}/{key}")
            pair_rows += 1
    design.require(len(seen) == expected, "duplicate or missing run accounting")
    report = {"schema": "heldout_composition_audit_v1", "status": "passed", "runs": len(results), "training_log_rows": rows_checked, "final_checkpoints": checkpoints, "max_abs_replay_error": max_error, "paired_channel_trajectory_rows": pair_rows, "rows": audit_rows}
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("schema", "status", "runs", "training_log_rows", "final_checkpoints", "max_abs_replay_error", "paired_channel_trajectory_rows")}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    audit(args.prepared, args.execution, args.out)
