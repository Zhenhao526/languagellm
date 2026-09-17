"""Independent replay and paired-stream audit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, runner


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def audit(prepared, execution, out):
    _, cfg = runner.verify(prepared)
    execution = Path(execution)
    results = json.loads((execution / "results.json").read_text())["results"]
    expected_keys = {(int(seed), condition) for seed in design.SEEDS for condition in design.CONDITIONS}
    design.require(results, "no runs")
    seen = set(); max_error = 0.0; rows_checked = 0; checkpoints = 0; pair_rows = 0; pairs = {}
    report_rows = []
    for result in results:
        key = (int(result["seed"]), result["condition"])
        design.require(key in expected_keys and key not in seen, f"invalid or duplicate run {key}")
        seen.add(key)
        representation, support, channel = design.parse_condition(result["condition"])
        seed = int(result["seed"])
        parent_path = runner.parent_path(cfg["parent_root"], seed)
        design.require(result["parent_checkpoint_sha256"] == runner.sha(parent_path), "parent checkpoint hash mismatch")
        parent = runner.load_checkpoint(parent_path)
        params = runner.policy.clone(parent)
        replacement = runner.base_policy.make_policy(seed + 191000, design.FORM)
        params["worker_logits"][design.TARGET_WORKER] = replacement["worker_logits"][0]
        design.require(result["parent_parameter_sha256"] == runner.base_policy.combined_parameter_hash(parent), "parent parameter hash mismatch")
        design.require(result["initial_parameter_sha256"] == runner.base_policy.combined_parameter_hash(params), "initial parameter hash mismatch")
        run_dir = execution / f"seed_{seed}_{result['condition']}"
        logged = load_rows(run_dir / "training.jsonl")
        design.require(len(logged) == int(result["updates"]), "training row count mismatch")
        for row in logged:
            update = int(row["update"])
            episode = design.episode_stream(seed, design.BATCH_SIZE, update=update, support=support, heldout_goal_index=design.heldout_goal(seed))
            if support == "leave_one_out":
                design.require(not np.any(base_goal_index(episode["goal"]) == design.heldout_goal(seed)), "held-out goal leaked")
            trajectory = runner.rollout(params, episode, representation, channel, sample=True)
            grad = runner.gradient(params, episode, trajectory, representation, design.entropy_coefficient(update))
            norm = float(np.sqrt(sum(float((value * value).sum()) for value in grad.values())))
            scale = min(1.0, 5.0 / max(norm, 1e-12))
            expected_return = float(trajectory["team_return"][trajectory["active"]].mean())
            max_error = max(max_error, abs(expected_return - float(row["return_mean"])), abs(norm - float(row["gradient_norm"])), abs(scale - float(row["gradient_clip_scale"])))
            for key_name, value in (("world_sha256", episode["site_type"]), ("goal_sha256", episode["goal"]), ("partner_sha256", episode["partner_id"]), ("message_uniform_sha256", episode["message_uniforms"]), ("action_uniform_sha256", episode["action_uniforms"])):
                design.require(row[key_name] == design.array_sha(value), f"stream mismatch {key_name}")
            params["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * grad["worker_logits"][design.TARGET_WORKER]
            design.require(row["parameter_sha256"] == runner.base_policy.combined_parameter_hash(params), "parameter replay mismatch")
            if update in result["checkpoints"]:
                checkpoint = run_dir / f"checkpoint_{update:04d}.npz"
                design.require(checkpoint.is_file() and row["checkpoint_sha256"] == runner.sha(checkpoint), "checkpoint receipt mismatch")
                with np.load(checkpoint, allow_pickle=False) as data:
                    for name in ("sender_logits_hidden", "sender_logits_visible", "worker_logits"):
                        max_error = max(max_error, float(np.max(np.abs(np.asarray(data[name]) - params[name]))))
                checkpoints += 1
            rows_checked += 1
        design.require(result["final_parameter_sha256"] == runner.base_policy.combined_parameter_hash(params), "final parameter mismatch")
        recomputed = runner.evaluate(params, seed, design.heldout_goal(seed), representation, channel)
        for kind in ("all", "seen", "heldout"):
            for mode in ("natural", "permuted"):
                max_error = max(max_error, abs(float(recomputed[kind][mode]["team_return_mean"]) - float(result["final"][kind][mode]["team_return_mean"])))
        pairs.setdefault((seed, representation, support), {})[channel] = logged
        report_rows.append({"seed": seed, "condition": result["condition"], "representation": representation, "support": support, "channel": channel, "heldout_goal": design.heldout_goal(seed), "heldout_natural": result["final"]["heldout"]["natural"]["team_return_mean"], "parent_composable": bool(cfg["parent_composable"][str(seed)])})
    for key, arms in pairs.items():
        design.require(set(arms) <= {"live", "silent"}, f"invalid channel arm {key}")
        if set(arms) != {"live", "silent"}:
            continue
        for left, right in zip(arms["live"], arms["silent"]):
            for name in ("world_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                design.require(left[name] == right[name], f"paired stream mismatch {key}/{name}")
            pair_rows += 1
    design.require(len(seen) == len(results), "run accounting mismatch")
    report = {"schema": "receiver_representation_audit_v1", "status": "passed", "runs": len(results), "training_log_rows": rows_checked, "final_checkpoints": checkpoints, "max_abs_replay_error": max_error, "paired_channel_trajectory_rows": pair_rows, "rows": report_rows}
    Path(out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("schema", "status", "runs", "training_log_rows", "final_checkpoints", "max_abs_replay_error", "paired_channel_trajectory_rows")}, ensure_ascii=False))


def base_goal_index(goal):
    array = np.asarray(goal, dtype=np.int64)
    return array[..., 0] * 2 + array[..., 1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepared", required=True); parser.add_argument("--execution", required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args(); audit(args.prepared, args.execution, args.out)
