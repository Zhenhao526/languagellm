"""Replay logged environment metadata and roles without importing a trainer.

Example, from the workspace root:
  .venv/bin/python -m pilot_round1.verify_replay --trials 2000

This checks the recorded random streams, not neural-network optimization or
pixel equality: the interaction log contains metadata but no image hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def verify(batch_dir: Path, run_dir: Path, trials: int) -> dict:
    if trials <= 0:
        raise ValueError("trials must be positive")
    batch = json.loads((batch_dir / "batch_config.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    snapshot = batch_dir / "source_snapshot"
    hashes = batch["source_sha256"]
    for name, expected in hashes.items():
        actual = hashlib.sha256((snapshot / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"source snapshot hash mismatch: {name}")

    for run_key, batch_key in (
        ("planned_steps", "steps_per_run"),
        ("rollout", "rollout"),
        ("eval_every", "eval_every"),
        ("eval_n", "eval_n"),
        ("lr", "learning_rate"),
        ("entropy", "entropy"),
        ("device", "communication_device"),
    ):
        if config[run_key] != batch[batch_key]:
            raise ValueError(f"batch/run configuration mismatch: {run_key}")
    if config["task"] not in batch["tasks"]:
        raise ValueError("task is outside the declared batch")
    if config["condition"] not in batch["conditions"]:
        raise ValueError("condition is outside the declared batch")
    seeds_key = "no_comm_seeds" if config["task"] == "no_comm" else "seeds"
    if config["seed"] not in batch[seeds_key]:
        raise ValueError("seed is outside the declared batch")
    if min(config["rollout"], config["eval_every"], config["planned_steps"]) <= 0:
        raise ValueError("invalid rollout, evaluation interval, or planned steps")
    if trials > config["planned_steps"]:
        raise ValueError("requested replay exceeds planned training steps")

    rows = []
    log_digest = hashlib.sha256()
    with (run_dir / "interactions.jsonl").open("rb") as stream:
        for raw in stream:
            # A live writer can leave a final partial line visible to readers.
            if not raw.endswith(b"\n"):
                break
            row = json.loads(raw)
            rows.append(row)
            log_digest.update(raw)
            if len(rows) >= trials:
                break
    if len(rows) < trials:
        return {
            "status": "insufficient_log",
            "run": str(run_dir),
            "requested_trials": trials,
            "complete_logged_trials": len(rows),
        }

    spec = importlib.util.spec_from_file_location(
        "_pilot_replay_snapshot_env", snapshot / "env.py"
    )
    if spec is None or spec.loader is None:
        raise ValueError("cannot load the verified environment snapshot")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed = int(config["seed"])
    world = module.VisualWorld(seed * 100000 + 1001)
    role_rng = np.random.default_rng(seed * 100000 + 1002)
    done = 0
    next_eval = int(config["eval_every"])
    checked = 0
    rollouts = 0
    evaluation_boundaries = []
    mismatches = []

    def compare(step, field, logged, replayed):
        if logged != replayed:
            mismatches.append(
                {"step": step, "field": field, "logged": logged, "replayed": replayed}
            )

    while checked < trials:
        # Do not truncate to the requested audit length before sampling. The
        # original full rollout size affects the environment RNG consumption.
        count = min(
            config["rollout"], config["planned_steps"] - done, next_eval - done
        )
        replay = world.sample(count)
        roles = role_rng.integers(0, 2, count)
        rollouts += 1
        for offset in range(min(count, trials - checked)):
            row = rows[checked]
            step = done + offset + 1
            compare(step, "step", row["step"], step)
            compare(step, "target_id", row["target_id"], int(replay["target_ids"][offset]))
            compare(
                step, "candidate_ids", row["candidate_ids"],
                replay["candidate_ids"][offset].tolist(),
            )
            compare(step, "sender", row["sender"], int(roles[offset]))
            correct_index = int(replay["correct_indices"][offset])
            logged_matches = [
                i for i, class_id in enumerate(row["candidate_ids"])
                if class_id == row["target_id"]
            ]
            compare(step, "inferred_correct_indices", logged_matches, [correct_index])
            # Some future log versions may explicitly record this index.
            if "correct_index" in row:
                compare(step, "correct_index", row["correct_index"], correct_index)
            compare(
                step, "selected_id", row["selected_id"],
                row["candidate_ids"][row["choice"]],
            )
            compare(step, "reward", row["reward"], float(row["choice"] == correct_index))
            checked += 1
            if mismatches:
                return {
                    "status": "failed", "run": str(run_dir),
                    "requested_trials": trials, "checked_trials": checked,
                    "mismatches": mismatches,
                }
        done += count
        if done == next_eval:
            evaluation_boundaries.append(done)
            # Evaluation constructs a fresh world and role generator. It
            # consumes no randomness from either of the training generators.
            next_eval += config["eval_every"]

    return {
        "status": "passed",
        "run": str(run_dir),
        "checked_trials": checked,
        "seed": seed,
        "rollouts_replayed": rollouts,
        "evaluation_boundaries_reached": evaluation_boundaries,
        "source_snapshot_hashes_match": True,
        "batch_and_run_config_match": True,
        "checks": [
            "step", "target_id", "candidate_ids", "sender",
            "correct_indices_inferred_from_logged_metadata", "selected_id", "reward",
        ],
        "correct_indices_separately_logged": "correct_index" in rows[0],
        "logged_prefix_sha256": log_digest.hexdigest(),
        "numpy_version": np.__version__,
        "numpy_matches_recorded_runtime": np.__version__ == batch["runtime"]["numpy"],
        "scope": "environment metadata and roles; no training, policy replay, or logged pixel hashes",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, default=Path("pilot_round1/results/batch_001"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--out", type=Path, help="Optional audit-result JSON output")
    args = parser.parse_args()
    run_dir = args.run or args.batch / "emergent_seed101_H0"
    try:
        result = verify(args.batch, run_dir, args.trials)
    except (ValueError, KeyError, OSError, IndexError) as exc:
        result = {"status": "failed", "run": str(run_dir), "error": str(exc)}
    output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(output)
    if args.out:
        args.out.write_text(output + "\n")
    return {"passed": 0, "insufficient_log": 2, "failed": 1}[result["status"]]


if __name__ == "__main__":
    raise SystemExit(main())
