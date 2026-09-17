"""Independent replay and paired-stream audit for ternary signaling."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design, runner


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    if isinstance(value, (float, int, np.number)): return math.isfinite(float(value))
    return True


def audit(prepared, execution):
    runner.verify(prepared); execution = Path(execution); progress = json.loads((execution / "progress.json").read_text()); files = sorted(execution.glob("seed_*/result.json")); assert progress["completed"] == progress["total"] == len(files)
    by = {}; logs = checkpoints = 0; max_error = 0.0
    for path in files:
        result = json.loads(path.read_text()); assert finite(result) and len(result["trajectory"]) == result["updates"]
        log = path.parent / "training.jsonl"; assert sha(log) == result["training_log_sha256"]; logs += sum(1 for _ in log.open())
        checkpoint = path.parent / f"checkpoint_{result['updates']:04d}.npz"; assert checkpoint.is_file() and sha(checkpoint) == result["final_checkpoint_sha256"]; checkpoints += 1
        with np.load(checkpoint, allow_pickle=False) as data:
            params = {key: np.asarray(data[key]).copy() for key in ("sender_logits_hidden", "sender_logits_visible", "worker_logits")}
        fresh = runner.evaluate(params, result["seed"], result["form"], result["task"], result["protocol"], result["channel"]); stored = result["final"]
        for worker, modes in stored["per_worker"].items():
            for mode, values in modes.items():
                for key in ("team_return_mean", "team_return_sd", "positive_episode_rate"):
                    x = values[key]; y = fresh["per_worker"][worker][mode][key]
                    if x is None or y is None: assert x is None and y is None
                    else: max_error = max(max_error, abs(float(x) - float(y)))
        assert stored["codebook"] == fresh["codebook"] and stored["composable"] == fresh["composable"]
        key = (int(result["seed"]), result["form"], result["task"], result["protocol"], result["channel"]); assert key not in by; by[key] = result
    pair_rows = 0
    hash_keys = ("world_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256")
    present_seeds = sorted({key[0] for key in by})
    for seed in present_seeds:
        for form in design.FORMS:
            for task in design.TASKS:
                for protocol in design.PROTOCOLS:
                    if (seed, form, task, protocol, "live") not in by or (seed, form, task, protocol, "silent") not in by:
                        continue
                    live = by[(seed, form, task, protocol, "live")]; silent = by[(seed, form, task, protocol, "silent")]
                    for left, right in zip(live["trajectory"], silent["trajectory"]):
                        for key in hash_keys: assert left[key] == right[key], (seed, form, task, protocol, key, left["update"])
                        pair_rows += 1
    assert max_error < 1e-12
    return {"schema": "ternary_composition_audit_v1", "status": "passed", "runs": len(files), "training_log_rows": logs, "final_checkpoints": checkpoints, "paired_channel_trajectory_rows": pair_rows, "max_abs_replay_error": max_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepared", required=True); parser.add_argument("--execution", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); result = audit(Path(args.prepared), Path(args.execution)); Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(result, ensure_ascii=False))
