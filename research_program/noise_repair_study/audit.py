"""Independent replay, pairing and freeze audit for the noisy study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design, runner


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, (float, int, np.number)):
        return math.isfinite(float(value))
    return True


def audit(prepared, execution):
    _, cfg = runner.verify(prepared)
    execution = Path(execution)
    progress = json.loads((execution / "progress.json").read_text())
    files = sorted(execution.glob("seed_*/result.json"))
    assert progress["completed"] == progress["total"] == len(files)
    by = {}
    logs = checkpoints = 0
    max_error = 0.0
    for path in files:
        result = json.loads(path.read_text())
        assert finite(result) and len(result["trajectory"]) == result["updates"]
        parent = Path(result["parent_checkpoint"])
        assert parent.is_file() and sha(parent) == result["parent_checkpoint_sha256"]
        log = path.parent / "training.jsonl"
        assert sha(log) == result["training_log_sha256"]
        logs += sum(1 for _ in log.open())
        checkpoint = path.parent / f"checkpoint_{result['updates']:04d}.npz"
        assert checkpoint.is_file() and sha(checkpoint) == result["final_checkpoint_sha256"]
        checkpoints += 1
        params = runner.load_checkpoint(checkpoint)
        fresh = runner.evaluate(params, result["seed"], result["representation"], result["noise_p"])
        stored = result["final"]
        for population in ("new_worker", "incumbent_workers_mean"):
            for mode in stored[population]:
                if mode in ("per_worker",):
                    continue
                for key in ("team_return_mean", "team_return_sd", "positive_episode_rate"):
                    x = stored[population][mode][key]
                    y = fresh[population][mode][key]
                    if x is None or y is None:
                        assert x is None and y is None
                    else:
                        max_error = max(max_error, abs(float(x) - float(y)))
        assert stored["sender_sequences"] == fresh["sender_sequences"]
        # The per-worker metrics are deterministic too; compare the compact
        # scalar fields without duplicating every value in the audit receipt.
        for worker, modes in stored["per_worker"].items():
            for mode, values in modes.items():
                for key in ("team_return_mean", "team_return_sd", "positive_episode_rate"):
                    max_error = max(max_error, abs(float(values[key]) - float(fresh["per_worker"][worker][mode][key])))
        key = (int(result["seed"]), result["adaptation"], result["representation"], result["noise_key"])
        assert key not in by
        by[key] = result
    pair_rows = 0
    hash_keys = ("world_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256", "noise_uniform_sha256")
    present_seeds = sorted({key[0] for key in by})
    present_adaptations = sorted({key[1] for key in by})
    present_representations = sorted({key[2] for key in by})
    present_noise = sorted({key[3] for key in by})
    for seed in present_seeds:
        for adaptation in present_adaptations:
            for representation in present_representations:
                available = [noise_key for noise_key in present_noise if (seed, adaptation, representation, noise_key) in by]
                reference = {}
                for noise_key in available:
                    result = by[(seed, adaptation, representation, noise_key)]
                    for row in result["trajectory"]:
                        if row["update"] not in reference:
                            reference[row["update"]] = row
                        else:
                            for key in hash_keys:
                                assert row[key] == reference[row["update"]][key], (seed, adaptation, representation, noise_key, key, row["update"])
                        pair_rows += 1
    freeze_rows = 0
    for seed in present_seeds:
        parent = runner.load_checkpoint(runner.parent_path(cfg["parent_root"], seed))
        for adaptation in present_adaptations:
            for representation in present_representations:
                for noise_key in present_noise:
                    if (seed, adaptation, representation, noise_key) not in by:
                        continue
                    result = by[(seed, adaptation, representation, noise_key)]
                    params = runner.load_checkpoint(result["final_checkpoint"])
                    assert np.array_equal(params["worker_logits"][1:], parent["worker_logits"][1:])
                    assert np.array_equal(params["sender_logits_visible"], parent["sender_logits_visible"])
                    if adaptation == "worker_only":
                        assert np.array_equal(params["sender_logits_hidden"], parent["sender_logits_hidden"])
                    freeze_rows += 1
    assert max_error < 1e-12
    return {"schema": "noise_repair_audit_v1", "status": "passed", "runs": len(files), "training_log_rows": logs, "final_checkpoints": checkpoints, "paired_noise_trajectory_rows": pair_rows, "freeze_checks": freeze_rows, "max_abs_replay_error": max_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepared", required=True); parser.add_argument("--execution", required=True); parser.add_argument("--out", required=True); args = parser.parse_args()
    result = audit(Path(args.prepared), Path(args.execution))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
