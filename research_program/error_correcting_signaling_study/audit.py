"""Independent replay and paired-stream audit for noisy redundancy."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design, runner


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, (float, int, np.number)):
        return math.isfinite(float(value))
    return True


def metric_keys():
    return ("team_return_mean", "team_return_sd", "positive_episode_rate")


def compare_final(stored, fresh):
    max_error = 0.0
    for worker, modes in stored["per_worker"].items():
        for mode, values in modes.items():
            for key in metric_keys():
                left = values[key]
                right = fresh["per_worker"][worker][mode][key]
                if left is None or right is None:
                    assert left is None and right is None
                else:
                    max_error = max(max_error, abs(float(left) - float(right)))
    assert stored["sender_codebook"] == fresh["sender_codebook"]
    assert stored["pairwise_min_hamming"] == fresh["pairwise_min_hamming"]
    return max_error


def audit(prepared, parents_path, execution):
    runner.verify(prepared)
    parents_path = Path(parents_path)
    execution = Path(execution)
    parent_payload = json.loads((parents_path / "parents.json").read_text())
    parent_files = sorted((parents_path / "parents").glob("seed_*_*/result.json"))
    present_parent_keys = {(int(row["seed"]), row["form"]) for row in parent_payload["results"]}
    assert len(parent_files) == len(parent_payload["results"])
    parent_by = {}
    parent_checkpoints = parent_logs = 0
    max_error = 0.0
    for path in parent_files:
        result = json.loads(path.read_text())
        assert finite(result) and len(result["trajectory"]) == result["updates"]
        log = path.parent / "training.jsonl"
        checkpoint = path.parent / f"checkpoint_{result['updates']:04d}.npz"
        assert sha(log) == result["training_log_sha256"] and checkpoint.is_file() and sha(checkpoint) == result["final_checkpoint_sha256"]
        parent_logs += sum(1 for _ in log.open())
        parent_checkpoints += 1
        params = runner.load_checkpoint(checkpoint)
        fresh = runner.evaluate(params, result["seed"], result["form"], 0.0)
        max_error = max(max_error, compare_final(result["final"], fresh))
        key = (int(result["seed"]), result["form"])
        assert key not in parent_by
        parent_by[key] = result
    progress = json.loads((execution / "progress.json").read_text())
    child_files = sorted(execution.glob("seed_*/result.json"))
    assert progress["completed"] == progress["total"] == len(child_files)
    child_logs = child_checkpoints = 0
    child_by = {}
    for path in child_files:
        result = json.loads(path.read_text())
        assert finite(result) and len(result["trajectory"]) == result["updates"]
        log = path.parent / "training.jsonl"
        checkpoint = path.parent / f"checkpoint_{result['updates']:04d}.npz"
        assert sha(log) == result["training_log_sha256"] and checkpoint.is_file() and sha(checkpoint) == result["final_checkpoint_sha256"]
        child_logs += sum(1 for _ in log.open())
        child_checkpoints += 1
        params = runner.load_checkpoint(checkpoint)
        fresh = runner.evaluate(params, result["seed"], result["form"], result["noise_p"])
        max_error = max(max_error, compare_final(result["final"], fresh))
        key = (int(result["seed"]), result["form"], result["adaptation"], result["noise_key"])
        assert key not in child_by
        child_by[key] = result
        if result["adaptation"] != "scratch":
            parent = parent_by[(int(result["seed"]), result["form"])]
            assert result["parent_parameter_sha256"] == parent["final_parameter_sha256"]
            assert result["parent_sender_codebook"] == parent["final"]["sender_codebook"]
    pair_rows = 0
    hash_keys = ("world_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256", "noise_uniform_sha256")
    present_child_keys = list(child_by)
    present_seeds = sorted({key[0] for key in present_child_keys})
    present_forms = sorted({key[1] for key in present_child_keys}, key=design.FORMS.index)
    present_adaptations = sorted({key[2] for key in present_child_keys}, key=design.ADAPTATIONS.index)
    for seed in present_seeds:
        for form in present_forms:
            for adaptation in present_adaptations:
                arms = [child_by[(seed, form, adaptation, key)] for key in design.NOISE_KEYS if (seed, form, adaptation, key) in child_by]
                if len(arms) < 2:
                    continue
                for left, right in zip(arms[0]["trajectory"], arms[1]["trajectory"]):
                    for key in hash_keys:
                        assert left[key] == right[key]
                    pair_rows += 1
                for left, right in zip(arms[0]["trajectory"], arms[2]["trajectory"]):
                    for key in hash_keys:
                        assert left[key] == right[key]
                    pair_rows += 1
    assert max_error < 1e-12
    return {"schema": "error_correcting_signaling_audit_v1", "status": "passed", "parent_runs": len(parent_files), "runs": len(child_files), "parent_training_log_rows": parent_logs, "training_log_rows": child_logs, "parent_final_checkpoints": parent_checkpoints, "final_checkpoints": child_checkpoints, "paired_noise_trajectory_rows": pair_rows, "max_abs_replay_error": max_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepared", required=True); parser.add_argument("--parents", required=True); parser.add_argument("--execution", required=True); parser.add_argument("--out", required=True); args = parser.parse_args()
    result = audit(Path(args.prepared), Path(args.parents), Path(args.execution))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
