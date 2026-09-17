"""Independent replay, pairing and freeze audit for the chain."""
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
    files = sorted(execution.glob("seed_*/**/result.json"))
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
        cp = path.parent / f"checkpoint_{result['updates']:04d}.npz"
        assert cp.is_file() and sha(cp) == result["final_checkpoint_sha256"]
        checkpoints += 1
        params = runner.load_checkpoint(cp)
        fresh = runner.evaluate(params, result["seed"], result["generation"], result["representation"])
        stored = result["final"]
        for mode in ("natural", "closed", "permuted", "recombined"):
            for key in ("team_return_mean", "team_return_sd", "positive_episode_rate"):
                x = stored["modes"][mode][key]; y = fresh["modes"][mode][key]
                if x is None or y is None:
                    assert x is None and y is None
                else:
                    max_error = max(max_error, abs(float(x) - float(y)))
        assert stored["codebook"] == fresh["codebook"]
        by[(int(result["seed"]), int(result["generation"]), result["lineage"], result["g1_channel"], result["channel"])] = result
    # Check deterministic channel pairing at each stage.
    pair_rows = 0
    present_seeds = sorted({key[0] for key in by})
    present_lineages = sorted({key[2] for key in by})
    for seed in present_seeds:
        for lineage in present_lineages:
            for generation in design.GENERATIONS:
                for g1 in design.CHANNELS:
                    if (seed, generation, lineage, g1, "live") not in by or (seed, generation, lineage, g1, "silent") not in by:
                        continue
                    live = by[(seed, generation, lineage, g1, "live")]
                    silent = by[(seed, generation, lineage, g1, "silent")]
                    for left, right in zip(live["trajectory"], silent["trajectory"]):
                        for key in ("world_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                            assert left[key] == right[key], (seed, lineage, generation, g1, key, left["update"])
                        pair_rows += 1
    # Check frozen parameters and the stage-to-stage parent link.
    freeze_rows = 0
    for seed in present_seeds:
        for lineage in present_lineages:
            parent = runner.load_checkpoint(runner.parent_path(cfg["parent_root"], seed))
            for g1 in design.CHANNELS:
                if (seed, 1, lineage, g1, g1) not in by:
                    continue
                s1 = by[(seed, 1, lineage, g1, g1)]
                p1 = runner.load_checkpoint(s1["final_checkpoint"])
                role1 = runner.stage_role(lineage, 1)
                if role1 == "worker":
                    assert np.array_equal(p1["sender_logits_hidden"], parent["sender_logits_hidden"])
                    assert np.array_equal(p1["worker_logits"][1:], parent["worker_logits"][1:])
                else:
                    assert np.array_equal(p1["worker_logits"], parent["worker_logits"])
                for g2 in design.CHANNELS:
                    if (seed, 2, lineage, g1, g2) not in by:
                        continue
                    s2 = by[(seed, 2, lineage, g1, g2)]
                    assert s2["parent_checkpoint_sha256"] == s1["final_checkpoint_sha256"]
                    p2 = runner.load_checkpoint(s2["final_checkpoint"])
                    role2 = runner.stage_role(lineage, 2)
                    if role2 == "worker":
                        assert np.array_equal(p2["sender_logits_hidden"], p1["sender_logits_hidden"])
                        assert np.array_equal(p2["worker_logits"][1:], p1["worker_logits"][1:])
                    else:
                        assert np.array_equal(p2["worker_logits"], p1["worker_logits"])
                    freeze_rows += 1
    assert max_error < 1e-12
    return {"schema": "multi_generation_transmission_audit_v1", "status": "passed", "runs": len(files), "training_log_rows": logs, "final_checkpoints": checkpoints, "paired_channel_trajectory_rows": pair_rows, "stage2_parent_links": freeze_rows, "max_abs_replay_error": max_error}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--prepared", required=True); parser.add_argument("--execution", required=True); parser.add_argument("--out", required=True); args = parser.parse_args()
    result = audit(Path(args.prepared), Path(args.execution))
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
