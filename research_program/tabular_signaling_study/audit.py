"""Independent audit for compact tabular signaling executions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design, environment, policy


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, (int, float, np.number)):
        return math.isfinite(float(value))
    return True


def load_policy(run_dir: Path, result: dict):
    p = policy.make_policy(result["seed"], result["task"], result["memory"])
    checkpoint = run_dir / f"checkpoint_{result['updates']:04d}.npz"
    with np.load(checkpoint) as z:
        p["sender_logits"] = z["sender_logits"]
        p["worker_logits"] = z["worker_logits"]
    return p, checkpoint


def verify_snapshot(prepared: Path) -> None:
    """Verify the frozen plan against its own source snapshot.

    Historical runs remain auditable after a later README or implementation
    change in the live working tree; the plan is the authority for the run.
    """
    plan = json.loads((prepared / "plan.json").read_text())
    config = json.loads((prepared / "prepared.json").read_text())
    freeze = json.loads((prepared / "freeze.json").read_text())
    require = lambda ok, msg: (_ for _ in ()).throw(AssertionError(msg)) if not ok else None
    require(sha(prepared / "plan.json") == freeze["plan_sha256"], "plan hash")
    require(sha(prepared / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "prepared hash")
    require(plan["config"] == config, "plan config")
    for rel, digest in plan["sources"].items():
        snapshot = prepared / "source_snapshot" / rel
        require(snapshot.is_file() and sha(snapshot) == digest, f"source snapshot: {rel}")


def replay_return(p, result, evaluation, mode):
    ep = design.episode_stream(result["seed"], result["scarcity"], result["task"],
                               4096, evaluation=evaluation)
    tr = environment.rollout(p, ep, result["information"], result["channel"],
                             message_mode=mode, sample=False)
    return float(tr["team_return"].mean())


def audit(prepared: Path, execution: Path) -> dict:
    verify_snapshot(prepared)
    progress = json.loads((execution / "progress.json").read_text())
    result_files = sorted(execution.glob("seed_*/result.json"))
    require = lambda ok, msg: (_ for _ in ()).throw(AssertionError(msg)) if not ok else None
    require(progress["completed"] == progress["total"] == len(result_files), "incomplete execution")
    rows = []
    log_count = 0
    checkpoint_count = 0
    max_abs_error = 0.0
    for path in result_files:
        result = json.loads(path.read_text())
        require(finite(result), f"nonfinite result: {path}")
        require(len(result["trajectory"]) == result["updates"], f"trajectory length: {path}")
        run_dir = path.parent
        log = run_dir / "training.jsonl"
        require(log.is_file(), f"missing log: {run_dir}")
        require(sha(log) == result["training_log_sha256"], f"log hash: {run_dir}")
        log_count += sum(1 for _ in log.open())
        checkpoint = run_dir / f"checkpoint_{result['updates']:04d}.npz"
        require(checkpoint.is_file(), f"missing final checkpoint: {run_dir}")
        checkpoint_count += 1
        p, _ = load_policy(run_dir, result)
        for evaluation, split in ((False, "training_support"), (True, "heldout")):
            for mode in ("natural", "closed", "permuted"):
                expected = result["final"][split][mode]["team_return_mean"]
                actual = replay_return(p, result, evaluation, mode)
                max_abs_error = max(max_abs_error, abs(actual - expected))
        h = result["final"]["heldout"]
        rows.append({
            "seed": result["seed"], "condition": result["condition"],
            "natural": h["natural"]["team_return_mean"],
            "closed": h["closed"]["team_return_mean"],
            "permuted": h["permuted"]["team_return_mean"],
            "natural_minus_closed": h["natural"]["team_return_mean"] - h["closed"]["team_return_mean"],
            "natural_minus_permuted": h["natural"]["team_return_mean"] - h["permuted"]["team_return_mean"],
            "message_type_mi": h["natural"]["message_type_mi"],
        })
    require(max_abs_error < 1e-12, f"replay mismatch {max_abs_error}")
    return {
        "schema": "tabular_signaling_audit_v1", "status": "passed",
        "prepared": str(prepared), "execution": str(execution),
        "runs": len(result_files), "training_log_rows": log_count,
        "final_checkpoints": checkpoint_count, "max_abs_replay_error": max_abs_error,
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    answer = audit(Path(args.prepared), Path(args.execution))
    Path(args.out).write_text(json.dumps(answer, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in answer.items() if k != "rows"}, ensure_ascii=False, sort_keys=True))
