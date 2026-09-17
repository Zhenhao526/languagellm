"""Independent replay audit for the compositional signaling study."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from . import design, runner


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(x):
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    if isinstance(x, (float, int, np.number)): return math.isfinite(float(x))
    return True


def verify_snapshot(prepared):
    prepared = Path(prepared); plan = json.loads((prepared / "plan.json").read_text()); cfg = json.loads((prepared / "prepared.json").read_text()); fr = json.loads((prepared / "freeze.json").read_text())
    assert sha(prepared / "plan.json") == fr["plan_sha256"]
    assert sha(prepared / "prepared.json") == fr["prepared_sha256"] == plan["prepared_sha256"]
    assert plan["config"] == cfg
    for rel, digest in plan["sources"].items(): assert sha(prepared / "source_snapshot" / rel) == digest, rel


def load_policy(path):
    with np.load(path) as z:
        return {"sender_logits_hidden": z["sender_logits_hidden"], "sender_logits_visible": z["sender_logits_visible"], "worker_logits": z["worker_logits"]}


def audit(prepared, execution):
    verify_snapshot(prepared); execution = Path(execution); progress = json.loads((execution / "progress.json").read_text())
    files = sorted(execution.glob("seed_*/result.json")); assert progress["completed"] == progress["total"] == len(files)
    logs = checkpoints = 0; max_error = 0.0; rows = []
    for path in files:
        r = json.loads(path.read_text()); assert finite(r); assert len(r["trajectory"]) == r["updates"]
        log = path.parent / "training.jsonl"; assert log.is_file(); assert sha(log) == r["training_log_sha256"]; logs += sum(1 for _ in log.open())
        cp = path.parent / f"checkpoint_{r['updates']:04d}.npz"; assert cp.is_file(); checkpoints += 1
        p = load_policy(cp); form, pm, pv, ch, task = design.parse_condition(r["condition"])
        for evaluation, split in ((False, "training_support"), (True, "heldout")):
            fresh = runner.evaluate_all(p, r["seed"], form, pm, pv, task, ch, evaluation=evaluation)
            for worker in r["final"][split]["workers"]:
                for mode in ("natural", "closed", "permuted"):
                    a = r["final"][split]["workers"][str(worker)][mode]["team_return_mean"]
                    b = fresh["workers"][str(worker)][mode]["team_return_mean"]
                    if a is None or b is None: assert a is None and b is None
                    else: max_error = max(max_error, abs(float(a) - float(b)))
                if "recombined" in r["final"][split]["workers"][str(worker)]:
                    a = r["final"][split]["workers"][str(worker)]["recombined"]["team_return_mean"]
                    b = fresh["workers"][str(worker)]["recombined"]["team_return_mean"]
                    if a is None or b is None: assert a is None and b is None
                    else: max_error = max(max_error, abs(float(a) - float(b)))
            assert r["final"][split]["codebook"] == fresh["codebook"]
        rows.append({"seed": r["seed"], "condition": r["condition"], "heldout": r["final"]["heldout"]})
    # Pair live/silent streams for every form/task/partner/visibility cell.
    by = {(r["seed"], r["condition"]): r for r in (json.loads(p.read_text()) for p in files)}; pair_checks = 0
    for (seed, cond), r in by.items():
        other = cond.replace("_live_", "_silent_") if "_live_" in cond else None
        if not other or (seed, other) not in by: continue
        s = by[(seed, other)]
        for a, b in zip(r["trajectory"], s["trajectory"]):
            for k in ("world_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"):
                assert a[k] == b[k], (seed, cond, other, k, a["update"])
            pair_checks += 1
    assert max_error < 1e-12
    return {"schema": "compositional_signaling_audit_v1", "status": "passed", "runs": len(files), "training_log_rows": logs, "final_checkpoints": checkpoints, "max_abs_replay_error": max_error, "live_silent_trajectory_pairs": pair_checks, "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prepared", required=True); ap.add_argument("--execution", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); result = audit(Path(args.prepared), Path(args.execution)); Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n"); print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
