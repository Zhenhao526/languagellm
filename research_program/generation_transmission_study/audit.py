"""Independent replay and freeze audit for generation-transmission runs."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from . import design, runner


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(x):
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    if isinstance(x, (float, int, np.number)): return bool(math.isfinite(float(x)))
    return True


def verify_snapshot(prepared):
    prepared = Path(prepared); plan = json.loads((prepared / "plan.json").read_text()); cfg = json.loads((prepared / "prepared.json").read_text()); fr = json.loads((prepared / "freeze.json").read_text())
    assert sha(prepared / "plan.json") == fr["plan_sha256"]
    assert sha(prepared / "prepared.json") == fr["prepared_sha256"] == plan["prepared_sha256"]
    assert plan["config"] == cfg
    for rel, digest in plan["sources"].items(): assert sha(prepared / "source_snapshot" / rel) == digest, rel


def _close(a, b, tol=1e-12):
    if a is None or b is None: assert a is None and b is None; return 0.0
    d = abs(float(a) - float(b)); assert d <= tol, (a, b, d); return d


def _check_metric_dict(a, b):
    err = 0.0
    for key in ("episodes", "team_return_mean", "team_return_sd", "positive_episode_rate", "oracle_team_return_mean"):
        if key in a or key in b: err = max(err, _close(a.get(key), b.get(key)))
    return err


def audit(prepared, execution):
    verify_snapshot(prepared); execution = Path(execution)
    progress = json.loads((execution / "progress.json").read_text())
    payload = json.loads((execution / "results.json").read_text())
    parents = {int(x["seed"]): x for x in payload["parents"]}; children = {(int(x["seed"]), x["condition"]): x for x in payload["children"]}
    assert progress["completed"] == progress["total"] == len(parents) + len(children)
    expected_seeds = tuple(int(x) for x in progress.get("seeds", sorted(parents)))
    expected_conditions = tuple(progress.get("conditions", sorted({c for _, c in children})))
    assert set(parents) == set(expected_seeds)
    assert set(children) == {(seed, condition) for seed in expected_seeds for condition in expected_conditions}
    logs = 0; checkpoints = 0; max_error = 0.0; replay_rows = []
    # Parent endpoint replay and integrity.
    for seed in expected_seeds:
        r = parents[seed]; assert finite(r); assert r["condition"] == "parent_live"; assert len(r["trajectory"]) == r["updates"]
        run = execution / f"parent_seed_{seed}"; log = run / "training.jsonl"; assert log.is_file(); assert sha(log) == r["training_log_sha256"]
        logs += sum(1 for _ in log.open()); cp = run / f"checkpoint_{r['updates']:04d}.npz"; assert cp.is_file(); checkpoints += 1
        p = runner.load_policy(cp); assert runner.policy.parameter_hash(p) == r["final_parameter_sha256"]
        for evaluation, split in ((False, "training_support"), (True, "heldout")):
            fresh = runner.evaluate_parent(p, seed, evaluation=evaluation)
            for worker in range(design.WORKERS):
                for mode in ("natural", "closed", "permuted"):
                    max_error = max(max_error, _check_metric_dict(r["final"][split]["workers"][str(worker)][mode], fresh["workers"][str(worker)][mode]))
        replay_rows.append({"seed": seed, "kind": "parent", "max_error": max_error})
    # Child endpoint, curve and frozen-component replay.
    for (seed, condition), r in sorted(children.items()):
        assert finite(r); role, channel = design.parse_condition(condition); assert r["role"] == role and r["channel"] == channel
        assert len(r["trajectory"]) == r["updates"]
        run = execution / f"seed_{seed}_{condition}"; log = run / "training.jsonl"; assert log.is_file(); assert sha(log) == r["training_log_sha256"]
        logs += sum(1 for _ in log.open()); cp = run / f"checkpoint_{r['updates']:04d}.npz"; assert cp.is_file(); checkpoints += 1
        p = runner.load_policy(cp); parent = runner.load_policy(execution / f"parent_seed_{seed}" / f"checkpoint_{parents[seed]['updates']:04d}.npz")
        assert runner.policy.parameter_hash(p) == r["final_parameter_sha256"]
        assert r["parent_parameter_sha256"] == runner.policy.parameter_hash(parent)
        assert r["frozen_initial_sha256"] == runner.frozen_hash(parent, role)
        assert r["frozen_final_sha256"] == runner.frozen_hash(p, role)
        fresh = runner.evaluate_child(p, seed, role, evaluation=True)
        got = r["final"]["heldout"]
        for mode in ("natural", "closed", "permuted", "scrambled"):
            max_error = max(max_error, _check_metric_dict(got["new_agent"][mode], fresh["new_agent"][mode]))
        max_error = max(max_error, _close(got["population_natural_mean"], fresh["population_natural_mean"]))
        max_error = max(max_error, _close(got["codebook"]["new_agent_semantic_success"], fresh["codebook"]["new_agent_semantic_success"]))
        for update in r["checkpoints"]:
            cp_i = run / f"checkpoint_{update:04d}.npz"; assert cp_i.is_file()
            pc = runner.load_policy(cp_i); ev = runner.evaluate_child(pc, seed, role, evaluation=True)
            curve = next(x for x in r["learning_curve"] if x["update"] == update)
            for key, evkey in (("new_agent_natural", "natural"), ("new_agent_closed", "closed"), ("new_agent_permuted", "permuted")):
                max_error = max(max_error, _close(curve[key], ev["new_agent"][evkey]["team_return_mean"]))
            max_error = max(max_error, _close(curve["codebook_semantic_success"], ev["codebook"]["new_agent_semantic_success"]))
        replay_rows.append({"seed": seed, "condition": condition, "max_error": max_error})
    # Verify every child stream is paired across its three channel arms for a
    # given replaced role, and that all arms use the same parent.
    paired_rows = 0
    for role in design.ROLES:
        present_channels = [ch for ch in design.CHANNELS if (expected_seeds[0], f"{role}_{ch}") in children]
        if len(present_channels) < 2:
            continue
        for seed in expected_seeds:
            if not all((seed, f"{role}_{ch}") in children for ch in present_channels):
                continue
            arms = [children[(seed, f"{role}_{ch}")] for ch in present_channels]
            for rows in zip(*(x["trajectory"] for x in arms)):
                for key in ("world_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256", "scramble_uniform_sha256", "active_count"):
                    assert all(row[key] == rows[0][key] for row in rows[1:]), (seed, role, key, rows[0]["update"])
                paired_rows += 1
    assert max_error < 1e-12
    return {"schema": "generation_transmission_audit_v1", "status": "passed", "parent_runs": len(parents), "child_runs": len(children),
            "training_log_rows": logs, "final_checkpoints": checkpoints, "max_abs_replay_error": max_error,
            "paired_child_trajectory_rows": paired_rows, "rows": replay_rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--prepared", required=True); ap.add_argument("--execution", required=True); ap.add_argument("--out", required=True)
    args = ap.parse_args(); result = audit(Path(args.prepared), Path(args.execution)); Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n"); print(json.dumps({k: v for k, v in result.items() if k != "rows"}, ensure_ascii=False))
