"""Independent replay audit for community-merge executions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, environment, policy, runner


def _close(a, b, tol=1e-12):
    return abs(float(a) - float(b)) <= tol


def _read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def _check_stream(row, ep):
    expected = runner._stream_hashes(ep)
    for key, value in expected.items():
        if row[key] != value:
            raise AssertionError(f"stream hash mismatch {key}")


def _parent_replay(execution, result):
    seed = int(result["seed"]); population = result["population"]; community = int(result["community"]); updates = int(result["updates"])
    run = Path(execution) / "parents" / f"seed_{seed}_{population}_community_{community}"
    rows = _read_rows(run / "training.jsonl")
    if len(rows) != updates:
        raise AssertionError("parent log length mismatch")
    communities, _ = runner.load_checkpoint(run / "checkpoint_0000.npz")
    params = communities[0]
    max_error = 0.0
    checkpoints = 0
    for update, row in enumerate(rows, start=1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, support="full", role="alternating", update=update)
        ep["partner_id"] = np.full(len(ep["goal"]), community * design.PARTNERS_PER_COMMUNITY, dtype=np.int8)
        ep["community_id"] = np.full(len(ep["goal"]), community, dtype=np.int8)
        _check_stream(row, ep)
        tr = environment.rollout([params], None, ep, population, "hidden", "alternating", sample=True)
        grad = runner.parent_gradient(params, ep, tr)
        norm, scale = runner.update_params(params, grad)
        max_error = max(max_error, abs(float(row["return_mean"]) - float(tr["team_return"].mean())), abs(float(row["gradient_norm"]) - norm), abs(float(row["gradient_clip_scale"]) - scale))
        if row["parameter_sha256"] != policy.parameter_hash(params):
            raise AssertionError("parent parameter hash mismatch")
        if update in result["checkpoints"]:
            path = run / f"checkpoint_{update:04d}.npz"
            if row.get("checkpoint_sha256") != runner.sha(path):
                raise AssertionError("parent checkpoint hash mismatch")
            checkpoints += 1
    if result["final_parameter_sha256"] != policy.parameter_hash(params):
        raise AssertionError("parent final hash mismatch")
    if result["final_checkpoint_sha256"] != runner.sha(run / f"checkpoint_{updates:04d}.npz"):
        raise AssertionError("parent terminal checkpoint mismatch")
    return len(rows), checkpoints, max_error


def _child_replay(execution, result):
    seed = int(result["seed"]); condition = result["condition"]
    population, visibility, adaptation, support, role = design.parse_child_condition(condition)
    updates = int(result["updates"])
    run = Path(execution) / "children" / f"seed_{seed}_{condition}"
    rows = _read_rows(run / "training.jsonl")
    if len(rows) != updates:
        raise AssertionError("child log length mismatch")
    communities, fresh = runner.load_checkpoint(run / "checkpoint_0000.npz")
    if fresh is None:
        raise AssertionError("child checkpoint has no fresh policy")
    max_error = 0.0
    checkpoints = 0
    for update, row in enumerate(rows, start=1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, support=support, role=role, update=update)
        goals = np.asarray(ep["goal_meaning"], dtype=np.int64)
        if support == "heldout_combo" and np.any(goals == design.heldout_combo(seed)):
            raise AssertionError("heldout combo leakage")
        if support == "heldout_value" and np.any(goals // design.VALUES == design.heldout_value(seed)):
            raise AssertionError("heldout value leakage")
        _check_stream(row, ep)
        tr = environment.rollout(communities, fresh, ep, population, visibility, role, sample=True)
        fg, cgs = runner.child_gradients(communities, fresh, ep, tr, visibility, adaptation)
        norms = [float(np.sqrt(sum(float((v * v).sum()) for v in fg.values())))] + [float(np.sqrt(sum(float((v * v).sum()) for v in g.values()))) for g in cgs]
        total_norm = float(np.sqrt(sum(x * x for x in norms)))
        scale = min(1.0, 5.0 / max(total_norm, 1e-12))
        for key in fresh:
            fresh[key] -= design.LEARNING_RATE * scale * fg[key]
        if adaptation == "coadapt":
            for params, grad in zip(communities, cgs):
                for key in params:
                    params[key] -= design.LEARNING_RATE * scale * grad[key]
        max_error = max(max_error, abs(float(row["return_mean"]) - float(tr["team_return"].mean())), abs(float(row["gradient_norm"]) - total_norm), abs(float(row["gradient_clip_scale"]) - scale))
        if row["parameter_sha256"] != policy.parameter_hash(fresh):
            raise AssertionError("child fresh parameter hash mismatch")
        if row["community_parameter_sha256"] != [policy.parameter_hash(p) for p in communities]:
            raise AssertionError("child community parameter hash mismatch")
        if update in result["checkpoints"]:
            path = run / f"checkpoint_{update:04d}.npz"
            if row.get("checkpoint_sha256") != runner.sha(path):
                raise AssertionError("child checkpoint hash mismatch")
            checkpoints += 1
    if result["final_parameter_sha256"] != policy.parameter_hash(fresh):
        raise AssertionError("child final hash mismatch")
    if result["final_community_parameter_sha256"] != [policy.parameter_hash(p) for p in communities]:
        raise AssertionError("child final community hash mismatch")
    if result["final_checkpoint_sha256"] != runner.sha(run / f"checkpoint_{updates:04d}.npz"):
        raise AssertionError("child terminal checkpoint mismatch")
    return len(rows), checkpoints, max_error


def _pair_rows(rows_a, rows_b, fields):
    if rows_a is None or rows_b is None:
        return 0
    a = {(int(r["seed"]), int(r["update"])): r for r in rows_a}
    b = {(int(r["seed"]), int(r["update"])): r for r in rows_b}
    keys = sorted(set(a) & set(b))
    for key in keys:
        for field in fields:
            if a[key].get(field) != b[key].get(field):
                raise AssertionError(f"pair mismatch {field} at {key}")
    return len(keys)


def audit(prepared, execution):
    prepared = Path(prepared); execution = Path(execution)
    _, cfg = runner.verify(prepared)
    payload = json.loads((execution / "results.json").read_text())
    parent_logs = {}
    child_logs = {}
    log_rows_parent = 0; log_rows_child = 0; checkpoints_parent = 0; checkpoints_child = 0; max_error = 0.0
    for result in payload["parents"]:
        key = (int(result["seed"]), result["population"], int(result["community"]))
        rows, cp, err = _parent_replay(execution, result); log_rows_parent += rows; checkpoints_parent += cp; max_error = max(max_error, err)
        parent_logs[key] = _read_rows(Path(execution) / "parents" / f"seed_{result['seed']}_{result['population']}_community_{result['community']}" / "training.jsonl")
    for result in payload["children"]:
        key = (int(result["seed"]), result["condition"])
        rows, cp, err = _child_replay(execution, result); log_rows_child += rows; checkpoints_child += cp; max_error = max(max_error, err)
        child_logs[key] = _read_rows(Path(execution) / "children" / f"seed_{result['seed']}_{result['condition']}" / "training.jsonl")

    def child_rows(seed, population, visibility, adaptation, support, role):
        condition = f"{population}_{visibility}_{adaptation}_{support}_{role}"
        return child_logs.get((int(seed), condition))

    visibility_pairs = adaptation_pairs = population_pairs = support_pairs = role_pairs = 0
    for seed in design.SEEDS:
        for population in design.POPULATIONS:
            for adaptation in design.ADAPTATIONS:
                for support in design.SUPPORTS:
                    for role in design.ROLES:
                        visibility_pairs += _pair_rows(child_rows(seed, population, "hidden", adaptation, support, role), child_rows(seed, population, "visible", adaptation, support, role), ["scene_sha256", "goal_sha256", "partner_sha256", "role_sha256", "message_uniform_sha256", "action_uniform_sha256"])
                for visibility in design.VISIBILITIES:
                    for support in design.SUPPORTS:
                        for role in design.ROLES:
                            adaptation_pairs += _pair_rows(child_rows(seed, population, visibility, "fresh_only", support, role), child_rows(seed, population, visibility, "coadapt", support, role), ["scene_sha256", "goal_sha256", "partner_sha256", "role_sha256", "message_uniform_sha256", "action_uniform_sha256"])
        for visibility in design.VISIBILITIES:
            for adaptation in design.ADAPTATIONS:
                for support in design.SUPPORTS:
                    for role in design.ROLES:
                        population_pairs += _pair_rows(child_rows(seed, "aligned", visibility, adaptation, support, role), child_rows(seed, "conflict", visibility, adaptation, support, role), ["scene_sha256", "goal_sha256", "partner_sha256", "role_sha256", "message_uniform_sha256", "action_uniform_sha256"])
        for population in design.POPULATIONS:
            for visibility in design.VISIBILITIES:
                for adaptation in design.ADAPTATIONS:
                    for role in design.ROLES:
                        full = child_rows(seed, population, visibility, adaptation, "full", role)
                        combo = child_rows(seed, population, visibility, adaptation, "heldout_combo", role)
                        value = child_rows(seed, population, visibility, adaptation, "heldout_value", role)
                        support_pairs += _pair_rows(full, combo, ["scene_sha256", "partner_sha256", "role_sha256", "message_uniform_sha256", "action_uniform_sha256"])
                        support_pairs += _pair_rows(full, value, ["scene_sha256", "partner_sha256", "role_sha256", "message_uniform_sha256", "action_uniform_sha256"])
        for population in design.POPULATIONS:
            for visibility in design.VISIBILITIES:
                for adaptation in design.ADAPTATIONS:
                    for support in design.SUPPORTS:
                        alternating = child_rows(seed, population, visibility, adaptation, support, "alternating")
                        sender_only = child_rows(seed, population, visibility, adaptation, support, "sender_only")
                        role_pairs += _pair_rows(alternating, sender_only, ["scene_sha256", "goal_sha256", "partner_sha256", "message_uniform_sha256", "action_uniform_sha256"])

    expected_parent = len(design.SEEDS) * len(design.POPULATIONS) * design.COMMUNITIES
    expected_child = len(design.SEEDS) * len(design.CHILD_CONDITIONS)
    complete = len(payload["parents"]) == expected_parent and len(payload["children"]) == expected_child
    status = "passed" if complete and max_error <= 1e-12 else ("partial" if max_error <= 1e-12 else "failed")
    return {
        "schema": "community_merge_audit_v1",
        "status": status,
        "prepared_schema": cfg["schema"],
        "parent_runs": len(payload["parents"]),
        "child_runs": len(payload["children"]),
        "expected_parent_runs": expected_parent,
        "expected_child_runs": expected_child,
        "parent_training_log_rows": log_rows_parent,
        "child_training_log_rows": log_rows_child,
        "parent_checkpoints": checkpoints_parent,
        "child_checkpoints": checkpoints_child,
        "paired_visibility_rows": visibility_pairs,
        "paired_adaptation_rows": adaptation_pairs,
        "paired_population_rows": population_pairs,
        "paired_support_rows": support_pairs,
        "paired_role_rows": role_pairs,
        "max_abs_replay_error": float(max_error),
        "raw_execution_tree": str(execution),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--execution", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = audit(args.prepared, args.execution)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))
