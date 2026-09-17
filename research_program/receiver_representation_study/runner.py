"""Train and audit joint-history versus slot-local receivers."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from . import design, environment, policy
from research_program.action_dependent_signaling_study import design as base_design
from research_program.action_dependent_signaling_study import environment as base_environment
from research_program.action_dependent_signaling_study import policy as base_policy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def source_hashes():
    rels = ("__init__.py", "design.py", "policy.py", "environment.py", "runner.py", "aggregate.py", "audit.py", "plan.md", "tests/test_game.py")
    return {str((HERE / rel).relative_to(ROOT)): sha(HERE / rel) for rel in rels}


def dependency_hashes():
    rels = (
        "research_program/action_dependent_signaling_study/design.py",
        "research_program/action_dependent_signaling_study/environment.py",
        "research_program/action_dependent_signaling_study/policy.py",
        "research_program/heldout_composition_study/design.py",
    )
    return {rel: sha(ROOT / rel) for rel in rels}


def parent_path(parent_root, seed):
    return Path(parent_root) / f"seed_{seed}_{design.PARENT_CONDITION}" / f"checkpoint_{design.CHECKPOINTS[-1]:04d}.npz"


def parent_hashes(parent_root):
    out = {}
    for seed in design.SEEDS:
        path = parent_path(parent_root, seed)
        design.require(path.is_file(), f"missing parent checkpoint {path}")
        out[str(seed)] = sha(path)
    return out


def parent_composable_from_analysis(path):
    payload = json.loads(Path(path).read_text())
    rows = [row for row in payload["rows"] if row["role"] == "worker" and row["channel"] == "live"]
    by_seed = {}
    for row in rows:
        seed = int(row["seed"])
        status = bool(row["parent_composable"])
        if seed in by_seed and by_seed[seed] != status:
            raise ValueError(f"inconsistent parent status for {seed}")
        by_seed[seed] = status
    if set(by_seed) != set(design.SEEDS):
        raise ValueError("parent analysis does not cover all formal seeds")
    return by_seed


def prepare(out, parent_root, parent_analysis):
    out = Path(out).resolve()
    design.require(not out.exists(), "refuse overwrite")
    parent_root = Path(parent_root).resolve()
    cfg = design.prepare(parent_root, parent_hashes(parent_root), parent_composable_from_analysis(parent_analysis))
    cfg.update({"runs": len(design.SEEDS) * len(design.CONDITIONS), "parent_runs": len(design.SEEDS), "child_runs": len(design.SEEDS) * len(design.CONDITIONS), "evaluation_episodes_per_goal": 1024})
    src = source_hashes()
    deps = dependency_hashes()
    out.mkdir(parents=True)
    for rel in src:
        target = out / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    for rel in deps:
        target = out / "dependency_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    shutil.copy2(parent_analysis, out / "parent_analysis.json")
    (out / "prepared.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    plan = {"schema": "receiver_representation_study_v1", "prepared_sha256": sha(out / "prepared.json"), "parent_analysis_sha256": sha(out / "parent_analysis.json"), "sources": src, "dependencies": deps, "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json"), "parent_analysis_sha256": sha(out / "parent_analysis.json")}, indent=2) + "\n")
    verify(out)
    return {"status": "prepared", "out": str(out), "plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}


def verify(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    cfg = json.loads((out / "prepared.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    design.require(sha(out / "plan.json") == freeze["plan_sha256"] and sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "freeze hash mismatch")
    design.require(sha(out / "parent_analysis.json") == freeze["parent_analysis_sha256"] == plan["parent_analysis_sha256"], "parent analysis changed")
    design.require(plan["config"] == cfg and plan["sources"] == source_hashes() and plan["dependencies"] == dependency_hashes(), "source or config changed")
    for rel, digest in plan["sources"].items():
        design.require(sha(out / "source_snapshot" / rel) == digest, "source snapshot changed " + rel)
    for rel, digest in plan["dependencies"].items():
        design.require(sha(out / "dependency_snapshot" / rel) == digest, "dependency snapshot changed " + rel)
    design.require(parent_hashes(cfg["parent_root"]) == cfg["parent_checkpoint_sha256"], "parent checkpoint changed")
    return plan, cfg


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in ("sender_logits_hidden", "sender_logits_visible", "worker_logits")}


def future_returns(rewards):
    return np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1)


def center(values, keys):
    values = np.asarray(values, dtype=np.float64)
    keys = np.asarray(keys)
    out = np.zeros_like(values)
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        if len(idx) > 1:
            out[idx] = values[idx] - (values[idx].sum() - values[idx]) / (len(idx) - 1)
    return out


def entropy_grad(prob):
    lp = np.log(np.maximum(prob, 1e-300))
    ent = -(prob * lp).sum(axis=-1, keepdims=True)
    return -prob * (lp + ent)


def rollout(params, episode, representation, channel, *, sample, message_mode="natural"):
    if representation == "joint_history":
        return base_environment.rollout(params, episode, design.FORM, design.PARTNER_MODE, design.VISIBILITY, design.TASK, channel, design.PROTOCOL, message_mode=message_mode, sample=sample, partner_filter=design.TARGET_WORKER)
    return environment.rollout_slot_local(params, episode, channel, message_mode=message_mode, sample=sample, partner_filter=design.TARGET_WORKER)


def gradient(params, episode, trajectory, representation, beta):
    grad = {key: np.zeros_like(value) for key, value in params.items()}
    idx = np.flatnonzero(trajectory["active"])
    if len(idx) == 0:
        return grad
    future = future_returns(trajectory["rewards"])
    for t in range(design.ACTION_START, design.HORIZON):
        ap = trajectory["action_probs"][t][idx]
        state = trajectory["state"][t][idx]
        local = trajectory["local"][t][idx]
        inventory = trajectory["inventory"][t][idx]
        keys = state * 1000 + t * 100 + local * 10 + inventory
        advantage = center(future[idx, t], keys)
        one_hot = np.zeros_like(ap)
        one_hot[np.arange(len(idx)), trajectory["actions"][idx, t]] = 1.0
        d = -(advantage[:, None] * (one_hot - ap)) / len(idx) - beta * entropy_grad(ap) / len(idx)
        np.add.at(grad["worker_logits"], (np.zeros(len(idx), dtype=np.int64), state, np.full(len(idx), t), local, inventory), d)
    return grad


def metrics(params, episode, representation, channel, message_mode="natural"):
    trajectory = rollout(params, episode, representation, channel, sample=False, message_mode=message_mode)
    mask = trajectory["active"]
    returns = trajectory["team_return"][mask]
    return {"episodes": int(mask.sum()), "team_return_mean": float(returns.mean()) if len(returns) else None, "team_return_sd": float(returns.std()) if len(returns) else None, "positive_episode_rate": float((returns > 0).mean()) if len(returns) else None}


def evaluate(params, seed, heldout_goal_index, representation, channel):
    split = {}
    for goal_kind in ("all", "seen", "heldout"):
        episode = design.balanced_eval_stream(seed + design.EVAL_SEED_OFFSET, 4096, goal_kind, heldout_goal_index)
        split[goal_kind] = {"natural": metrics(params, episode, representation, channel, "natural"), "permuted": metrics(params, episode, representation, channel, "permuted")}
    return split


def train_one(seed, condition, execution, parent_checkpoint, updates=None):
    representation, support, channel = design.parse_condition(condition)
    updates = design.UPDATES if updates is None else int(updates)
    run = Path(execution) / f"seed_{seed}_{condition}"
    run.mkdir(parents=True, exist_ok=False)
    parent = load_checkpoint(parent_checkpoint)
    params = policy.clone(parent)
    replacement = base_policy.make_policy(seed + 191000, design.FORM)
    params["worker_logits"][design.TARGET_WORKER] = replacement["worker_logits"][0]
    parent_parameter_hash = base_policy.combined_parameter_hash(parent)
    initial_parameter_hash = base_policy.combined_parameter_hash(params)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    rows = []
    start = time.perf_counter()
    if 0 in checkpoints:
        np.savez_compressed(run / "checkpoint_0000.npz", update=np.array(0, dtype=np.int64), sender_logits_hidden=params["sender_logits_hidden"], sender_logits_visible=params["sender_logits_visible"], worker_logits=params["worker_logits"])
    for update in range(1, updates + 1):
        heldout_goal_index = design.heldout_goal(seed)
        episode = design.episode_stream(seed, design.BATCH_SIZE, update=update, support=support, heldout_goal_index=heldout_goal_index)
        trajectory = rollout(params, episode, representation, channel, sample=True)
        beta = design.entropy_coefficient(update)
        grad = gradient(params, episode, trajectory, representation, beta)
        norm = float(np.sqrt(sum(float((value * value).sum()) for value in grad.values())))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        params["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * grad["worker_logits"][design.TARGET_WORKER]
        row = {"update": update, "seed": seed, "condition": condition, "representation": representation, "support": support, "channel": channel, "heldout_goal": heldout_goal_index, "world_sha256": design.array_sha(episode["site_type"]), "goal_sha256": design.array_sha(episode["goal"]), "partner_sha256": design.array_sha(episode["partner_id"]), "message_uniform_sha256": design.array_sha(episode["message_uniforms"]), "action_uniform_sha256": design.array_sha(episode["action_uniforms"]), "return_mean": float(trajectory["team_return"][trajectory["active"]].mean()) if trajectory["active"].any() else 0.0, "active_count": int(trajectory["active"].sum()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": base_policy.combined_parameter_hash(params), "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            path = run / f"checkpoint_{update:04d}.npz"
            np.savez_compressed(path, update=np.array(update, dtype=np.int64), sender_logits_hidden=params["sender_logits_hidden"], sender_logits_visible=params["sender_logits_visible"], worker_logits=params["worker_logits"])
            row["checkpoint_sha256"] = sha(path)
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    heldout_goal_index = design.heldout_goal(seed)
    result = {"seed": seed, "condition": condition, "representation": representation, "support": support, "channel": channel, "heldout_goal": heldout_goal_index, "parent_checkpoint": str(parent_checkpoint), "parent_checkpoint_sha256": sha(parent_checkpoint), "parent_parameter_sha256": parent_parameter_hash, "initial_parameter_sha256": initial_parameter_hash, "updates": updates, "final": evaluate(params, seed, heldout_goal_index, representation, channel), "checkpoints": checkpoints, "trajectory": rows, "training_log_sha256": sha(log), "final_parameter_sha256": base_policy.combined_parameter_hash(params)}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def run_grid(prepared, out, updates=None, seeds=None, conditions=None):
    _, cfg = verify(prepared)
    out = Path(out)
    execution = out / "execution"
    execution.mkdir(parents=True)
    seeds = tuple(design.SEEDS if seeds is None else seeds)
    conditions = tuple(design.CONDITIONS if conditions is None else conditions)
    design.require(set(seeds) <= set(design.SEEDS) and set(conditions) <= set(design.CONDITIONS), "invalid subset")
    results = []
    for seed in seeds:
        for condition in conditions:
            results.append(train_one(seed, condition, execution, parent_path(cfg["parent_root"], seed), updates))
            (execution / "progress.json").write_text(json.dumps({"completed": len(results), "total": len(seeds) * len(conditions), "seeds": list(seeds), "conditions": list(conditions)}, indent=2))
    (execution / "results.json").write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--out", required=True); a.add_argument("--parent-root", required=True); a.add_argument("--parent-analysis", required=True)
    a = sub.add_parser("execute"); a.add_argument("--out", required=True); a.add_argument("--prepared", required=True); a.add_argument("--updates", type=int, default=None); a.add_argument("--seeds"); a.add_argument("--conditions")
    args = parser.parse_args()
    if args.cmd == "prepare":
        print(json.dumps(prepare(args.out, args.parent_root, args.parent_analysis), ensure_ascii=False))
    else:
        seeds = None if args.seeds is None else tuple(int(value) for value in args.seeds.split(",") if value)
        conditions = None if args.conditions is None else tuple(value for value in args.conditions.split(",") if value)
        print(json.dumps({"status": "completed", "runs": len(run_grid(args.prepared, args.out, args.updates, seeds, conditions))}, ensure_ascii=False))
