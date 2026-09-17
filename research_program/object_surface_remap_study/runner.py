"""Train, evaluate, freeze and replay the object-surface remapping study."""
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def source_hashes():
    rels = ("__init__.py", "README.md", "design.py", "policy.py", "environment.py", "runner.py", "aggregate.py", "audit.py", "plan.md", "tests/test_game.py")
    return {str((HERE / rel).relative_to(ROOT)): sha(HERE / rel) for rel in rels}


def prepare(out):
    out = Path(out).resolve()
    design.require(not out.exists(), "refuse overwrite")
    cfg = design.prepare()
    sources = source_hashes()
    out.mkdir(parents=True)
    for rel in sources:
        target = out / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (out / "prepared.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    plan = {"schema": "object_surface_remap_study_v1", "prepared_sha256": sha(out / "prepared.json"), "sources": sources, "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}, indent=2) + "\n")
    verify(out)
    return {"status": "prepared", "out": str(out), "plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}


def verify(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    cfg = json.loads((out / "prepared.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    design.require(sha(out / "plan.json") == freeze["plan_sha256"], "plan hash mismatch")
    design.require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "prepared hash mismatch")
    design.require(plan["config"] == cfg and plan["sources"] == source_hashes(), "source or config changed")
    for rel, digest in plan["sources"].items():
        design.require(sha(out / "source_snapshot" / rel) == digest, "source snapshot changed " + rel)
    return plan, cfg


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


def gradient(params, episode, trajectory, form, representation, channel, *, child=False):
    grad = {key: np.zeros_like(value) for key, value in params.items()}
    future = future_returns(trajectory["rewards"])
    n = len(episode["goal"])
    if not child and channel == "live":
        context = policy.sender_context(episode["goal"])
        for slot, arrival in enumerate(design.message_arrival_times(form)):
            sp = trajectory["message_probs"][slot]
            one = np.zeros_like(sp)
            one[np.arange(n), trajectory["selected_messages"][:, slot]] = 1.0
            advantage = center(future[:, arrival + 1 :].sum(axis=1), context)
            beta = design.entropy_coefficient(episode["update"], child=child)
            d = -(advantage[:, None] * (one - sp)) / n - beta * entropy_grad(sp) / n
            np.add.at(grad["sender_logits"][:, slot, :], context, d)
    worker = episode["partner_id"].astype(np.int64)
    for t in range(design.ACTION_START, design.HORIZON):
        ap = trajectory["action_probs"][t]
        state = trajectory["state"][t]
        scene = trajectory["scene"][t]
        inventory = trajectory["inventory"][t]
        keys = worker * 100000000 + state * 1000000 + t * 10000 + scene * 100 + inventory
        advantage = center(future[:, t], keys)
        one = np.zeros_like(ap)
        one[np.arange(n), trajectory["actions"][:, t]] = 1.0
        beta = design.entropy_coefficient(episode["update"], child=child)
        d = -(advantage[:, None] * (one - ap)) / n - beta * entropy_grad(ap) / n
        np.add.at(grad["worker_logits"], (worker, state, np.full(n, t), scene, inventory), d)
    return grad


def _metric(trajectory):
    values = trajectory["team_return"][trajectory["active"]]
    return {"episodes": int(len(values)), "team_return_mean": float(values.mean()) if len(values) else None, "team_return_sd": float(values.std()) if len(values) else None, "positive_episode_rate": float((values > 0).mean()) if len(values) else None, "oracle_team_return_mean": float(environment.oracle_team_return())}


def sender_codebook(params, form):
    return [environment.deterministic_sequence(params, form, goal).tolist() for goal in design.GOAL_PAIRS]


def pairwise_min_hamming(codebook):
    return int(min(sum(a != b for a, b in zip(left, right)) for i, left in enumerate(codebook) for right in codebook[i + 1:]))


def _eval_episode(seed, task, goal_kind, heldout, mapping):
    return design.balanced_eval_stream(seed + design.EVAL_SEED_OFFSET, design.WORKERS * 4096, goal_kind, task, heldout, mapping)


def evaluate(params, seed, form, task, representation, channel, heldout, mapping):
    codebook = sender_codebook(params, form)
    result = {"sender_codebook": codebook, "pairwise_min_hamming": pairwise_min_hamming(codebook), "mapping": mapping, "channel": channel}
    for goal_kind in ("all", "seen", "heldout"):
        ep = _eval_episode(seed, task, goal_kind, heldout, mapping)
        natural = environment.rollout(params, ep, form, task, representation, channel, sample=False, message_mode="natural", partner_filter=design.TARGET_WORKER)
        modes = {"natural": _metric(natural)}
        if channel == "live":
            permuted = environment.rollout(params, ep, form, task, representation, channel, sample=False, message_mode="permuted", partner_filter=design.TARGET_WORKER)
            modes["permuted"] = _metric(permuted)
            if form == "dual2":
                raw = environment.rollout(params, ep, form, task, representation, channel, sample=False, message_override=environment.recombination_override(params, ep, form, task, basis="raw"), partner_filter=design.TARGET_WORKER)
                modes["raw_recombined"] = _metric(raw)
        else:
            modes["permuted"] = dict(modes["natural"], reused_natural=True)
        result[goal_kind] = modes
    return result


def parent_path(execution, seed, form, task, updates=None):
    updates = design.PARENT_UPDATES if updates is None else int(updates)
    return Path(execution) / "parents" / f"seed_{seed}_{form}_{task}" / f"checkpoint_{updates:04d}.npz"


def save_checkpoint(path, params, update, form, representation):
    np.savez_compressed(path, update=np.array(update, dtype=np.int64), sender_logits=params["sender_logits"], worker_logits=params["worker_logits"], form=np.array(form), representation=np.array(representation))
    return sha(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return {"sender_logits": np.asarray(data["sender_logits"]).copy(), "worker_logits": np.asarray(data["worker_logits"]).copy()}


def _stream_hashes(ep):
    return {"site_semantic_sha256": design.array_sha(ep["site_semantic"]), "site_surface_sha256": design.array_sha(ep["site_surface"]), "goal_sha256": design.array_sha(ep["goal"]), "partner_sha256": design.array_sha(ep["partner_id"]), "message_uniform_sha256": design.array_sha(ep["message_uniforms"]), "action_uniform_sha256": design.array_sha(ep["action_uniforms"])}


def train_parent(seed, form, task, execution, updates=None):
    updates = design.PARENT_UPDATES if updates is None else int(updates)
    run = Path(execution) / "parents" / f"seed_{seed}_{form}_{task}"
    run.mkdir(parents=True, exist_ok=False)
    params = policy.make_policy(seed, form, "joint_history")
    initial_hash = policy.parameter_hash(params)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", params, 0, form, "joint_history")
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, task=task, update=update, support="full", mapping="identity")
        tr = environment.rollout(params, ep, form, task, "joint_history", "live", sample=True)
        grad = gradient(params, ep, tr, form, "joint_history", "live", child=False)
        norm = float(np.sqrt(sum(float((x * x).sum()) for x in grad.values())))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        for key in params:
            params[key] -= design.LEARNING_RATE * scale * grad[key]
        row = {"update": update, "seed": seed, "form": form, "task": task, "mapping": "identity", **_stream_hashes(ep), "return_mean": float(tr["team_return"].mean()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.parameter_hash(params), "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            row["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", params, update, form, "joint_history")
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    heldout = design.heldout_goal(seed)
    result = {"kind": "parent", "seed": seed, "form": form, "task": task, "mapping": "identity", "representation": "joint_history", "updates": updates, "initial_parameter_sha256": initial_hash, "final_parameter_sha256": policy.parameter_hash(params), "final": evaluate(params, seed, form, task, "joint_history", "live", heldout, "identity"), "heldout_goal": heldout, "checkpoints": checkpoints, "training_log_sha256": sha(log), "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"), "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz")}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def train_child(seed, condition, execution, parent_checkpoint, updates=None):
    representation, form, task, mapping, support, channel = design.parse_child_condition(condition)
    updates = design.CHILD_UPDATES if updates is None else int(updates)
    run = Path(execution) / "children" / f"seed_{seed}_{condition}"
    run.mkdir(parents=True, exist_ok=False)
    parent = load_checkpoint(parent_checkpoint)
    params = policy.make_policy(seed + 191000, form, representation, sender_logits=parent["sender_logits"])
    parent_hash = policy.parameter_hash(parent)
    initial_hash = policy.parameter_hash(params)
    heldout = design.heldout_goal(seed)
    initial_eval = evaluate(params, seed, form, task, representation, channel, heldout, mapping)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", params, 0, form, representation)
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, task=task, update=update, support=support, heldout=heldout, mapping=mapping)
        if support == "leave_one_out":
            design.require(not np.any(design.goal_index(ep["goal"]) == heldout), f"heldout leakage {condition}/{update}")
        tr = environment.rollout(params, ep, form, task, representation, channel, sample=True, partner_filter=design.TARGET_WORKER)
        grad = gradient(params, ep, tr, form, representation, channel, child=True)
        norm = float(np.sqrt(sum(float((x * x).sum()) for x in grad.values())))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        params["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * grad["worker_logits"][design.TARGET_WORKER]
        active = tr["active"]
        row = {"update": update, "seed": seed, "condition": condition, "representation": representation, "form": form, "task": task, "mapping": mapping, "support": support, "channel": channel, "heldout_goal": heldout, **_stream_hashes(ep), "return_mean": float(tr["team_return"][active].mean()) if active.any() else 0.0, "active_count": int(active.sum()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.parameter_hash(params), "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            row["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", params, update, form, representation)
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result = {"kind": "child", "seed": seed, "condition": condition, "representation": representation, "form": form, "task": task, "mapping": mapping, "support": support, "channel": channel, "heldout_goal": heldout, "parent_checkpoint": str(parent_checkpoint), "parent_parameter_sha256": parent_hash, "initial_parameter_sha256": initial_hash, "updates": updates, "initial": initial_eval, "final_parameter_sha256": policy.parameter_hash(params), "final": evaluate(params, seed, form, task, representation, channel, heldout, mapping), "checkpoints": checkpoints, "training_log_sha256": sha(log), "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"), "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz")}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def run_grid(prepared, out, updates=None, seeds=None, conditions=None):
    verify(prepared)
    out = Path(out).resolve()
    execution = out / "execution"
    (execution / "parents").mkdir(parents=True, exist_ok=True)
    (execution / "children").mkdir(parents=True, exist_ok=True)
    seeds = tuple(design.SEEDS if seeds is None else seeds)
    conditions = tuple(design.CHILD_CONDITIONS if conditions is None else conditions)
    design.require(set(seeds) <= set(design.SEEDS), "invalid seed subset")
    design.require(set(conditions) <= set(design.CHILD_CONDITIONS), "invalid condition subset")
    parent_conditions = sorted({(design.parse_child_condition(c)[1], design.parse_child_condition(c)[2]) for c in conditions})
    parent_results = []
    for seed in seeds:
        for form, task in parent_conditions:
            parent_results.append(train_parent(seed, form, task, execution, updates))
    child_results = []
    for seed in seeds:
        for condition in conditions:
            _, form, task, _, _, _ = design.parse_child_condition(condition)
            child_results.append(train_child(seed, condition, execution, parent_path(execution, seed, form, task, updates), updates))
            (execution / "progress.json").write_text(json.dumps({"completed_children": len(child_results), "total_children": len(seeds) * len(conditions), "seeds": list(seeds), "conditions": list(conditions)}, indent=2))
    payload = {"parents": parent_results, "children": child_results}
    (execution / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare"); prep.add_argument("--out", required=True)
    exe = sub.add_parser("execute"); exe.add_argument("--out", required=True); exe.add_argument("--prepared", required=True); exe.add_argument("--updates", type=int); exe.add_argument("--seeds"); exe.add_argument("--conditions")
    args = parser.parse_args()
    if args.cmd == "prepare":
        print(json.dumps(prepare(args.out), ensure_ascii=False))
    else:
        seeds = None if args.seeds is None else tuple(int(x) for x in args.seeds.split(",") if x)
        conditions = None if args.conditions is None else tuple(x for x in args.conditions.split(",") if x)
        payload = run_grid(args.prepared, args.out, args.updates, seeds, conditions)
        print(json.dumps({"status": "completed", "parents": len(payload["parents"]), "children": len(payload["children"])}, ensure_ascii=False))
