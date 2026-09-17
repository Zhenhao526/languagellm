"""Train, evaluate, freeze and replay the multi-partner sender-alignment study."""
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
    plan = {"schema": "partner_exposure_study_v1", "prepared_sha256": sha(out / "prepared.json"), "sources": sources, "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
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


def gradient(params, episode, trajectory, visibility, *, train_sender=True, train_workers=True):
    """REINFORCE gradient with explicit sender visibility and worker exposure."""
    grad = {key: np.zeros_like(value) for key, value in params.items()}
    future = future_returns(trajectory["rewards"])
    n = len(episode["goal"])
    worker = np.asarray(episode["partner_id"], dtype=np.int64)
    context = design.goal_index(episode["goal"]).astype(np.int64)
    beta = design.entropy_coefficient(episode["update"])
    if train_sender:
        sender_keys = context if visibility == "hidden" else worker * len(design.GOAL_PAIRS) + context
        for slot, arrival in enumerate(design.ARRIVAL_TIMES):
            sp = trajectory["message_probs"][slot]
            one = np.zeros_like(sp)
            one[np.arange(n), trajectory["selected_messages"][:, slot]] = 1.0
            advantage = center(future[:, arrival + 1 :].sum(axis=1), sender_keys)
            d = -(advantage[:, None] * (one - sp)) / n - beta * entropy_grad(sp) / n
            if visibility == "hidden":
                np.add.at(grad["sender_logits_hidden"][:, slot, :], context, d)
            else:
                np.add.at(grad["sender_logits_visible"][:, :, slot, :], (worker, context), d)
    if train_workers:
        for t in range(design.ACTION_START, design.HORIZON):
            ap = trajectory["action_probs"][t]
            state = trajectory["state"][t]
            scene = trajectory["scene"][t]
            inventory = trajectory["inventory"][t]
            keys = worker * 100000000 + state * 1000000 + t * 10000 + scene * 100 + inventory
            advantage = center(future[:, t], keys)
            one = np.zeros_like(ap)
            one[np.arange(n), trajectory["actions"][:, t]] = 1.0
            d = -(advantage[:, None] * (one - ap)) / n - beta * entropy_grad(ap) / n
            np.add.at(grad["worker_logits"], (worker, state, np.full(n, t), scene, inventory), d)
    return grad


def update_params(params, grad, *, visibility, train_sender, train_workers):
    norm = float(np.sqrt(sum(float((value * value).sum()) for value in grad.values())))
    scale = min(1.0, 5.0 / max(norm, 1e-12))
    if train_sender:
        key = "sender_logits_hidden" if visibility == "hidden" else "sender_logits_visible"
        params[key] -= design.LEARNING_RATE * scale * grad[key]
    if train_workers:
        params["worker_logits"] -= design.LEARNING_RATE * scale * grad["worker_logits"]
    return norm, scale


def _metric(trajectory):
    values = trajectory["team_return"][trajectory["active"]]
    return {"episodes": int(len(values)), "team_return_mean": float(values.mean()) if len(values) else None, "team_return_sd": float(values.std()) if len(values) else None, "positive_episode_rate": float((values > 0).mean()) if len(values) else None, "oracle_team_return_mean": float(environment.oracle_team_return())}


def sender_codebook(params, visibility):
    if visibility == "hidden":
        return [policy.deterministic_sequence(params, goal, "hidden").tolist() for goal in design.GOAL_PAIRS]
    return [[policy.deterministic_sequence(params, goal, "visible", worker).tolist() for goal in design.GOAL_PAIRS] for worker in range(design.WORKERS)]


def pairwise_min_hamming(codebook, visibility):
    books = [codebook] if visibility == "hidden" else codebook
    mins = []
    for book in books:
        mins.append(min(sum(a != b for a, b in zip(left, right)) for i, left in enumerate(book) for right in book[i + 1:]))
    return int(min(mins))


def sender_consistency(params, visibility):
    if visibility == "hidden":
        return 1.0
    equal = 0
    total = 0
    for goal in design.GOAL_PAIRS:
        seqs = [tuple(policy.deterministic_sequence(params, goal, "visible", worker).tolist()) for worker in range(design.WORKERS)]
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                equal += int(seqs[i] == seqs[j])
                total += 1
    return float(equal / total) if total else None


def _eval_episode(seed, goal_kind, heldout, population_mode):
    return design.balanced_eval_stream(seed + design.EVAL_SEED_OFFSET, design.WORKERS * 4096, goal_kind, heldout, population_mode)


def evaluate(params, seed, population_mode, visibility, heldout):
    codebook = sender_codebook(params, visibility)
    result = {"sender_codebook": codebook, "sender_consistency": sender_consistency(params, visibility), "pairwise_min_hamming": pairwise_min_hamming(codebook, visibility), "visibility": visibility}
    for goal_kind in ("all", "seen", "heldout"):
        ep = _eval_episode(seed, goal_kind, heldout, population_mode)
        natural = environment.rollout(params, ep, visibility, channel="live", sample=False, message_mode="natural")
        permuted = environment.rollout(params, ep, visibility, channel="live", sample=False, message_mode="permuted")
        silent = environment.rollout(params, ep, visibility, channel="silent", sample=False, message_mode="natural")
        raw = environment.rollout(params, ep, visibility, channel="live", sample=False, message_override=environment.recombination_override(params, ep, visibility))
        modes = {"natural": _metric(natural), "permuted": _metric(permuted), "silent": _metric(silent), "raw_recombined": _metric(raw)}
        result[goal_kind] = modes
    all_ep = _eval_episode(seed, "all", heldout, population_mode)
    result["all_by_partner"] = [_metric(environment.rollout(params, all_ep, visibility, channel="live", sample=False, partner_filter=worker)) for worker in range(design.WORKERS)]
    return result


def parent_path(execution, seed, population_mode, updates=None):
    updates = design.PARENT_UPDATES if updates is None else int(updates)
    return Path(execution) / "parents" / f"seed_{seed}_{population_mode}" / f"checkpoint_{updates:04d}.npz"


def save_checkpoint(path, params, update):
    np.savez_compressed(path, update=np.array(update, dtype=np.int64), sender_logits_hidden=params["sender_logits_hidden"], sender_logits_visible=params["sender_logits_visible"], worker_logits=params["worker_logits"])
    return sha(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]).copy() for key in ("sender_logits_hidden", "sender_logits_visible", "worker_logits")}


def _stream_hashes(ep):
    return {"site_semantic_sha256": design.array_sha(ep["site_semantic"]), "site_surface_sha256": design.array_sha(ep["site_surface"]), "goal_sha256": design.array_sha(ep["goal"]), "partner_sha256": design.array_sha(ep["partner_id"]), "message_uniform_sha256": design.array_sha(ep["message_uniforms"]), "action_uniform_sha256": design.array_sha(ep["action_uniforms"])}


def train_parent(seed, population_mode, execution, updates=None):
    updates = design.PARENT_UPDATES if updates is None else int(updates)
    run = Path(execution) / "parents" / f"seed_{seed}_{population_mode}"
    run.mkdir(parents=True, exist_ok=False)
    params = policy.make_policy(seed)
    initial_hash = policy.parameter_hash(params)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", params, 0)
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, support="full", population_mode=population_mode, update=update)
        tr = environment.rollout(params, ep, "hidden", channel="live", sample=True)
        grad = gradient(params, ep, tr, "hidden", train_sender=True, train_workers=True)
        norm, scale = update_params(params, grad, visibility="hidden", train_sender=True, train_workers=True)
        row = {"update": update, "seed": seed, "population_mode": population_mode, "visibility": "hidden", **_stream_hashes(ep), "return_mean": float(tr["team_return"].mean()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.parameter_hash(params), "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            row["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", params, update)
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result = {"kind": "parent", "seed": seed, "population_mode": population_mode, "visibility": "hidden", "form": design.FORM, "task": design.TASK, "updates": updates, "initial_parameter_sha256": initial_hash, "final_parameter_sha256": policy.parameter_hash(params), "final": evaluate(params, seed, population_mode, "hidden", design.heldout_goal(seed)), "heldout_goal": design.heldout_goal(seed), "checkpoints": checkpoints, "training_log_sha256": sha(log), "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"), "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz")}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def child_initial_params(seed, parent):
    params = policy.make_policy(seed + 191000)
    params["worker_logits"] = parent["worker_logits"].copy()
    return params


def train_child(seed, condition, execution, parent_checkpoint, updates=None):
    population_mode, visibility, adaptation, support = design.parse_child_condition(condition)
    updates = design.CHILD_UPDATES if updates is None else int(updates)
    run = Path(execution) / "children" / f"seed_{seed}_{condition}"
    run.mkdir(parents=True, exist_ok=False)
    parent = load_checkpoint(parent_checkpoint)
    params = child_initial_params(seed, parent)
    parent_hash = policy.parameter_hash(parent)
    initial_hash = policy.parameter_hash(params)
    heldout = design.heldout_goal(seed)
    initial_eval = evaluate(params, seed, population_mode, visibility, heldout)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", params, 0)
    rows = []
    start = time.perf_counter()
    train_workers = adaptation == "coadapt"
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, support=support, heldout=heldout, population_mode=population_mode, update=update)
        if support == "leave_one_out":
            design.require(not np.any(design.goal_index(ep["goal"]) == heldout), f"heldout leakage {condition}/{update}")
        tr = environment.rollout(params, ep, visibility, channel="live", sample=True)
        grad = gradient(params, ep, tr, visibility, train_sender=True, train_workers=train_workers)
        norm, scale = update_params(params, grad, visibility=visibility, train_sender=True, train_workers=train_workers)
        row = {"update": update, "seed": seed, "condition": condition, "population_mode": population_mode, "visibility": visibility, "adaptation": adaptation, "support": support, **_stream_hashes(ep), "return_mean": float(tr["team_return"].mean()), "active_count": int(tr["active"].sum()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.parameter_hash(params), "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            row["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", params, update)
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result = {"kind": "child", "seed": seed, "condition": condition, "population_mode": population_mode, "visibility": visibility, "adaptation": adaptation, "support": support, "form": design.FORM, "task": design.TASK, "heldout_goal": heldout, "parent_checkpoint": str(parent_checkpoint), "parent_parameter_sha256": parent_hash, "initial_parameter_sha256": initial_hash, "updates": updates, "initial": initial_eval, "final_parameter_sha256": policy.parameter_hash(params), "final": evaluate(params, seed, population_mode, visibility, heldout), "checkpoints": checkpoints, "training_log_sha256": sha(log), "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"), "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz")}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def run_grid(prepared, out, updates=None, seeds=None, conditions=None):
    _, cfg = verify(prepared)
    out = Path(out).resolve()
    execution = out / "execution"
    (execution / "parents").mkdir(parents=True, exist_ok=True)
    (execution / "children").mkdir(parents=True, exist_ok=True)
    seeds = tuple(design.SEEDS if seeds is None else seeds)
    conditions = tuple(design.CHILD_CONDITIONS if conditions is None else conditions)
    design.require(set(seeds) <= set(design.SEEDS), "invalid seed subset")
    design.require(set(conditions) <= set(design.CHILD_CONDITIONS), "invalid condition subset")
    modes = sorted({design.parse_child_condition(c)[0] for c in conditions})
    parents = []
    for seed in seeds:
        for mode in modes:
            parents.append(train_parent(seed, mode, execution, updates))
    children = []
    for seed in seeds:
        for condition in conditions:
            mode, _, _, _ = design.parse_child_condition(condition)
            children.append(train_child(seed, condition, execution, parent_path(execution, seed, mode, updates), updates))
            (execution / "progress.json").write_text(json.dumps({"completed_children": len(children), "total_children": len(seeds) * len(conditions), "seeds": list(seeds), "conditions": list(conditions)}, indent=2))
    payload = {"parents": parents, "children": children}
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
