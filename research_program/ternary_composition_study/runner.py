"""Train and audit the ternary message-capacity extension."""
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


def finite(value):
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, list): return all(finite(v) for v in value)
    if isinstance(value, (float, int, np.number)): return bool(np.isfinite(float(value)))
    return True


def source_hashes():
    rels = ("__init__.py", "design.py", "policy.py", "environment.py", "runner.py", "aggregate.py", "audit.py", "plan.md", "tests/test_game.py")
    return {str((HERE / rel).relative_to(ROOT)): sha(HERE / rel) for rel in rels}


def prepare(out):
    out = Path(out).resolve(); design.require(not out.exists(), "refuse overwrite")
    cfg = design.prepare(); cfg.update({"checkpoints": list(design.CHECKPOINTS), "runs": len(design.SEEDS) * len(design.CONDITIONS), "evaluation_episodes_per_worker": 4096})
    sources = source_hashes(); out.mkdir(parents=True)
    for rel in sources:
        target = out / "source_snapshot" / rel; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, target)
    (out / "prepared.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    plan = {"schema": "ternary_composition_study_v1", "prepared_sha256": sha(out / "prepared.json"), "sources": sources, "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}, indent=2) + "\n")
    verify(out); return {"status": "prepared", "out": str(out), "plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}


def verify(out):
    out = Path(out); plan = json.loads((out / "plan.json").read_text()); cfg = json.loads((out / "prepared.json").read_text()); freeze = json.loads((out / "freeze.json").read_text())
    design.require(sha(out / "plan.json") == freeze["plan_sha256"] and sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "freeze hash mismatch")
    design.require(plan["config"] == cfg and plan["sources"] == source_hashes(), "source or config changed")
    for rel, digest in plan["sources"].items(): design.require(sha(out / "source_snapshot" / rel) == digest, "source snapshot changed " + rel)
    return plan, cfg


def save_checkpoint(path, params, update, form):
    np.savez_compressed(path, update=np.array(update, dtype=np.int64), sender_logits_hidden=params["sender_logits_hidden"], sender_logits_visible=params["sender_logits_visible"], worker_logits=params["worker_logits"], form=np.array(form))
    return sha(path)


def future_returns(rewards): return np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1)


def center(values, keys):
    values = np.asarray(values, dtype=np.float64); keys = np.asarray(keys); out = np.zeros_like(values)
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        if len(idx) > 1: out[idx] = values[idx] - (values[idx].sum() - values[idx]) / (len(idx) - 1)
    return out


def entropy_grad(prob):
    lp = np.log(np.maximum(prob, 1e-300)); ent = -(prob * lp).sum(axis=-1, keepdims=True); return -prob * (lp + ent)


def gradient(params, episode, trajectory, form, task, protocol, channel, beta):
    g = {key: np.zeros_like(value) for key, value in params.items()}
    future = future_returns(trajectory["rewards"])
    if channel == "live":
        context = design.goal_index(episode["goal"])
        arrivals = design.message_arrival_times(protocol, form)
        for slot, arrival in enumerate(arrivals):
            sp = trajectory["message_probs"][slot]; one = np.zeros_like(sp); one[np.arange(len(context)), trajectory["selected_messages"][:, slot]] = 1
            advantage = center(future[:, arrival + 1:].sum(axis=1), context)
            d = -(advantage[:, None] * (one - sp)) / len(context) - beta * entropy_grad(sp) / len(context)
            np.add.at(g["sender_logits_hidden"][:, slot, :], context, d)
    worker = episode["partner_id"].astype(np.int64)
    for t in range(design.ACTION_START, design.HORIZON):
        ap = trajectory["action_probs"][t]; state = trajectory["state"][t]; local = trajectory["local"][t]; inventory = trajectory["inventory"][t]
        keys = worker * 100000000 + state * 1000000 + t * 10000 + local * 100 + inventory
        advantage = center(future[:, t], keys)
        one = np.zeros_like(ap); one[np.arange(len(worker)), trajectory["actions"][:, t]] = 1
        d = -(advantage[:, None] * (one - ap)) / len(worker) - beta * entropy_grad(ap) / len(worker)
        np.add.at(g["worker_logits"], (worker, state, np.full(len(worker), t), local, inventory), d)
    return g


def sequence(params, form, goal):
    context = int(goal[0]) * design.OBJECT_TYPES + int(goal[1])
    return np.array([policy.softmax(params["sender_logits_hidden"][context, slot]).argmax() for slot in range(design.message_length(form))], dtype=np.int64)


def sender_codebook(params, form):
    return [sequence(params, form, (a, b)).tolist() for a in range(design.OBJECT_TYPES) for b in range(design.OBJECT_TYPES)]


def recombination_override(params, episode, form, task):
    if form != "tri3": return None
    goals = np.asarray([[a, b] for a in range(design.OBJECT_TYPES) for b in range(design.OBJECT_TYPES)], dtype=np.int8)
    out = np.zeros((len(episode["goal"]), design.MESSAGE_SLOTS), dtype=np.int64)
    for i, goal in enumerate(episode["goal"]):
        target = design.target_bits(np.asarray(goal, dtype=np.int8)[None, :], task)[0]
        donor0 = goals[np.flatnonzero(design.target_bits(goals, task)[:, 0] == target[0])[0]]
        donor1 = goals[np.flatnonzero(design.target_bits(goals, task)[:, 1] == target[1])[0]]
        out[i, 0] = sequence(params, form, donor0)[0]; out[i, 1] = sequence(params, form, donor1)[1]
    return out


def metric(trajectory):
    returns = trajectory["team_return"][trajectory["active"]]
    return {"episodes": int(len(returns)), "team_return_mean": float(returns.mean()) if len(returns) else None, "team_return_sd": float(returns.std()) if len(returns) else None, "positive_episode_rate": float((returns > 0).mean()) if len(returns) else None}


def evaluate_one(params, episode, form, task, protocol, channel, worker, mode, override=None):
    message_mode = "natural" if mode == "silent" else mode
    tr = environment.rollout(params, episode, form, task, protocol, channel, message_mode=message_mode, sample=False, partner_filter=worker, message_override=override)
    return metric(tr)


def evaluate(params, seed, form, task, protocol, channel):
    episode = design.episode_stream(seed + design.EVAL_SEED_OFFSET, design.WORKERS * 4096, form, task, evaluation=True)
    override = recombination_override(params, episode, form, task)
    per_worker = {}
    for worker in range(design.WORKERS):
        modes = {"natural": evaluate_one(params, episode, form, task, protocol, channel, worker, "natural")}
        if channel == "live":
            modes["silent"] = evaluate_one(params, episode, form, task, protocol, "silent", worker, "silent")
            modes["permuted"] = evaluate_one(params, episode, form, task, protocol, "live", worker, "permuted")
            if override is not None: modes["recombined"] = evaluate_one(params, episode, form, task, protocol, "live", worker, "natural", override)
        else:
            modes["silent"] = modes["natural"]
            modes["permuted"] = modes["natural"]
        per_worker[str(worker)] = modes
    codebook = {"sender_sequence_by_goal": sender_codebook(params, form), "recombination_defined": form == "tri3"}
    live_values = [per_worker[str(w)]["natural"]["team_return_mean"] for w in range(design.WORKERS)]
    out = {"split": "heldout", "form": form, "task": task, "protocol": protocol, "channel": channel, "per_worker": per_worker, "codebook": codebook, "natural_mean_all_workers": float(np.mean(live_values))}
    if form == "tri3" and channel == "live":
        vals = [per_worker[str(w)]["recombined"]["team_return_mean"] - per_worker[str(w)]["natural"]["team_return_mean"] for w in range(design.WORKERS)]
        out["recombined_minus_natural_mean"] = float(np.mean(vals))
    else:
        out["recombined_minus_natural_mean"] = None
    out["composable"] = bool(form == "tri3" and channel == "live" and out["natural_mean_all_workers"] >= 0.60 and abs(out["recombined_minus_natural_mean"]) <= 0.02)
    return out


def train_one(seed, condition, execution, updates=None):
    form, task, protocol, channel = design.parse_condition(condition); updates = design.UPDATES if updates is None else int(updates)
    run = Path(execution) / f"seed_{seed}_{condition}"; run.mkdir(parents=True, exist_ok=False)
    params = policy.make_policy(seed, form); initial_hash = policy.combined_parameter_hash(params)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints: save_checkpoint(run / "checkpoint_0000.npz", params, 0, form)
    rows = []; start = time.perf_counter()
    for update in range(1, updates + 1):
        episode = design.episode_stream(seed, design.BATCH_SIZE, form, task, update=update); trajectory = environment.rollout(params, episode, form, task, protocol, channel, sample=True); beta = design.entropy_coefficient(update); g = gradient(params, episode, trajectory, form, task, protocol, channel, beta)
        norm = float(np.sqrt(sum(float((v * v).sum()) for v in g.values()))); scale = min(1.0, 5.0 / max(norm, 1e-12))
        for key in ("sender_logits_hidden", "worker_logits"):
            params[key] -= design.LEARNING_RATE * scale * g[key]
        rows.append({"update": update, "seed": seed, "condition": condition, "form": form, "task": task, "protocol": protocol, "channel": channel, "world_sha256": design.array_sha(episode["site_type"]), "goal_sha256": design.array_sha(episode["goal"]), "partner_sha256": design.array_sha(episode["partner_id"]), "message_uniform_sha256": design.array_sha(episode["message_uniforms"]), "action_uniform_sha256": design.array_sha(episode["action_uniforms"]), "return_mean": float(trajectory["team_return"].mean()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.combined_parameter_hash(params), "elapsed_seconds": time.perf_counter() - start})
        if update in checkpoints: rows[-1]["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", params, update, form)
    log = run / "training.jsonl"; log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result = {"seed": seed, "condition": condition, "form": form, "task": task, "protocol": protocol, "channel": channel, "updates": updates, "initial_parameter_sha256": initial_hash, "final": evaluate(params, seed, form, task, protocol, channel), "checkpoints": checkpoints, "trajectory": rows, "training_log_sha256": sha(log), "final_parameter_sha256": policy.combined_parameter_hash(params), "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"), "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz")}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def run_grid(prepared, out, updates=None, seeds=None, conditions=None):
    _, cfg = verify(prepared); out = Path(out); execution = out / "execution"; execution.mkdir(parents=True)
    seeds = tuple(design.SEEDS if seeds is None else seeds); conditions = tuple(design.CONDITIONS if conditions is None else conditions); design.require(set(seeds) <= set(design.SEEDS) and set(conditions) <= set(design.CONDITIONS), "invalid subset")
    rows = []; total = len(seeds) * len(conditions)
    for seed in seeds:
        for condition in conditions:
            rows.append(train_one(seed, condition, execution, updates)); (execution / "progress.json").write_text(json.dumps({"completed": len(rows), "total": total, "seeds": list(seeds), "conditions": list(conditions)}, indent=2))
    (execution / "results.json").write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2) + "\n"); return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--out", required=True)
    a = sub.add_parser("execute"); a.add_argument("--out", required=True); a.add_argument("--prepared", required=True); a.add_argument("--updates", type=int, default=None); a.add_argument("--seeds"); a.add_argument("--conditions")
    args = parser.parse_args()
    if args.cmd == "prepare": print(json.dumps(prepare(args.out), ensure_ascii=False))
    else:
        seeds = None if args.seeds is None else tuple(int(value) for value in args.seeds.split(",") if value); conditions = None if args.conditions is None else tuple(value for value in args.conditions.split(",") if value); print(json.dumps({"status": "completed", "runs": len(run_grid(args.prepared, args.out, args.updates, seeds, conditions))}, ensure_ascii=False))
