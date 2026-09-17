"""Train, evaluate and freeze noisy protocol transmission and repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path

import numpy as np

from . import design, environment
from research_program.action_dependent_signaling_study import design as base_design
from research_program.action_dependent_signaling_study import policy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, (float, int, np.number)):
        return bool(np.isfinite(float(value)))
    return True


def source_hashes():
    rels = ("__init__.py", "design.py", "environment.py", "runner.py", "aggregate.py", "audit.py", "plan.md", "tests/test_game.py")
    return {str((HERE / rel).relative_to(ROOT)): sha(HERE / rel) for rel in rels}


def dependency_hashes():
    rels = (
        "research_program/action_dependent_signaling_study/design.py",
        "research_program/action_dependent_signaling_study/environment.py",
        "research_program/action_dependent_signaling_study/policy.py",
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
    statuses = {}
    for row in payload["rows"]:
        if row.get("role") != "worker" or row.get("channel") != "live":
            continue
        seed = int(row["seed"])
        if seed not in design.SEEDS:
            continue
        status = bool(row["parent_composable"])
        if seed in statuses and statuses[seed] != status:
            raise ValueError(f"inconsistent parent status {seed}")
        statuses[seed] = status
    design.require(set(statuses) == set(design.SEEDS), "parent analysis does not cover fixed seeds")
    design.require(all(statuses.values()), "noise study requires the pre-registered composable parent stratum")
    return statuses


def prepare(out, parent_root, parent_analysis):
    out = Path(out).resolve()
    design.require(not out.exists(), "refuse overwrite")
    parent_root = Path(parent_root).resolve()
    analysis_path = Path(parent_analysis).resolve()
    parent_composable = parent_composable_from_analysis(analysis_path)
    cfg = design.prepare(parent_root, parent_hashes(parent_root), sha(analysis_path))
    cfg.update({"parent_composable": {str(k): bool(v) for k, v in parent_composable.items()}, "runs": len(design.SEEDS) * len(design.CONDITIONS), "evaluation_episodes_total": 16384, "evaluation_episodes_per_worker_expected": 4096})
    sources = source_hashes()
    deps = dependency_hashes()
    out.mkdir(parents=True)
    for rel in sources:
        target = out / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    for rel in deps:
        target = out / "dependency_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    shutil.copy2(analysis_path, out / "parent_analysis.json")
    (out / "prepared.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    plan = {"schema": "noise_repair_study_v1", "prepared_sha256": sha(out / "prepared.json"), "parent_analysis_sha256": sha(out / "parent_analysis.json"), "sources": sources, "dependencies": deps, "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
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


def save_checkpoint(path, params, update):
    np.savez_compressed(path, update=np.array(update, dtype=np.int64), sender_logits_hidden=params["sender_logits_hidden"], sender_logits_visible=params["sender_logits_visible"], worker_logits=params["worker_logits"])
    return sha(path)


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


def worker_gradient(trajectory, beta):
    grad = np.zeros_like(trajectory["_params"]["worker_logits"])
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
        np.add.at(grad, (np.zeros(len(idx), dtype=np.int64), state, np.full(len(idx), t), local, inventory), d)
    return grad


def sender_gradient(params, episode, trajectory, beta):
    grad = np.zeros_like(params["sender_logits_hidden"])
    idx = np.flatnonzero(trajectory["active"])
    if len(idx) == 0:
        return grad
    future = future_returns(trajectory["rewards"])
    context = base_design.goal_index(episode["goal"])[idx]
    arrivals = base_design.message_arrival_times("staged", "dual2")
    for slot, arrival in enumerate(arrivals):
        sp = trajectory["message_probs"][slot][idx]
        one = np.zeros_like(sp)
        one[np.arange(len(idx)), trajectory["selected_messages"][idx, slot]] = 1.0
        advantage = center(future[idx, arrival + 1 :].sum(axis=1), context)
        d = -(advantage[:, None] * (one - sp)) / len(idx) - beta * entropy_grad(sp) / len(idx)
        np.add.at(grad[:, slot, :], context, d)
    return grad


def sequence(params, goal):
    context = int(goal[0]) * 2 + int(goal[1])
    return np.array([policy.softmax(params["sender_logits_hidden"][context, slot]).argmax() for slot in range(design.MESSAGE_SLOTS)], dtype=np.int64)


def sender_codebook(params):
    goals = ((0, 0), (0, 1), (1, 0), (1, 1))
    return [sequence(params, goal).tolist() for goal in goals]


def recombination_override(params, episode):
    donors = np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.int8)
    out = np.zeros((len(episode["goal"]), design.MESSAGE_SLOTS), dtype=np.int64)
    for i, goal in enumerate(episode["goal"]):
        target = np.asarray(goal, dtype=np.int8)
        d0 = donors[np.flatnonzero(donors[:, 0] == target[0])[0]]
        d1 = donors[np.flatnonzero(donors[:, 1] == target[1])[0]]
        out[i, 0] = sequence(params, d0)[0]
        out[i, 1] = sequence(params, d1)[1]
    return out


def _metric(trajectory):
    returns = trajectory["team_return"][trajectory["active"]]
    return {"episodes": int(len(returns)), "team_return_mean": float(returns.mean()) if len(returns) else None, "team_return_sd": float(returns.std()) if len(returns) else None, "positive_episode_rate": float((returns > 0).mean()) if len(returns) else None}


def evaluate_worker(params, episode, worker, representation, noise_p, *, mode="natural", override=None):
    tr = environment.rollout(params, episode, representation if worker == design.TARGET_WORKER else "joint_history", "live" if mode != "silent" else "silent", noise_p if mode != "silent" else 0.0, message_mode="permuted" if mode == "permuted" else "natural", sample=False, partner_filter=worker, message_override=override)
    return _metric(tr)


def evaluate(params, seed, representation, noise_p):
    episode = design.episode_stream(int(seed) + design.EVAL_SEED_OFFSET, design.WORKERS * 4096, evaluation=True)
    override = recombination_override(params, episode)
    per_worker = {}
    for worker in range(design.WORKERS):
        per_worker[str(worker)] = {
            "natural_at_train_noise": evaluate_worker(params, episode, worker, representation, noise_p, mode="natural"),
            "clean_natural": evaluate_worker(params, episode, worker, representation, 0.0, mode="natural"),
            "silent": evaluate_worker(params, episode, worker, representation, 0.0, mode="silent"),
            "permuted_at_train_noise": evaluate_worker(params, episode, worker, representation, noise_p, mode="permuted"),
            "recombined_at_train_noise": evaluate_worker(params, episode, worker, representation, noise_p, mode="natural", override=override),
        }
    def mean_metric(name):
        vals = [per_worker[str(w)][name]["team_return_mean"] for w in range(1, design.WORKERS)]
        return {"episodes": int(sum(per_worker[str(w)][name]["episodes"] for w in range(1, design.WORKERS))), "team_return_mean": float(np.mean(vals)), "team_return_sd": float(np.mean([per_worker[str(w)][name]["team_return_sd"] for w in range(1, design.WORKERS)])), "positive_episode_rate": float(np.mean([per_worker[str(w)][name]["positive_episode_rate"] for w in range(1, design.WORKERS)]))}
    new_worker = per_worker[str(design.TARGET_WORKER)]
    incumbents = {name: mean_metric(name) for name in new_worker}
    return {"noise_p": float(noise_p), "new_worker": new_worker, "incumbent_workers_mean": incumbents, "per_worker": per_worker, "sender_sequences": sender_codebook(params)}


def replace_worker(params, seed):
    fresh = policy.make_policy(int(seed) + 192000, design.FORM)
    params["worker_logits"][design.TARGET_WORKER] = fresh["worker_logits"][0]


def train_one(seed, condition, execution, parent_checkpoint, updates=None):
    adaptation, representation, noise_p, noise_key = design.parse_condition(condition)
    updates = design.UPDATES if updates is None else int(updates)
    run = Path(execution) / f"seed_{seed}_{condition}"
    run.mkdir(parents=True, exist_ok=False)
    parent = load_checkpoint(parent_checkpoint)
    params = policy.clone(parent)
    replace_worker(params, seed)
    parent_hash = policy.combined_parameter_hash(parent)
    initial_hash = policy.combined_parameter_hash(params)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", params, 0)
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        episode = design.episode_stream(seed, design.BATCH_SIZE, update=update)
        trajectory = environment.rollout(params, episode, representation, "live", noise_p, sample=True, partner_filter=design.TARGET_WORKER)
        trajectory["_params"] = params
        beta = design.entropy_coefficient(update)
        gw = worker_gradient(trajectory, beta)
        gs = sender_gradient(params, episode, trajectory, beta) if adaptation == "coadapt" else np.zeros_like(params["sender_logits_hidden"])
        norm = float(np.sqrt(float((gw * gw).sum()) + float((gs * gs).sum())))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        params["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * gw[design.TARGET_WORKER]
        if adaptation == "coadapt":
            params["sender_logits_hidden"] -= design.LEARNING_RATE * scale * gs
        rows.append({"update": update, "seed": seed, "condition": condition, "adaptation": adaptation, "representation": representation, "noise_p": float(noise_p), "noise_key": noise_key, "world_sha256": design.array_sha(episode["site_type"]), "goal_sha256": design.array_sha(episode["goal"]), "partner_sha256": design.array_sha(episode["partner_id"]), "message_uniform_sha256": design.array_sha(episode["message_uniforms"]), "action_uniform_sha256": design.array_sha(episode["action_uniforms"]), "noise_uniform_sha256": design.array_sha(episode["noise_uniforms"]), "return_mean": float(trajectory["team_return"][trajectory["active"]].mean()), "active_count": int(trajectory["active"].sum()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.combined_parameter_hash(params), "elapsed_seconds": time.perf_counter() - start})
        if update in checkpoints:
            rows[-1]["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", params, update)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    final = evaluate(params, seed, representation, noise_p)
    result = {"seed": seed, "condition": condition, "adaptation": adaptation, "representation": representation, "noise_p": float(noise_p), "noise_key": noise_key, "parent_checkpoint": str(parent_checkpoint), "parent_checkpoint_sha256": sha(parent_checkpoint), "parent_parameter_sha256": parent_hash, "initial_parameter_sha256": initial_hash, "updates": updates, "final": final, "parent_sender_sequences": sender_codebook(parent), "checkpoints": checkpoints, "trajectory": rows, "training_log_sha256": sha(log), "final_parameter_sha256": policy.combined_parameter_hash(params), "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"), "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz")}
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
    records = []
    total = len(seeds) * len(conditions)
    for seed in seeds:
        for condition in conditions:
            records.append(train_one(seed, condition, execution, parent_path(cfg["parent_root"], seed), updates))
            (execution / "progress.json").write_text(json.dumps({"completed": len(records), "total": total, "seeds": list(seeds), "conditions": list(conditions)}, indent=2))
    (execution / "results.json").write_text(json.dumps({"results": records}, ensure_ascii=False, indent=2) + "\n")
    return records


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
