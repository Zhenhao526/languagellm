"""Train clean parents and noisy children for emergent error correction."""
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
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
    rels = ("__init__.py", "design.py", "policy.py", "environment.py", "runner.py", "aggregate.py", "audit.py", "plan.md", "tests/test_game.py")
    return {str((HERE / rel).relative_to(ROOT)): sha(HERE / rel) for rel in rels}


def prepare(out):
    out = Path(out).resolve()
    design.require(not out.exists(), "refuse overwrite")
    cfg = design.prepare()
    cfg.update({"runs": len(design.SEEDS) * len(design.CONDITIONS), "parent_runs": len(design.SEEDS) * len(design.FORMS), "evaluation_episodes_per_worker": 4096})
    sources = source_hashes()
    out.mkdir(parents=True)
    for rel in sources:
        target = out / "source_snapshot" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
    (out / "prepared.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    plan = {"schema": "repetition_pressure_study_v1", "prepared_sha256": sha(out / "prepared.json"), "sources": sources, "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}, indent=2) + "\n")
    verify(out)
    return {"status": "prepared", "out": str(out), "plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}


def verify(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    cfg = json.loads((out / "prepared.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    design.require(sha(out / "plan.json") == freeze["plan_sha256"] and sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "freeze hash mismatch")
    design.require(plan["config"] == cfg and plan["sources"] == source_hashes(), "source or config changed")
    for rel, digest in plan["sources"].items():
        design.require(sha(out / "source_snapshot" / rel) == digest, "source snapshot changed " + rel)
    return plan, cfg


def save_checkpoint(path, params, update, form):
    np.savez_compressed(path, update=np.array(update, dtype=np.int64), form=np.array(form), sender_logits_hidden=params["sender_logits_hidden"], sender_logits_visible=params["sender_logits_visible"], worker_logits=params["worker_logits"])
    return sha(path)


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
    log_prob = np.log(np.maximum(prob, 1e-300))
    entropy = -(prob * log_prob).sum(axis=-1, keepdims=True)
    return -prob * (log_prob + entropy)


def gradient(params, episode, trajectory, form, *, train_sender, train_workers, beta):
    grad = {key: np.zeros_like(value) for key, value in params.items()}
    active_idx = np.flatnonzero(trajectory["active"])
    if len(active_idx) == 0:
        return grad
    future = future_returns(trajectory["rewards"])
    if train_sender:
        context = design.goal_index(episode["goal"])[active_idx]
        for slot, arrival in enumerate(design.arrival_times(form)):
            sp = trajectory["message_probs"][slot][active_idx]
            one = np.zeros_like(sp)
            one[np.arange(len(active_idx)), trajectory["selected_messages"][active_idx, slot]] = 1
            advantage = center(future[active_idx, arrival:].sum(axis=1), context)
            d = -(advantage[:, None] * (one - sp)) / len(active_idx) - beta * entropy_grad(sp) / len(active_idx)
            np.add.at(grad["sender_logits_hidden"][:, slot, :], context, d)
    for t in range(design.ACTION_START, design.HORIZON):
        ap = trajectory["action_probs"][t][active_idx]
        state = trajectory["state"][t][active_idx]
        local = trajectory["local"][t][active_idx]
        inventory = trajectory["inventory"][t][active_idx]
        worker = episode["partner_id"][active_idx].astype(np.int64)
        keys = worker * 100000000 + state * 1000000 + t * 10000 + local * 100 + inventory
        advantage = center(future[active_idx, t], keys)
        one = np.zeros_like(ap)
        one[np.arange(len(active_idx)), trajectory["actions"][active_idx, t]] = 1
        d = -(advantage[:, None] * (one - ap)) / len(active_idx) - beta * entropy_grad(ap) / len(active_idx)
        np.add.at(grad["worker_logits"], (worker, state, np.full(len(active_idx), t), local, inventory), d)
    if train_workers != "all":
        keep = int(train_workers)
        grad["worker_logits"][np.arange(design.WORKERS) != keep] = 0
    return grad


def metric(trajectory):
    returns = trajectory["team_return"][trajectory["active"]]
    return {"episodes": int(len(returns)), "team_return_mean": float(returns.mean()) if len(returns) else None, "team_return_sd": float(returns.std()) if len(returns) else None, "positive_episode_rate": float((returns > 0).mean()) if len(returns) else None}


def sender_codebook(params, form):
    out = []
    for goal in ((0, 0), (0, 1), (1, 0), (1, 1)):
        context = int(goal[0]) * 2 + int(goal[1])
        out.append([int(policy.softmax(params["sender_logits_hidden"][context, slot]).argmax()) for slot in range(design.message_length(form))])
    return out


def pairwise_min_hamming(codebook):
    return int(min(sum(a != b for a, b in zip(left, right)) for i, left in enumerate(codebook) for right in codebook[i + 1:]))


def task_class_hamming(codebook, task):
    """Return geometry relevant to the task's repeated meanings."""
    if task != "shared4":
        return pairwise_min_hamming(codebook), 0
    class0 = (codebook[0], codebook[3])
    class1 = (codebook[1], codebook[2])
    cross = min(sum(a != b for a, b in zip(left, right)) for left in class0 for right in class1)
    within = max(sum(a != b for a, b in zip(class0[0], class0[1])), sum(a != b for a, b in zip(class1[0], class1[1])))
    return int(cross), int(within)


def codebook_distance(left, right):
    if left is None or right is None or len(left) != len(right):
        return None
    return float(np.mean([sum(a != b for a, b in zip(a0, b0)) for a0, b0 in zip(left, right)]))


def evaluate_one(params, episode, form, worker, channel, noise_p, *, message_mode="natural"):
    tr = environment.rollout(params, episode, form, channel, noise_p, message_mode=message_mode, sample=False, partner_filter=worker)
    return metric(tr)


def evaluate(params, seed, form, task, noise_p):
    episode = design.episode_stream(int(seed) + design.EVAL_SEED_OFFSET, design.WORKERS * 4096, form, task, evaluation=True)
    per_worker = {}
    for worker in range(design.WORKERS):
        per_worker[str(worker)] = {
            "natural_at_train_noise": evaluate_one(params, episode, form, worker, "live", noise_p),
            "clean_natural": evaluate_one(params, episode, form, worker, "live", 0.0),
            "silent": evaluate_one(params, episode, form, worker, "silent", 0.0),
            "permuted_at_train_noise": evaluate_one(params, episode, form, worker, "live", noise_p, message_mode="permuted"),
        }
    def mean_metric(name):
        vals = [per_worker[str(worker)][name]["team_return_mean"] for worker in range(1, design.WORKERS)]
        return {"episodes": int(sum(per_worker[str(worker)][name]["episodes"] for worker in range(1, design.WORKERS)),), "team_return_mean": float(np.mean(vals)), "team_return_sd": float(np.mean([per_worker[str(worker)][name]["team_return_sd"] for worker in range(1, design.WORKERS)])), "positive_episode_rate": float(np.mean([per_worker[str(worker)][name]["positive_episode_rate"] for worker in range(1, design.WORKERS)]))}
    codebook = sender_codebook(params, form)
    class_hamming, within_class_hamming = task_class_hamming(codebook, task)
    new = per_worker["0"]
    incumbent = {name: mean_metric(name) for name in new}
    return {
        "noise_p": float(noise_p),
        "atomic_noise_p": float(design.atomic_corruption_probability(noise_p)),
        "new_worker": new,
        "incumbent_workers_mean": incumbent,
        "per_worker": per_worker,
        "sender_codebook": codebook,
        "pairwise_min_hamming": pairwise_min_hamming(codebook),
        "task_class_hamming": class_hamming,
        "within_class_hamming": within_class_hamming,
    }


def _run_train(params, seed, form, task, noise_p, run, *, updates, train_sender, train_workers, partner_filter=None):
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", params, 0, form)
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        episode = design.episode_stream(seed, design.BATCH_SIZE, form, task, update=update)
        trajectory = environment.rollout(params, episode, form, "live", noise_p, sample=True, partner_filter=partner_filter)
        beta = design.entropy_coefficient(update)
        grad = gradient(params, episode, trajectory, form, train_sender=train_sender, train_workers=train_workers, beta=beta)
        norm = float(np.sqrt(sum(float((value * value).sum()) for value in grad.values())))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        if train_sender:
            params["sender_logits_hidden"] -= design.LEARNING_RATE * scale * grad["sender_logits_hidden"]
        if train_workers == "all":
            params["worker_logits"] -= design.LEARNING_RATE * scale * grad["worker_logits"]
        else:
            params["worker_logits"][int(train_workers)] -= design.LEARNING_RATE * scale * grad["worker_logits"][int(train_workers)]
        rows.append({"update": update, "seed": int(seed), "form": form, "noise_p": float(noise_p), "world_sha256": design.array_sha(episode["site_type"]), "goal_sha256": design.array_sha(episode["goal"]), "partner_sha256": design.array_sha(episode["partner_id"]), "message_uniform_sha256": design.array_sha(episode["message_uniforms"]), "action_uniform_sha256": design.array_sha(episode["action_uniforms"]), "noise_uniform_sha256": design.array_sha(episode["noise_uniforms"]), "return_mean": float(trajectory["team_return"][trajectory["active"]].mean()), "active_count": int(trajectory["active"].sum()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.combined_parameter_hash(params), "elapsed_seconds": time.perf_counter() - start})
        if update in checkpoints:
            rows[-1]["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", params, update, form)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    return rows, checkpoints, sha(log)


def train_parent(seed, task, form, parents_root, updates=None):
    updates = design.PARENT_UPDATES if updates is None else int(updates)
    run = Path(parents_root) / f"seed_{seed}_{task}_{form}"
    run.mkdir(parents=True, exist_ok=False)
    params = policy.make_policy(int(seed) + 100000, form)
    initial_hash = policy.combined_parameter_hash(params)
    rows, checkpoints, log_hash = _run_train(params, seed, form, task, 0.0, run, updates=updates, train_sender=True, train_workers="all")
    final_checkpoint = run / f"checkpoint_{updates:04d}.npz"
    final = evaluate(params, seed, form, task, 0.0)
    result = {"seed": int(seed), "task": task, "form": form, "updates": updates, "initial_parameter_sha256": initial_hash, "final_parameter_sha256": policy.combined_parameter_hash(params), "final_checkpoint_sha256": sha(final_checkpoint), "training_log_sha256": log_hash, "checkpoints": checkpoints, "final": final, "trajectory": rows}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def replace_worker(params, seed, form):
    fresh = policy.make_policy(int(seed) + 192000 + design.FORMS.index(form), form)
    params["worker_logits"][0] = fresh["worker_logits"][0]


def train_child(seed, condition, execution, parent_result, parent_params, updates=None):
    task, form, adaptation, noise_p, noise_key = design.parse_condition(condition)
    updates = design.UPDATES if updates is None else int(updates)
    run = Path(execution) / f"seed_{seed}_{condition}"
    run.mkdir(parents=True, exist_ok=False)
    if adaptation == "scratch":
        params = policy.make_policy(int(seed) + 300000 + design.FORMS.index(form), form)
        parent_parameter_hash = None
        parent_codebook = None
        replace_worker_flag = False
    else:
        params = policy.clone(parent_params)
        parent_parameter_hash = policy.combined_parameter_hash(parent_params)
        parent_codebook = parent_result["final"]["sender_codebook"]
        replace_worker(params, seed, form)
        replace_worker_flag = True
    initial_hash = policy.combined_parameter_hash(params)
    train_sender = adaptation in ("scratch", "coadapt")
    train_workers = "all" if adaptation == "scratch" else 0
    rows, checkpoints, log_hash = _run_train(params, seed, form, task, noise_p, run, updates=updates, train_sender=train_sender, train_workers=train_workers, partner_filter=None if adaptation == "scratch" else 0)
    final_checkpoint = run / f"checkpoint_{updates:04d}.npz"
    final = evaluate(params, seed, form, task, noise_p)
    result = {"seed": int(seed), "condition": condition, "task": task, "form": form, "adaptation": adaptation, "noise_p": float(noise_p), "noise_key": noise_key, "updates": updates, "parent_parameter_sha256": parent_parameter_hash, "parent_sender_codebook": parent_codebook, "replaced_worker": replace_worker_flag, "initial_parameter_sha256": initial_hash, "final_parameter_sha256": policy.combined_parameter_hash(params), "final_checkpoint_sha256": sha(final_checkpoint), "training_log_sha256": log_hash, "checkpoints": checkpoints, "final": final, "trajectory": rows, "sender_codebook_distance_to_parent": codebook_distance(final["sender_codebook"], parent_codebook)}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def compact_result(result):
    """Drop the per-run trajectory before retaining the merged index in memory."""
    out = dict(result)
    out.pop("trajectory", None)
    return out


def run_grid(prepared, out, updates=None, parent_updates=None, seeds=None, conditions=None):
    _, cfg = verify(prepared)
    out = Path(out)
    execution = out / "execution"
    parents = out / "parents"
    execution.mkdir(parents=True)
    parents.mkdir(parents=True)
    seeds = tuple(design.SEEDS if seeds is None else seeds)
    conditions = tuple(design.CONDITIONS if conditions is None else conditions)
    design.require(set(seeds) <= set(design.SEEDS) and set(conditions) <= set(design.CONDITIONS), "invalid subset")
    parent_records = []
    parent_by = {}
    selected_tasks = sorted(set(design.parse_condition(c)[0] for c in conditions), key=design.TASKS.index)
    parent_forms = sorted(set(design.parse_condition(c)[1] for c in conditions), key=design.FORMS.index)
    total_parents = len(seeds) * len(selected_tasks) * len(parent_forms)
    for seed in seeds:
        for task in selected_tasks:
            for form in parent_forms:
                full_record = train_parent(seed, task, form, parents, updates=parent_updates)
                record = compact_result(full_record)
                del full_record
                parent_records.append(record)
                parent_by[(seed, task, form)] = record
                (out / "parent_progress.json").write_text(json.dumps({"completed": len(parent_records), "total": total_parents, "seeds": list(seeds), "tasks": selected_tasks, "forms": parent_forms}, indent=2))
    child_records = []
    total = len(seeds) * len(conditions)
    for seed in seeds:
        for condition in conditions:
            task, form = design.parse_condition(condition)[:2]
            parent_path = parents / f"seed_{seed}_{task}_{form}" / f"checkpoint_{(design.PARENT_UPDATES if parent_updates is None else int(parent_updates)):04d}.npz"
            parent_params = load_checkpoint(parent_path)
            full_record = train_child(seed, condition, execution, parent_by[(seed, task, form)], parent_params, updates=updates)
            child_records.append(compact_result(full_record))
            del full_record
            (execution / "progress.json").write_text(json.dumps({"completed": len(child_records), "total": total, "seeds": list(seeds), "tasks": selected_tasks, "conditions": list(conditions)}, indent=2))
    (out / "parents.json").write_text(json.dumps({"results": parent_records}, ensure_ascii=False, indent=2) + "\n")
    (execution / "results.json").write_text(json.dumps({"results": child_records}, ensure_ascii=False, indent=2) + "\n")
    return parent_records, child_records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--out", required=True)
    a = sub.add_parser("execute"); a.add_argument("--out", required=True); a.add_argument("--prepared", required=True); a.add_argument("--updates", type=int, default=None); a.add_argument("--parent-updates", type=int, default=None); a.add_argument("--seeds"); a.add_argument("--conditions")
    args = parser.parse_args()
    if args.cmd == "prepare":
        print(json.dumps(prepare(args.out), ensure_ascii=False))
    else:
        seeds = None if args.seeds is None else tuple(int(value) for value in args.seeds.split(",") if value)
        conditions = None if args.conditions is None else tuple(value for value in args.conditions.split(",") if value)
        parents, children = run_grid(args.prepared, args.out, args.updates, args.parent_updates, seeds, conditions)
        print(json.dumps({"status": "completed", "parents": len(parents), "runs": len(children)}, ensure_ascii=False))
