"""Execute and audit a two-generation component-substitution chain."""
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
from research_program.action_dependent_signaling_study import environment as joint_environment
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


def prepare(out, parent_root, parent_analysis):
    out = Path(out).resolve()
    design.require(not out.exists(), "refuse overwrite")
    parent_root = Path(parent_root).resolve()
    analysis_path = Path(parent_analysis).resolve()
    cfg = design.prepare(parent_root, parent_hashes(parent_root), sha(analysis_path))
    cfg.update({"runs": len(design.SEEDS) * len(design.LINEAGES) * 6, "stage1_runs": len(design.SEEDS) * len(design.LINEAGES) * 2, "stage2_runs": len(design.SEEDS) * len(design.LINEAGES) * 4, "evaluation_episodes": 4096})
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
    shutil.copy2(analysis_path, out / "parent_analysis.json")
    (out / "prepared.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    plan = {"schema": "multi_generation_transmission_study_v1", "prepared_sha256": sha(out / "prepared.json"), "parent_analysis_sha256": sha(out / "parent_analysis.json"), "sources": src, "dependencies": deps, "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
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


def stage_role(lineage, generation):
    first, second = lineage.split("_then_")
    return first if generation == 1 else second


def representation(lineage, generation):
    # The worker created in either lineage uses slot-local state.  A parent
    # worker is joint-history until it is replaced in generation 2.
    if lineage == "worker_then_sender":
        return "slot_local"
    return "joint_history" if generation == 1 else "slot_local"


def rollout(params, episode, rep, channel, *, sample, message_mode="natural", message_override=None):
    if rep == "slot_local":
        return environment.rollout_slot_local(params, episode, channel, message_mode=message_mode, sample=sample, partner_filter=design.TARGET_WORKER, message_override=message_override)
    return joint_environment.rollout(params, episode, design.FORM, design.PARTNER_MODE, design.VISIBILITY, design.TASK, channel, design.PROTOCOL, message_mode=message_mode, sample=sample, partner_filter=design.TARGET_WORKER, message_override=message_override)


def worker_gradient(params, episode, trajectory, beta):
    grad = np.zeros_like(params["worker_logits"])
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


def sender_gradient(params, episode, trajectory, channel, beta):
    grad = np.zeros_like(params["sender_logits_hidden"])
    if channel == "silent":
        return grad
    future = future_returns(trajectory["rewards"])
    context = base_design.goal_index(episode["goal"])
    arrival = base_design.message_arrival_times("staged", "dual2")
    for slot in range(design.MESSAGE_SLOTS):
        sp = policy.softmax(params["sender_logits_hidden"][context, slot])
        one = np.zeros_like(sp)
        one[np.arange(len(context)), trajectory["selected_messages"][:, slot]] = 1.0
        advantage = center(future[:, arrival[slot] + 1 :].sum(axis=1), context)
        d = -(advantage[:, None] * (one - sp)) / len(context) - beta * entropy_grad(sp) / len(context)
        np.add.at(grad[:, slot, :], context, d)
    return grad


def sequence(params, goal):
    context = int(goal[0]) * 2 + int(goal[1])
    return np.array([policy.softmax(params["sender_logits_hidden"][context, slot]).argmax() for slot in range(design.MESSAGE_SLOTS)], dtype=np.int64)


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


def metrics(params, episode, rep, *, mode="natural", override=None):
    tr = rollout(params, episode, rep, "live", sample=False, message_mode=mode, message_override=override)
    returns = tr["team_return"][tr["active"]]
    return {"episodes": int(len(returns)), "team_return_mean": float(returns.mean()) if len(returns) else None, "team_return_sd": float(returns.std()) if len(returns) else None, "positive_episode_rate": float((returns > 0).mean()) if len(returns) else None}


def semantic(params, rep):
    # A compact codebook probe: test all factor combinations with fixed local
    # site patterns and average success over the two sub-tasks.
    values = []
    for goal in ((0, 0), (0, 1), (1, 0), (1, 1)):
        ep = design.episode_stream(880000 + 10 * goal[0] + goal[1], 1, 4, evaluation=True)
        ep["goal"][:] = np.asarray(goal, dtype=np.int8)
        ep["target_bits"] = base_design.target_bits(ep["goal"], design.TASK)
        tr = rollout(params, ep, rep, "live", sample=False)
        hit = 0
        for t in range(design.ACTION_START, design.HORIZON):
            stage = (t - design.ACTION_START) // design.STEPS_PER_SUBTASK
            action = tr["actions"][:, t]
            site = action - 1
            valid = action > 0
            chosen_type = np.zeros(len(ep["goal"]), dtype=np.int8)
            chosen_type[valid] = ep["site_type"][valid, stage, site[valid]]
            hit += int(np.sum(valid & (chosen_type == ep["target_bits"][:, stage])))
        values.append(float(hit / (len(ep["goal"]) * design.SUBTASKS * design.STEPS_PER_SUBTASK)))
    return {"semantic_success_by_goal": values, "semantic_success_mean": float(np.mean(values)), "sender_sequences": [sequence(params, goal).tolist() for goal in ((0, 0), (0, 1), (1, 0), (1, 1))]}


def evaluate(params, seed, generation, rep):
    episode = design.episode_stream(seed, generation, 4096, evaluation=True)
    modes = {mode: metrics(params, episode, rep, mode=mode) for mode in ("natural", "closed", "permuted")}
    modes["recombined"] = metrics(params, episode, rep, override=recombination_override(params, episode))
    return {"generation": generation, "representation": rep, "modes": modes, "codebook": semantic(params, rep)}


def replacement(params, seed, generation, role):
    fresh = policy.make_policy(int(seed) + 100000 * generation + (17 if role == "worker" else 31), design.FORM)
    if role == "worker":
        params["worker_logits"][design.TARGET_WORKER] = fresh["worker_logits"][0]
    else:
        params["sender_logits_hidden"] = fresh["sender_logits_hidden"]
        params["sender_logits_visible"] = fresh["sender_logits_visible"]


def train_stage(seed, condition, generation, execution, parent_checkpoint, updates=None):
    lineage, channel = design.stage_condition(condition, generation)
    _, g1_channel, g2_channel = design.parse_condition(condition)
    role = stage_role(lineage, generation)
    rep = representation(lineage, generation)
    updates = design.UPDATES if updates is None else int(updates)
    run = Path(execution)
    run.mkdir(parents=True, exist_ok=False)
    parent = load_checkpoint(parent_checkpoint)
    params = policy.clone(parent)
    replacement(params, seed, generation, role)
    parent_hash = policy.combined_parameter_hash(parent)
    initial_hash = policy.combined_parameter_hash(params)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", params, 0)
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        episode = design.episode_stream(seed, generation, design.BATCH_SIZE, update=update)
        trajectory = rollout(params, episode, rep, channel, sample=True, message_mode="natural")
        beta = design.entropy_coefficient(update)
        if role == "worker":
            grad = worker_gradient(params, episode, trajectory, beta)
            norm = float(np.sqrt(float((grad * grad).sum())))
            scale = min(1.0, 5.0 / max(norm, 1e-12))
            params["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * grad[design.TARGET_WORKER]
        else:
            grad = sender_gradient(params, episode, trajectory, channel, beta)
            norm = float(np.sqrt(float((grad * grad).sum())))
            scale = min(1.0, 5.0 / max(norm, 1e-12))
            params["sender_logits_hidden"] -= design.LEARNING_RATE * scale * grad
            params["sender_logits_visible"] -= design.LEARNING_RATE * scale * np.repeat(grad, design.WORKERS, axis=0)
        row = {"update": update, "seed": seed, "condition": condition, "generation": generation, "lineage": lineage, "role": role, "channel": channel, "representation": rep, "world_sha256": design.array_sha(episode["site_type"]), "goal_sha256": design.array_sha(episode["goal"]), "partner_sha256": design.array_sha(episode["partner_id"]), "message_uniform_sha256": design.array_sha(episode["message_uniforms"]), "action_uniform_sha256": design.array_sha(episode["action_uniforms"]), "return_mean": float(trajectory["team_return"][trajectory["active"]].mean()) if trajectory["active"].any() else 0.0, "active_count": int(trajectory["active"].sum()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.combined_parameter_hash(params), "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            row["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", params, update)
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result = {"seed": seed, "condition": condition, "generation": generation, "lineage": lineage, "g1_channel": g1_channel, "g2_channel": g2_channel, "role": role, "channel": channel, "representation": rep, "parent_checkpoint": str(parent_checkpoint), "parent_checkpoint_sha256": sha(parent_checkpoint), "parent_parameter_sha256": parent_hash, "initial_parameter_sha256": initial_hash, "updates": updates, "final": evaluate(params, seed, generation, rep), "checkpoints": checkpoints, "trajectory": rows, "training_log_sha256": sha(log), "final_parameter_sha256": policy.combined_parameter_hash(params), "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"), "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz")}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def run_grid(prepared, out, updates=None, seeds=None, lineages=None):
    _, cfg = verify(prepared)
    out = Path(out)
    execution = out / "execution"
    execution.mkdir(parents=True)
    seeds = tuple(design.SEEDS if seeds is None else seeds)
    lineages = tuple(design.LINEAGES if lineages is None else lineages)
    design.require(set(seeds) <= set(design.SEEDS) and set(lineages) <= set(design.LINEAGES), "invalid subset")
    records = []
    total = len(seeds) * len(lineages) * 6
    for seed in seeds:
        for lineage in lineages:
            for g1 in design.CHANNELS:
                # g2 is irrelevant to the generation-1 state; choose the
                # canonical live spelling while retaining the requested g1.
                condition = f"{lineage}_g1{g1}_g2live"
                parent = parent_path(cfg["parent_root"], seed)
                stage_dir = execution / f"seed_{seed}" / lineage / f"g1{g1}"
                result = train_stage(seed, condition, 1, stage_dir, parent, updates)
                records.append(result)
                (execution / "progress.json").write_text(json.dumps({"completed": len(records), "total": total, "seeds": list(seeds), "lineages": list(lineages)}, indent=2))
                for g2 in design.CHANNELS:
                    full_condition = f"{lineage}_g1{g1}_g2{g2}"
                    stage2_dir = execution / f"seed_{seed}" / lineage / f"g1{g1}_g2{g2}"
                    result2 = train_stage(seed, full_condition, 2, stage2_dir, stage_dir / f"checkpoint_{updates or design.UPDATES:04d}.npz", updates)
                    records.append(result2)
                    (execution / "progress.json").write_text(json.dumps({"completed": len(records), "total": total, "seeds": list(seeds), "lineages": list(lineages)}, indent=2))
    (execution / "results.json").write_text(json.dumps({"results": records}, ensure_ascii=False, indent=2) + "\n")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--out", required=True); a.add_argument("--parent-root", required=True); a.add_argument("--parent-analysis", required=True)
    a = sub.add_parser("execute"); a.add_argument("--out", required=True); a.add_argument("--prepared", required=True); a.add_argument("--updates", type=int, default=None); a.add_argument("--seeds"); a.add_argument("--lineages")
    args = parser.parse_args()
    if args.cmd == "prepare":
        print(json.dumps(prepare(args.out, args.parent_root, args.parent_analysis), ensure_ascii=False))
    else:
        seeds = None if args.seeds is None else tuple(int(value) for value in args.seeds.split(",") if value)
        lineages = None if args.lineages is None else tuple(value for value in args.lineages.split(",") if value)
        print(json.dumps({"status": "completed", "runs": len(run_grid(args.prepared, args.out, args.updates, seeds, lineages))}, ensure_ascii=False))
