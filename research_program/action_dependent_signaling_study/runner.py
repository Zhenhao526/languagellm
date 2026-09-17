"""Train, evaluate and freeze the action-dependent signaling control."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, time
from pathlib import Path
import numpy as np
from . import design, environment, policy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKPOINTS = design.CHECKPOINTS


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_bytes(x): return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


def finite(x):
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    if isinstance(x, (float, int, np.number)): return bool(np.isfinite(float(x)))
    return True


def source_hashes():
    rels = ("__init__.py", "README.md", "design.py", "policy.py", "environment.py", "runner.py", "aggregate.py", "audit.py", "plan.md", "tests/test_game.py")
    files = [HERE / x for x in rels]
    if not all(x.is_file() for x in files): raise ValueError("missing source file")
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def prepare(out):
    out = Path(out).resolve()
    if out.exists(): raise ValueError("refuse overwrite")
    cfg = design.prepare(); cfg.update({"checkpoints": list(CHECKPOINTS), "runtime_policy": "float64 NumPy only", "runs": len(design.SEEDS) * len(design.CONDITIONS), "evaluation_episodes_per_worker": 4096})
    sources = source_hashes(); out.mkdir(parents=True)
    for rel in sources:
        q = out / "source_snapshot" / rel; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, q)
    (out / "prepared.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    plan = {"schema": "action_dependent_signaling_v1", "prepared_sha256": sha(out / "prepared.json"), "sources": sources, "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}, indent=2) + "\n")
    verify(out); return {"status": "prepared", "out": str(out), "plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}


def verify(out):
    out = Path(out); plan = json.loads((out / "plan.json").read_text()); cfg = json.loads((out / "prepared.json").read_text()); fr = json.loads((out / "freeze.json").read_text())
    if sha(out / "plan.json") != fr["plan_sha256"] or sha(out / "prepared.json") != fr["prepared_sha256"] or fr["prepared_sha256"] != plan["prepared_sha256"]: raise ValueError("freeze hash mismatch")
    if plan["config"] != cfg or plan["sources"] != source_hashes(): raise ValueError("design or source changed")
    for rel, digest in plan["sources"].items():
        if sha(out / "source_snapshot" / rel) != digest: raise ValueError("snapshot changed " + rel)
    return plan, cfg


def entropy_grad(prob):
    lp = np.log(np.maximum(prob, 1e-300)); ent = -(prob * lp).sum(axis=-1, keepdims=True)
    return -prob * (lp + ent)


def center(values, keys):
    values = np.asarray(values, dtype=np.float64); keys = np.asarray(keys); out = np.zeros_like(values)
    for key in np.unique(keys):
        idx = np.flatnonzero(keys == key)
        if len(idx) > 1: out[idx] = values[idx] - (values[idx].sum() - values[idx]) / (len(idx) - 1)
    return out


def sequence_index(messages, form):
    k = design.alphabet_size(form); out = np.zeros(len(messages), dtype=np.int64)
    for i in range(design.message_length(form)): out = out * k + np.asarray(messages)[:, i]
    return out


def _sender_context(ep, pv):
    goal = design.goal_index(ep["goal"])
    if pv == "visible": return goal * design.WORKERS + ep["partner_id"].astype(np.int64)
    return goal


def train_one(seed, condition, out, updates=None):
    form, pm, pv, ch, task, protocol = design.parse_condition(condition); updates = design.UPDATES if updates is None else updates
    run = Path(out) / f"seed_{seed}_{condition}"; run.mkdir(parents=True, exist_ok=False)
    p = policy.make_policy(seed, form); init_hash = policy.combined_parameter_hash(p)
    checkpoints = sorted(set([u for u in CHECKPOINTS if u <= updates] + [updates]))

    def save(path, update):
        np.savez_compressed(path, update=np.array(update, dtype=np.int64), sender_logits_hidden=p["sender_logits_hidden"], sender_logits_visible=p["sender_logits_visible"], worker_logits=p["worker_logits"], form=np.array(form))
        return sha(path)

    if 0 in checkpoints: save(run / "checkpoint_0000.npz", 0)
    rows = []; start = time.perf_counter(); length = design.message_length(form)
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, pm, pv, task, design.BATCH_SIZE, update=update)
        tr = environment.rollout(p, ep, form, pm, pv, task, ch, protocol, sample=True)
        g = {k: np.zeros_like(v) for k, v in p.items()}; beta = design.entropy_coefficient(update)
        future_all = np.flip(np.cumsum(np.flip(tr["rewards"], axis=1), axis=1), axis=1)
        context = _sender_context(ep, pv)
        for slot in range(length):
            sp = tr["message_probs"][slot]
            one = np.zeros_like(sp); one[np.arange(len(ep["goal"])), tr["selected_messages"][:, slot]] = 1
            arrival = design.message_arrival_times(protocol, form)
            adv = center(future_all[:, arrival[slot] + 1:].sum(axis=1), context)
            d = -(adv[:, None] * (one - sp)) / len(ep["goal"]) - beta * entropy_grad(sp) / len(ep["goal"])
            key = "sender_logits_visible" if pv == "visible" else "sender_logits_hidden"
            np.add.at(g[key][:, slot, :], context, d)
        worker = ep["partner_id"].astype(np.int64)
        for t in range(design.ACTION_START, design.HORIZON):
            ap = tr["action_probs"][t]; st = tr["state"][t]; local = tr["local"][t]; inv = tr["inventory"][t]
            returns_t = future_all[:, t]
            keys = worker * 10000000 + st * 100000 + t * 1000 + local * 10 + inv
            adv = center(returns_t, keys)
            oa = np.zeros_like(ap); oa[np.arange(len(ep["goal"])), tr["actions"][:, t]] = 1
            d = -(adv[:, None] * (oa - ap)) / len(ep["goal"]) - beta * entropy_grad(ap) / len(ep["goal"])
            np.add.at(g["worker_logits"], (worker, st, np.full(len(worker), t), local, inv), d)
        norm = float(np.sqrt(sum(float((x * x).sum()) for x in g.values()))); scale = min(1.0, 5.0 / max(norm, 1e-12))
        for key in p: p[key] -= design.LEARNING_RATE * scale * g[key]
        row = {"update": update, "seed": seed, "condition": condition, "protocol": protocol, "world_sha256": design.array_sha(ep["site_type"]), "goal_sha256": design.array_sha(ep["goal"]), "target_sha256": design.array_sha(ep["target_bits"]), "partner_sha256": design.array_sha(ep["partner_id"]), "message_uniform_sha256": design.array_sha(ep["message_uniforms"]), "action_uniform_sha256": design.array_sha(ep["action_uniforms"]), "return_mean": float(tr["team_return"].mean()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.combined_parameter_hash(p), "elapsed_seconds": time.perf_counter() - start}
        rows.append(row)
        if update in checkpoints and update > 0: row["checkpoint_sha256"] = save(run / f"checkpoint_{update:04d}.npz", update)
    final = {}
    for evaluation, label in ((False, "training_support"), (True, "heldout")):
        final[label] = evaluate_all(p, seed, form, pm, pv, task, ch, protocol, evaluation=evaluation)
    result = {"seed": seed, "condition": condition, "form": form, "partner_mode": pm, "partner_visibility": pv, "channel": ch, "task": task, "protocol": protocol, "updates": updates, "initial_parameter_sha256": init_hash, "trajectory": rows, "final": final, "checkpoints": checkpoints}
    (run / "training.jsonl").write_bytes(b"".join(json_bytes(x) for x in rows)); result["training_log_sha256"] = sha(run / "training.jsonl")
    result["final_parameter_sha256"] = policy.combined_parameter_hash(p)
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def _entropy(arr, k):
    x = np.bincount(np.asarray(arr).ravel(), minlength=k).astype(float); x /= max(x.sum(), 1); x = x[x > 0]
    return float(-(x * np.log2(x)).sum())


def _mi(a, b, ka, kb):
    joint = np.zeros((ka, kb), dtype=float)
    for x, y in zip(a, b): joint[int(x), int(y)] += 1
    joint /= max(joint.sum(), 1); px = joint.sum(1); py = joint.sum(0); val = 0.0
    for i in range(ka):
        for j in range(kb):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0: val += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return float(val)


def evaluate_one(p, ep, form, pm, pv, task, ch, protocol, mode, worker, *, message_override=None):
    tr = environment.rollout(p, ep, form, pm, pv, task, ch, protocol, message_mode=mode, sample=False, partner_filter=worker, message_override=message_override)
    mask = tr["active"]; r = tr["team_return"][mask]; goal = ep["goal"][mask]
    oracle = environment.oracle_team_return({k: (v[mask] if hasattr(v, "__len__") and not isinstance(v, str) else v) for k, v in ep.items()})
    if len(r):
        raw = tr["messages"][mask, :design.message_length(form)]
        seq = sequence_index(raw, form)
        null_seq = design.alphabet_size(form) ** design.message_length(form)
        seq[np.any(raw < 0, axis=1)] = null_seq
        seq_k = null_seq + 1
    else:
        seq = np.zeros(0, dtype=np.int64); seq_k = design.alphabet_size(form) ** design.message_length(form) + 1
    return {"episodes": int(mask.sum()), "team_return_mean": float(r.mean()) if len(r) else None, "team_return_sd": float(r.std()) if len(r) else None, "positive_episode_rate": float((r > 0).mean()) if len(r) else None, "oracle_team_return_mean": float(oracle.mean()) if len(r) else None, "normalized_return_mean_on_oracle_positive": float(np.mean(r[oracle > 1e-12] / oracle[oracle > 1e-12])) if np.any(oracle > 1e-12) else None, "message_entropy": _entropy(seq, seq_k) if len(r) else None, "message_goal_mi": _mi(seq, design.goal_index(goal), seq_k, 4) if len(r) else None, "message_histogram": np.bincount(seq, minlength=seq_k).tolist() if len(r) else [], "action_histogram": np.bincount(tr["actions"][mask].ravel(), minlength=design.ACTION_COUNT).tolist() if len(r) else []}


def _deterministic_sequence(p, form, pv, goal, partner):
    idx = int(goal[0]) * 2 + int(goal[1]); context = idx * design.WORKERS + partner if pv == "visible" else idx
    return np.array([policy.softmax(p["sender_logits_visible" if pv == "visible" else "sender_logits_hidden"][context, slot]).argmax() for slot in range(design.message_length(form))], dtype=np.int64)


def recombination_override(p, ep, form, pv, task):
    n = len(ep["goal"]); out = np.zeros((n, design.message_length(form)), dtype=np.int64)
    for i, goal in enumerate(ep["goal"]):
        partner = int(ep["partner_id"][i]); g0, g1 = int(goal[0]), int(goal[1])
        if form == "dual2":
            # Select donors by the action-relevant target bits. In the
            # entangled control, target[1] is g0 XOR g1, so raw labels would
            # test the wrong notion of compositionality.
            target = design.target_bits(np.asarray([[g0, g1]], dtype=np.int8), task)[0]
            donors = np.asarray([[a, b] for a in (0, 1) for b in (0, 1)], dtype=np.int8)
            donor0 = donors[np.flatnonzero(design.target_bits(donors, task)[:, 0] == target[0])[0]]
            donor1 = donors[np.flatnonzero(design.target_bits(donors, task)[:, 1] == target[1])[0]]
            out[i, 0] = _deterministic_sequence(p, form, pv, donor0, partner)[0]
            out[i, 1] = _deterministic_sequence(p, form, pv, donor1, partner)[1]
        else:
            out[i] = _deterministic_sequence(p, form, pv, goal, partner)
    return out


def evaluate_recombined(p, ep, form, pm, pv, task, ch, protocol, worker):
    if form != "dual2": return {"episodes": 0, "team_return_mean": None, "defined": False}
    override = recombination_override(p, ep, form, pv, task)
    return {**evaluate_one(p, ep, form, pm, pv, task, ch, protocol, "natural", worker, message_override=override), "defined": True}


def _semantic_codebook(p, form, pm, pv, task, protocol):
    active = [0] if pm == "fixed" else list(range(design.WORKERS))
    sender = np.zeros((4, len(active), design.message_length(form)), dtype=np.int64)
    semantic = np.zeros((4, len(active)), dtype=np.float64)
    local_patterns = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.int8)
    for goal_idx, goal in enumerate(local_patterns):
        for wi, worker in enumerate(active):
            sender[goal_idx, wi] = _deterministic_sequence(p, form, pv, goal, worker)
            cases = []
            for locals_ in local_patterns:
                ep = {"site_type": np.stack([locals_, 1 - locals_], axis=-1)[None, :, :].repeat(1, axis=0),
                      "goal": np.asarray(goal, dtype=np.int8)[None, :], "target_bits": design.target_bits(np.asarray(goal, dtype=np.int8)[None, :], task),
                      "partner_id": np.array([worker], dtype=np.int8), "message_uniforms": np.zeros((1, design.MESSAGE_SLOTS)), "action_uniforms": np.zeros((1, design.HORIZON)),
                      "capacity": np.full((1, design.SUBTASKS, 2), design.CAPACITY, dtype=np.int8)}
                tr = environment.rollout(p, ep, form, pm, pv, task, "live", protocol, sample=False, partner_filter=None)
                target = ep["target_bits"][0]
                hit = 0
                for t in range(design.ACTION_START, design.HORIZON):
                    stage = (t - design.ACTION_START) // design.STEPS_PER_SUBTASK; action = int(tr["actions"][0, t]); site = action - 1
                    hit += int(action > 0 and int(ep["site_type"][0, stage, site]) == int(target[stage]))
                cases.append(hit / (design.SUBTASKS * design.STEPS_PER_SUBTASK))
            semantic[goal_idx, wi] = float(np.mean(cases))
    agreement = []
    for goal_idx in range(4):
        agreement.append(float(np.mean([np.all(sender[goal_idx, w] == sender[goal_idx, 0]) for w in range(len(active))])))
    return {"active_workers": active, "sender_sequence_by_goal_worker": sender.tolist(), "semantic_success_by_goal_worker": semantic.tolist(), "sender_sequence_agreement": float(np.mean(agreement)), "semantic_success_mean": float(semantic.mean()), "semantic_success_min": float(semantic.min())}


def evaluate_all(p, seed, form, pm, pv, task, ch, protocol, *, evaluation):
    count = 4096 if pm == "fixed" else 4096 * design.WORKERS
    ep = design.episode_stream(seed, pm, pv, task, count, evaluation=evaluation)
    out = {"split": "heldout" if evaluation else "training_support", "workers": {}, "codebook": _semantic_codebook(p, form, pm, pv, task, protocol)}
    worker_ids = [0] if pm == "fixed" else list(range(design.WORKERS))
    for worker in worker_ids:
        if pm == "fixed":
            ep_worker = dict(ep, partner_id=np.full(len(ep["goal"]), worker, dtype=np.int8)); selected_worker = None
        else:
            ep_worker = ep; selected_worker = worker
        modes = {"natural": evaluate_one(p, ep_worker, form, pm, pv, task, ch, protocol, "natural", selected_worker)}
        if ch == "live":
            modes["closed"] = evaluate_one(p, ep_worker, form, pm, pv, task, ch, protocol, "closed", selected_worker)
            modes["permuted"] = evaluate_one(p, ep_worker, form, pm, pv, task, ch, protocol, "permuted", selected_worker)
        else:
            modes["closed"] = dict(modes["natural"], reused_natural=True)
            modes["permuted"] = dict(modes["natural"], reused_natural=True)
        if form == "dual2" and ch == "live": modes["recombined"] = evaluate_recombined(p, ep_worker, form, pm, pv, task, ch, protocol, selected_worker)
        out["workers"][str(worker)] = modes
    return out


def run_grid(prepared, out, updates=None, seeds=None, conditions=None):
    verify(prepared); out = Path(out); execution = out / "execution"; execution.mkdir(parents=True)
    seeds = tuple(design.SEEDS if seeds is None else seeds); conditions = tuple(design.CONDITIONS if conditions is None else conditions)
    if not set(seeds).issubset(design.SEEDS) or not set(conditions).issubset(design.CONDITIONS): raise ValueError("invalid subset")
    results = []; total = len(seeds) * len(conditions)
    for seed in seeds:
        for condition in conditions:
            results.append(train_one(seed, condition, execution, updates))
            (execution / "progress.json").write_text(json.dumps({"completed": len(results), "total": total, "seeds": list(seeds), "conditions": list(conditions)}, indent=2))
    (execution / "results.json").write_text(json.dumps({"results": results}, ensure_ascii=False, indent=2) + "\n")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--out", required=True)
    a = sub.add_parser("execute"); a.add_argument("--out", required=True); a.add_argument("--prepared", required=True); a.add_argument("--updates", type=int, default=None); a.add_argument("--seeds", default=None); a.add_argument("--conditions", default=None)
    args = ap.parse_args()
    if args.cmd == "prepare": print(json.dumps(prepare(args.out), ensure_ascii=False))
    else:
        seeds = None if args.seeds is None else tuple(int(x) for x in args.seeds.split(",") if x)
        conditions = None if args.conditions is None else tuple(x for x in args.conditions.split(",") if x)
        print(json.dumps({"status": "completed", "runs": len(run_grid(args.prepared, args.out, args.updates, seeds, conditions))}, ensure_ascii=False))
