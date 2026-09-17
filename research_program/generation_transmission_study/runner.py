"""Train a parent code and measure recovery after one-agent replacement."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, time
from pathlib import Path
import numpy as np
from . import design, environment, policy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKPOINTS = design.CHECKPOINTS


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(x):
    return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()


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
    cfg = design.prepare(); cfg.update({
        "runs": len(design.SEEDS) * len(design.CONDITIONS),
        "parent_runs": len(design.SEEDS),
        "child_runs": len(design.SEEDS) * len(design.CONDITIONS),
        "evaluation_episodes_per_worker": 4096,
    })
    sources = source_hashes(); out.mkdir(parents=True)
    for rel in sources:
        q = out / "source_snapshot" / rel; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, q)
    (out / "prepared.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    plan = {"schema": "generation_transmission_v1", "prepared_sha256": sha(out / "prepared.json"), "sources": sources,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}, indent=2) + "\n")
    verify(out)
    return {"status": "prepared", "out": str(out), "plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}


def verify(out):
    out = Path(out); plan = json.loads((out / "plan.json").read_text()); cfg = json.loads((out / "prepared.json").read_text()); freeze = json.loads((out / "freeze.json").read_text())
    if sha(out / "plan.json") != freeze["plan_sha256"] or sha(out / "prepared.json") != freeze["prepared_sha256"] or freeze["prepared_sha256"] != plan["prepared_sha256"]: raise ValueError("freeze hash mismatch")
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


def future_returns(rewards):
    return np.flip(np.cumsum(np.flip(rewards, axis=1), axis=1), axis=1)


def save_policy(path, p, update):
    np.savez_compressed(path, update=np.array(update, dtype=np.int64), sender_logits=p["sender_logits"], worker_logits=p["worker_logits"])
    return sha(path)


def load_policy(path):
    with np.load(path) as z:
        return {"sender_logits": np.asarray(z["sender_logits"], dtype=np.float64), "worker_logits": np.asarray(z["worker_logits"], dtype=np.float64)}


def frozen_hash(p, role):
    h = hashlib.sha256(); h.update(role.encode())
    if role == "worker":
        h.update(np.asarray(p["sender_logits"], dtype=np.float64).tobytes()); h.update(np.asarray(p["worker_logits"][1:], dtype=np.float64).tobytes())
    else:
        h.update(np.asarray(p["worker_logits"], dtype=np.float64).tobytes())
    return h.hexdigest()


def _training_row(seed, condition, update, ep, tr, p, norm, scale, active_count, start):
    return {"update": update, "seed": seed, "condition": condition,
            "world_sha256": design.array_sha(ep["site_type"]), "goal_sha256": design.array_sha(ep["goal"]),
            "partner_sha256": design.array_sha(ep["partner_id"]), "message_uniform_sha256": design.array_sha(ep["message_uniforms"]),
            "action_uniform_sha256": design.array_sha(ep["action_uniforms"]), "scramble_uniform_sha256": design.array_sha(ep["scramble_uniforms"]),
            "active_count": int(active_count), "return_mean": float(tr["team_return"][tr["active"]].mean()) if active_count else 0.0,
            "gradient_norm": float(norm), "gradient_clip_scale": float(scale), "parameter_sha256": policy.parameter_hash(p),
            "elapsed_seconds": time.perf_counter() - start}


def train_parent(seed, execution, updates=None):
    updates = design.UPDATES if updates is None else int(updates)
    run = Path(execution) / f"parent_seed_{seed}"; run.mkdir(parents=True, exist_ok=False)
    p = policy.make_policy(seed, "parent"); init_hash = policy.parameter_hash(p)
    checkpoints = sorted(set([u for u in CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints: save_policy(run / "checkpoint_0000.npz", p, 0)
    rows = []; start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, "parent", design.BATCH_SIZE, update=update)
        tr = environment.rollout(p, ep, "live", message_mode="natural", sample=True)
        g = {"sender_logits": np.zeros_like(p["sender_logits"]), "worker_logits": np.zeros_like(p["worker_logits"])}
        beta = design.entropy_coefficient(update); future = future_returns(tr["rewards"])
        sender_adv = center(future[:, design.ACTION_START:].sum(axis=1), ep["goal"].astype(np.int64))
        sp = tr["message_probs"][0]; one = np.zeros_like(sp); one[np.arange(len(ep["goal"])), tr["selected_messages"]] = 1
        d = -(sender_adv[:, None] * (one - sp)) / len(ep["goal"]) - beta * entropy_grad(sp) / len(ep["goal"])
        np.add.at(g["sender_logits"], ep["goal"].astype(np.int64), d)
        worker = ep["partner_id"].astype(np.int64)
        for t in range(design.ACTION_START, design.HORIZON):
            ap = tr["action_probs"][t]; st = tr["state"][t]; local = tr["local"][t]; inv = tr["inventory"][t]
            adv = center(future[:, t], worker * 100000 + st * 1000 + t * 100 + local * 10 + inv)
            oa = np.zeros_like(ap); oa[np.arange(len(worker)), tr["actions"][:, t]] = 1
            d = -(adv[:, None] * (oa - ap)) / len(worker) - beta * entropy_grad(ap) / len(worker)
            np.add.at(g["worker_logits"], (worker, st, np.full(len(worker), t), local, inv), d)
        norm = float(np.sqrt(sum(float((x * x).sum()) for x in g.values()))); scale = min(1.0, 5.0 / max(norm, 1e-12))
        for k in p: p[k] -= design.LEARNING_RATE * scale * g[k]
        row = _training_row(seed, "parent_live", update, ep, tr, p, norm, scale, len(ep["goal"]), start); rows.append(row)
        if update in checkpoints and update > 0: row["checkpoint_sha256"] = save_policy(run / f"checkpoint_{update:04d}.npz", p, update)
    final = evaluate_parent(p, seed, evaluation=False), evaluate_parent(p, seed, evaluation=True)
    result = {"role": "parent", "seed": seed, "condition": "parent_live", "updates": updates, "initial_parameter_sha256": init_hash,
              "final": {"training_support": final[0], "heldout": final[1]}, "checkpoints": checkpoints,
              "training_log_sha256": None, "final_parameter_sha256": policy.parameter_hash(p)}
    log = run / "training.jsonl"; log.write_bytes(b"".join(json_bytes(x) for x in rows)); result["training_log_sha256"] = sha(log)
    result["trajectory"] = rows
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def _child_grad_worker(p, ep, tr, beta):
    g = {"sender_logits": np.zeros_like(p["sender_logits"]), "worker_logits": np.zeros_like(p["worker_logits"])}
    idx = np.flatnonzero(tr["active"]); future = future_returns(tr["rewards"])
    for t in range(design.ACTION_START, design.HORIZON):
        if len(idx) == 0: break
        ap = tr["action_probs"][t][idx]; st = tr["state"][t][idx]; local = tr["local"][t][idx]; inv = tr["inventory"][t][idx]
        adv = center(future[idx, t], st * 1000 + t * 100 + local * 10 + inv)
        oa = np.zeros_like(ap); oa[np.arange(len(idx)), tr["actions"][idx, t]] = 1
        d = -(adv[:, None] * (oa - ap)) / len(idx) - beta * entropy_grad(ap) / len(idx)
        np.add.at(g["worker_logits"], (np.zeros(len(idx), dtype=np.int64), st, np.full(len(idx), t), local, inv), d)
    return g


def _child_grad_sender(p, ep, tr, beta, channel):
    g = {"sender_logits": np.zeros_like(p["sender_logits"]), "worker_logits": np.zeros_like(p["worker_logits"])}
    if channel == "silent": return g
    future = future_returns(tr["rewards"]); adv = center(future[:, design.ACTION_START:].sum(axis=1), ep["goal"].astype(np.int64))
    sp = tr["message_probs"][0]; one = np.zeros_like(sp); one[np.arange(len(ep["goal"])), tr["selected_messages"]] = 1
    d = -(adv[:, None] * (one - sp)) / len(ep["goal"]) - beta * entropy_grad(sp) / len(ep["goal"])
    np.add.at(g["sender_logits"], ep["goal"].astype(np.int64), d)
    return g


def train_child(seed, condition, execution, parent_checkpoint, updates=None):
    role, channel = design.parse_condition(condition); updates = design.UPDATES if updates is None else int(updates)
    run = Path(execution) / f"seed_{seed}_{condition}"; run.mkdir(parents=True, exist_ok=False)
    parent = load_policy(parent_checkpoint); p = policy.clone(parent)
    replacement = policy.make_replacement(seed, role)
    if role == "worker":
        p["worker_logits"][design.TARGET_WORKER] = replacement
    else:
        p["sender_logits"] = replacement
    init_hash = policy.parameter_hash(p); replacement_hash = hashlib.sha256(np.asarray(replacement, dtype=np.float64).tobytes()).hexdigest()
    parent_hash = policy.parameter_hash(parent); frozen_initial = frozen_hash(p, role)
    checkpoints = sorted(set([u for u in CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints: save_policy(run / "checkpoint_0000.npz", p, 0)
    rows = []; start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, role, design.BATCH_SIZE, update=update)
        train_mode = "natural" if channel == "live" else channel
        tr = environment.rollout(p, ep, channel, message_mode=train_mode, sample=True, partner_filter=design.TARGET_WORKER if role == "worker" else None)
        beta = design.entropy_coefficient(update)
        g = _child_grad_worker(p, ep, tr, beta) if role == "worker" else _child_grad_sender(p, ep, tr, beta, channel)
        # Explicitly update only the replaced component; all parent components
        # are frozen and checked again by the independent audit.
        norm = float(np.sqrt(sum(float((x * x).sum()) for x in g.values()))); scale = min(1.0, 5.0 / max(norm, 1e-12))
        if role == "worker": p["worker_logits"][design.TARGET_WORKER] -= design.LEARNING_RATE * scale * g["worker_logits"][design.TARGET_WORKER]
        else: p["sender_logits"] -= design.LEARNING_RATE * scale * g["sender_logits"]
        active_count = int(tr["active"].sum()); row = _training_row(seed, condition, update, ep, tr, p, norm, scale, active_count, start); rows.append(row)
        if update in checkpoints and update > 0: row["checkpoint_sha256"] = save_policy(run / f"checkpoint_{update:04d}.npz", p, update)
    final_training = evaluate_child(p, seed, role, evaluation=False); final_heldout = evaluate_child(p, seed, role, evaluation=True)
    curve = []
    for update in checkpoints:
        cp = run / f"checkpoint_{update:04d}.npz"; pc = load_policy(cp)
        ev = evaluate_child(pc, seed, role, evaluation=True)
        curve.append({"update": update, "new_agent_natural": ev["new_agent"]["natural"]["team_return_mean"],
                      "new_agent_closed": ev["new_agent"]["closed"]["team_return_mean"],
                      "new_agent_permuted": ev["new_agent"]["permuted"]["team_return_mean"],
                      "codebook_semantic_success": ev["codebook"]["new_agent_semantic_success"]})
    log = run / "training.jsonl"; log.write_bytes(b"".join(json_bytes(x) for x in rows))
    result = {"role": role, "seed": seed, "condition": condition, "channel": channel, "updates": updates,
              "parent_checkpoint": str(Path(parent_checkpoint).relative_to(Path(execution))), "parent_parameter_sha256": parent_hash,
              "initial_parameter_sha256": init_hash, "replacement_sha256": replacement_hash, "frozen_initial_sha256": frozen_initial,
              "final": {"training_support": final_training, "heldout": final_heldout}, "learning_curve": curve,
              "protocol_fidelity": _fidelity(parent, p, role), "frozen_final_sha256": frozen_hash(p, role),
              "checkpoints": checkpoints, "training_log_sha256": sha(log), "final_parameter_sha256": policy.parameter_hash(p),
              "trajectory": rows}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def _metrics(tr, ep):
    mask = tr["active"]; r = tr["team_return"][mask]
    return {"episodes": int(mask.sum()), "team_return_mean": float(r.mean()) if len(r) else None,
            "team_return_sd": float(r.std()) if len(r) else None,
            "positive_episode_rate": float((r > 0).mean()) if len(r) else None,
            "oracle_team_return_mean": float(environment.oracle_team_return({"goal": ep["goal"][mask]})[0]) if len(r) else None,
            "message_histogram": np.bincount(tr["messages"][mask, 0], minlength=design.ALPHABET_SIZE + 1).tolist() if len(r) else [],
            "action_histogram": np.bincount(tr["actions"][mask].ravel(), minlength=design.ACTION_COUNT).tolist() if len(r) else []}


def evaluate_one(p, ep, mode, worker):
    tr = environment.rollout(p, ep, "live", message_mode=mode, sample=False, partner_filter=worker)
    return _metrics(tr, ep)


def _codebook_for_worker(p, worker):
    sender_tokens = [int(p["sender_logits"][goal].argmax()) for goal in (0, 1)]
    semantic = []
    responses = []
    for goal, token in enumerate(sender_tokens):
        good = []; acts = []
        for local in (0, 1):
            action = int(policy.softmax(p["worker_logits"][worker, token, design.ACTION_START, local, design.CAPACITY]).argmax())
            acts.append(action)
            site_type = local if action == 1 else (1 - local if action == 2 else -1)
            good.append(float(site_type == goal))
        semantic.append(float(np.mean(good))); responses.append(acts)
    return {"sender_tokens": sender_tokens, "worker": int(worker), "semantic_success": float(np.mean(semantic)), "semantic_by_goal": semantic, "actions_by_goal_local": responses}


def _parent_codebook(p):
    per = [_codebook_for_worker(p, w) for w in range(design.WORKERS)]
    agree = [float(np.mean([x["sender_tokens"][goal] == per[0]["sender_tokens"][goal] for x in per])) for goal in (0, 1)]
    return {"by_worker": per, "sender_token_agreement": float(np.mean(agree)), "semantic_success_mean": float(np.mean([x["semantic_success"] for x in per]))}


def evaluate_parent(p, seed, *, evaluation):
    ep = design.episode_stream(seed, "parent", 4096 * design.WORKERS, evaluation=evaluation)
    workers = {}
    for worker in range(design.WORKERS):
        workers[str(worker)] = {m: evaluate_one(p, ep, m, worker) for m in ("natural", "closed", "permuted")}
    return {"split": "heldout" if evaluation else "training_support", "workers": workers, "codebook": _parent_codebook(p)}


def _fidelity(parent, child, role):
    if role == "worker":
        old = parent["worker_logits"][design.TARGET_WORKER]; new = child["worker_logits"][design.TARGET_WORKER]
        # Compare greedy actions on both local layouts, all message states and
        # action rounds.  This is a descriptive protocol-fidelity readout.
        a = policy.softmax(old).argmax(axis=-1); b = policy.softmax(new).argmax(axis=-1)
        return {"worker_greedy_agreement": float(np.mean(a == b)),
                "sender_greedy_agreement": 1.0}
    a = policy.softmax(parent["sender_logits"]).argmax(axis=-1); b = policy.softmax(child["sender_logits"]).argmax(axis=-1)
    return {"worker_greedy_agreement": 1.0,
            "sender_greedy_agreement": float(np.mean(a == b))}


def evaluate_child(p, seed, role, *, evaluation):
    phase = role; ep = design.episode_stream(seed, phase, 4096 * design.WORKERS, evaluation=evaluation)
    if role == "worker":
        new = {m: evaluate_one(p, ep, m, design.TARGET_WORKER) for m in ("natural", "closed", "permuted", "scrambled")}
        old = {str(w): evaluate_one(p, ep, "natural", w) for w in range(1, design.WORKERS)}
        cb = _codebook_for_worker(p, design.TARGET_WORKER)
        return {"split": "heldout" if evaluation else "training_support", "new_agent": new, "old_workers": old,
                "population_natural_mean": float(np.mean([new["natural"]["team_return_mean"]] + [x["team_return_mean"] for x in old.values()])),
                "codebook": {"new_agent_sender_tokens": cb["sender_tokens"], "new_agent_semantic_success": cb["semantic_success"], "new_agent_semantic_by_goal": cb["semantic_by_goal"], "new_agent_actions_by_goal_local": cb["actions_by_goal_local"]}}
    modes = {m: {str(w): evaluate_one(p, ep, m, w) for w in range(design.WORKERS)} for m in ("natural", "closed", "permuted", "scrambled")}
    means = {m: float(np.mean([v["team_return_mean"] for v in modes[m].values()])) for m in modes}
    cb = [_codebook_for_worker(p, w) for w in range(design.WORKERS)]
    return {"split": "heldout" if evaluation else "training_support", "new_agent": {m: {"team_return_mean": means[m], "workers": modes[m]} for m in modes},
            "old_workers": {}, "population_natural_mean": means["natural"],
            "codebook": {"new_agent_sender_tokens": [int(x) for x in p["sender_logits"].argmax(axis=1)], "new_agent_semantic_success": float(np.mean([x["semantic_success"] for x in cb])), "new_agent_semantic_by_worker": [x["semantic_success"] for x in cb]}}


def run_grid(prepared, out, updates=None, seeds=None, conditions=None):
    verify(prepared); out = Path(out); execution = out / "execution"; execution.mkdir(parents=True)
    seeds = tuple(design.SEEDS if seeds is None else seeds); conditions = tuple(design.CONDITIONS if conditions is None else conditions)
    if not set(seeds).issubset(design.SEEDS) or not set(conditions).issubset(design.CONDITIONS): raise ValueError("invalid subset")
    parents = []; children = []; total = len(seeds) * (1 + len(conditions)); completed = 0
    for seed in seeds:
        parent = train_parent(seed, execution, updates); parents.append(parent); completed += 1
        parent_cp = execution / f"parent_seed_{seed}" / f"checkpoint_{(updates if updates is not None else design.UPDATES):04d}.npz"
        for condition in conditions:
            children.append(train_child(seed, condition, execution, parent_cp, updates)); completed += 1
            (execution / "progress.json").write_text(json.dumps({"completed": completed, "total": total, "seeds": list(seeds), "conditions": list(conditions)}, indent=2))
    payload = {"schema": "generation_transmission_execution_v1", "parents": parents, "children": children}
    (execution / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare"); a.add_argument("--out", required=True)
    a = sub.add_parser("execute"); a.add_argument("--out", required=True); a.add_argument("--prepared", required=True); a.add_argument("--updates", type=int, default=None); a.add_argument("--seeds", default=None); a.add_argument("--conditions", default=None)
    args = ap.parse_args()
    if args.cmd == "prepare": print(json.dumps(prepare(args.out), ensure_ascii=False))
    else:
        seeds = None if args.seeds is None else tuple(int(x) for x in args.seeds.split(",") if x)
        conditions = None if args.conditions is None else tuple(x for x in args.conditions.split(",") if x)
        payload = run_grid(args.prepared, args.out, args.updates, seeds, conditions)
        print(json.dumps({"status": "completed", "parents": len(payload["parents"]), "children": len(payload["children"])}, ensure_ascii=False))
