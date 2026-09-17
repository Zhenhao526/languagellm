"""Train and evaluate the fixed-versus-rotating partner study."""
from __future__ import annotations
import argparse, hashlib, json, platform, shutil, time
from pathlib import Path
import numpy as np
from . import design, environment, policy

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKPOINTS = (0, 500, 1000, 2000, 3000)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_bytes(x): return (json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
def finite(x):
    if isinstance(x, dict): return all(finite(v) for v in x.values())
    if isinstance(x, list): return all(finite(v) for v in x)
    if isinstance(x, (float, int, np.number)): return np.isfinite(float(x))
    return True


def combined_parameter_hash(p):
    h = hashlib.sha256()
    for key in ("sender_logits_hidden", "sender_logits_visible", "worker_logits"):
        h.update(key.encode()); h.update(np.asarray(p[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def source_hashes():
    files = [HERE / x for x in ("__init__.py", "README.md", "design.py", "policy.py", "environment.py", "runner.py", "plan.md", "tests/test_game.py")]
    if not all(x.is_file() for x in files):
        raise ValueError("missing source file")
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def prepare(out):
    out = Path(out).resolve()
    if out.exists(): raise ValueError("refuse overwrite")
    cfg = design.prepare(); cfg.update({"checkpoints": list(CHECKPOINTS), "runtime_policy": "float64 NumPy only",
        "runs": len(design.SEEDS) * len(design.CONDITIONS), "evaluation_episodes_per_worker": 4096})
    sources = source_hashes(); out.mkdir(parents=True)
    for rel in sources:
        q = out / "source_snapshot" / rel; q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(ROOT / rel, q)
    (out / "prepared.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")
    plan = {"schema": "population_signaling_v1", "prepared_sha256": sha(out / "prepared.json"), "sources": sources,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__}, "config": cfg}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}, indent=2) + "\n")
    verify(out); return {"status": "prepared", "out": str(out), "plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}


def verify(out):
    out = Path(out); plan = json.loads((out / "plan.json").read_text()); cfg = json.loads((out / "prepared.json").read_text()); freeze = json.loads((out / "freeze.json").read_text())
    if sha(out / "plan.json") != freeze["plan_sha256"] or sha(out / "prepared.json") != freeze["prepared_sha256"] != plan["prepared_sha256"]: raise ValueError("freeze hash mismatch")
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


def train_one(seed, condition, out, updates=None):
    pm, pv, ch, sc = design.parse_condition(condition); updates = design.UPDATES if updates is None else updates
    run = Path(out) / f"seed_{seed}_{condition}"; run.mkdir(parents=True, exist_ok=False)
    p0 = policy.make_policy(seed); p = {"sender_logits_hidden": p0["sender_logits"].copy(),
                                      # Start every partner-specific visible row
                                      # from the same hidden row.  Visibility
                                      # then adds conditioning capacity without
                                      # changing the initial codebook.
                                      "sender_logits_visible": np.repeat(p0["sender_logits"], design.WORKERS, axis=0).copy(),
                                      "worker_logits": p0["worker_logits"].copy()}
    init_hash = combined_parameter_hash(p)
    # Always retain the actual terminal state so every execution, including a
    # short smoke run, can be replay-audited.
    checkpoints = sorted(set([u for u in CHECKPOINTS if u <= updates] + [updates]))
    def save(path, update):
        np.savez_compressed(path, update=np.array(update, dtype=np.int64), sender_logits_hidden=p["sender_logits_hidden"], sender_logits_visible=p["sender_logits_visible"], worker_logits=p["worker_logits"])
        return sha(path)
    if 0 in checkpoints: save(run / "checkpoint_0000.npz", 0)
    rows = []; start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, pm, pv, sc, design.BATCH_SIZE, update=update)
        tr = environment.rollout(p, ep, pm, pv, sc, ch, sample=True)
        g = {"sender_logits_hidden": np.zeros_like(p["sender_logits_hidden"]), "sender_logits_visible": np.zeros_like(p["sender_logits_visible"]), "worker_logits": np.zeros_like(p["worker_logits"])}
        beta = design.entropy_coefficient(update)
        future = np.flip(np.cumsum(np.flip(tr["rewards"], axis=1), axis=1), axis=1)[:, 1:].sum(axis=1)
        sender_adv = center(future, ep["goal"].astype(np.int64) * 100 + ep["partner_id"].astype(np.int64) * (1 if pv == "visible" else 0))
        sp = tr["message_probs"][0]; one = np.zeros_like(sp)
        # A silent channel delivers NULL, which has no sender-logit column.
        # Map it to an arbitrary column because the sender gradient is disabled
        # in that arm; live episodes always contain a real token.
        tok = np.where(tr["messages"][:, 0] == design.NULL_MESSAGE, 0, tr["messages"][:, 0])
        one[np.arange(len(ep["goal"])), tok] = 1
        if ch == "live":
            if pv == "visible":
                context = ep["goal"].astype(np.int64) * design.WORKERS + ep["partner_id"].astype(np.int64)
                np.add.at(g["sender_logits_visible"], context, -(sender_adv[:, None] * (one - sp)) / len(ep["goal"]) - beta * entropy_grad(sp) / len(ep["goal"]))
            else:
                np.add.at(g["sender_logits_hidden"], ep["goal"].astype(np.int64), -(sender_adv[:, None] * (one - sp)) / len(ep["goal"]) - beta * entropy_grad(sp) / len(ep["goal"]))
        for t in range(design.ACTION_START, design.HORIZON):
            ap = tr["action_probs"][t]; st = tr["state"][t]; local = tr["local"][t]; inv = tr["inventory"][t]; worker = ep["partner_id"]
            returns_t = np.flip(np.cumsum(np.flip(tr["rewards"], axis=1), axis=1), axis=1)[:, t]
            act_adv = center(returns_t, worker.astype(np.int64) * 10000 + st.astype(np.int64) * 100 + t * 10 + local.astype(np.int64) * 2 + inv.astype(np.int64))
            oa = np.zeros_like(ap); oa[np.arange(len(ep["goal"])), tr["actions"][:, t]] = 1
            d = -(act_adv[:, None] * (oa - ap)) / len(ep["goal"]) - beta * entropy_grad(ap) / len(ep["goal"])
            np.add.at(g["worker_logits"], (worker, st, np.full(len(worker), t), local, inv), d)
        # The helper expects only the two actual arrays; use the hidden/visible arrays explicitly.
        norm = float(np.sqrt(sum(float((g[k] * g[k]).sum()) for k in g))); scale = min(1., 5. / max(norm, 1e-12))
        for k in g: p[k] -= design.LEARNING_RATE * scale * g[k]
        row = {"update": update, "seed": seed, "condition": condition, "world_sha256": design.array_sha(ep["site_type"]), "goal_sha256": design.array_sha(ep["goal"]), "partner_sha256": design.array_sha(ep["partner_id"]), "message_uniform_sha256": design.array_sha(ep["message_uniforms"]), "action_uniform_sha256": design.array_sha(ep["action_uniforms"]), "return_mean": float(tr["team_return"].mean()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": combined_parameter_hash(p), "elapsed_seconds": time.perf_counter() - start}
        rows.append(row)
        if update in checkpoints and update > 0: row["checkpoint_sha256"] = save(run / f"checkpoint_{update:04d}.npz", update)
    final = {}
    for eval_split, label in ((False, "training_support"), (True, "heldout")):
        final[label] = evaluate_all(p, seed, pm, pv, sc, ch, evaluation=eval_split)
    result = {"seed": seed, "condition": condition, "partner_mode": pm, "partner_visibility": pv, "channel": ch, "scarcity": sc, "updates": updates, "initial_parameter_sha256": init_hash, "trajectory": rows, "final": final, "checkpoints": checkpoints}
    (run / "training.jsonl").write_bytes(b"".join(json_bytes(x) for x in rows)); result["training_log_sha256"] = sha(run / "training.jsonl")
    result["final_parameter_sha256"] = combined_parameter_hash(p)
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def _entropy(arr, k):
    x = np.bincount(np.asarray(arr).ravel(), minlength=k).astype(float); x /= max(x.sum(), 1); x = x[x > 0]
    return float(-(x * np.log2(x)).sum())


def _mi(a, b, ka, kb):
    joint = np.zeros((ka, kb), dtype=float)
    for x, y in zip(a, b): joint[int(x), int(y)] += 1
    joint /= max(joint.sum(), 1); px = joint.sum(1); py = joint.sum(0); val = 0.
    for i in range(ka):
        for j in range(kb):
            if joint[i, j] > 0 and px[i] > 0 and py[j] > 0: val += joint[i, j] * np.log2(joint[i, j] / (px[i] * py[j]))
    return float(val)


def evaluate_one(p, ep, pm, pv, sc, ch, mode, worker):
    tr = environment.rollout(p, ep, pm, pv, sc, ch, message_mode=mode, sample=False, partner_filter=worker)
    mask = tr["active"]; r = tr["team_return"][mask]; goal = ep["goal"][mask]; token = tr["messages"][mask, 0]
    oracle = environment.oracle_team_return({k: (v[mask] if hasattr(v, "__len__") and not isinstance(v, str) else v) for k, v in ep.items()})
    return {"episodes": int(mask.sum()), "team_return_mean": float(r.mean()) if len(r) else None, "team_return_sd": float(r.std()) if len(r) else None, "positive_episode_rate": float((r > 0).mean()) if len(r) else None, "oracle_team_return_mean": float(oracle.mean()) if len(r) else None, "normalized_return_mean_on_oracle_positive": float(np.mean(r[oracle > 1e-12] / oracle[oracle > 1e-12])) if np.any(oracle > 1e-12) else None, "message_entropy": _entropy(token, design.ALPHABET_SIZE + 1) if len(r) else None, "message_goal_mi": _mi(token, goal, design.ALPHABET_SIZE + 1, 2) if len(r) else None, "message_histogram": np.bincount(token, minlength=design.ALPHABET_SIZE + 1).tolist() if len(r) else [], "action_histogram": np.bincount(tr["actions"][mask].ravel(), minlength=design.ACTION_COUNT).tolist() if len(r) else []}


def evaluate_all(p, seed, pm, pv, sc, ch, *, evaluation):
    ep = design.episode_stream(seed, pm, pv, sc, 4096 * design.WORKERS, evaluation=evaluation)
    out = {"split": "heldout" if evaluation else "training_support", "workers": {}, "codebook": codebook(p, pm, pv, sc)}
    for worker in range(design.WORKERS):
        # For a fixed-partner evaluation, replay the identical worlds once for
        # each hypothetical worker.  In the rotating arm, retain the sampled
        # partner assignment and select that worker's episodes.
        if pm == "fixed":
            ep_worker = dict(ep, partner_id=np.full(len(ep["goal"]), worker, dtype=np.int8))
            selected_worker = None
        else:
            ep_worker = ep
            selected_worker = worker
        modes = {"natural": evaluate_one(p, ep_worker, pm, pv, sc, ch, "natural", selected_worker)}
        if ch == "live":
            modes["closed"] = evaluate_one(p, ep_worker, pm, pv, sc, ch, "closed", selected_worker)
            modes["permuted"] = evaluate_one(p, ep_worker, pm, pv, sc, ch, "permuted", selected_worker)
        else:
            modes["closed"] = dict(modes["natural"], message_mode="closed", reused_natural=True)
            modes["permuted"] = dict(modes["natural"], message_mode="permuted", reused_natural=True)
        out["workers"][str(worker)] = modes
    return out


def codebook(p, pm, pv, sc):
    """Read out a causal, partner-level codebook from the final tables.

    For each sender-selected token and target, test both local site
    permutations.  This avoids counting a token as semantic merely because it
    works for one private layout.  The readout is descriptive and is never fed
    back into training.
    """
    cap = design.capacity(sc)
    sender_tokens = np.zeros((2, design.WORKERS), dtype=np.int64)
    semantic = np.zeros((2, design.WORKERS), dtype=np.float64)
    worker_best = np.zeros((2, design.WORKERS), dtype=np.int64)
    for goal in range(2):
        for worker in range(design.WORKERS):
            if pv == "visible":
                # A visible sender may intentionally choose a
                # partner-specific token; the fixed arm only trains row 0.
                token = int(p["sender_logits_visible"][goal * design.WORKERS + worker].argmax())
            else:
                token = int(p["sender_logits_hidden"][goal].argmax())
            sender_tokens[goal, worker] = token
            scores = []
            for candidate in range(design.ALPHABET_SIZE):
                good = []
                for local in (0, 1):
                    probs = policy.softmax(p["worker_logits"][worker, candidate, 1, local, cap])
                    action = int(probs.argmax())
                    site_type = local if action == 1 else (1 - local if action == 2 else -1)
                    good.append(float(site_type == goal))
                scores.append(np.mean(good))
            worker_best[goal, worker] = int(np.argmax(scores))
            semantic[goal, worker] = float(scores[token])
    agreement = []
    for goal in range(2):
        x = sender_tokens[goal]
        counts = np.bincount(x, minlength=design.ALPHABET_SIZE)
        agreement.append(float(counts.max() / design.WORKERS))
    return {"sender_token_by_goal_worker": sender_tokens.tolist(),
            "semantic_success_by_goal_worker": semantic.tolist(),
            "best_token_by_goal_worker": worker_best.tolist(),
            "sender_token_agreement": float(np.mean(agreement)),
            "semantic_success_mean": float(semantic.mean()),
            "semantic_success_min": float(semantic.min())}


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
