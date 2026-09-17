"""Train, evaluate, freeze and replay the community-merge study."""
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
    plan = {
        "schema": "community_merge_study_v1",
        "prepared_sha256": sha(out / "prepared.json"),
        "sources": sources,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "config": cfg,
    }
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
    return np.asarray(rewards, dtype=np.float64)


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


def _sender_update(grad, tr, ep, mask, params_kind, params, visibility, selected):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return
    meaning = np.asarray(ep["goal_meaning"], dtype=np.int64)[idx]
    partner = np.asarray(ep["partner_id"], dtype=np.int64)[idx]
    probs = np.asarray(tr["sender_probs"], dtype=np.float64)[idx]
    selected = np.asarray(selected, dtype=np.int64)[idx]
    context = meaning if visibility == "hidden" else partner * design.MEANINGS + meaning
    advantage = center(np.asarray(tr["rewards"], dtype=np.float64)[idx], context)
    beta = design.entropy_coefficient(int(ep["update"]))
    for slot in range(design.MESSAGE_LENGTH):
        one = np.zeros_like(probs[:, slot])
        one[np.arange(len(idx)), selected[:, slot]] = 1.0
        d = -(advantage[:, None] * (one - probs[:, slot])) / len(ep["goal"]) - beta * entropy_grad(probs[:, slot]) / len(ep["goal"])
        if params_kind == "hidden":
            np.add.at(grad["sender_hidden"][:, slot, :], meaning, d)
        else:
            np.add.at(grad["sender_visible"][:, :, slot, :], (partner, meaning), d)


def _receiver_update(grad, tr, ep, mask, params_kind, visibility):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return
    state = np.asarray(tr["message_state"], dtype=np.int64)[idx]
    partner = np.asarray(ep["partner_id"], dtype=np.int64)[idx]
    scene = np.asarray(ep["scene_meanings"], dtype=np.int64)[idx]
    probs = np.asarray(tr["receiver_probs"], dtype=np.float64)[idx]
    action = np.asarray(tr["actions"], dtype=np.int64)[idx]
    context = state if visibility == "hidden" else partner * design.MESSAGE_STATE_COUNT + state
    advantage = center(np.asarray(tr["rewards"], dtype=np.float64)[idx], context)
    beta = design.entropy_coefficient(int(ep["update"]))
    one = np.zeros_like(probs)
    one[np.arange(len(idx)), action] = 1.0
    d = -(advantage[:, None] * (one - probs)) / len(ep["goal"]) - beta * entropy_grad(probs) / len(ep["goal"])
    if params_kind == "hidden":
        for position in range(design.SCENE_SIZE):
            np.add.at(grad["receiver_hidden"], (state, scene[:, position]), d[:, position])
    else:
        for position in range(design.SCENE_SIZE):
            np.add.at(grad["receiver_visible"], (partner, state, scene[:, position]), d[:, position])


def parent_gradient(params, ep, tr):
    grad = policy.empty_grad(params)
    full = np.ones(len(ep["goal"]), dtype=bool)
    _sender_update(grad, tr, ep, full, "hidden", params, "hidden", tr["selected_canonical"])
    _receiver_update(grad, tr, ep, full, "hidden", "hidden")
    return grad


def child_gradients(communities, fresh, ep, tr, visibility, adaptation):
    fresh_grad = policy.empty_grad(fresh)
    community_grads = [policy.empty_grad(params) for params in communities]
    sender_fresh = np.asarray(tr["sender_fresh"], dtype=bool)
    receiver_fresh = np.asarray(tr["receiver_fresh"], dtype=bool)
    if np.any(sender_fresh):
        _sender_update(fresh_grad, tr, ep, sender_fresh, "hidden" if visibility == "hidden" else "visible", fresh, visibility, tr["selected_canonical"])
    if np.any(receiver_fresh):
        _receiver_update(fresh_grad, tr, ep, receiver_fresh, "hidden" if visibility == "hidden" else "visible", visibility)
    if adaptation == "coadapt":
        community = np.asarray(ep["community_id"], dtype=np.int64)
        incumbent_sender = ~sender_fresh
        incumbent_receiver = ~receiver_fresh
        for cid in range(design.COMMUNITIES):
            smask = incumbent_sender & (community == cid)
            rmask = incumbent_receiver & (community == cid)
            if np.any(smask):
                _sender_update(community_grads[cid], tr, ep, smask, "hidden", communities[cid], "hidden", tr["selected_canonical"])
            if np.any(rmask):
                _receiver_update(community_grads[cid], tr, ep, rmask, "hidden", "hidden")
    return fresh_grad, community_grads


def update_params(params, grad):
    norm = float(np.sqrt(sum(float((value * value).sum()) for value in grad.values())))
    scale = min(1.0, 5.0 / max(norm, 1e-12))
    for key in params:
        params[key] -= design.LEARNING_RATE * scale * grad[key]
    return norm, scale


def _metric(trajectory):
    values = np.asarray(trajectory["team_return"], dtype=np.float64)
    return {
        "episodes": int(len(values)),
        "team_return_mean": float(values.mean()) if len(values) else None,
        "team_return_sd": float(values.std()) if len(values) else None,
        "positive_episode_rate": float((values > 0).mean()) if len(values) else None,
        "oracle_return_mean": float(environment.oracle_return()),
    }


def _stream_hashes(ep):
    return {
        "scene_sha256": design.array_sha(ep["scene_meanings"]),
        "goal_sha256": design.array_sha(ep["goal_meaning"]),
        "partner_sha256": design.array_sha(ep["partner_id"]),
        "role_sha256": design.array_sha(ep["fresh_role"]),
        "message_uniform_sha256": design.array_sha(ep["message_uniforms"]),
        "action_uniform_sha256": design.array_sha(ep["action_uniforms"]),
    }


def _eval_episode(seed, goal_kind, support, combo, value, role):
    return design.balanced_eval_stream(seed + design.EVAL_SEED_OFFSET, design.PARTNERS * 2 * 512, goal_kind, support, combo, value, role)


def _codebooks(communities, fresh, population, visibility):
    community_codes = []
    for cid, cp in enumerate(communities):
        perm = design.surface_permutation(population, cid)
        community_codes.append([policy.sender_sequence(cp, meaning, visibility="hidden", external=True, permutation=perm).tolist() for meaning in range(design.MEANINGS)])
    fresh_codes = []
    for partner in range(design.PARTNERS):
        fresh_codes.append([policy.sender_sequence(fresh, meaning, partner_id=partner, visibility=visibility).tolist() for meaning in range(design.MEANINGS)])
    pairs = []
    for meaning in range(design.MEANINGS):
        seqs = [tuple(fresh_codes[p][meaning]) for p in range(design.PARTNERS)]
        pairs.append(sum(seqs[i] == seqs[j] for i in range(len(seqs)) for j in range(i + 1, len(seqs))) / 6.0)
    community_hamming = np.mean([
        sum(a != b for a, b in zip(community_codes[0][meaning], community_codes[1][meaning]))
        for meaning in range(design.MEANINGS)
    ])
    return {
        "community_external_codebooks": community_codes,
        "fresh_external_codebooks": fresh_codes,
        "fresh_sender_consistency": float(np.mean(pairs)),
        "community_codebook_hamming": float(community_hamming),
    }


def evaluate_parent(params, seed, population, community):
    combo = design.heldout_combo(seed)
    value = design.heldout_value(seed)
    result = {"community": int(community), "population": population}
    for goal_kind in ("all", "seen"):
        ep = _eval_episode(seed, goal_kind, "full", combo, value, "alternating")
        tr = environment.rollout([params], None, ep, population, "hidden", "alternating", sample=False)
        result[goal_kind] = {
            "natural": _metric(tr),
            "permuted": _metric(environment.rollout([params], None, ep, population, "hidden", "alternating", sample=False, message_mode="permuted")),
            "silent": _metric(environment.rollout([params], None, ep, population, "hidden", "alternating", sample=False, channel="silent")),
        }
    result.update(_codebooks([params, params], params, population, "hidden"))
    return result


def evaluate_child(communities, fresh, seed, population, visibility, support, role):
    combo = design.heldout_combo(seed)
    value = design.heldout_value(seed)
    result = {"population": population, "visibility": visibility, "support": support, "role": role}
    for goal_kind in ("all", "seen", "heldout_combo", "heldout_value"):
        ep = _eval_episode(seed, goal_kind, support, combo, value, role)
        natural = environment.rollout(communities, fresh, ep, population, visibility, role, sample=False)
        permuted = environment.rollout(communities, fresh, ep, population, visibility, role, sample=False, message_mode="permuted")
        silent = environment.rollout(communities, fresh, ep, population, visibility, role, sample=False, channel="silent")
        recombined = environment.rollout(communities, fresh, ep, population, visibility, role, sample=False, message_override=environment.recombination_messages(communities, fresh, ep, population, visibility))
        modes = {"natural": _metric(natural), "permuted": _metric(permuted), "silent": _metric(silent), "recombined": _metric(recombined)}
        modes["by_role"] = {
            "fresh_sender": _metric_subset(natural, np.asarray(ep["fresh_role"]) == 0),
            "fresh_receiver": _metric_subset(natural, np.asarray(ep["fresh_role"]) == 1),
        }
        result[goal_kind] = modes
    result.update(_codebooks(communities, fresh, population, visibility))
    return result


def _metric_subset(trajectory, mask):
    values = np.asarray(trajectory["team_return"])[np.asarray(mask, dtype=bool)]
    return {
        "episodes": int(len(values)),
        "team_return_mean": float(values.mean()) if len(values) else None,
        "team_return_sd": float(values.std()) if len(values) else None,
        "positive_episode_rate": float((values > 0).mean()) if len(values) else None,
        "oracle_return_mean": float(environment.oracle_return()),
    }


def parent_path(execution, seed, population, community, updates=None):
    updates = design.PARENT_UPDATES if updates is None else int(updates)
    return Path(execution) / "parents" / f"seed_{seed}_{population}_community_{community}" / f"checkpoint_{updates:04d}.npz"


def save_checkpoint(path, communities, fresh, update):
    payload = {"update": np.array(update, dtype=np.int64)}
    for cid, params in enumerate(communities):
        for key, value in params.items():
            payload[f"community{cid}_{key}"] = np.asarray(value)
    if fresh is not None:
        for key, value in fresh.items():
            payload[f"fresh_{key}"] = np.asarray(value)
    np.savez_compressed(path, **payload)
    return sha(path)


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as data:
        communities = []
        for cid in range(design.COMMUNITIES):
            communities.append({key: np.asarray(data[f"community{cid}_{key}"]).copy() for key in ("sender_hidden", "sender_visible", "receiver_hidden", "receiver_visible")})
        fresh_keys = [key for key in data.files if key.startswith("fresh_")]
        fresh = None
        if fresh_keys:
            fresh = {key: np.asarray(data[f"fresh_{key}"]).copy() for key in ("sender_hidden", "sender_visible", "receiver_hidden", "receiver_visible")}
    return communities, fresh


def train_parent(seed, population, community, execution, updates=None):
    updates = design.PARENT_UPDATES if updates is None else int(updates)
    run = Path(execution) / "parents" / f"seed_{seed}_{population}_community_{community}"
    run.mkdir(parents=True, exist_ok=False)
    # Use the same frozen initialization and paired stream for both community
    # copies.  This makes ``aligned`` a genuine shared-code baseline; the
    # conflict arm then differs only by the recorded external surface map.
    params = policy.make_policy(seed)
    initial_hash = policy.parameter_hash(params)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", [params, params], None, 0)
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, support="full", role="alternating", update=update)
        ep["partner_id"] = np.full(len(ep["goal"]), int(community * design.PARTNERS_PER_COMMUNITY), dtype=np.int8)
        ep["community_id"] = np.full(len(ep["goal"]), int(community), dtype=np.int8)
        tr = environment.rollout([params], None, ep, population, "hidden", "alternating", sample=True)
        grad = parent_gradient(params, ep, tr)
        norm, scale = update_params(params, grad)
        row = {"update": update, "seed": seed, "population": population, "community": int(community), **_stream_hashes(ep), "return_mean": float(tr["team_return"].mean()), "gradient_norm": norm, "gradient_clip_scale": scale, "parameter_sha256": policy.parameter_hash(params), "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            row["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", [params, params], None, update)
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result = {
        "kind": "parent",
        "seed": seed,
        "population": population,
        "community": int(community),
        "updates": updates,
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": policy.parameter_hash(params),
        "final": evaluate_parent(params, seed, population, community),
        "heldout_combo": design.heldout_combo(seed),
        "heldout_value": design.heldout_value(seed),
        "checkpoints": checkpoints,
        "training_log_sha256": sha(log),
        "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"),
        "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz"),
    }
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def train_child(seed, condition, execution, parent_checkpoints, updates=None):
    population, visibility, adaptation, support, role = design.parse_child_condition(condition)
    updates = design.CHILD_UPDATES if updates is None else int(updates)
    run = Path(execution) / "children" / f"seed_{seed}_{condition}"
    run.mkdir(parents=True, exist_ok=False)
    communities = []
    for path in parent_checkpoints:
        parent, _ = load_checkpoint(path)
        communities.append(parent[0])
    fresh = policy.make_policy(seed + 191000, visible=visibility == "visible")
    initial_hash = policy.parameter_hash(fresh)
    parent_hashes = [policy.parameter_hash(cp) for cp in communities]
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", communities, fresh, 0)
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, support=support, role=role, update=update)
        if support == "heldout_combo":
            design.require(not np.any(np.asarray(ep["goal_meaning"]) == design.heldout_combo(seed)), f"heldout combo leakage {condition}/{update}")
        elif support == "heldout_value":
            design.require(not np.any((np.asarray(ep["goal_meaning"]) // design.VALUES) == design.heldout_value(seed)), f"heldout value leakage {condition}/{update}")
        tr = environment.rollout(communities, fresh, ep, population, visibility, role, sample=True)
        fg, cgs = child_gradients(communities, fresh, ep, tr, visibility, adaptation)
        norms = [float(np.sqrt(sum(float((v * v).sum()) for v in fg.values())))] + [float(np.sqrt(sum(float((v * v).sum()) for v in g.values()))) for g in cgs]
        total_norm = float(np.sqrt(sum(x * x for x in norms)))
        scale = min(1.0, 5.0 / max(total_norm, 1e-12))
        for key in fresh:
            fresh[key] -= design.LEARNING_RATE * scale * fg[key]
        if adaptation == "coadapt":
            for params, grad in zip(communities, cgs):
                for key in params:
                    params[key] -= design.LEARNING_RATE * scale * grad[key]
        row = {"update": update, "seed": seed, "condition": condition, "population": population, "visibility": visibility, "adaptation": adaptation, "support": support, "role": role, **_stream_hashes(ep), "return_mean": float(tr["team_return"].mean()), "fresh_sender_count": int(np.asarray(tr["sender_fresh"]).sum()), "fresh_receiver_count": int(np.asarray(tr["receiver_fresh"]).sum()), "gradient_norm": total_norm, "gradient_clip_scale": scale, "parameter_sha256": policy.parameter_hash(fresh), "community_parameter_sha256": [policy.parameter_hash(p) for p in communities], "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            row["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", communities, fresh, update)
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result = {
        "kind": "child",
        "seed": seed,
        "condition": condition,
        "population": population,
        "visibility": visibility,
        "adaptation": adaptation,
        "support": support,
        "role": role,
        "heldout_combo": design.heldout_combo(seed),
        "heldout_value": design.heldout_value(seed),
        "updates": updates,
        "parent_parameter_sha256": parent_hashes,
        "initial_parameter_sha256": initial_hash,
        "final_parameter_sha256": policy.parameter_hash(fresh),
        "final_community_parameter_sha256": [policy.parameter_hash(p) for p in communities],
        "final": evaluate_child(communities, fresh, seed, population, visibility, support, role),
        "checkpoints": checkpoints,
        "training_log_sha256": sha(log),
        "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"),
        "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz"),
    }
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
    design.require(set(seeds) <= set(design.SEEDS) and set(conditions) <= set(design.CHILD_CONDITIONS), "invalid subset")
    populations = sorted({design.parse_child_condition(c)[0] for c in conditions})
    parents = []
    for seed in seeds:
        for population in populations:
            for community in range(design.COMMUNITIES):
                parents.append(train_parent(seed, population, community, execution, updates))
    children = []
    for seed in seeds:
        for condition in conditions:
            population = design.parse_child_condition(condition)[0]
            parent_paths = [parent_path(execution, seed, population, community, updates) for community in range(design.COMMUNITIES)]
            children.append(train_child(seed, condition, execution, parent_paths, updates))
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
