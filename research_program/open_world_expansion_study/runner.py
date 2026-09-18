"""Train, evaluate, freeze and replay the learned-routing study."""
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
    rels = ("__init__.py", "README.md", "design.py", "policy.py", "environment.py", "runner.py",
            "aggregate.py", "audit.py", "plan.md", "tests/test_game.py")
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
    plan = {"schema": "open_world_expansion_study_v1", "prepared_sha256": sha(out / "prepared.json"),
            "sources": sources, "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "config": cfg}
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"),
                                                   "prepared_sha256": sha(out / "prepared.json")}, indent=2) + "\n")
    verify(out)
    return {"status": "prepared", "out": str(out), "plan_sha256": sha(out / "plan.json"),
            "prepared_sha256": sha(out / "prepared.json")}


def verify(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text())
    cfg = json.loads((out / "prepared.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    design.require(sha(out / "plan.json") == freeze["plan_sha256"], "plan hash mismatch")
    design.require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"],
                   "prepared hash mismatch")
    design.require(plan["config"] == cfg and plan["sources"] == source_hashes(), "source/config changed")
    for rel, digest in plan["sources"].items():
        design.require(sha(out / "source_snapshot" / rel) == digest, "source snapshot changed " + rel)
    return plan, cfg


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


def _sender_update(grad, tr, ep, mask, params, visibility, selected):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return
    architecture = params["architecture"]
    meaning = np.asarray(ep["goal_meaning"], dtype=np.int64)[idx]
    partner = np.asarray(ep["partner_id"], dtype=np.int64)[idx]
    probs = np.asarray(tr["sender_probs"], dtype=np.float64)[idx]
    selected = np.asarray(selected, dtype=np.int64)[idx]
    context = meaning if visibility == "hidden" else partner * design.MEANINGS + meaning
    advantage = center(np.asarray(tr["rewards"], dtype=np.float64)[idx], context)
    beta = design.entropy_coefficient(int(ep["update"]))
    attrs = design.attrs_from_meaning(meaning)
    dlogits = np.zeros_like(probs)
    for slot in range(design.MESSAGE_LENGTH):
        one = np.zeros_like(probs[:, slot])
        one[np.arange(len(idx)), selected[:, slot]] = 1.0
        dlogits[:, slot] = (-(advantage[:, None] * (one - probs[:, slot]))
                            - beta * entropy_grad(probs[:, slot])) / len(ep["goal"])

    groups = [(None, np.arange(len(idx), dtype=np.int64))]
    if visibility != "hidden":
        groups = [(partner_id, np.flatnonzero(partner == partner_id)) for partner_id in range(design.PARTNERS)]
    for owner, local in groups:
        if not len(local):
            continue
        if architecture == "holistic":
            if visibility == "hidden":
                np.add.at(grad["sender_hidden"], meaning[local], dlogits[local])
            else:
                np.add.at(grad["sender_visible"], (partner[local], meaning[local]), dlogits[local])
            continue
        if architecture == "factorized":
            for slot in range(design.MESSAGE_LENGTH):
                if visibility == "hidden":
                    np.add.at(grad["sender_hidden"], (slot, attrs[local, slot]), dlogits[local, slot])
                else:
                    np.add.at(grad["sender_visible"], (partner[local], slot, attrs[local, slot]), dlogits[local, slot])
            continue

        if visibility == "hidden":
            factors = params["sender_hidden"]
            route_raw = params["route_hidden"] if architecture == "tied_routed" else params["sender_route_hidden"]
            routes = policy.softmax(design.ROUTE_TEMPERATURE * route_raw, axis=-1)
            grad_factors = grad["sender_hidden"]
            grad_routes = grad["route_hidden"] if architecture == "tied_routed" else grad["sender_route_hidden"]
        else:
            factors = params["sender_visible"][owner]
            route_key = "route_visible" if architecture == "tied_routed" else "sender_route_visible"
            routes = policy.softmax(design.ROUTE_TEMPERATURE * params[route_key][owner], axis=-1)
            grad_factors = grad["sender_visible"][owner]
            grad_routes = grad[route_key][owner]
        for slot in range(design.MESSAGE_LENGTH):
            d = dlogits[local, slot]
            route_derivative = np.zeros((len(local), design.ATTRIBUTES), dtype=np.float64)
            for attr in range(design.ATTRIBUTES):
                values = attrs[local, attr]
                np.add.at(grad_factors, (attr, values), routes[slot, attr] * d)
                route_derivative[:, attr] = np.sum(d * factors[attr, values], axis=-1)
            # Apply the softmax Jacobian per example, then sum examples.
            grad_routes[slot] = design.ROUTE_TEMPERATURE * np.sum(routes[slot] *
                                       (route_derivative - np.sum(route_derivative * routes[slot], axis=1, keepdims=True)), axis=0)


def _receiver_update(grad, tr, ep, mask, params, visibility):
    idx = np.flatnonzero(mask)
    if not len(idx):
        return
    architecture = params["architecture"]
    state = np.asarray(tr["message_state"], dtype=np.int64)[idx]
    partner = np.asarray(ep["partner_id"], dtype=np.int64)[idx]
    scene = np.asarray(ep["scene_meanings"], dtype=np.int64)[idx]
    scene_attrs = design.attrs_from_meaning(scene)
    probs = np.asarray(tr["receiver_probs"], dtype=np.float64)[idx]
    action = np.asarray(tr["actions"], dtype=np.int64)[idx]
    context = state if visibility == "hidden" else partner * design.MESSAGE_STATE_COUNT + state
    advantage = center(np.asarray(tr["rewards"], dtype=np.float64)[idx], context)
    beta = design.entropy_coefficient(int(ep["update"]))
    one = np.zeros_like(probs)
    one[np.arange(len(idx)), action] = 1.0
    dscores = (-(advantage[:, None] * (one - probs)) - beta * entropy_grad(probs)) / len(ep["goal"])

    groups = [(None, np.arange(len(idx), dtype=np.int64))]
    if visibility != "hidden":
        groups = [(partner_id, np.flatnonzero(partner == partner_id)) for partner_id in range(design.PARTNERS)]
    tokens = design.decode_message_state(state)
    for owner, local in groups:
        if not len(local):
            continue
        if architecture == "holistic":
            if visibility == "hidden":
                for pos in range(design.SCENE_SIZE):
                    np.add.at(grad["receiver_hidden"], (state[local], scene[local, pos]), dscores[local, pos])
            else:
                for pos in range(design.SCENE_SIZE):
                    np.add.at(grad["receiver_visible"], (partner[local], state[local], scene[local, pos]), dscores[local, pos])
            continue
        if architecture == "factorized":
            for slot in range(design.MESSAGE_LENGTH):
                tok = tokens[local, slot]
                valid = tok >= 0
                if not np.any(valid):
                    continue
                for pos in range(design.SCENE_SIZE):
                    values = scene_attrs[local, pos, slot]
                    if visibility == "hidden":
                        np.add.at(grad["receiver_hidden"], (slot, tok[valid], values[valid]), dscores[local[valid], pos])
                    else:
                        np.add.at(grad["receiver_visible"],
                                  (partner[local[valid]], slot, tok[valid], values[valid]), dscores[local[valid], pos])
            continue

        if visibility == "hidden":
            table = params["receiver_hidden"]
            route_raw = params["route_hidden"] if architecture == "tied_routed" else params["receiver_route_hidden"]
            routes = policy.softmax(design.ROUTE_TEMPERATURE * route_raw, axis=-1)
            grad_table = grad["receiver_hidden"]
            grad_routes = grad["route_hidden"] if architecture == "tied_routed" else grad["receiver_route_hidden"]
        else:
            table = params["receiver_visible"][owner]
            route_key = "route_visible" if architecture == "tied_routed" else "receiver_route_visible"
            routes = policy.softmax(design.ROUTE_TEMPERATURE * params[route_key][owner], axis=-1)
            grad_table = grad["receiver_visible"][owner]
            grad_routes = grad[route_key][owner]
        for slot in range(design.MESSAGE_LENGTH):
            tok = tokens[local, slot]
            valid = tok >= 0
            if not np.any(valid):
                continue
            local_valid = local[valid]
            tok_valid = tok[valid]
            route_derivative = np.zeros((len(local_valid), design.ATTRIBUTES), dtype=np.float64)
            for attr in range(design.ATTRIBUTES):
                for pos in range(design.SCENE_SIZE):
                    values = scene_attrs[local_valid, pos, attr]
                    d = dscores[local_valid, pos]
                    np.add.at(grad_table, (slot, tok_valid, attr, values), routes[slot, attr] * d)
                    route_derivative[:, attr] += d * table[slot, tok_valid, attr, values]
            grad_routes[slot] += design.ROUTE_TEMPERATURE * np.sum(routes[slot] *
                                        (route_derivative - np.sum(route_derivative * routes[slot], axis=1, keepdims=True)), axis=0)


def parent_gradient(params, ep, tr):
    grad = policy.empty_grad(params)
    mask = np.ones(len(ep["goal"]), dtype=bool)
    _sender_update(grad, tr, ep, mask, params, "hidden", tr["selected_canonical"])
    _receiver_update(grad, tr, ep, mask, params, "hidden")
    return grad


def child_gradient(fresh, ep, tr, visibility):
    grad = policy.empty_grad(fresh)
    sender = np.asarray(tr["sender_fresh"], dtype=bool)
    receiver = np.asarray(tr["receiver_fresh"], dtype=bool)
    if np.any(sender):
        _sender_update(grad, tr, ep, sender, fresh, visibility, tr["selected_canonical"])
    if np.any(receiver):
        _receiver_update(grad, tr, ep, receiver, fresh, visibility)
    return grad


def update_params(params, grad):
    norm = float(np.sqrt(sum(float((value * value).sum()) for value in grad.values())))
    scale = min(1.0, 5.0 / max(norm, 1e-12))
    for key in grad:
        params[key] -= design.LEARNING_RATE * scale * grad[key]
    return norm, scale


def _metric(tr):
    values = np.asarray(tr["team_return"], dtype=np.float64)
    return {"episodes": int(len(values)), "team_return_mean": float(values.mean()),
            "team_return_sd": float(values.std()), "positive_episode_rate": float((values > 0).mean()),
            "oracle_return_mean": float(environment.oracle_return())}


def _stream_hashes(ep):
    return {"scene_sha256": design.array_sha(ep["scene_meanings"]),
            "goal_sha256": design.array_sha(ep["goal_meaning"]),
            "partner_sha256": design.array_sha(ep["partner_id"]),
            "role_sha256": design.array_sha(ep["fresh_role"]),
            "message_uniform_sha256": design.array_sha(ep["message_uniforms"]),
            "action_uniform_sha256": design.array_sha(ep["action_uniforms"])}


def _eval_episode(seed, goal_kind, support, combo):
    return design.balanced_eval_stream(seed + design.EVAL_SEED_OFFSET,
                                       design.PARTNERS * 2 * 512,
                                       goal_kind, support, combo)


def _route_summary(params, side, visibility):
    route = policy.routing_probabilities(params, side=side, visibility=visibility)
    if route is None:
        return None
    identity = float(np.mean([route[0, 0], route[1, 1]]))
    swapped = float(np.mean([route[0, 1], route[1, 0]]))
    entropy = float(-(route * np.log(np.maximum(route, 1e-300))).sum(axis=-1).mean())
    return {"matrix": route.tolist(), "best_alignment": max(identity, swapped), "identity_alignment": identity,
            "swap_alignment": swapped, "mean_entropy": entropy}


def _codebooks(communities, fresh, architecture, population, visibility):
    community_codes = []
    for cid, params in enumerate(communities):
        perm = design.surface_permutation(population, cid)
        community_codes.append([policy.sender_sequence(params, meaning, visibility="hidden", external=True,
                                                       permutation=perm).tolist() for meaning in range(design.MEANINGS)])
    fresh_codes = []
    for partner in range(design.PARTNERS):
        fresh_codes.append([policy.sender_sequence(fresh, meaning, partner_id=partner, visibility=visibility).tolist()
                            for meaning in range(design.MEANINGS)])
    consistency = []
    for meaning in range(design.MEANINGS):
        sequences = [tuple(fresh_codes[p][meaning]) for p in range(design.PARTNERS)]
        consistency.append(sum(sequences[i] == sequences[j] for i in range(len(sequences))
                               for j in range(i + 1, len(sequences))) / 6.0)
    def category_consistency(indices):
        values = [consistency[int(i)] for i in indices]
        return float(np.mean(values)) if values else None

    result = {"community_external_codebooks": community_codes, "fresh_external_codebooks": fresh_codes,
              "fresh_sender_consistency": float(np.mean(consistency)),
              "fresh_sender_consistency_old": category_consistency(range(design.OLD_MEANINGS)),
              "fresh_sender_consistency_new_single": category_consistency(
                  [m for m in range(design.MEANINGS) if design.is_new_single(m)]),
              "fresh_sender_consistency_new_double": category_consistency([design.NEW_VALUE * design.VALUES + design.NEW_VALUE]),
              "community_codebook_hamming": float(np.mean([
                  sum(a != b for a, b in zip(community_codes[0][m], community_codes[1][m]))
                  for m in range(design.MEANINGS)]))}
    if architecture == "factorized":
        sender = fresh["sender_hidden"] if visibility == "hidden" else fresh["sender_visible"][0]
        result["factor_slot_unique_fraction"] = float(np.mean([
            len(set(sender[slot].argmax(axis=-1).tolist())) / design.VALUES
            for slot in range(design.MESSAGE_LENGTH)]))
    else:
        result["factor_slot_unique_fraction"] = None
    if architecture in ("routed", "tied_routed"):
        result["sender_route"] = _route_summary(fresh, "sender", visibility)
        result["receiver_route"] = _route_summary(fresh, "receiver", visibility)
    else:
        result["sender_route"] = None
        result["receiver_route"] = None
    return result


def evaluate_parent(params, seed, architecture, population, community):
    combo = design.heldout_combo(seed)
    result = {"architecture": architecture, "population": population, "community": community}
    for kind in design.EVAL_KINDS:
        support = "old_combo" if kind in ("old_seen", "old_combo") else "expanded_alternating"
        ep = _eval_episode(seed, kind, support, combo)
        natural = environment.rollout([params], None, ep, architecture, population, "hidden", sample=False)
        result[kind] = {"natural": _metric(natural),
                        "permuted": _metric(environment.rollout([params], None, ep, architecture, population,
                                                                 "hidden", sample=False, message_mode="permuted")),
                        "silent": _metric(environment.rollout([params], None, ep, architecture, population,
                                                               "hidden", sample=False, channel="silent"))}
    result.update(_codebooks([params, params], params, architecture, population, "hidden"))
    return result


def _metric_subset(tr, mask):
    values = np.asarray(tr["team_return"])[np.asarray(mask, dtype=bool)]
    return {"episodes": int(len(values)), "team_return_mean": float(values.mean()),
            "team_return_sd": float(values.std()), "positive_episode_rate": float((values > 0).mean()),
            "oracle_return_mean": float(environment.oracle_return())}


def evaluate_child(communities, fresh, seed, architecture, population, visibility, support):
    combo = design.heldout_combo(seed)
    result = {"architecture": architecture, "population": population, "visibility": visibility, "support": support}
    for kind in design.EVAL_KINDS:
        ep = _eval_episode(seed, kind, support, combo)
        natural = environment.rollout(communities, fresh, ep, architecture, population, visibility, sample=False)
        result[kind] = {
            "natural": _metric(natural),
            "permuted": _metric(environment.rollout(communities, fresh, ep, architecture, population, visibility,
                                                     sample=False, message_mode="permuted")),
            "silent": _metric(environment.rollout(communities, fresh, ep, architecture, population, visibility,
                                                   sample=False, channel="silent")),
            "recombined": _metric(environment.rollout(
                communities, fresh, ep, architecture, population, visibility, sample=False,
                message_override=environment.recombination_messages(communities, fresh, ep, architecture,
                                                                     population, visibility))),
            "fresh_sender": _metric_subset(natural, np.isin(np.asarray(ep["fresh_role"]), (0, 2))),
            "fresh_receiver": _metric_subset(natural, np.isin(np.asarray(ep["fresh_role"]), (1, 2))),
        }
    result.update(_codebooks(communities, fresh, architecture, population, visibility))
    return result


def parent_path(execution, seed, architecture, population, community, updates=None):
    updates = design.PARENT_UPDATES if updates is None else int(updates)
    return Path(execution) / "parents" / f"seed_{seed}_{architecture}_{population}_community_{community}" / f"checkpoint_{updates:04d}.npz"


def save_checkpoint(path, communities, fresh, update):
    payload = {"update": np.array(update, dtype=np.int64)}
    for cid, params in enumerate(communities):
        for key, value in params.items():
            if key != "architecture":
                payload[f"community{cid}_{key}"] = np.asarray(value)
    if fresh is not None:
        for key, value in fresh.items():
            if key != "architecture":
                payload[f"fresh_{key}"] = np.asarray(value)
    np.savez_compressed(path, **payload)
    return sha(path)


def load_checkpoint(path, architecture):
    with np.load(path, allow_pickle=False) as data:
        communities = []
        for cid in range(design.COMMUNITIES):
            prefix = f"community{cid}_"
            params = {key[len(prefix):]: np.asarray(data[key]).copy() for key in data.files if key.startswith(prefix)}
            params["architecture"] = architecture
            communities.append(params)
        fresh = None
        fresh_keys = [key for key in data.files if key.startswith("fresh_")]
        if fresh_keys:
            fresh = {key[len("fresh_"):]: np.asarray(data[key]).copy() for key in fresh_keys}
            fresh["architecture"] = architecture
    return communities, fresh


def train_parent(seed, architecture, population, community, execution, updates=None):
    updates = design.PARENT_UPDATES if updates is None else int(updates)
    run = Path(execution) / "parents" / f"seed_{seed}_{architecture}_{population}_community_{community}"
    run.mkdir(parents=True, exist_ok=False)
    params = policy.make_policy(seed, architecture)
    initial = policy.parameter_hash(params)
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", [params, params], None, 0)
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, support="old_world", update=update)
        ep["partner_id"] = np.full(len(ep["goal"]), community * design.PARTNERS_PER_COMMUNITY, dtype=np.int8)
        ep["community_id"] = np.full(len(ep["goal"]), community, dtype=np.int8)
        tr = environment.rollout([params], None, ep, architecture, population, "hidden")
        grad = parent_gradient(params, ep, tr)
        norm, scale = update_params(params, grad)
        row = {"update": update, "seed": seed, "architecture": architecture, "population": population,
               "community": community, **_stream_hashes(ep), "return_mean": float(tr["team_return"].mean()),
               "gradient_norm": norm, "gradient_clip_scale": scale,
               "parameter_sha256": policy.parameter_hash(params),
               "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            row["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", [params, params], None, update)
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result = {"kind": "parent", "seed": seed, "architecture": architecture, "population": population,
              "community": community, "updates": updates, "initial_parameter_sha256": initial,
              "final_parameter_sha256": policy.parameter_hash(params),
              "final": evaluate_parent(params, seed, architecture, population, community),
              "heldout_combo": design.heldout_combo(seed), "new_value": design.NEW_VALUE,
              "checkpoints": checkpoints, "training_log_sha256": sha(log),
              "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"),
              "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz")}
    (run / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result


def train_child(seed, condition, execution, parent_checkpoints, updates=None):
    architecture, population, visibility, support = design.parse_child_condition(condition)
    updates = design.CHILD_UPDATES if updates is None else int(updates)
    run = Path(execution) / "children" / f"seed_{seed}_{condition}"
    run.mkdir(parents=True, exist_ok=False)
    communities = [load_checkpoint(path, architecture)[0][0] for path in parent_checkpoints]
    fresh = policy.make_policy(seed + 191000, architecture, visible=visibility == "visible")
    initial = policy.parameter_hash(fresh)
    parent_hashes = [policy.parameter_hash(params) for params in communities]
    checkpoints = sorted(set([u for u in design.CHECKPOINTS if u <= updates] + [updates]))
    if 0 in checkpoints:
        save_checkpoint(run / "checkpoint_0000.npz", communities, fresh, 0)
    rows = []
    start = time.perf_counter()
    for update in range(1, updates + 1):
        ep = design.episode_stream(seed, design.BATCH_SIZE, support=support, update=update)
        if support == "old_combo":
            design.require(not np.any(np.asarray(ep["goal_meaning"]) == design.heldout_combo(seed)),
                           f"heldout combo leakage {condition}/{update}")
        tr = environment.rollout(communities, fresh, ep, architecture, population, visibility)
        grad = child_gradient(fresh, ep, tr, visibility)
        norm, scale = update_params(fresh, grad)
        row = {"update": update, "seed": seed, "condition": condition, "architecture": architecture,
               "population": population, "visibility": visibility, "support": support, **_stream_hashes(ep),
               "return_mean": float(tr["team_return"].mean()),
               "fresh_sender_count": int(np.asarray(tr["sender_fresh"]).sum()),
               "fresh_receiver_count": int(np.asarray(tr["receiver_fresh"]).sum()), "gradient_norm": norm,
               "gradient_clip_scale": scale, "parameter_sha256": policy.parameter_hash(fresh),
               "community_parameter_sha256": [policy.parameter_hash(params) for params in communities],
               "elapsed_seconds": time.perf_counter() - start}
        if update in checkpoints:
            row["checkpoint_sha256"] = save_checkpoint(run / f"checkpoint_{update:04d}.npz", communities, fresh, update)
        rows.append(row)
    log = run / "training.jsonl"
    log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result = {"kind": "child", "seed": seed, "condition": condition, "architecture": architecture,
              "population": population, "visibility": visibility, "support": support,
              "heldout_combo": design.heldout_combo(seed), "new_value": design.NEW_VALUE,
              "updates": updates, "parent_parameter_sha256": parent_hashes,
              "initial_parameter_sha256": initial, "final_parameter_sha256": policy.parameter_hash(fresh),
              "final_community_parameter_sha256": [policy.parameter_hash(params) for params in communities],
              "final": evaluate_child(communities, fresh, seed, architecture, population, visibility, support),
              "checkpoints": checkpoints, "training_log_sha256": sha(log),
              "final_checkpoint": str(run / f"checkpoint_{updates:04d}.npz"),
              "final_checkpoint_sha256": sha(run / f"checkpoint_{updates:04d}.npz")}
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
    design.require(set(seeds) <= set(design.SEEDS) and set(conditions) <= set(design.CHILD_CONDITIONS),
                   "invalid subset")
    parents = []
    parent_keys = sorted({design.parse_child_condition(condition)[:2] for condition in conditions})
    for seed in seeds:
        for architecture, population in parent_keys:
            for community in range(design.COMMUNITIES):
                parents.append(train_parent(seed, architecture, population, community, execution, updates))
    children = []
    for seed in seeds:
        for condition in conditions:
            architecture, population, _, _ = design.parse_child_condition(condition)
            paths = [parent_path(execution, seed, architecture, population, community, updates)
                     for community in range(design.COMMUNITIES)]
            children.append(train_child(seed, condition, execution, paths, updates))
            (execution / "progress.json").write_text(json.dumps(
                {"completed_children": len(children), "total_children": len(seeds) * len(conditions),
                 "seeds": list(seeds), "conditions": list(conditions)}, indent=2))
    payload = {"parents": parents, "children": children}
    (execution / "results.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--out", required=True)
    exe = sub.add_parser("execute")
    exe.add_argument("--out", required=True)
    exe.add_argument("--prepared", required=True)
    exe.add_argument("--updates", type=int)
    exe.add_argument("--seeds")
    exe.add_argument("--conditions")
    args = parser.parse_args()
    if args.cmd == "prepare":
        print(json.dumps(prepare(args.out), ensure_ascii=False))
    else:
        seeds = None if args.seeds is None else tuple(int(x) for x in args.seeds.split(",") if x)
        conditions = None if args.conditions is None else tuple(x for x in args.conditions.split(",") if x)
        payload = run_grid(args.prepared, args.out, args.updates, seeds, conditions)
        print(json.dumps({"status": "completed", "parents": len(payload["parents"]),
                          "children": len(payload["children"])}, ensure_ascii=False))
