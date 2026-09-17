"""CPU-only exact policy-gradient capability control; prepare never trains.

Three independent actors, all 17 actions, original triadic D1 rewards.
No Torch/MLX, teacher policy, successful-action mask, or joint-plan targets.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path
import platform
import random
import shutil
import sys
import time

import numpy as np

from research_program.triadic_task import environment as env

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
AGENTS = env.AGENTS
ACTIONS = [env.all_actions(a) for a in AGENTS]
SEEDS = (43101, 43102, 43103, 43104)
DEMAND_SPLIT_SEED = 2026091901
LAYOUT_SPLIT_SEED = 2026091902
MONITOR_SEED = 2026091903
PARTITIONS = ("train", "new_needs", "new_layouts", "new_needs_and_layouts")
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
FEATURES = 54
CONFIG = {
    "seeds": list(SEEDS), "updates": 6000, "batch_size": 256,
    "hidden_sizes": [64, 64], "features": FEATURES, "actions_per_actor": 17,
    "learning_rate": 0.001, "adam_beta1": 0.9, "adam_beta2": 0.999,
    "adam_epsilon": 1e-8, "global_gradient_clip": 5.0,
    "entropy_initial": 0.001, "entropy_zero_after_updates": 1000,
    "entropy_reduction": "mean_over_three_actors", "dtype": "float64",
    "checkpoints": list(CHECKPOINTS), "monitor_worlds_per_partition": 1024,
    "evaluation_batch_size": 1024, "deployment": "independent_argmax_first_index_on_tie",
    "training_objective": "exact_native_expected_reward_plus_fixed_entropy",
    "full_information": True, "require_match": True,
    "candidate_min_full_success_rate_each_heldout_partition_each_seed": 0.99,
    "automatic_symbolic_stage": False,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def json_hash(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json_bytes(value))


def array_sha(value):
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(json_bytes({"shape": list(value.shape), "dtype": value.dtype.str}))
    h.update(value.tobytes())
    return h.hexdigest()


def finite(value, label):
    require(np.isfinite(value).all(), label + " contains a nonfinite value")


def joint_action_indices():
    rows = []
    for i, j in combinations(range(3), 2):
        for site, destination in product(env.SITES, env.DESTINATIONS):
            row = [0, 0, 0]
            for a, b in ((i, j), (j, i)):
                action = {"kind": "transport", "site": site, "destination": destination, "partner": AGENTS[b]}
                row[a] = ACTIONS[a].index(action)
            rows.append(row)
    require(len(rows) == len(set(map(tuple, rows))) == 24, "Structural action support is not24")
    return np.array(rows, dtype=np.int64)


JOINT_ACTIONS = joint_action_indices()


def encode_observations(observations):
    """Only official observations enter actor features; researcher states do not.

    observations is a batch of dictionaries keyed A/B/C. The same 54 columns
    can represent local views: absent needs/materials have known=0 and zero
    attributes. This run uses full-information observations exclusively.
    """
    encoded = np.zeros((len(observations), 3, FEATURES), dtype=np.float64)
    allowed = {"self", "public_site", "private_view_owners", "own_need", "visible_materials",
               "shared_needs", "information_control"}
    categories = {("kind", "wood"): 0, ("kind", "fiber"): 1,
                  ("length", "short"): 2, ("length", "long"): 3}
    for b, views in enumerate(observations):
        require(set(views) == set(AGENTS), "Missing actor observation")
        for i, a in enumerate(AGENTS):
            view = views[a]
            require(set(view) <= allowed and view["self"] == a and view["public_site"] == "S0",
                    "Unexpected observation field, identity, or public site")
            x = encoded[b, i]
            needs = deepcopy(view.get("shared_needs", {}))
            require(set(needs) <= set(AGENTS), "Unknown need owner")
            require(a not in needs or needs[a] == view["own_need"], "Conflicting own need")
            needs[a] = view["own_need"]
            for who, need in needs.items():
                require(set(need) == {"factor", "value", "destinations"}, "Unexpected need field")
                index = categories.get((need["factor"], need["value"]))
                destinations = need["destinations"]
                require(index is not None and destinations and len(set(destinations)) == len(destinations)
                        and set(destinations) <= {"L", "R"}, "Invalid semantic need")
                offset = 7 * AGENTS.index(who)
                x[offset] = 1
                x[offset + 1 + index] = 1
                for destination in destinations:
                    x[offset + 5 + env.DESTINATIONS.index(destination)] = 1
            seen = set()
            for item in view["visible_materials"]:
                require(set(item) == {"site", "kind", "length"} and item["site"] not in seen,
                        "Unexpected/duplicate visible item")
                require(item["site"] in env.SITES and item["kind"] in ("wood", "fiber")
                        and item["length"] in ("short", "long"), "Invalid visible item")
                seen.add(item["site"])
                offset = 21 + 5 * env.SITES.index(item["site"])
                x[offset] = 1
                x[offset + 1 + (item["kind"] == "fiber")] = 1
                x[offset + 3 + (item["length"] == "long")] = 1
            owners = view["private_view_owners"]
            require(set(owners) == {"S1", "S2", "S3"} and set(owners.values()) == set(AGENTS),
                    "Invalid public allocation")
            for site, who in owners.items():
                x[41 + 3 * (env.SITES.index(site) - 1) + AGENTS.index(who)] = 1
            x[50 + i] = 1
            control = view.get("information_control")
            require(control in (None, "full_information"), "Unknown information control")
            x[53] = control == "full_information"
    return encoded


def reward_terms(states):
    """All24 structural configurations, including zero/half/full reward.

    Uses only the native reward's semantic predicates. No maximizing plan or
    witness is computed. Independent tests compare against all4913 settlements.
    """
    result = np.zeros((len(states), 24), dtype=np.float64)
    for b, state in enumerate(states):
        for t, indices in enumerate(JOINT_ACTIONS):
            for i, action_index in enumerate(indices):
                action = ACTIONS[i][action_index]
                if action["kind"] == "transport":
                    material = state.layout[env.SITES.index(action["site"])]
                    destination = env.DESTINATIONS.index(action["destination"])
                    result[b, t] += 0.5 * env.accepts(state.needs[i], material, destination)
    return result


def policy_distribution(logits):
    finite(logits, "logits")
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probabilities = exp / exp.sum(axis=-1, keepdims=True)
    log_probabilities = shifted - np.log(exp.sum(axis=-1, keepdims=True))
    finite(probabilities, "probabilities")
    finite(log_probabilities, "log_probabilities")
    return probabilities, log_probabilities


def expected_reward_and_logit_gradient(probabilities, rewards):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    require(probabilities.ndim == 3 and probabilities.shape[1:] == (3, 17)
            and rewards.shape == (len(probabilities), 24), "Wrong probability/reward shape")
    finite(probabilities, "probabilities"); finite(rewards, "rewards")
    require((probabilities >= 0).all() and np.allclose(probabilities.sum(-1), 1, atol=1e-12, rtol=0)
            and np.isin(rewards, [0, 0.5, 1]).all(), "Invalid policy/native reward")
    joint = np.ones((len(probabilities), 24), dtype=np.float64)
    for i in range(3):
        joint *= probabilities[:, i, JOINT_ACTIONS[:, i]]
    weighted = joint * rewards
    expected = weighted.sum(axis=1)
    gradient = -probabilities * expected[:, None, None]
    for t in range(24):
        for i in range(3):
            gradient[:, i, JOINT_ACTIONS[t, i]] += weighted[:, t]
    return expected, gradient


def exact_statistics(probabilities, rewards):
    expected, _ = expected_reward_and_logit_gradient(probabilities, rewards)
    joint = np.ones((len(probabilities), 24), dtype=np.float64)
    for i in range(3):
        joint *= probabilities[:, i, JOINT_ACTIONS[:, i]]
    return expected, (joint * (rewards == 1)).sum(1), joint.sum(1)


def entropy_and_logit_gradient(probabilities, log_probabilities):
    entropy = -(probabilities * log_probabilities).sum(-1)
    gradient = -probabilities * (log_probabilities + entropy[:, :, None]) / 3
    return entropy.mean(1), gradient


def entropy_coefficient(update):
    require(1 <= update <= CONFIG["updates"], "Invalid update")
    return CONFIG["entropy_initial"] * max(0, 1 - (update - 1) / CONFIG["entropy_zero_after_updates"])


def make_actor(seed):
    rng = np.random.default_rng(seed)
    dimensions = (FEATURES, 64, 64, 17)
    actor = {}
    for layer, (left, right) in enumerate(zip(dimensions, dimensions[1:]), 1):
        actor[f"W{layer}"] = rng.normal(0, math.sqrt(2 / (left + right)), (left, right))
        actor[f"b{layer}"] = np.zeros(right, dtype=np.float64)
    return actor


def actor_forward(actor, x):
    h1 = np.tanh(x @ actor["W1"] + actor["b1"])
    h2 = np.tanh(h1 @ actor["W2"] + actor["b2"])
    logits = h2 @ actor["W3"] + actor["b3"]
    return logits, (x, h1, h2)


def actor_backward(actor, cache, d_logits):
    x, h1, h2 = cache
    gradients = {"W3": h2.T @ d_logits, "b3": d_logits.sum(0)}
    d2 = (d_logits @ actor["W3"].T) * (1 - h2 * h2)
    gradients.update(W2=h1.T @ d2, b2=d2.sum(0))
    d1 = (d2 @ actor["W2"].T) * (1 - h1 * h1)
    gradients.update(W1=x.T @ d1, b1=d1.sum(0))
    return gradients


def make_adam(actors):
    return [{"m": {k: np.zeros_like(v) for k, v in a.items()},
             "v": {k: np.zeros_like(v) for k, v in a.items()}} for a in actors]


def adam_step(actors, gradients, optimizer, update):
    norm = math.sqrt(sum(float((g * g).sum()) for actor in gradients for g in actor.values()))
    require(math.isfinite(norm), "Nonfinite gradient norm")
    scale = min(1.0, CONFIG["global_gradient_clip"] / max(norm, 1e-300))
    b1, b2 = CONFIG["adam_beta1"], CONFIG["adam_beta2"]
    for actor, gradient, opt in zip(actors, gradients, optimizer):
        for key in actor:
            g = gradient[key] * scale
            opt["m"][key] *= b1
            opt["m"][key] += (1-b1) * g
            opt["v"][key] *= b2
            opt["v"][key] += (1-b2) * g * g
            actor[key] -= CONFIG["learning_rate"] * (opt["m"][key] / (1-b1**update)) / (
                np.sqrt(opt["v"][key] / (1-b2**update)) + CONFIG["adam_epsilon"])
            finite(actor[key], "updated parameter")
    return norm, scale


def make_prepared():
    demands = list(env.support())
    require(len(demands) == 996, "Unexpected development support")
    multisets = sorted({tuple(sorted(n)) for n in demands})
    require(len(multisets) == 212, "Unexpected multiset support")
    random.Random(DEMAND_SPLIT_SEED).shuffle(multisets)
    train_multisets, held_multisets = multisets[:159], multisets[159:]
    train_set = set(train_multisets)
    train_needs = [n for n in demands if tuple(sorted(n)) in train_set]
    held_needs = [n for n in demands if tuple(sorted(n)) not in train_set]
    layouts = list(permutations(range(4)))
    random.Random(LAYOUT_SPLIT_SEED).shuffle(layouts)
    assignment = list(permutations((1, 2, 3)))
    parts = {}
    for name, needs, ls in (("train", train_needs, layouts[:18]), ("new_needs", held_needs, layouts[:18]),
                            ("new_layouts", train_needs, layouts[18:]), ("new_needs_and_layouts", held_needs, layouts[18:])):
        count = len(needs) * len(ls) * len(assignment)
        # Different named streams; no outcome-based selection.
        monitor_rng = random.Random(f"{MONITOR_SEED}:{name}")
        parts[name] = {"needs": needs, "layouts": ls, "private_sites": assignment,
                       "world_count": count, "monitor_indices": sorted(monitor_rng.sample(range(count), 1024))}
    require(sum(p["world_count"] for p in parts.values()) == 996 * 24 * 6, "Incomplete Cartesian support")
    return {"schema": "triadic_learning_v1", "config": deepcopy(CONFIG),
            "split_seeds": {"demands": DEMAND_SPLIT_SEED, "layouts": LAYOUT_SPLIT_SEED, "monitors": MONITOR_SEED},
            "train_need_multisets": train_multisets, "heldout_need_multisets": held_multisets,
            "partitions": parts, "actions": ACTIONS, "structural_joint_actions": JOINT_ACTIONS.tolist(),
            "training_state_samples_per_seed": CONFIG["updates"] * CONFIG["batch_size"],
            "training_state_samples_total": len(SEEDS) * CONFIG["updates"] * CONFIG["batch_size"],
            "no_unique_correct_actions_or_plans": True}


def states_for_partition(spec):
    return [env.State(tuple(n), tuple(l), tuple(p)) for n, l, p in product(spec["needs"], spec["layouts"], spec["private_sites"])]


def build_arrays(spec):
    states = states_for_partition(spec)
    require(len(states) == spec["world_count"], "Partition count differs")
    x = np.empty((len(states), 3, FEATURES), dtype=np.float64)
    reward = np.empty((len(states), 24), dtype=np.float64)
    for start in range(0, len(states), 1024):
        chunk = states[start:start+1024]
        observations = [{a: env.observe(s, a, shared_needs=True, full_information=True) for a in AGENTS} for s in chunk]
        x[start:start+len(chunk)] = encode_observations(observations)
        reward[start:start+len(chunk)] = reward_terms(chunk)
    packed = np.array([s.needs + s.layout + s.private_sites for s in states], dtype=np.int16)
    return {"states": states, "x": x, "rewards": reward, "packed_states": packed}


def evaluate(actors, arrays, indices=None, *, save_path=None):
    require(save_path is None or indices is None, "Saved final evaluation must contain full partition")
    if indices is None:
        indices = np.arange(len(arrays["states"]), dtype=np.int64)
    else:
        indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    probabilities = np.empty((n, 3, 17), dtype=np.float64)
    choices = np.empty((n, 3), dtype=np.int16)
    rewards = np.empty(n, dtype=np.float64)
    satisfied = np.empty((n, 3), dtype=bool)
    executed = np.empty((n, 3), dtype=bool)
    expectations, full_probabilities, execution_probabilities = [], [], []
    ties = 0
    for start in range(0, n, CONFIG["evaluation_batch_size"]):
        ids = indices[start:start+CONFIG["evaluation_batch_size"]]
        logits = np.stack([actor_forward(actors[a], arrays["x"][ids, a])[0] for a in range(3)], axis=1)
        probs, _ = policy_distribution(logits)
        greedy = np.argmax(probs, axis=-1)
        ties += int(((probs == probs.max(-1, keepdims=True)).sum(-1) > 1).sum())
        probabilities[start:start+len(ids)] = probs
        choices[start:start+len(ids)] = greedy
        er, ef, ee = exact_statistics(probs, arrays["rewards"][ids])
        expectations.extend(er); full_probabilities.extend(ef); execution_probabilities.extend(ee)
        for k, index in enumerate(ids):
            actions = {a: ACTIONS[i][greedy[k, i]] for i, a in enumerate(AGENTS)}
            outcome = env.settle(arrays["states"][index], actions, require_match=True)
            row = start+k
            rewards[row] = outcome["reward"]
            for i, a in enumerate(AGENTS):
                satisfied[row, i] = outcome["individual_feedback"][a]["own_need_satisfied"]
                executed[row, i] = outcome["individual_feedback"][a]["executed"]
    attempted = (choices != 0).sum(1)
    physically_executed = executed.sum(1)
    result = {"worlds": n, "greedy_reward_sum": float(rewards.sum()), "greedy_reward_mean": float(rewards.mean()),
        "greedy_full_successes": int((rewards == 1).sum()), "greedy_full_success_rate": float((rewards == 1).mean()),
        "greedy_non_full_success_worlds": int((rewards != 1).sum()),
        "greedy_reward_counts": {str(r): int((rewards == r).sum()) for r in (0.0, 0.5, 1.0)},
        "greedy_attempted_transports": int(attempted.sum()), "greedy_executed_transports": int(executed.sum()),
        "greedy_satisfied_agents": int(satisfied.sum()), "greedy_active_count_worlds": {str(k): int((attempted == k).sum()) for k in range(4)},
        "greedy_agent_argmax_ties": ties, "exact_stochastic_expected_reward_mean": float(np.mean(expectations)),
        "exact_stochastic_full_success_probability_mean": float(np.mean(full_probabilities)),
        "exact_stochastic_physical_execution_probability_mean": float(np.mean(execution_probabilities)),
        "greedy_failure_categories": {"all_wait": int((attempted == 0).sum()),
            "single_transport_proposal": int((attempted == 1).sum()), "overload": int((attempted == 3).sum()),
            "two_unmatched": int(((attempted == 2) & (physically_executed == 0)).sum()),
            "matched_no_need_satisfied": int(((physically_executed == 2) & (rewards == 0)).sum()),
            "matched_one_need_satisfied": int((rewards == 0.5).sum())},
        "greedy_executed_pair_worlds": {AGENTS[i]+AGENTS[j]: int((executed[:, i] & executed[:, j]).sum()) for i, j in combinations(range(3), 2)},
        "state_indices_sha256": array_sha(indices), "packed_states_sha256": array_sha(arrays["packed_states"][indices])}
    if save_path is not None:
        require(not Path(save_path).exists(), "Evaluation output exists")
        np.savez_compressed(save_path, states=arrays["packed_states"], action_indices=choices,
            action_probabilities=probabilities, greedy_reward=rewards, executed=executed, satisfied=satisfied,
            exact_expected_reward=np.asarray(expectations), exact_full_success_probability=np.asarray(full_probabilities),
            exact_execution_probability=np.asarray(execution_probabilities))
        result["data_sha256"] = sha(save_path)
    return result


def save_checkpoint(path, actors, optimizer, update, batch_rng):
    require(not path.exists(), "Checkpoint exists")
    payload = {f"agent{i}_{key}": value for i, actor in enumerate(actors) for key, value in actor.items()}
    for i, opt in enumerate(optimizer):
        for moment in ("m", "v"):
            payload.update({f"adam_agent{i}_{moment}_{key}": value for key, value in opt[moment].items()})
    payload["update"] = np.array(update, dtype=np.int64)
    payload["batch_rng_json"] = np.array(json.dumps(batch_rng.bit_generator.state, sort_keys=True))
    np.savez_compressed(path, **payload)
    return sha(path)


def sources():
    paths = [Path(__file__).resolve(), HERE / "plan.md", HERE / "tests/test_runner.py", Path(env.__file__).resolve()]
    require(all(p.is_file() for p in paths), "Missing source/plan/test file before freezing")
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite preparation")
    source_hashes = sources()
    prepared = make_prepared()
    plan = {"schema": "triadic_learning_v1", "prepared_at": now(), "config": deepcopy(CONFIG),
            "prepared_sha256": json_hash(prepared), "sources": source_hashes,
            "runtime": {"python": platform.python_version(), "numpy": np.__version__},
            "no_training_executed_by_prepare": True}
    output.mkdir(parents=True, exist_ok=False)
    for name in source_hashes:
        target = output / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    write_new(output / "prepared.json", prepared)
    write_new(output / "plan.json", plan)
    write_new(output / "freeze.json", {"plan_sha256": sha(output / "plan.json"), "prepared_sha256": sha(output / "prepared.json")})
    verify(output)
    return {"status": "prepared_not_trained", "output": str(output), "plan_sha256": sha(output / "plan.json"),
            "partition_counts": {k: p["world_count"] for k, p in prepared["partitions"].items()}}


def verify(output):
    output = Path(output).resolve()
    plan, freeze, prepared = [read(output / name) for name in ("plan.json", "freeze.json", "prepared.json")]
    require(sha(output / "plan.json") == freeze["plan_sha256"], "Plan changed")
    require(sha(output / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared split changed")
    require(json_hash(make_prepared()) == plan["prepared_sha256"], "Runtime preparation differs")
    require(plan["config"] == CONFIG and plan["runtime"] == {"python": platform.python_version(), "numpy": np.__version__},
            "Configuration or numerical runtime changed")
    require(plan["sources"] == sources(), "Current source hashes changed")
    for name, expected in plan["sources"].items():
        require(sha(output / "source_snapshot" / name) == expected, "Frozen source snapshot changed")
    return plan, prepared


def train_seed(seed, prepared, arrays, output):
    output.mkdir(exist_ok=False)
    actors = [make_actor(np.random.SeedSequence([seed, i, 100])) for i in range(3)]
    # Independent objects and independent initialization streams; no weight tying.
    for i, j in combinations(range(3), 2):
        require(all(not np.shares_memory(actors[i][k], actors[j][k]) for k in actors[i]), "Shared parameters")
    optimizer = make_adam(actors)
    batch_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
    history = []
    started = time.perf_counter()

    def checkpoint(update):
        checkpoint_file = output / f"checkpoint_{update:04d}.npz"
        digest = save_checkpoint(checkpoint_file, actors, optimizer, update, batch_rng)
        monitor = {name: evaluate(actors, arrays[name], prepared["partitions"][name]["monitor_indices"]) for name in PARTITIONS}
        row = {"update": update, "checkpoint_sha256": digest, "monitor": monitor,
               "elapsed_seconds": time.perf_counter()-started}
        history.append(row)
        with (output / "monitor.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json_bytes(row).decode())

    checkpoint(0)
    with (output / "training.jsonl").open("x", encoding="utf-8") as stream:
        for update in range(1, CONFIG["updates"] + 1):
            ids = batch_rng.integers(0, len(arrays["train"]["states"]), size=CONFIG["batch_size"], dtype=np.int64)
            inputs = arrays["train"]["x"][ids]
            cached = [actor_forward(actors[i], inputs[:, i]) for i in range(3)]
            logits = np.stack([p[0] for p in cached], axis=1)
            probs, log_probs = policy_distribution(logits)
            expected, reward_gradient = expected_reward_and_logit_gradient(probs, arrays["train"]["rewards"][ids])
            entropy, entropy_gradient = entropy_and_logit_gradient(probs, log_probs)
            beta = entropy_coefficient(update)
            loss = -float(expected.mean() + beta * entropy.mean())
            derivative = -(reward_gradient + beta * entropy_gradient) / CONFIG["batch_size"]
            gradients = [actor_backward(actors[i], cached[i][1], derivative[:, i]) for i in range(3)]
            norm, clip_scale = adam_step(actors, gradients, optimizer, update)
            row = {"update": update, "batch_indices_sha256": array_sha(ids),
                   "batch_states_sha256": array_sha(arrays["train"]["packed_states"][ids]),
                   "native_expected_reward": float(expected.mean()), "mean_actor_entropy": float(entropy.mean()),
                   "entropy_coefficient": beta, "loss": loss, "gradient_norm": norm, "gradient_clip_scale": clip_scale,
                   "elapsed_seconds": time.perf_counter()-started}
            stream.write(json_bytes(row).decode())
            if update in CHECKPOINTS:
                stream.flush()
                checkpoint(update)
    final = {name: evaluate(actors, arrays[name], save_path=output / f"final_{name}.npz") for name in PARTITIONS}
    candidate = all(final[name]["greedy_full_success_rate"] >= CONFIG["candidate_min_full_success_rate_each_heldout_partition_each_seed"]
                    for name in PARTITIONS if name != "train")
    result = {"seed": seed, "updates": CONFIG["updates"], "training_world_samples": CONFIG["updates"] * CONFIG["batch_size"],
              "final": final, "candidate_threshold_met": candidate, "monitor": history,
              "elapsed_seconds": time.perf_counter()-started, "training_log_sha256": sha(output / "training.jsonl"),
              "final_checkpoint_sha256": sha(output / "checkpoint_6000.npz")}
    write_new(output / "result.json", result)
    return result


def execute(output):
    output = Path(output).resolve()
    plan, prepared = verify(output)
    execution = output / "execution"
    require(not execution.exists(), "Never resume/overwrite a started execution")
    execution.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_new(execution / "started.json", {"started_at": now(), "plan_sha256": sha(output / "plan.json"),
              "device": "cpu_numpy", "pid": os.getpid(), "thread_environment": {k: os.environ.get(k) for k in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    try:
        arrays = {name: build_arrays(prepared["partitions"][name]) for name in PARTITIONS}
        write_new(execution / "input_arrays.json", {name: {"worlds": len(a["states"]), "features_sha256": array_sha(a["x"]),
            "native_rewards_sha256": array_sha(a["rewards"]), "states_sha256": array_sha(a["packed_states"])} for name, a in arrays.items()})
        results = []
        for seed in SEEDS:
            result = train_seed(seed, prepared, arrays, execution / f"seed_{seed}")
            results.append(result)
            print(json.dumps({"seed_completed": seed, "heldout_full_success_rates": {k: v["greedy_full_success_rate"] for k, v in result["final"].items()},
                              "elapsed_seconds": time.perf_counter()-started}), flush=True)
        verify(output)
        result = {"status": "completed", "completed_at": now(), "plan_sha256": sha(output / "plan.json"),
                  "seeds": results, "completed_seed_count": len(results), "updates_total": len(SEEDS)*CONFIG["updates"],
                  "training_world_samples_total": prepared["training_state_samples_total"],
                  "all_seed_candidates_meet_threshold": all(r["candidate_threshold_met"] for r in results),
                  "symbolic_phase_unlocked": False, "elapsed_seconds": time.perf_counter()-started,
                  "scope": "Full-information centralized exact-return training with independently parameterized decentralized actors; not language emergence."}
        write_new(execution / "results.json", result)
        write_new(execution / "status.json", {"status": "completed", "completed_at": now(), "completed_seeds": len(results)})
    except BaseException as error:
        write_new(execution / "failure.json", {"status": "failed", "failed_at": now(), "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.perf_counter()-started})
        write_new(execution / "status.json", {"status": "failed", "failed_at": now()})
        raise
    return {"status": "completed", "output": str(execution), "all_seed_candidates_meet_threshold": result["all_seed_candidates_meet_threshold"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.out)
    elif args.command == "verify":
        verify(args.out); result = {"status": "verified_not_executed", "output": str(args.out.resolve())}
    else:
        result = execute(args.out)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
