"""Paired completed-run audit; execution only after root authorization.

Pure numerical helpers are adapted from the prior independent execution audit,
not the training runner. Final saved policies will be forwarded and counted;
no optimizer replay, training or model activity occurs on import.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations, permutations, product
import json
import math
from pathlib import Path
import platform
import random
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
PARTITIONS = ("train", "new_needs", "new_layouts", "new_needs_and_layouts")
SEEDS = (45101, 45102, 45103, 45104)
OBJECTIVES = ("mean_J", "mean_log_J")
BASE_RUNNER_SHA = "1f3cae631bc15e8f2004e27187370fcc8f9448ac6f99495e867c6ac40e2e0d6a"
SHAPES = {"W1": (54, 64), "b1": (64,), "W2": (64, 64), "b2": (64,), "W3": (64, 17), "b3": (17,)}


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def array_sha(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(json_bytes({"shape": list(a.shape), "dtype": a.dtype.str}))
    h.update(a.tobytes())
    return h.hexdigest()


def close(a, b, message, *, atol=2e-12, rtol=2e-12):
    require(np.allclose(a, b, atol=atol, rtol=rtol), message)


def finite_tree(value):
    if isinstance(value, dict):
        for v in value.values():
            finite_tree(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            finite_tree(v)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite recorded scalar")


def accepts(need, material, destination):
    resource, target = divmod(need, 3)
    return material in ((0, 1), (2, 3), (0, 2), (1, 3))[resource] and destination in ((0,), (1,), (0, 1))[target]


def support():
    return [n for n in product(range(12), repeat=3) if sum(
        any(accepts(n[i], m, d) and accepts(n[j], m, d) for m, d in product(range(4), range(2)))
        for i, j in combinations(range(3), 2)) >= 2]


def split_specs():
    needs = support(); require(len(needs) == 996, "Independent support count")
    groups = sorted({tuple(sorted(n)) for n in needs})
    random.Random(2026091901).shuffle(groups)
    train_set = set(groups[:159])
    train = [n for n in needs if tuple(sorted(n)) in train_set]
    held = [n for n in needs if tuple(sorted(n)) not in train_set]
    layouts = list(permutations(range(4))); random.Random(2026091902).shuffle(layouts)
    specs = {}
    for name, ns, ls in (("train", train, layouts[:18]), ("new_needs", held, layouts[:18]),
                         ("new_layouts", train, layouts[18:]), ("new_needs_and_layouts", held, layouts[18:])):
        owners = list(permutations((1, 2, 3)))
        n = len(ns)*len(ls)*len(owners)
        monitor = sorted(random.Random("2026091903:"+name).sample(range(n), 1024))
        specs[name] = {"needs": ns, "layouts": ls, "private_sites": owners, "world_count": n, "monitor_indices": monitor}
    return json.loads(json.dumps(specs)), json.loads(json.dumps(groups[:159])), json.loads(json.dumps(groups[159:]))


def packed(spec):
    return np.asarray([tuple(n)+tuple(l)+tuple(p) for n, l, p in product(spec["needs"], spec["layouts"], spec["private_sites"])], dtype=np.int16)


def features(states):
    n = len(states); rows = np.arange(n); base = np.zeros((n, 54), dtype=np.float64)
    for who in range(3):
        ids = states[:, who]
        base[:, 7*who] = 1
        base[rows, 7*who+1+ids//3] = 1
        base[:, 7*who+5] = (ids % 3 != 1)
        base[:, 7*who+6] = (ids % 3 != 0)
    for site in range(4):
        material = states[:, 3+site]
        base[:, 21+5*site] = 1
        base[rows, 21+5*site+1+material//2] = 1
        base[rows, 21+5*site+3+material%2] = 1
    for who in range(3):
        base[rows, 41+3*(states[:, 7+who]-1)+who] = 1
    base[:, 53] = 1
    result = np.repeat(base[:, None, :], 3, axis=1)
    for who in range(3):
        result[:, who, 50+who] = 1
    return result


def action_index(who, partner, site, destination):
    others = [i for i in range(3) if i != who]
    return 1 + 4*site + 2*destination + others.index(partner)


def decode_joint(indices):
    result = {}
    for who, index in enumerate(indices):
        if index == 0:
            action = {"action": "wait"}
        else:
            value = index-1
            action = {"action": "transport", "site": f"S{value//4}",
                      "destination": ("L", "R")[(value//2) % 2],
                      "partner": "ABC"[[i for i in range(3) if i != who][value % 2]]}
        result["ABC"[who]] = action
    return result


def joint_indices():
    result = []
    for i, j in combinations(range(3), 2):
        for site, dest in product(range(4), range(2)):
            row = [0, 0, 0]
            row[i] = action_index(i, j, site, dest); row[j] = action_index(j, i, site, dest)
            result.append(row)
    return np.asarray(result, dtype=np.int64)


def rewards(states):
    values = np.zeros((len(states), 24), dtype=np.float64)
    resource_ok = np.asarray([[m in group for m in range(4)] for group in ((0, 1), (2, 3), (0, 2), (1, 3))])
    dest_ok = np.asarray([[d in group for d in range(2)] for group in ((0,), (1,), (0, 1))])
    col = 0
    for i, j in combinations(range(3), 2):
        for site, dest in product(range(4), range(2)):
            material = states[:, 3+site]
            for who in (i, j):
                need = states[:, who]
                values[:, col] += .5*(resource_ok[need//3, material] & dest_ok[need%3, dest])
            col += 1
    return values


def forward(parameters, x):
    h1 = np.tanh(x @ parameters["W1"] + parameters["b1"])
    h2 = np.tanh(h1 @ parameters["W2"] + parameters["b2"])
    logits = h2 @ parameters["W3"] + parameters["b3"]
    require(np.isfinite(logits).all(), "Nonfinite recomputed final logits")
    exp = np.exp(logits-logits.max(axis=-1, keepdims=True))
    return exp/exp.sum(axis=-1, keepdims=True)


def checkpoint(path, expected_update):
    with np.load(path, allow_pickle=False) as z:
        keys = {f"agent{i}_{k}" for i in range(3) for k in SHAPES}
        keys |= {f"adam_agent{i}_{moment}_{k}" for i in range(3) for moment in ("m", "v") for k in SHAPES}
        keys |= {"update", "batch_rng_json"}
        require(set(z.files) == keys and z["update"].shape == () and int(z["update"]) == expected_update,
                "Checkpoint keys/update differ")
        actors = []
        for who in range(3):
            actor = {}
            for key, shape in SHAPES.items():
                v = z[f"agent{who}_{key}"]
                require(v.shape == shape and v.dtype == np.float64 and np.isfinite(v).all(), "Parameter shape/type/finite differs")
                actor[key] = v.copy()
                for moment in ("m", "v"):
                    value = z[f"adam_agent{who}_{moment}_{key}"]
                    require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(), "Optimizer array differs")
                    if moment == "v":
                        require((value >= 0).all(), "Negative Adam second moment")
                    if expected_update == 0:
                        require(not value.any(), "Nonzero initial Adam moments")
            actors.append(actor)
        rng = json.loads(str(z["batch_rng_json"].item()))
    return actors, rng


def summary_from_saved(z, ids):
    p, choices, r = z["action_probabilities"][ids], z["action_indices"][ids], z["greedy_reward"][ids]
    attempted = (choices != 0).sum(1)
    physical = z["executed"][ids].sum(1)
    return {"worlds": len(ids), "greedy_reward_sum": float(r.sum()), "greedy_reward_mean": float(r.mean()),
        "greedy_full_successes": int((r == 1).sum()), "greedy_full_success_rate": float((r == 1).mean()),
        "greedy_non_full_success_worlds": int((r != 1).sum()),
        "greedy_reward_counts": {str(v): int((r == v).sum()) for v in (0., .5, 1.)},
        "greedy_attempted_transports": int(attempted.sum()), "greedy_executed_transports": int(z["executed"][ids].sum()),
        "greedy_satisfied_agents": int(z["satisfied"][ids].sum()),
        "greedy_active_count_worlds": {str(k): int((attempted == k).sum()) for k in range(4)},
        "greedy_agent_argmax_ties": int(((p == p.max(-1, keepdims=True)).sum(-1) > 1).sum()),
        "exact_stochastic_expected_reward_mean": float(z["exact_expected_reward"][ids].mean()),
        "exact_stochastic_full_success_probability_mean": float(z["exact_full_success_probability"][ids].mean()),
        "exact_stochastic_physical_execution_probability_mean": float(z["exact_execution_probability"][ids].mean()),
        "greedy_failure_categories": {"all_wait": int((attempted == 0).sum()),
            "single_transport_proposal": int((attempted == 1).sum()), "overload": int((attempted == 3).sum()),
            "two_unmatched": int(((attempted == 2) & (physical == 0)).sum()),
            "matched_no_need_satisfied": int(((physical == 2) & (r == 0)).sum()),
            "matched_one_need_satisfied": int((r == .5).sum())},
        "greedy_executed_pair_worlds": {"ABC"[i]+"ABC"[j]: int((z["executed"][ids, i] & z["executed"][ids, j]).sum())
                                       for i, j in combinations(range(3), 2)},
        "state_indices_sha256": array_sha(np.asarray(ids, dtype=np.int64)), "packed_states_sha256": array_sha(z["states"][ids])}


def compare_summary(actual, expected):
    # Compare every independently constructed summary field, retaining failures.
    for key, value in expected.items():
        require(key in actual, "Missing final/monitor summary: " + key)
        if isinstance(value, float):
            close(actual[key], value, "Final/monitor summary differs: " + key)
        else:
            require(actual[key] == value, "Final/monitor summary differs: " + key)


def semantic_actions():
    return [[{"kind": "wait"}] + [{"kind": "transport", "site": f"S{s}",
        "destination": "LR"[d], "partner": partner}
        for s, d, partner in product(range(4), range(2), [a for a in "ABC" if a != who])]
        for who in "ABC"]


def settle_saved(states, choices):
    """Independent D1 formula for saved joint choices, not all4913 enumeration."""
    n = len(states)
    require(states.shape == (n, 10) and choices.shape == (n, 3), "Saved physical input shape")
    require(np.issubdtype(choices.dtype, np.integer) and ((choices >= 0) & (choices < 17)).all(), "Action index domain")
    active = choices != 0
    local = np.maximum(choices-1, 0)
    site, dest = local//4, (local//2) % 2
    partner = np.empty_like(choices)
    for who in range(3):
        partner[:, who] = np.asarray([i for i in range(3) if i != who])[local[:, who] % 2]
    executed = np.zeros((n, 3), dtype=bool)
    for i, j in combinations(range(3), 2):
        match = (active.sum(1) == 2) & active[:, i] & active[:, j]
        match &= (partner[:, i] == j) & (partner[:, j] == i)
        match &= (site[:, i] == site[:, j]) & (dest[:, i] == dest[:, j])
        executed[:, i] |= match
        executed[:, j] |= match
    satisfied = np.zeros((n, 3), dtype=bool)
    resource = np.asarray([[m in g for m in range(4)] for g in ((0, 1), (2, 3), (0, 2), (1, 3))])
    destinations = np.asarray([[d in g for d in range(2)] for g in ((0,), (1,), (0, 1))])
    for who in range(3):
        need = states[:, who]
        material = states[np.arange(n), 3+site[:, who]]
        satisfied[:, who] = executed[:, who] & resource[need//3, material] & destinations[need % 3, dest[:, who]]
    require((executed.sum(1) <= 2).all(), "Independent physical execution exceeds capacity")
    return satisfied.sum(1)/2, executed, satisfied


def exact_statistics(probabilities, native_rewards):
    selected = np.ones((len(probabilities), 24), dtype=np.float64)
    joint = joint_indices()
    for who in range(3):
        selected *= probabilities[:, who, joint[:, who]]
    return ((selected*native_rewards).sum(1), (selected*(native_rewards == 1)).sum(1), selected.sum(1))


def parameters_sha(actors):
    return hashlib.sha256(json_bytes({f"agent{i}_{key}": array_sha(v)
        for i, actor in enumerate(actors) for key, v in actor.items()})).hexdigest()


def validate_preparation(run):
    plan, freeze, prepared = [read(run/n) for n in ("plan.json", "freeze.json", "prepared.json")]
    require(sha(run/"plan.json") == freeze["plan_sha256"], "Plan SHA mismatch")
    require(sha(run/"prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared SHA mismatch")
    require(plan["config"] == prepared["config"], "Plan/prepared config mismatch")
    require(plan["no_training_executed_by_prepare"] is True, "Preparation cannot train")
    config = plan["config"]
    expected = {"seeds": list(SEEDS), "objectives": list(OBJECTIVES), "updates": 6000, "batch_size": 256,
        "features": 54, "hidden_sizes": [64, 64], "actions_per_actor": 17, "learning_rate": .001,
        "adam_beta1": .9, "adam_beta2": .999, "adam_epsilon": 1e-8, "global_gradient_clip": 5.,
        "entropy_initial": .001, "entropy_zero_after_updates": 1000, "entropy_reduction": "mean_over_three_actors",
        "dtype": "float64", "checkpoints": list(CHECKPOINTS), "monitor_worlds_per_partition": 1024,
        "evaluation_batch_size": 1024, "full_information": True, "require_match": True,
        "candidate_min_full_success_rate_each_heldout_partition_each_seed": .99,
        "automatic_symbolic_stage": False, "run_order": "seed_outer_objective_inner",
        "log_J_epsilon_or_floor": None, "baseline_runner_sha256": BASE_RUNNER_SHA,
        "training_objective": "paired_state_aggregation_intervention",
        "deployment": "independent_argmax_first_index_on_tie"}
    for key, value in expected.items():
        require(config[key] == value, "Fixed config mismatch: "+key)
    require(plan["runtime"] == {"python": platform.python_version(), "numpy": np.__version__}, "Numerical runtime mismatch")
    source_hashes = {}
    for name, expected_sha in plan["sources"].items():
        for path in (ROOT/name, run/"source_snapshot"/name):
            require(sha(path) == expected_sha, "Frozen/current source mismatch: "+str(path))
            source_hashes[str(path)] = expected_sha
    require(plan["sources"]["research_program/triadic_learning_baseline/runner.py"] == BASE_RUNNER_SHA,
            "Baseline evaluator/actor implementation changed")
    specs, train, held = split_specs()
    require(prepared["partitions"] == specs and prepared["train_need_multisets"] == train and prepared["heldout_need_multisets"] == held,
            "Independent split/monitor reconstruction mismatch")
    require(prepared["baseline_partitions_sha256"] == hashlib.sha256(json_bytes(specs)).hexdigest(), "Baseline partition anchor mismatch")
    require(prepared["actions"] == semantic_actions(), "Complete17 action heads mismatch")
    require(np.array_equal(prepared["structural_joint_actions"], joint_indices()), "24 structural indices mismatch")
    runs = [{"seed": seed, "objective": obj, "directory": f"seed_{seed}_{obj}"} for seed in SEEDS for obj in OBJECTIVES]
    require(prepared["runs"] == runs, "Eight paired-run order mismatch")
    for key, value in {"training_state_samples_per_run": 1536000, "training_state_samples_total": 12288000,
        "weighted_structural_action_contributions": 294912000, "offline_reward_table_entries": 3442176,
        "complete_final_world_evaluations": 1147392}.items():
        require(prepared[key] == value, "Prepared budget mismatch: "+key)
    return plan, prepared, specs, source_hashes


def audit(run):
    run = Path(run).resolve()
    execution = run/"execution"
    # Refuse incomplete runs before loading any policy array or score payload.
    status = read(execution/"status.json")
    require(status["status"] == "completed" and status["completed_runs"] == 8, "Only the completed eight-run batch may be audited")
    require(not (execution/"failure.json").exists(), "Completed batch has an execution failure artifact")
    plan, prepared, specs, sources = validate_preparation(run)
    result = read(execution/"results.json")
    require(result["status"] == "completed" and result["plan_sha256"] == sha(run/"plan.json"), "Final status/plan mismatch")
    require(result["completed_run_count"] == 8 and result["paired_seed_count"] == 4
            and result["updates_total"] == 48000 and result["training_world_samples_total"] == 12288000
            and result["weighted_structural_action_contributions"] == 294912000
            and result["symbolic_phase_unlocked"] is False, "Final budget or symbolic gate mismatch")
    require([(r["seed"], r["objective"]) for r in result["runs"]] == [(s, o) for s in SEEDS for o in OBJECTIVES], "Missing/reordered run")
    require({p.name for p in execution.glob("seed_*")} == {r["directory"] for r in prepared["runs"]}, "Extra/missing run directory")
    started = read(execution/"started.json")
    require(started["plan_sha256"] == sha(run/"plan.json") and started["device"] == "cpu_numpy"
        and set(started["thread_environment"]) == {"OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"}
        and all(v == "1" for v in started["thread_environment"].values()), "Frozen runtime record mismatch")
    finite_tree(result)
    states = {name: packed(spec) for name, spec in specs.items()}
    input_receipt = read(execution/"input_arrays.json")
    require(set(input_receipt) == set(PARTITIONS), "Input partition omitted")
    for name, state in states.items():
        require(input_receipt[name] == {"worlds": len(state), "features_sha256": array_sha(features(state)),
            "native_rewards_sha256": array_sha(rewards(state)), "states_sha256": array_sha(state)}, "Independent full-observation input mismatch: "+name)
    artifacts = {str(p): sha(p) for p in (run/"plan.json", run/"freeze.json", run/"prepared.json",
        execution/"results.json", execution/"status.json", execution/"started.json", execution/"input_arrays.json")}
    completed, pair_cache, pair_checks = [], {}, []
    checked_updates = checkpoints_count = final_count = forward_samples = forward_batches = 0
    max_p_error = max_statistics_error = 0.
    for row in result["runs"]:
        seed, objective = row["seed"], row["objective"]
        directory = execution/f"seed_{seed}_{objective}"
        require(read(directory/"result.json") == row, "Per-run/main result mismatch")
        artifacts[str(directory/"result.json")] = sha(directory/"result.json")
        require(row["updates"] == 6000 and row["training_world_samples"] == 1536000, "Incomplete per-run budget")
        history = [json.loads(line) for line in (directory/"monitor.jsonl").read_text().splitlines() if line.strip()]
        artifacts[str(directory/"monitor.jsonl")] = sha(directory/"monitor.jsonl")
        require(history == row["monitor"] and [m["update"] for m in history] == list(CHECKPOINTS), "Monitor schedule mismatch")
        require({p.name for p in directory.glob("checkpoint_*.npz")} == {f"checkpoint_{u:04d}.npz" for u in CHECKPOINTS}, "Extra/missing checkpoint")
        require({p.name for p in directory.glob("final_*.npz")} == {f"final_{name}.npz" for name in PARTITIONS}, "Extra/missing final partition")
        checkpoint_data = {}
        for monitor in history:
            u = monitor["update"]
            path = directory/f"checkpoint_{u:04d}.npz"
            require(sha(path) == monitor["checkpoint_sha256"], "Checkpoint SHA mismatch")
            artifacts[str(path)] = sha(path)
            checkpoint_data[u] = checkpoint(path, u)
            require(set(monitor["monitor"]) == set(PARTITIONS), "Monitor omitted a partition")
            for name, m in monitor["monitor"].items():
                ids = np.asarray(specs[name]["monitor_indices"], dtype=np.int64)
                require(m["worlds"] == 1024 and m["state_indices_sha256"] == array_sha(ids)
                        and m["packed_states_sha256"] == array_sha(states[name][ids]), "Monitor world support mismatch")
            finite_tree(monitor)
            checkpoints_count += 1
        initial = checkpoint_data[0][0]
        for who in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([seed, who, 100]))
            for layer, (left, right) in enumerate(((54, 64), (64, 64), (64, 17)), 1):
                expected = rng.normal(0, math.sqrt(2/(left+right)), (left, right))
                require(np.array_equal(initial[who][f"W{layer}"], expected)
                        and not initial[who][f"b{layer}"].any(), "Independent initialization mismatch")
        for i, j in combinations(range(3), 2):
            require(all(not np.shares_memory(initial[i][key], initial[j][key]) for key in SHAPES), "Loaded parameters alias")
            require(all(not np.array_equal(initial[i][f"W{k}"], initial[j][f"W{k}"]) for k in (1, 2, 3)), "Actors copied same weights")
        initial_sha = parameters_sha(initial)
        require(initial_sha == row["initial_parameter_sha256"], "Initial parameter hash mismatch")
        batch_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
        require(batch_rng.bit_generator.state == checkpoint_data[0][1], "Initial batch RNG mismatch")
        batches = []
        log_path = directory/"training.jsonl"
        require(sha(log_path) == row["training_log_sha256"], "Training log SHA mismatch")
        artifacts[str(log_path)] = sha(log_path)
        previous_elapsed = -1.
        with log_path.open() as stream:
            count = 0
            for count, line in enumerate(stream, 1):
                log = json.loads(line)
                finite_tree(log)
                require(log["update"] == count <= 6000 and log["objective"] == objective, "Update/objective order mismatch")
                ids = batch_rng.integers(0, len(states["train"]), size=256, dtype=np.int64)
                require(log["batch_indices_sha256"] == array_sha(ids)
                        and log["batch_states_sha256"] == array_sha(states["train"][ids]), "Reconstructed batch/state hash mismatch")
                beta = .001*max(0, 1-(count-1)/1000)
                require(log["entropy_coefficient"] == beta, "Fixed entropy schedule mismatch")
                selected = log["mean_J"] if objective == "mean_J" else log["mean_log_J"]
                require(log["selected_objective_mean"] == selected, "Selected logged objective mismatch")
                close(log["loss"], -(selected+beta*log["mean_actor_entropy"]), "Recorded scalar loss algebra mismatch")
                require(0 <= log["mean_J"] <= 1+2e-12 and log["min_log_J"] <= log["mean_log_J"] <= log["max_log_J"] <= 2e-12
                    and 0 <= log["zero_float_J_states"] <= 256 and log["gradient_norm"] >= 0, "Logged value domain mismatch")
                close(log["gradient_clip_scale"], min(1., 5/max(log["gradient_norm"], 1e-300)), "Gradient clip scalar mismatch")
                require(log["elapsed_seconds"] >= previous_elapsed, "Elapsed time went backwards")
                previous_elapsed = log["elapsed_seconds"]
                batches.append((log["update"], log["batch_indices_sha256"], log["batch_states_sha256"], beta))
                if count in checkpoint_data:
                    require(batch_rng.bit_generator.state == checkpoint_data[count][1], "Checkpoint batch RNG mismatch")
                checked_updates += 1
            require(count == 6000, "Missing training updates")
        if objective == OBJECTIVES[0]:
            pair_cache[seed] = (initial, batches, history[0]["monitor"], initial_sha)
        else:
            first_initial, first_batches, first_monitor, first_sha = pair_cache.pop(seed)
            require(initial_sha == first_sha and batches == first_batches
                    and history[0]["monitor"] == first_monitor, "Cross-arm pairing mismatch")
            require(all(np.array_equal(initial[a][k], first_initial[a][k]) for a in range(3) for k in SHAPES), "Paired initial arrays differ")
            pair_checks.append({"seed": seed, "same_initial_parameters": True, "same_initial_monitor": True,
                "paired_batch_and_state_hashes": 6000, "same_fixed_entropy_schedule": True})
        final_actor = checkpoint_data[6000][0]
        require(sha(directory/"checkpoint_6000.npz") == row["final_checkpoint_sha256"]
                and parameters_sha(final_actor) == row["final_parameter_sha256"], "Final parameter/checkpoint anchor mismatch")
        summaries, action_counts = {}, Counter()
        require(set(row["final"]) == set(PARTITIONS), "Omitted final partition")
        for name in PARTITIONS:
            path = directory/f"final_{name}.npz"
            require(sha(path) == row["final"][name]["data_sha256"], "Final NPZ SHA mismatch")
            artifacts[str(path)] = sha(path)
            with np.load(path, allow_pickle=False) as loaded:
                z = {k: loaded[k] for k in loaded.files}
            require(set(z) == {"states", "action_indices", "action_probabilities", "greedy_reward", "executed", "satisfied",
                "exact_expected_reward", "exact_full_success_probability", "exact_execution_probability"}, "Final schema mismatch")
            state, n = states[name], len(states[name])
            require(z["states"].dtype == np.int16 and np.array_equal(z["states"], state), "Full final state order mismatch")
            require(z["action_probabilities"].shape == (n, 3, 17) and z["action_probabilities"].dtype == np.float64
                    and z["action_indices"].shape == (n, 3) and z["action_indices"].dtype == np.int16, "Final action/probability shape or type mismatch")
            for key, value in z.items():
                require(np.isfinite(value).all(), "Nonfinite saved array: "+key)
            require((z["action_probabilities"] >= 0).all(), "Negative saved probability")
            close(z["action_probabilities"].sum(-1), 1, "Saved probability normalization mismatch")
            require(z["executed"].shape == z["satisfied"].shape == (n, 3)
                    and z["executed"].dtype == z["satisfied"].dtype == bool, "Final feedback shape/type mismatch")
            for key in ("greedy_reward", "exact_expected_reward", "exact_full_success_probability", "exact_execution_probability"):
                require(z[key].shape == (n,) and z[key].dtype == np.float64, "Final reward/statistic shape/type mismatch")
            r, executed, satisfied = settle_saved(state, z["action_indices"])
            require(np.array_equal(r, z["greedy_reward"]) and np.array_equal(executed, z["executed"])
                    and np.array_equal(satisfied, z["satisfied"]), "Independent saved-choice physical settlement mismatch")
            for start in range(0, n, 1024):
                chunk = state[start:start+1024]
                x = features(chunk)
                prob = np.stack([forward(final_actor[i], np.ascontiguousarray(x[:, i])) for i in range(3)], axis=1)
                saved = z["action_probabilities"][start:start+len(chunk)]
                close(prob, saved, "Independent final probability mismatch")
                max_p_error = max(max_p_error, float(np.max(np.abs(prob-saved))))
                require(np.array_equal(prob.argmax(-1), z["action_indices"][start:start+len(chunk)]), "Final independent argmax mismatch")
                require(np.array_equal(prob == prob.max(-1, keepdims=True), saved == saved.max(-1, keepdims=True)), "Final exact argmax tie set mismatch")
                values = exact_statistics(prob, rewards(chunk))
                for key, values_ in zip(("exact_expected_reward", "exact_full_success_probability", "exact_execution_probability"), values):
                    saved_ = z[key][start:start+len(chunk)]
                    close(values_, saved_, "Final exact statistic mismatch: "+key)
                    max_statistics_error = max(max_statistics_error, float(np.max(np.abs(values_-saved_))))
                forward_samples += 3*len(chunk)
                forward_batches += 3
            summary = summary_from_saved(z, np.arange(n, dtype=np.int64))
            compare_summary(row["final"][name], summary)
            compare_summary(history[-1]["monitor"][name], summary_from_saved(z, np.asarray(specs[name]["monitor_indices"], dtype=np.int64)))
            summaries[name] = summary
            distinct, counts = np.unique(z["action_indices"], axis=0, return_counts=True)
            distribution = [{"action_indices": a.tolist(), "worlds": int(c)} for a, c in zip(distinct, counts)]
            require(row["final"][name]["distinct_joint_argmax_actions"] == len(distinct)
                and row["final"][name]["joint_argmax_action_counts"] == distribution, "Final action-coverage summary mismatch")
            action_counts.update({tuple(map(int, a)): int(c) for a, c in zip(distinct, counts)})
            final_count += 1
        require(row["full_domain_actions"] == {"worlds": sum(action_counts.values()),
            "distinct_joint_argmax_actions": len(action_counts),
            "joint_argmax_action_counts": [{"action_indices": list(a), "worlds": c} for a, c in sorted(action_counts.items())]},
            "Complete-domain action-coverage summary mismatch")
        candidate = all(summaries[name]["greedy_full_success_rate"] >= .99 for name in PARTITIONS if name != "train")
        require(row["candidate_threshold_met"] is candidate, "Unchanged99% per-run threshold mismatch")
        completed.append({"seed": seed, "objective": objective, "candidate_threshold_met": candidate,
            "initial_parameter_sha256": initial_sha, "final": summaries,
            "distinct_complete_domain_joint_argmax": len(action_counts),
            "joint_argmax_action_counts": [{"indices": list(k), "count": v} for k, v in sorted(action_counts.items())]})
    require(not pair_cache and pair_checks == result["pairing_checks"], "Paired checks incomplete/mismatched")
    require(checked_updates == 48000 and checkpoints_count == 48 and final_count == 32
            and forward_samples == 3442176, "Audit count incomplete")
    candidate_by_objective = {obj: all(r["candidate_threshold_met"] for r in completed if r["objective"] == obj) for obj in OBJECTIVES}
    require(result["all_seed_candidates_by_objective"] == candidate_by_objective, "Aggregate candidate flags mismatch")
    primary_rows = []
    index = {(r["seed"], r["objective"]): r for r in completed}
    for seed in SEEDS:
        first, second = [index[(seed, obj)]["final"]["new_needs_and_layouts"] for obj in OBJECTIVES]
        require(first["worlds"] == second["worlds"] == 8244, "Primary denominator mismatch")
        primary_rows.append({"seed": seed, "mean_J_full_success_rate": first["greedy_full_success_rate"],
            "mean_log_J_full_success_rate": second["greedy_full_success_rate"],
            "paired_difference_log_minus_mean": second["greedy_full_success_rate"]-first["greedy_full_success_rate"]})
    primary = {"partition": "new_needs_and_layouts", "endpoint_update": 6000, "metric": "greedy_full_success_rate",
        "expected_direction": "mean_log_J greater than mean_J", "seed_pairs": primary_rows,
        "equal_weight_mean_paired_difference": sum(r["paired_difference_log_minus_mean"] for r in primary_rows)/4,
        "independent_paired_initializations": 4, "significance_test": None}
    require(result["primary_comparison"] == primary, "Primary heldout paired contrast mismatch")
    for filename, expected_sha in sources.items():
        require(sha(filename) == expected_sha, "Source changed during audit")
    for filename, expected_sha in artifacts.items():
        require(sha(filename) == expected_sha, "Artifact changed during audit")
    return {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(), "audit_source_sha256": sha(__file__),
        "source_sha256": sources, "artifact_sha256": artifacts, "pairing_checks": pair_checks, "runs": completed,
        "all_seed_candidates_by_objective": candidate_by_objective, "primary_comparison": primary, "symbolic_phase_unlocked": False,
        "counts": {"paired_seed_blocks": 4, "runs": 8, "training_updates_verified": checked_updates,
            "state_samples_reconstructed": 12288000, "within_pair_matched_batch_hashes": 24000,
            "checkpoints": checkpoints_count, "full_final_npz": final_count,
            "independently_settled_final_worlds": 1147392, "actual_actor_forward_samples": forward_samples,
            "actual_actor_forward_batches": forward_batches, "optimizer_or_training_replays": 0},
        "maximum_probability_absolute_error": max_p_error, "maximum_exact_statistic_absolute_error": max_statistics_error,
        "numerical_tolerance": {"atol": 2e-12, "rtol": 2e-12, "argmax_tie_sets_actions_rewards": "exact"},
        "limits": ["Actually forwards final saved policies; not a zero-inference audit.",
            "Checks all batch RNG streams and logged scalar loss algebra, not every parameter update; no training replay.",
            "Intermediate checkpoint shapes, finite values and RNG anchors checked, but intermediate policies not forwarded.",
            "Historical parameter non-aliasing additionally relies on frozen actor-construction source.",
            "Does not replace the independent objective-gradient audit or establish a scientific mechanism."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.run/"audit_paired_execution_001"
    require(not out.exists(), "Refuse overwrite or automatic retry")
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args.run)
        (out/"verification.json").write_bytes(json_bytes(result))
        report = f'''# 配对执行独立核验

核验通过：4个配对种子、8次固定训练、48000个更新与12288000个状态抽样哈希重建一致。每对初始参数逐值相同，初始化各actor的SeedSequence独立；6000个批次、状态哈希、熵时间表及初始监测逐项配对一致。48个检查点与32份四格完整终点保存齐全，原99%标准和全部失败世界保留。

本次实际加载终点参数，以独立NumPy前向重算1147392个世界、3442176个actor样本。概率最大绝对差{result['maximum_probability_absolute_error']:.3g}，随机策略精确统计最大差{result['maximum_exact_statistic_absolute_error']:.3g}；概率采用atol=rtol=2e-12，argmax、精确并列集合及原R／执行／满足逐项完全相同。没有重放优化器或新增训练。

两目标是否全部达到原候选标准：{result['all_seed_candidates_by_objective']}。symbolic_phase_unlocked仍为false。逐种子、四格完整行为计数及来源哈希见verification.json；4个配对种子是重复单位，不把32格或世界数当成独立训练重复。

本核查不重算每一步训练梯度和参数更新，也不前向中间检查点；它不能替代独立数学差分审查，不能证明目标替换效应由某一机制单独造成，不能充当语言形成证据。
'''
        (out/"独立核验.md").write_text(report)
        print(json.dumps({k: result[k] for k in ("status", "counts", "maximum_probability_absolute_error")}, ensure_ascii=False))
    except BaseException as error:
        (out/"failure.json").write_bytes(json_bytes({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
            "audit_source_sha256": sha(__file__), "error": str(error), "traceback": traceback.format_exc(), "automatic_retry": False}))
        raise


if __name__ == "__main__":
    main()
