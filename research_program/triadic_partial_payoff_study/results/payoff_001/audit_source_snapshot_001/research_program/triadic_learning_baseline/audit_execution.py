"""Audit completed training without rerunning optimization.

Loads saved initial/checkpoint/final arrays. Independently forwards final
policies on every saved final world, explicitly counting actor samples.
The separate frozen behavior analysis owns full physical classification.
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
SEEDS = (43101, 43102, 43103, 43104)
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
PARTITIONS = ("train", "new_needs", "new_layouts", "new_needs_and_layouts")
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
    return {"worlds": len(ids), "greedy_reward_sum": float(r.sum()), "greedy_reward_mean": float(r.mean()),
        "greedy_full_successes": int((r == 1).sum()), "greedy_full_success_rate": float((r == 1).mean()),
        "greedy_reward_counts": {str(v): int((r == v).sum()) for v in (0., .5, 1.)},
        "greedy_attempted_transports": int(attempted.sum()), "greedy_executed_transports": int(z["executed"][ids].sum()),
        "greedy_satisfied_agents": int(z["satisfied"][ids].sum()),
        "greedy_active_count_worlds": {str(k): int((attempted == k).sum()) for k in range(4)},
        "greedy_agent_argmax_ties": int(((p == p.max(-1, keepdims=True)).sum(-1) > 1).sum()),
        "exact_stochastic_expected_reward_mean": float(z["exact_expected_reward"][ids].mean()),
        "exact_stochastic_full_success_probability_mean": float(z["exact_full_success_probability"][ids].mean()),
        "exact_stochastic_physical_execution_probability_mean": float(z["exact_execution_probability"][ids].mean()),
        "state_indices_sha256": array_sha(np.asarray(ids, dtype=np.int64)), "packed_states_sha256": array_sha(z["states"][ids])}


def compare_summary(actual, expected):
    # Root's independently frozen behavior analyzer verifies extra pair/error fields.
    for key, value in expected.items():
        require(key in actual, "Missing final/monitor summary: " + key)
        if isinstance(value, float):
            close(actual[key], value, "Final/monitor summary differs: " + key)
        else:
            require(actual[key] == value, "Final/monitor summary differs: " + key)


def audit(run, behavior_path=None):
    run = Path(run).resolve(); execution = run / "execution"
    plan, frozen, prepared = read(run / "plan.json"), read(run / "freeze.json"), read(run / "prepared.json")
    results, status = read(execution / "results.json"), read(execution / "status.json")
    require(results["status"] == status["status"] == "completed", "Training must be complete before audit")
    require(not (execution / "failure.json").exists(), "Training failure artifact present")
    require(sha(run / "plan.json") == frozen["plan_sha256"] == results["plan_sha256"], "Frozen plan differs")
    require(sha(run / "prepared.json") == frozen["prepared_sha256"] == plan["prepared_sha256"], "Prepared data differs")
    require(plan["config"] == prepared["config"], "Config inconsistent")
    config = plan["config"]
    for key, value in {"seeds": list(SEEDS), "updates": 6000, "batch_size": 256, "features": 54, "hidden_sizes": [64, 64],
                       "actions_per_actor": 17, "learning_rate": .001, "adam_beta1": .9, "adam_beta2": .999,
                       "adam_epsilon": 1e-8, "global_gradient_clip": 5., "entropy_initial": .001,
                       "entropy_zero_after_updates": 1000, "entropy_reduction": "mean_over_three_actors",
                       "dtype": "float64", "checkpoints": list(CHECKPOINTS), "full_information": True, "require_match": True,
                       "candidate_min_full_success_rate_each_heldout_partition_each_seed": .99}.items():
        require(config[key] == value, "Fixed config differs: " + key)
    require(plan["runtime"] == {"python": platform.python_version(), "numpy": np.__version__}, "Audit runtime differs from frozen numerical runtime")
    source_hashes = {}
    for name, digest in plan["sources"].items():
        for path in (ROOT / name, run / "source_snapshot" / name):
            require(sha(path) == digest, "Frozen/current source differs: " + str(path))
            source_hashes[str(path)] = digest
    review_path = run / "preflight" / "root_review.json"
    require(sha(review_path) == "76c469af0a0841ebeddf4759be37d3f17a58f53a160c78b18a880a0136468397", "Preflight root receipt differs")
    review = read(review_path)
    require(review["status"] == "root_review_passed_before_training" and review["plan_sha256"] == frozen["plan_sha256"]
            and review["prepared_sha256"] == frozen["prepared_sha256"] and review["training_runs_started"] == 0, "Preflight binding differs")
    source_hashes[str(review_path)] = sha(review_path)
    for name, digest in review["sources"].items():
        path = run / "preflight" / name
        require(sha(path) == digest, "Archived preflight source differs: " + name)
        source_hashes[str(path)] = digest
    gradient_receipt = read(run / "preflight" / "independent_environment_audit.json")
    require(gradient_receipt["status"] == "passed" and gradient_receipt["runner_sha256"] == sha(HERE / "runner.py"), "Pretraining derivative audit source differs")
    runner_source = (HERE / "runner.py").read_text()
    require('loss = -float(expected.mean() + beta * entropy.mean())' in runner_source
            and 'derivative = -(reward_gradient + beta * entropy_gradient) / CONFIG["batch_size"]' in runner_source
            and 'actor[key] -= CONFIG["learning_rate"]' in runner_source, "Reviewed loss/Adam sign construction differs")
    sign_review = {"negative_expected_reward_plus_entropy_loss": True, "adam_subtracts_loss_gradient": True,
                   "interpretation": "Gradient descent on the negative objective; no sign reversal found in frozen source",
                   "pretraining_gradient_receipt_sha256": sha(run / "preflight" / "independent_environment_audit.json"),
                   "logit_finite_difference_max_error": gradient_receipt["gradient_max_abs_error"],
                   "network_parameter_finite_difference_max_error": gradient_receipt["random_matrix_parameter_gradient_max_abs_error"]}
    specs, train_multisets, held_multisets = split_specs()
    require(prepared["partitions"] == specs and prepared["train_need_multisets"] == train_multisets
            and prepared["heldout_need_multisets"] == held_multisets, "Independent split/monitor reconstruction differs")
    joint = joint_indices()
    require(np.array_equal(prepared["structural_joint_actions"], joint), "Structural indices differ")
    require(prepared["training_state_samples_per_seed"] == 1536000 and prepared["training_state_samples_total"] == 6144000,
            "Sample budget differs")
    require(results["completed_seed_count"] == status["completed_seeds"] == 4 and results["updates_total"] == 24000
            and results["training_world_samples_total"] == 6144000 and results["symbolic_phase_unlocked"] is False,
            "Completion budget or symbol flag differs")
    require([r["seed"] for r in results["seeds"]] == list(SEEDS), "Missing/reordered training seed")
    require({p.name for p in execution.glob("seed_*")} == {f"seed_{s}" for s in SEEDS}, "Extra/missing seed directory")
    started = read(execution / "started.json")
    require(started["plan_sha256"] == frozen["plan_sha256"] and started["device"] == "cpu_numpy"
            and all(v == "1" for v in started["thread_environment"].values()), "Device/thread/freeze record differs")
    state_arrays = {name: packed(spec) for name, spec in specs.items()}
    input_receipt = read(execution / "input_arrays.json")
    for name, state in state_arrays.items():
        x, r = features(state), rewards(state)
        require(input_receipt[name] == {"worlds": len(state), "features_sha256": array_sha(x),
                "native_rewards_sha256": array_sha(r), "states_sha256": array_sha(state)}, "Precomputed input arrays differ: " + name)
        del x, r
    finite_tree(results)
    max_probability_error, max_expectation_error = 0., 0.
    forward_samples = forward_batches = checked_batches = checkpoint_count = final_files = initial_parameter_values = 0
    audit_seeds, artifacts, constant_rows = [], {}, []
    for result in results["seeds"]:
        seed = result["seed"]; directory = execution / f"seed_{seed}"
        require(read(directory / "result.json") == result, "Seed result differs from main result")
        require(result["updates"] == 6000 and result["training_world_samples"] == 1536000, "Seed training incomplete")
        history = [json.loads(s) for s in (directory / "monitor.jsonl").read_text().splitlines() if s.strip()]
        require(history == result["monitor"] and [r["update"] for r in history] == list(CHECKPOINTS), "Checkpoint schedule differs")
        require({p.name for p in directory.glob("checkpoint_*.npz")} == {f"checkpoint_{u:04d}.npz" for u in CHECKPOINTS}, "Extra/missing checkpoint")
        checkpoints = {}
        for row in history:
            path = directory / f"checkpoint_{row['update']:04d}.npz"
            digest = sha(path); require(digest == row["checkpoint_sha256"], "Checkpoint SHA differs")
            artifacts[str(path)] = digest
            actor, rng_state = checkpoint(path, row["update"])
            checkpoints[row["update"]] = (actor, rng_state)
            require(set(row["monitor"]) == set(PARTITIONS), "Monitor partition omitted")
            for name, monitor in row["monitor"].items():
                require(monitor["worlds"] == 1024 and monitor["state_indices_sha256"] == array_sha(np.asarray(specs[name]["monitor_indices"], dtype=np.int64))
                        and monitor["packed_states_sha256"] == array_sha(state_arrays[name][specs[name]["monitor_indices"]]), "Monitor state support differs")
            finite_tree(row); checkpoint_count += 1
        initial = checkpoints[0][0]
        for who in range(3):
            rng = np.random.default_rng(np.random.SeedSequence([seed, who, 100]))
            for layer, (left, right) in enumerate(((54, 64), (64, 64), (64, 17)), 1):
                expected = rng.normal(0, math.sqrt(2/(left+right)), (left, right))
                require(np.array_equal(initial[who][f"W{layer}"], expected) and not initial[who][f"b{layer}"].any(), "Initial independent RNG stream differs")
            initial_parameter_values += sum(v.size for v in initial[who].values())
        for i, j in combinations(range(3), 2):
            require(all(not np.shares_memory(initial[i][key], initial[j][key]) for key in SHAPES), "Loaded actor arrays alias")
            require(all(not np.array_equal(initial[i][f"W{k}"], initial[j][f"W{k}"]) for k in (1, 2, 3)), "Initial actor weights copied")
        batch_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
        require(batch_rng.bit_generator.state == checkpoints[0][1], "Initial batch RNG differs")
        log_path = directory / "training.jsonl"
        artifacts[str(log_path)] = sha(log_path)
        require(artifacts[str(log_path)] == result["training_log_sha256"], "Training log SHA differs")
        previous_elapsed = -1.
        with log_path.open() as stream:
            row_count = 0
            for row_count, line in enumerate(stream, 1):
                row = json.loads(line); require(row["update"] == row_count <= 6000, "Update sequence differs")
                ids = batch_rng.integers(0, len(state_arrays["train"]), size=256, dtype=np.int64)
                require(row["batch_indices_sha256"] == array_sha(ids) and row["batch_states_sha256"] == array_sha(state_arrays["train"][ids]), "Training RNG/state hash differs")
                beta = .001*max(0, 1-(row_count-1)/1000)
                require(row["entropy_coefficient"] == beta, "Entropy schedule differs")
                finite_tree(row)
                require(0 <= row["native_expected_reward"] <= 1 and 0 <= row["mean_actor_entropy"] <= math.log(17)+1e-12,
                        "Reward/entropy outside bounds")
                close(row["loss"], -(row["native_expected_reward"]+beta*row["mean_actor_entropy"]), "Training scalar loss algebra differs")
                norm = row["gradient_norm"]
                require(norm >= 0, "Negative gradient norm")
                close(row["gradient_clip_scale"], min(1., 5/max(norm, 1e-300)), "Gradient clip scalar differs")
                require(row["elapsed_seconds"] >= previous_elapsed, "Training elapsed goes backwards")
                previous_elapsed = row["elapsed_seconds"]
                if row_count in CHECKPOINTS:
                    require(batch_rng.bit_generator.state == checkpoints[row_count][1], "Checkpoint batch RNG differs")
                checked_batches += 1
            require(row_count == 6000, "Missing training rows")
        require(result["final_checkpoint_sha256"] == artifacts[str(directory / "checkpoint_6000.npz")], "Final parameter anchor differs")
        require({p.name for p in directory.glob("final_*.npz")} == {f"final_{n}.npz" for n in PARTITIONS}, "Extra/missing final partition")
        final_actor = checkpoints[6000][0]
        summaries, action_counts = {}, Counter()
        for name in PARTITIONS:
            path = directory / f"final_{name}.npz"; digest = sha(path)
            require(digest == result["final"][name]["data_sha256"], "Final NPZ SHA differs")
            artifacts[str(path)] = digest
            with np.load(path, allow_pickle=False) as loaded:
                z = {k: loaded[k] for k in loaded.files}
            require(set(z) == {"states", "action_indices", "action_probabilities", "greedy_reward", "executed", "satisfied",
                              "exact_expected_reward", "exact_full_success_probability", "exact_execution_probability"}, "Final NPZ fields differ")
            states = state_arrays[name]; n = len(states)
            require(z["states"].dtype == np.int16 and np.array_equal(z["states"], states), "Final world order differs")
            require(z["action_probabilities"].shape == (n, 3, 17) and z["action_indices"].shape == (n, 3), "Final output shapes differ")
            for key, value in z.items():
                require(np.isfinite(value).all(), "Nonfinite final array: " + key)
            require(z["action_indices"].dtype == np.int16 and ((z["action_indices"] >= 0) & (z["action_indices"] < 17)).all(), "Invalid final17 action")
            distinct, counts = np.unique(z["action_indices"], axis=0, return_counts=True)
            action_counts.update({tuple(map(int, row)): int(count) for row, count in zip(distinct, counts)})
            require(z["executed"].shape == z["satisfied"].shape == (n, 3) and z["executed"].dtype == z["satisfied"].dtype == bool, "Feedback shape/type differs")
            require(np.isin(z["greedy_reward"], [0., .5, 1.]).all(), "Non-native saved reward")
            for start in range(0, n, 1024):
                chunk = states[start:start+1024]; x = features(chunk)
                probabilities = np.stack([forward(final_actor[i], np.ascontiguousarray(x[:, i])) for i in range(3)], axis=1)
                saved_p = z["action_probabilities"][start:start+len(chunk)]
                error = float(np.max(np.abs(probabilities-saved_p))); max_probability_error = max(max_probability_error, error)
                close(probabilities, saved_p, "Independent final probabilities differ")
                require(np.array_equal(np.argmax(probabilities, axis=-1), z["action_indices"][start:start+len(chunk)]), "Independent final argmax differs")
                require(np.array_equal((probabilities == probabilities.max(-1, keepdims=True)).sum(-1),
                                       (saved_p == saved_p.max(-1, keepdims=True)).sum(-1)), "Argmax tie sets differ")
                native = rewards(chunk)
                mass = probabilities[:, 0, joint[:, 0]]*probabilities[:, 1, joint[:, 1]]*probabilities[:, 2, joint[:, 2]]
                expected = {"exact_expected_reward": (mass*native).sum(1),
                            "exact_full_success_probability": (mass*(native == 1)).sum(1), "exact_execution_probability": mass.sum(1)}
                for key, value in expected.items():
                    stored = z[key][start:start+len(chunk)]
                    close(value, stored, "Exact final policy statistic differs: " + key)
                    max_expectation_error = max(max_expectation_error, float(np.max(np.abs(value-stored))))
                forward_samples += 3*len(chunk); forward_batches += 3
            all_ids = np.arange(n, dtype=np.int64)
            summary = summary_from_saved(z, all_ids)
            compare_summary(result["final"][name], summary)
            compare_summary(history[-1]["monitor"][name], summary_from_saved(z, np.asarray(specs[name]["monitor_indices"], dtype=np.int64)))
            summaries[name] = {"worlds": n, "greedy_full_successes": summary["greedy_full_successes"],
                               "greedy_full_success_rate": summary["greedy_full_success_rate"], "data_sha256": digest}
            final_files += 1
        candidate = all(summaries[name]["greedy_full_success_rate"] >= .99 for name in PARTITIONS if name != "train")
        require(result["candidate_threshold_met"] is candidate, "Per-seed candidate threshold differs")
        audit_seeds.append({"seed": seed, "final": summaries, "candidate_threshold_met": candidate})
        distinct_actions = sorted(action_counts)
        constant_rows.append({"seed": seed, "worlds": sum(action_counts.values()), "distinct_joint_argmax_actions": len(distinct_actions),
                              "action_indices": [list(v) for v in distinct_actions], "counts": [action_counts[v] for v in distinct_actions],
                              "decoded": [decode_joint(v) for v in distinct_actions]})
    require(checked_batches == 24000 and checkpoint_count == 24 and final_files == 16 and forward_samples == 1721088, "Audit scope incomplete")
    require(results["all_seed_candidates_meet_threshold"] is all(r["candidate_threshold_met"] for r in audit_seeds), "All-seed candidate flag differs")
    behavior_review = {"provided": behavior_path is not None, "full_physical_analysis_repeated": False}
    constant_path = run / "constant_action_check.json"
    constant = read(constant_path)
    require(constant["seeds"] == constant_rows, "Independent complete-domain action counts or decoding differs")
    for name, digest in constant["source_sha256"].items():
        require(artifacts[str(ROOT / name)] == digest, "Constant action check source differs")
    artifacts[str(constant_path)] = sha(constant_path)
    if behavior_path is not None:
        behavior_path = Path(behavior_path).resolve(); behavior = read(behavior_path)
        require(behavior["source_results_sha256"] == sha(execution / "results.json"), "Behavior source result differs")
        require(behavior["analysis_code_sha256"] == sha(HERE / "analyze_behavior.py"), "Behavior analyzer source differs")
        require([s["seed"] for s in behavior["seeds"]] == list(SEEDS), "Behavior seed grid differs")
        for observed, source in zip(behavior["seeds"], results["seeds"]):
            require(set(observed["partitions"]) == set(PARTITIONS), "Behavior partition grid differs")
            for name, b in observed["partitions"].items():
                s = source["final"][name]
                require(b["data_sha256"] == s["data_sha256"] and b["worlds"] == s["worlds"] and
                        sum(b["categories"].values()) == s["worlds"] and
                        sum(b["successful_pair_counts"].values()) == s["greedy_full_successes"] and
                        b["all_argmax_rewards_execution_satisfaction_verified"] is True, "Bound behavior totals differ")
        behavior_review.update({"path": str(behavior_path), "sha256": sha(behavior_path), "linked_source_and_totals_verified": True})
    artifacts.update({str(p): sha(p) for p in (run / "plan.json", run / "freeze.json", run / "prepared.json",
                    execution / "results.json", execution / "status.json", execution / "started.json", execution / "input_arrays.json")})
    return {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(), "audit_source_sha256": sha(__file__),
            "source_sha256": source_hashes, "artifacts_sha256": artifacts, "seeds": audit_seeds,
            "counts": {"independent_training_seeds": 4, "verified_training_updates": checked_batches, "reconstructed_state_samples": 6144000,
                       "checkpoints": checkpoint_count, "final_npz": final_files, "initial_parameter_values_verified": initial_parameter_values,
                       "forward_worlds": forward_samples//3, "actual_final_actor_forward_samples": forward_samples,
                       "actual_final_actor_forward_batches": forward_batches, "training_or_optimizer_replays": 0},
            "max_final_probability_abs_error": max_probability_error, "max_final_exact_statistic_abs_error": max_expectation_error,
            "numerical_tolerance": {"absolute": 2e-12, "relative": 2e-12}, "behavior_review": behavior_review,
            "gradient_sign_review": sign_review, "posthoc_constant_argmax": constant_rows,
            "all_seed_candidates_meet_threshold": results["all_seed_candidates_meet_threshold"], "symbolic_phase_unlocked": False,
            "limits": ["Final saved policies were actually forwarded; this is not a zero-forward audit",
                       "Training batch RNG, scalar loss algebra, checkpoint shapes/finite values verified, but optimizer updates were not replayed",
                       "No forward pass of intermediate checkpoints; only their recorded support/counts and final monitor subset were checked",
                       "Loaded arrays do not share memory and match independent initial RNG streams; historical runtime aliasing additionally relies on frozen construction source",
                       "Physical classifications are owned by the separately bound behavior analysis, not independently repeated here"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--behavior", type=Path)
    args = parser.parse_args(); out = args.out or args.run / "audit_execution_001"
    require(not out.exists(), "Do not overwrite or automatically retry an audit")
    out.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args.run, args.behavior)
        (out / "verification.json").write_bytes(json_bytes(result))
        report = f'''# 可训练D1基线执行独立核验

核验通过。四个种子的24000次更新记录及6144000个状态抽样哈希重建一致；24个检查点、16份完整终点NPZ和冻结来源相符。初始权重逐项匹配独立SeedSequence，Adam初值为0；所有参数、日志与保存数组的有限值及固定熵时间表通过。

本次实际加载保存末点参数，以独立NumPy矩阵计算重算573696个世界、1721088个actor样本的概率和argmax；没有训练或重放优化。概率最大绝对误差{result['max_final_probability_abs_error']:.3g}，精确随机政策统计最大误差{result['max_final_exact_statistic_abs_error']:.3g}，容差为atol=rtol=2e-12。所有实际argmax及并列数相同，最终监测子集与完整终点一致。

全部种子都满足既定99%候选筛查：{result['all_seed_candidates_meet_threshold']}。逐种子、逐格原始计数保存在[verification.json](verification.json)，没有删除失败或改变门槛，symbolic_phase_unlocked仍为false。物理分类另由根任务的独立行为分析负责；本次是否绑定并核对其来源和总数：{result['behavior_review']['provided']}。

冻结代码的梯度符号没有反转：loss为负的期望回报与熵奖励之和，Adam减去该loss梯度；训练前的logit及网络参数有限差分核验与本批源码一致。本次没有重放优化器来重新验证每步更新。

事后常动作记录独立核对一致：每个种子的全部143424个世界恰有1个联合argmax，分别为43101的BC/S2/R、43102的AC/S1/R、43103的AC/S2/L、43104的AB/S0/R。该结论不表示动作概率或隐藏表示完全忽略输入，也不确定造成这种结果的机制。

本核验没有重放优化器、重新计算每一训练步的中间政策，不能证明所有未保存中间参数值。初值与当前载入数组的独立性已核，历史运行时是否共享内存还依赖冻结构造代码。四个训练种子才是独立重复；完整状态数不增加训练重复数。此批是完整信息、反事实富反馈能力控制，没有新符号训练或语言形成结论。
'''
        (out / "独立核验.md").write_text(report)
        print(json.dumps({k: result[k] for k in ("status", "counts", "max_final_probability_abs_error", "all_seed_candidates_meet_threshold")}, ensure_ascii=False, indent=2))
    except BaseException as error:
        (out / "failure.json").write_bytes(json_bytes({"status": "failed", "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": str(error), "traceback": traceback.format_exc(), "audit_source_sha256": sha(__file__), "automatic_retry": False}))
        raise


if __name__ == "__main__":
    main()
