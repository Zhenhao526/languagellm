"""Audit completed16-run message learning without optimizer replay.

Only saved final networks are forwarded. Intermediate monitoring arrays are
checked as records and native settlements, not regenerated from their weights.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import platform
import re
import time
import traceback
import numpy as np

from research_program.triadic_learning_baseline import audit_execution as pure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (47101, 47102, 47103, 47104)
CONDITIONS = ("FI_silent", "FI_live", "PI_silent", "PI_live")
PARTITIONS = pure.PARTITIONS
CHECKPOINTS = pure.CHECKPOINTS
MODULES = ("sender1", "sender2", "action")
DIMENSIONS = ((54, 64, 64, 32), (153, 64, 64, 32), (252, 64, 64, 17))
PARAMETER_KEYS = ("W1", "b1", "W2", "b2", "W3", "b3")
TOL = 2e-12
require, read, sha, array_sha, json_bytes, close, finite_tree = (
    pure.require, pure.read, pure.sha, pure.array_sha, pure.json_bytes, pure.close, pure.finite_tree)


def json_sha(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def observed_features(states, full):
    x = pure.features(states)
    if not full:
        x[:, :, 53] = 0
        for who in range(3):
            for other in range(3):
                if other != who:
                    x[:, who, 7*other:7*(other+1)] = 0
            for site in (1, 2, 3):
                hidden = states[:, 7+who] != site
                x[hidden, who, 21+5*site:21+5*(site+1)] = 0
    return x


def probabilities(network, inputs):
    first = np.tanh(inputs@network["W1"]+network["b1"])
    second = np.tanh(first@network["W2"]+network["b2"])
    logits = second@network["W3"]+network["b3"]
    if logits.shape[-1] == 32:
        logits = logits.reshape(len(inputs), 4, 8)
    require(np.isfinite(logits).all(), "Nonfinite independently computed logits")
    exp = np.exp(logits-logits.max(-1, keepdims=True))
    return exp/exp.sum(-1, keepdims=True)


def route(messages, live):
    # All96 payload coordinates precede3 visibility coordinates; no interleaving.
    result = np.zeros((len(messages), 3, 99), dtype=np.float64)
    rows = np.arange(len(messages))
    for viewer in range(3):
        for sender in range(3):
            if live or sender == viewer:
                result[:, viewer, 96+sender] = 1
                for position in range(4):
                    result[rows, viewer, 32*sender+8*position+messages[:, sender, position]] = 1
    return result


def final_forward(networks, states, condition):
    full, live = condition.startswith("FI_"), condition.endswith("_live")
    x = observed_features(states, full)
    messages = []; inputs = x; token_ties = 0
    for window in range(2):
        p = np.stack([probabilities(networks[3*who+window], inputs[:, who]) for who in range(3)], axis=1)
        token_ties += int(((p == p.max(-1, keepdims=True)).sum(-1) > 1).sum())
        m = p.argmax(-1).astype(np.int8); messages.append(m)
        if window == 0:
            inputs = np.concatenate((x, route(m, live)), axis=-1)
    inputs = np.concatenate((x, route(messages[0], live), route(messages[1], live)), axis=-1)
    actions = np.stack([probabilities(networks[3*who+2], inputs[:, who]) for who in range(3)], axis=1)
    return np.stack(messages, axis=1), actions, token_ties


def native_settlement(states, choices):
    active = choices != 0; active_count = active.sum(1)
    value = np.maximum(choices-1, 0)
    sites, destination = value//4, (value//2) % 2
    partners = np.empty_like(choices)
    for who in range(3):
        partners[:, who] = np.asarray([j for j in range(3) if j != who])[value[:, who] % 2]
    executed = np.zeros_like(active)
    for i, j in combinations(range(3), 2):
        matching = ((active_count == 2) & active[:, i] & active[:, j] & (partners[:, i] == j)
                    & (partners[:, j] == i) & (sites[:, i] == sites[:, j]) & (destination[:, i] == destination[:, j]))
        executed[:, i] |= matching; executed[:, j] |= matching
    satisfied = np.zeros_like(executed)
    resource = np.asarray([[m in group for m in range(4)] for group in ((0, 1), (2, 3), (0, 2), (1, 3))])
    target = np.asarray([[d in group for d in range(2)] for group in ((0,), (1,), (0, 1))])
    for who in range(3):
        material = states[np.arange(len(states)), 3+sites[:, who]]; need = states[:, who]
        satisfied[:, who] = executed[:, who] & resource[need//3, material] & target[need % 3, destination[:, who]]
    return satisfied.sum(1)/2., executed, satisfied


def network_hash(networks):
    return json_sha({f"agent{who}_{module}_{key}": array_sha(networks[3*who+m][key])
                     for who in range(3) for m, module in enumerate(MODULES) for key in PARAMETER_KEYS})


def checkpoint(path, update, seed):
    networks = []; expected_keys = {"update", "batch_rng_json", "message_rngs_json"}
    with np.load(path, allow_pickle=False) as z:
        require(z["update"].dtype == np.int64 and z["update"].shape == () and int(z["update"]) == update, "Checkpoint update/type")
        for who in range(3):
            for module_id, module in enumerate(MODULES):
                dims = DIMENSIONS[module_id]; net = {}
                rng = np.random.default_rng(np.random.SeedSequence([seed, who, module_id, 100])) if update == 0 else None
                for layer, (left, right) in enumerate(zip(dims, dims[1:]), 1):
                    expected_weight = rng.normal(0, math.sqrt(2/(left+right)), (left, right)) if rng is not None else None
                    for prefix, shape in (("W", (left, right)), ("b", (right,))):
                        key = f"{prefix}{layer}"; name = f"agent{who}_{module}_{key}"
                        expected_keys.add(name); value = z[name]
                        require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(), "Checkpoint parameter shape/finite")
                        if update == 0:
                            require(np.array_equal(value, expected_weight if prefix == "W" else np.zeros(right)), "Independent initial stream differs")
                        net[key] = value.copy()
                        for moment in ("m", "v"):
                            name = f"adam_agent{who}_{module}_{moment}_{key}"; expected_keys.add(name); value = z[name]
                            require(value.shape == shape and value.dtype == np.float64 and np.isfinite(value).all(), "Adam shape/finite")
                            if moment == "v":
                                require((value >= 0).all(), "Negative Adam second moment")
                            if update == 0:
                                require(not value.any(), "Nonzero initial optimizer state")
                networks.append(net)
        require(set(z.files) == expected_keys, "Extra/missing checkpoint fields")
        world_rng = json.loads(str(z["batch_rng_json"].item())); message_rng = json.loads(str(z["message_rngs_json"].item()))
    for i, j in combinations(range(9), 2):
        require(all(not np.shares_memory(networks[i][key], networks[j][key]) for key in PARAMETER_KEYS), "Loaded modules share memory")
    return networks, world_rng, message_rng


def summary(values, condition):
    a, p, reward = values["action_indices"], values["action_probabilities"], values["greedy_reward"]
    active = (a != 0).sum(1); executed = values["executed"].sum(1); n = len(a)
    unique, counts = np.unique(a, axis=0, return_counts=True)
    result = {"worlds": n, "condition": condition, "message_mode": "greedy", "action_mode": "greedy",
        "greedy_reward_sum": float(reward.sum()), "greedy_reward_mean": float(reward.mean()),
        "greedy_full_successes": int((reward == 1).sum()), "greedy_full_success_rate": float((reward == 1).mean()),
        "greedy_non_full_success_worlds": int((reward != 1).sum()),
        "greedy_reward_counts": {str(r): int((reward == r).sum()) for r in (0., .5, 1.)},
        "greedy_attempted_transports": int(active.sum()), "greedy_executed_transports": int(values["executed"].sum()),
        "greedy_satisfied_agents": int(values["satisfied"].sum()), "greedy_active_count_worlds": {str(k): int((active == k).sum()) for k in range(4)},
        "greedy_agent_argmax_ties": int(((p == p.max(-1, keepdims=True)).sum(-1) > 1).sum()),
        "conditional_exact_expected_reward_mean": float(values["conditional_exact_expected_reward"].mean()),
        "conditional_exact_full_success_probability_mean": float(values["conditional_exact_full_success_probability"].mean()),
        "conditional_exact_physical_execution_probability_mean": float(values["conditional_exact_execution_probability"].mean()),
        "greedy_failure_categories": {"all_wait": int((active == 0).sum()), "single_transport_proposal": int((active == 1).sum()),
             "overload": int((active == 3).sum()), "two_unmatched": int(((active == 2)&(executed == 0)).sum()),
             "matched_no_need_satisfied": int(((executed == 2)&(reward == 0)).sum()), "matched_one_need_satisfied": int((reward == .5).sum())},
        "greedy_executed_pair_worlds": {"ABC"[i]+"ABC"[j]: int((values["executed"][:, i]&values["executed"][:, j]).sum()) for i, j in combinations(range(3), 2)},
        "distinct_joint_argmax_actions": len(unique), "joint_argmax_action_counts": [{"action_indices": row.tolist(), "worlds": int(count)} for row, count in zip(unique, counts)],
        "state_indices_sha256": array_sha(values["state_indices"]), "packed_states_sha256": array_sha(values["states"]),
        "messages_sha256": array_sha(values["messages"]), "action_indices_sha256": array_sha(a)}
    return result


def compare_mapping(actual, expected, prefix):
    for key, value in expected.items():
        require(key in actual, "Missing summary field "+prefix+key)
        if isinstance(value, float):
            close(actual[key], value, "Summary mismatch "+prefix+key)
        else:
            require(actual[key] == value, "Summary mismatch "+prefix+key)


def check_evaluation(path, expected_states, expected_indices, recorded, condition, networks=None):
    digest = sha(path); require(digest == recorded["data_sha256"], "Evaluation SHA differs")
    with np.load(path, allow_pickle=False) as z:
        values = {key: z[key] for key in z.files}
    expected_keys = {"states", "state_indices", "messages", "action_indices", "action_probabilities", "greedy_reward", "executed", "satisfied",
                     "conditional_exact_expected_reward", "conditional_exact_full_success_probability", "conditional_exact_execution_probability"}
    require(set(values) == expected_keys, "Evaluation fields differ")
    require(values["states"].dtype == np.int16 and np.array_equal(values["states"], expected_states)
            and values["state_indices"].dtype == np.int64 and np.array_equal(values["state_indices"], expected_indices), "Evaluation state order differs")
    n = len(expected_states); p, a, m = values["action_probabilities"], values["action_indices"], values["messages"]
    require(p.shape == (n, 3, 17) and p.dtype == np.float64 and (p >= 0).all() and np.allclose(p.sum(-1), 1., atol=TOL, rtol=TOL), "Action probability shape/normalization")
    require(a.shape == (n, 3) and a.dtype == np.int16 and ((a >= 0)&(a < 17)).all() and np.array_equal(a, p.argmax(-1)), "Saved action not full17 argmax")
    require(m.shape == (n, 2, 3, 4) and m.dtype == np.int8 and ((m >= 0)&(m < 8)).all(), "Saved messages outside2window4position8category")
    for key, value in values.items():
        require(np.isfinite(value).all(), "Nonfinite evaluation array "+key)
    require(values["executed"].dtype == values["satisfied"].dtype == bool and values["executed"].shape == values["satisfied"].shape == (n, 3), "Feedback arrays differ")
    for key in ("greedy_reward", "conditional_exact_expected_reward", "conditional_exact_full_success_probability", "conditional_exact_execution_probability"):
        require(values[key].dtype == np.float64 and values[key].shape == (n,), "Scalar evaluation array differs "+key)
    r, e, s = native_settlement(expected_states, a)
    require(np.array_equal(r, values["greedy_reward"]) and np.array_equal(e, values["executed"])
            and np.array_equal(s, values["satisfied"]), "Independent native settlement differs")
    max_probability_error = max_stat_error = 0.; sender_ties = 0; forward_batches = 0
    joint = pure.joint_indices()
    for start in range(0, n, 1024):
        states = expected_states[start:start+1024]; saved_p = p[start:start+len(states)]
        if networks is not None:
            messages, recomputed, ties = final_forward(networks, states, condition)
            require(np.array_equal(messages, m[start:start+len(states)]), "Independent greedy message differs")
            close(recomputed, saved_p, "Independent final action probability differs")
            require(np.array_equal(recomputed.argmax(-1), a[start:start+len(states)]), "Independent final action differs")
            require(np.array_equal((recomputed == recomputed.max(-1, keepdims=True)).sum(-1),
                                   (saved_p == saved_p.max(-1, keepdims=True)).sum(-1)), "Final action tie count differs")
            max_probability_error = max(max_probability_error, float(np.max(np.abs(recomputed-saved_p))))
            sender_ties += ties; forward_batches += 9
        masses = saved_p[:, 0, joint[:, 0]]*saved_p[:, 1, joint[:, 1]]*saved_p[:, 2, joint[:, 2]]
        rewards = pure.rewards(states)
        statistics = {"conditional_exact_expected_reward": (masses*rewards).sum(1),
                      "conditional_exact_full_success_probability": (masses*(rewards == 1)).sum(1), "conditional_exact_execution_probability": masses.sum(1)}
        for key, value in statistics.items():
            saved = values[key][start:start+len(states)]; close(saved, value, "Conditional exact statistic differs")
            max_stat_error = max(max_stat_error, float(np.max(np.abs(saved-value))))
    observed = summary(values, condition); compare_mapping(recorded, observed, str(path)+":")
    require("conditional" in recorded["exact_statistics_scope"], "Conditional-statistic scope missing")
    require(type(recorded["greedy_sender_token_argmax_ties"]) == int and 0 <= recorded["greedy_sender_token_argmax_ties"] <= n*24, "Recorded token ties outside bounds")
    if networks is not None:
        require(sender_ties == recorded["greedy_sender_token_argmax_ties"], "Final sender ties differ")
    return values, observed, {"sha256": digest, "max_probability_error": max_probability_error, "max_conditional_statistic_error": max_stat_error,
                              "actual_forward_network_batches": forward_batches, "actual_forward_network_samples": 9*n if networks is not None else 0}


def audit(run):
    started = time.perf_counter(); run = Path(run).resolve(); execution = run/"execution"
    plan, prepared, frozen = [read(run/name) for name in ("plan.json", "prepared.json", "freeze.json")]
    results, status = read(execution/"results.json"), read(execution/"status.json")
    require(results["status"] == status["status"] == "completed", "Wait for all16 runs before audit")
    require(not list(execution.rglob("failure.json")), "Failure artifact present")
    require(sha(run/"plan.json") == frozen["plan_sha256"] == results["plan_sha256"], "Plan changed")
    require(sha(run/"prepared.json") == frozen["prepared_sha256"] == plan["prepared_sha256"], "Prepared changed")
    require(plan["runtime"] == {"python": platform.python_version(), "numpy": np.__version__}, "Audit numerical runtime differs")
    require(plan["config"] == prepared["config"], "Prepared/config mismatch")
    config = plan["config"]
    expected_config = {"seeds": list(SEEDS), "conditions": list(CONDITIONS), "updates": 6000, "batch_size": 256,
        "checkpoints": list(CHECKPOINTS), "dimensions": {module: list(DIMENSIONS[i]) for i, module in enumerate(MODULES)},
        "sender_windows": 2, "sender_tokens_per_window": 4, "alphabet": ["@", "#", "$", "%", "&", "*", "+", "~"],
        "trajectories_per_state": 2, "sender_entropy_coefficient": 0, "same_uniforms_across_conditions": True,
        "silent_retains_own_messages": True, "learning_rate": .001, "adam_beta1": .9, "adam_beta2": .999,
        "adam_epsilon": 1e-8, "global_gradient_clip": 5., "entropy_initial": .001, "entropy_zero_after_updates": 1000,
        "dtype": "float64", "evaluation_batch_size": 1024, "deployment": "greedy_messages_then_independent_greedy_actions"}
    compare_mapping(config, expected_config, "config:")
    artifacts = {}; source_hashes = {}
    for name, digest in plan["sources"].items():
        for path in (ROOT/name, run/"source_snapshot"/name):
            require(sha(path) == digest, "Frozen/current source changed "+str(path)); source_hashes[str(path)] = digest
    prior = ROOT/"research_program/triadic_learning_baseline/results/learning_001/audit_execution_001/verification.json"
    require(sha(pure.__file__) == read(prior)["audit_source_sha256"], "Reused independent helper differs from audited version")
    source_hashes[str(Path(pure.__file__))] = sha(pure.__file__); artifacts[str(prior)] = sha(prior)
    gradient_path = HERE/"gradient_audit_001/verification.json"; gradient = read(gradient_path)
    require(gradient["status"] == "passed", "Missing mathematical preflight")
    nodes = ast.parse((HERE/"runner.py").read_text()).body
    for name, digest in gradient["core_function_AST_sha256"].items():
        node = next(n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == name)
        require(hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest() == digest, "Gradient-audited core changed "+name)
    artifacts[str(gradient_path)] = sha(gradient_path)
    if (run/"preflight").exists():
        for path in sorted((run/"preflight").rglob("*")):
            if path.is_file():
                artifacts[str(path)] = sha(path)
    specs, train_need, held_need = pure.split_specs()
    require(prepared["partitions"] == specs and prepared["train_need_multisets"] == train_need and prepared["heldout_need_multisets"] == held_need, "Independent partition reconstruction differs")
    expected_runs = [{"seed": seed, "condition": condition, "directory": f"seed_{seed}_{condition}"} for seed in SEEDS for condition in CONDITIONS]
    require(prepared["runs"] == expected_runs and [(r["seed"], r["condition"]) for r in results["runs"]]
            == [(r["seed"], r["condition"]) for r in expected_runs], "16run complete ordered grid differs")
    require({p.name for p in execution.glob("seed_*")} == {r["directory"] for r in expected_runs}, "Extra/missing run directory")
    require({p.name for p in execution.glob("worker_seed_*")} == {f"worker_seed_{seed}" for seed in SEEDS}, "Extra/missing worker directory")
    counts_expected = {"completed_run_count": 16, "paired_seed_count": 4, "updates_total": 96000,
        "training_world_samples_total": 24576000, "sampled_complete_message_trajectories_total": 49152000,
        "weighted_structural_action_contributions": 1179648000, "offline_reward_table_entries_actual": 13768704,
        "language_or_convention_claim_automatically_supported": False}
    compare_mapping(results, counts_expected, "result:"); require(status["completed_runs"] == 16, "Terminal completed runs")
    require(prepared["categorical_message_samples_total"] == 1179648000 and prepared["complete_final_world_evaluations"] == 2294784, "Prepared total budget")
    finite_tree(results)
    top_started = read(execution/"started.json")
    require(top_started["plan_sha256"] == frozen["plan_sha256"] and top_started["device"] == "cpu_numpy"
            and all(v == "1" for v in top_started["thread_environment"].values()), "Started source/device/thread")
    require(len(results["worker_processes"]) == 4 and len({r["pid"] for r in results["worker_processes"]}) == 4
            and all(r["exitcode"] == 0 for r in results["worker_processes"]), "Four workers did not exit0")
    states = {name: pure.packed(spec) for name, spec in specs.items()}
    independent_inputs = {}
    for name, state in states.items():
        full = observed_features(state, True); private = observed_features(state, False)
        independent_inputs[name] = {"worlds": len(state), "full_information_features_sha256": array_sha(full),
            "private_information_features_sha256": array_sha(private), "native_rewards_sha256": array_sha(pure.rewards(state)), "states_sha256": array_sha(state)}
        del full, private
    input_summary = read(execution/"input_arrays.json")
    require(input_summary == {"worker_count": 4, "all_worker_arrays_identical": True, "arrays": independent_inputs,
                             "actual_offline_reward_table_entries": 13768704}, "Worker FI/PI/reward arrays differ from independent encoding")
    run_index = {(r["seed"], r["condition"]): r for r in results["runs"]}
    logs_checked = checkpoints_checked = final_checked = monitor_checked = final_network_samples = network_batches = physical_worlds = 0
    max_probability_error = max_stat_error = 0.; summaries = []; pair_receipts = []
    for seed in SEEDS:
        worker = execution/f"worker_seed_{seed}"
        worker_started, worker_results, worker_status = [read(worker/name) for name in ("started.json", "results.json", "status.json")]
        process = next(p for p in results["worker_processes"] if p["name"] == f"triadic_messages_seed_{seed}")
        require(worker_started["seed"] == seed and worker_started["pid"] == process["pid"] and worker_started["parent_pid"] == top_started["pid"]
                and worker_started["plan_sha256"] == frozen["plan_sha256"] and all(v == "1" for v in worker_started["thread_environment"].values()), "Worker identity/source/threads")
        require(worker_results["status"] == worker_status["status"] == "completed" and worker_results["seed"] == seed
                and worker_results["runs"] == [run_index[(seed, c)] for c in CONDITIONS], "Worker result aggregation differs")
        require(read(worker/"input_arrays.json") == independent_inputs and worker_results["offline_reward_table_entries"] == 3442176, "Worker input/precompute count")
        for path in worker.glob("*.json"):
            artifacts[str(path)] = sha(path)
        histories = {}; checkpoint_rng = {}; initial_hashes = {}; final_networks = {}
        for condition in CONDITIONS:
            directory = execution/f"seed_{seed}_{condition}"; row = run_index[(seed, condition)]
            require(read(directory/"result.json") == row and row["updates"] == 6000 and row["training_world_samples"] == 1536000
                    and row["sampled_complete_message_trajectories"] == 3072000, "Run result/budget differs")
            monitor = [json.loads(line) for line in (directory/"monitor.jsonl").read_text().splitlines() if line.strip()]
            require(monitor == row["monitor"] and [v["update"] for v in monitor] == list(CHECKPOINTS), "Monitor/checkpoint sequence differs")
            histories[condition] = monitor; checkpoint_rng[condition] = {}
            require({p.name for p in directory.glob("checkpoint_*.npz")} == {f"checkpoint_{u:04d}.npz" for u in CHECKPOINTS}, "Checkpoint inventory")
            require({p.name for p in directory.glob("monitor_*.npz")} == {f"monitor_{u:04d}_{p}.npz" for u in CHECKPOINTS for p in PARTITIONS}, "Monitor NPZ inventory")
            require({p.name for p in directory.glob("final_*.npz")} == {f"final_{p}.npz" for p in PARTITIONS}, "Final NPZ inventory")
            for entry in monitor:
                update = entry["update"]; path = directory/f"checkpoint_{update:04d}.npz"; digest = sha(path)
                require(digest == entry["checkpoint_sha256"], "Checkpoint SHA")
                nets, world_rng, message_rng = checkpoint(path, update, seed)
                checkpoint_rng[condition][update] = (world_rng, message_rng); artifacts[str(path)] = digest
                checkpoints_checked += 1
                if update == 0:
                    initial_hashes[condition] = network_hash(nets)
                    require(initial_hashes[condition] == row["initial_parameter_sha256"], "Initial parameter hash")
                if update == 6000:
                    require(network_hash(nets) == row["final_parameter_sha256"] and digest == row["final_checkpoint_sha256"], "Final parameter hash")
                    final_networks[condition] = nets
                for name in PARTITIONS:
                    ids = np.asarray(specs[name]["monitor_indices"], dtype=np.int64)
                    mpath = directory/f"monitor_{update:04d}_{name}.npz"
                    data, _, receipt = check_evaluation(mpath, states[name][ids], ids, entry["monitor"][name], condition)
                    artifacts[str(mpath)] = receipt["sha256"]; max_stat_error = max(max_stat_error, receipt["max_conditional_statistic_error"])
                    monitor_checked += 1; physical_worlds += len(ids)
                    del data
            for name in ("result.json", "monitor.jsonl", "training.jsonl"):
                artifacts[str(directory/name)] = sha(directory/name)
            require(artifacts[str(directory/"training.jsonl")] == row["training_log_sha256"], "Training log SHA")
        require(len(set(initial_hashes.values())) == 1, "Conditions do not share initial parameters")
        world = np.random.default_rng(np.random.SeedSequence([seed, 200]))
        message_rngs = {(t, w, a): np.random.default_rng(np.random.SeedSequence([seed, a, t, w, 300]))
                        for t in range(2) for w in range(2) for a in range(3)}
        def verify_rng_at(update):
            expected_message = {f"t{t}_w{w}_a{a}": rng.bit_generator.state for (t, w, a), rng in message_rngs.items()}
            for condition in CONDITIONS:
                wrng, mrng = checkpoint_rng[condition][update]
                require(wrng == world.bit_generator.state and mrng == expected_message, "Saved world/message RNG state differs")
        verify_rng_at(0)
        streams = {c: (execution/f"seed_{seed}_{c}/training.jsonl").open() for c in CONDITIONS}
        previous_time = {c: -1. for c in CONDITIONS}
        try:
            for update in range(1, 6001):
                ids = world.integers(0, len(states["train"]), size=256, dtype=np.int64)
                uniforms = np.empty((2, 256, 2, 3, 4), dtype=np.float64)
                for (t, w, a), rng in message_rngs.items():
                    uniforms[t, :, w, a] = rng.random((256, 4))
                expected_log = {"update": update, "batch_indices_sha256": array_sha(ids), "batch_states_sha256": array_sha(states["train"][ids]),
                                "sample_uniforms_sha256": array_sha(uniforms), "entropy_coefficient": .001*max(0, 1-(update-1)/1000)}
                for condition in CONDITIONS:
                    line = streams[condition].readline(); require(bool(line), "Truncated training log")
                    row = json.loads(line); compare_mapping(row, expected_log, "training:"); finite_tree(row)
                    require(row["condition"] == condition and 0 <= row["mean_J"] <= 1 and row["min_log_J"] <= row["mean_log_J"] <= row["max_log_J"] <= TOL,
                            "Training objective/domain fields")
                    require(0 <= row["mean_actor_entropy"] <= math.log(17)+TOL and 0 <= row["zero_float_J_states"] <= 512, "Training entropy/zeroJ count")
                    close(row["mean_F"], row["mean_log_J"]+row["entropy_coefficient"]*row["mean_actor_entropy"], "Training F algebra")
                    close(row["receiver_loss"], -row["mean_F"], "Training receiver loss sign")
                    close(row["sender_advantage_mean"], 0., "LOO advantages not opposite")
                    require(row["sender_advantage_abs_mean"] >= 0 and row["sender_advantage_max_abs"] >= row["sender_advantage_abs_mean"]-TOL
                            and row["sender_advantage_squared_mean"] >= row["sender_advantage_abs_mean"]**2-TOL, "LOO summary bounds")
                    require(row["sender_mean_complete_log_score"] <= TOL and re.fullmatch(r"[0-9a-f]{64}", row["sampled_messages_sha256"]), "Complete score/message receipt malformed")
                    require(row["gradient_norm"] >= 0 and row["elapsed_seconds"] >= previous_time[condition], "Gradient/time invalid")
                    close(row["gradient_clip_scale"], min(1., 5/max(row["gradient_norm"], 1e-300)), "Gradient clipping algebra")
                    previous_time[condition] = row["elapsed_seconds"]; logs_checked += 1
                if update in CHECKPOINTS:
                    verify_rng_at(update)
            require(all(not f.read() for f in streams.values()), "Extra training log rows")
        finally:
            for stream in streams.values(): stream.close()
        pair_receipts.append({"seed": seed, "same_initial_parameters": True, "paired_world_and_message_uniform_updates": 6000,
                              "same_entropy_schedule": True, "initial_policy_outputs_not_required_equal": True})
        for condition in CONDITIONS:
            directory = execution/f"seed_{seed}_{condition}"; row = run_index[(seed, condition)]
            finals = {}; action_counts = Counter()
            for name in PARTITIONS:
                path = directory/f"final_{name}.npz"; ids = np.arange(len(states[name]), dtype=np.int64)
                values, observed, receipt = check_evaluation(path, states[name], ids, row["final"][name], condition, final_networks[condition])
                artifacts[str(path)] = receipt["sha256"]; final_checked += 1; physical_worlds += len(ids)
                final_network_samples += receipt["actual_forward_network_samples"]; network_batches += receipt["actual_forward_network_batches"]
                max_probability_error = max(max_probability_error, receipt["max_probability_error"]); max_stat_error = max(max_stat_error, receipt["max_conditional_statistic_error"])
                monitor_ids = np.asarray(specs[name]["monitor_indices"], dtype=np.int64)
                with np.load(directory/f"monitor_6000_{name}.npz", allow_pickle=False) as loaded:
                    for field in values:
                        selected = values[field][monitor_ids]
                        if values[field].dtype.kind == "f":
                            close(loaded[field], selected, "Final monitor subset differs "+field)
                        else:
                            require(np.array_equal(loaded[field], selected), "Final monitor subset differs "+field)
                for item in observed["joint_argmax_action_counts"]:
                    action_counts[tuple(item["action_indices"])] += item["worlds"]
                finals[name] = {"worlds": len(ids), "full_successes": observed["greedy_full_successes"], "full_success_rate": observed["greedy_full_success_rate"],
                                "mean_native_reward": observed["greedy_reward_mean"], "conditional_expected_reward": observed["conditional_exact_expected_reward_mean"]}
                del values
            domain = {"worlds": sum(action_counts.values()), "distinct_joint_argmax_actions": len(action_counts),
                      "joint_argmax_action_counts": [{"action_indices": list(a), "worlds": n} for a, n in sorted(action_counts.items())]}
            require(domain == row["full_domain_actions"], "Full-domain action inventory differs")
            candidate = all(finals[p]["full_success_rate"] >= .99 for p in PARTITIONS if p != "train")
            require(row["candidate_threshold_met"] is candidate, "Candidate threshold differs")
            summaries.append({"seed": seed, "condition": condition, "final": finals, "candidate_threshold_met": candidate})
        del final_networks
    require(logs_checked == 96000 and checkpoints_checked == 96 and monitor_checked == 384 and final_checked == 64
            and final_network_samples == 20653056 and physical_worlds == 2688000, "Incomplete audit budget")
    require(results["pairing_checks"] == pair_receipts, "Main paired receipt differs")
    compare_rows = []
    for seed in SEEDS:
        rates = {c: next(r for r in summaries if r["seed"] == seed and r["condition"] == c)["final"]["new_needs_and_layouts"]["full_success_rate"] for c in CONDITIONS}
        pi, fi = rates["PI_live"]-rates["PI_silent"], rates["FI_live"]-rates["FI_silent"]
        compare_rows.append({"seed": seed, "full_success_rates": rates, "PI_live_minus_silent": pi, "FI_live_minus_silent": fi,
                             "difference_in_differences_PI_minus_FI": pi-fi})
    primary = {"partition": "new_needs_and_layouts", "endpoint_update": 6000, "metric": "greedy_full_success_rate", "primary": "PI_live_minus_silent",
        "seed_pairs": compare_rows, "equal_weight_means": {key: sum(r[key] for r in compare_rows)/4 for key in
             ("PI_live_minus_silent", "FI_live_minus_silent", "difference_in_differences_PI_minus_FI")},
        "independent_paired_initializations": 4, "significance_test": None}
    require(primary == results["primary_comparison"], "Independent PI/FI/DiD primary contrast differs")
    require(results["all_seed_candidates_by_condition"] == {c: all(r["candidate_threshold_met"] for r in summaries if r["condition"] == c) for c in CONDITIONS}, "Condition candidate flags differ")
    for path in (run/"plan.json", run/"prepared.json", run/"freeze.json", execution/"results.json", execution/"status.json", execution/"started.json", execution/"input_arrays.json"):
        artifacts[str(path)] = sha(path)
    for path, digest in source_hashes.items():
        require(sha(path) == digest, "Audited source changed during audit")
    for path, digest in artifacts.items():
        require(sha(path) == digest, "Audited record changed during audit")
    return {"status": "passed", "checked_at": datetime.now(timezone.utc).isoformat(), "elapsed_seconds": time.perf_counter()-started,
        "audit_source_sha256": sha(__file__), "sources_sha256": source_hashes, "artifacts_sha256": artifacts,
        "counts": {"paired_training_seeds": 4, "completed_runs": 16, "verified_training_records": logs_checked,
            "reconstructed_distinct_world_and_uniform_updates": 24000, "verified_condition_world_draws": 24576000,
            "verified_condition_symbol_draws": 1179648000, "checkpoint_files": checkpoints_checked, "monitor_npz": monitor_checked,
            "full_final_npz": final_checked, "independent_native_settlements": physical_worlds,
            "actual_full_final_forward_worlds": final_network_samples//9, "actual_network_forward_samples": final_network_samples,
            "actual_network_forward_batches": network_batches, "optimizer_replays": 0, "training_updates": 0},
        "max_action_probability_abs_error": max_probability_error, "max_conditional_statistic_abs_error": max_stat_error,
        "tolerance": {"absolute": TOL, "relative": TOL}, "runs": summaries, "primary_comparison": primary,
        "limits": ["Saved final networks were actually forwarded; this is not a zero-forward audit",
            "Intermediate monitor arrays were checked for support, native settlement and conditional statistics, not regenerated from intermediate networks",
            "Intermediate sender tie counts are only bounded as recorded; final sender ties are independently recomputed",
            "Training U and world RNG were reconstructed; sampled training token hashes were bound and format-checked, not regenerated without intermediate forward/optimizer replay",
            "No optimizer replay or recalculation of every training gradient; pure gradient audit and frozen code provide separate evidence",
            "Loaded module independence and exact initial streams are checked; historical parameter aliasing additionally relies on frozen construction code",
            "Conditional exact action statistics condition on greedy messages and do not integrate stochastic communication trajectories",
            "Four paired seeds are independent repetitions; large finite state counts do not establish language or population-wide significance"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", type=Path, required=True); parser.add_argument("--out", type=Path)
    args = parser.parse_args(); out = args.out or args.run/"audit_execution_001"
    require(not out.exists(), "No overwrite or automatic retry")
    out.mkdir(parents=True)
    try:
        result = audit(args.run); (out/"verification.json").write_bytes(json_bytes(result))
        (out/"独立核验.md").write_text(f'''# 有限符号实验执行独立核验

核验通过。16运行、96000更新的world与12条消息随机流逐批重建一致，96检查点、384监测NPZ、64完整终点与冻结来源一致。FI/PI观察权限及live/silent路由独立实现；初值与四格配对相符。

本次实际载入保存末点参数，重算2294784世界、20653056个网络输入样本的两窗greedy消息及完整17行动；没有训练或重放优化。最大行动概率绝对误差{result['max_action_probability_abs_error']:.3g}，conditional精确统计最大误差{result['max_conditional_statistic_abs_error']:.3g}，容差atol=rtol=2e-12。全部消息、动作及终点并列计数一致。含监测记录的2688000个原生结算独立重算一致。

四种子的PI差、FI差及差中差与主结果一致，完整数值见[verification.json](verification.json)。不以平均值删去失败或改旧门槛。

没有重算中间检查点的神经前向，也没有重放每步梯度；训练token哈希只绑定记录，未靠末点参数伪造其再现。中间监测的sender并列数仅检查记录范围；末点实际重算。conditional动作期望只给定greedy消息，不是对随机通信轨迹的积分。数学与执行核验不证明语义或语言形成。
''')
        print(json.dumps({k: result[k] for k in ("status", "counts", "max_action_probability_abs_error", "max_conditional_statistic_abs_error")}, ensure_ascii=False))
    except BaseException as error:
        (out/"failure.json").write_bytes(json_bytes({"status": "failed", "error": str(error), "traceback": traceback.format_exc(),
            "audit_source_sha256": sha(__file__), "automatic_retry": False}))
        raise


if __name__ == "__main__":
    main()
