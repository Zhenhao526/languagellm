"""Paired mean J versus mean log J intervention; CPU only, no teacher.

Frozen baseline functions are reused without changing their source or config.
Only the state aggregation objective changes. prepare never trains.
"""
from __future__ import annotations

import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from copy import deepcopy
from itertools import combinations, zip_longest
import json
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from research_program.triadic_learning_baseline import runner as base

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SEEDS = (45101, 45102, 45103, 45104)
OBJECTIVES = ("mean_J", "mean_log_J")
PARTITIONS = base.PARTITIONS
CHECKPOINTS = base.CHECKPOINTS
BASE_RUNNER_SHA = "1f3cae631bc15e8f2004e27187370fcc8f9448ac6f99495e867c6ac40e2e0d6a"
CONFIG = deepcopy(base.CONFIG)
CONFIG.update(seeds=list(SEEDS), objectives=list(OBJECTIVES),
    training_objective="paired_state_aggregation_intervention", baseline_runner_sha256=BASE_RUNNER_SHA,
    run_order="seed_outer_objective_inner", log_J_epsilon_or_floor=None)

require, finite, sha = base.require, base.finite, base.sha
read, write_new, json_hash, json_bytes = base.read, base.write_new, base.json_hash, base.json_bytes
array_sha, now = base.array_sha, base.now


def objective_terms(logits, rewards):
    """Exact native J and stable logJ, with both analytical logit gradients.

    Zero native J from floating-point underflow is allowed. We never compute
    logJ as log(J), divide by J, add epsilon, clip, or choose a best joint plan.
    Positive native reward support is required; the fixed domain satisfies it.
    """
    logits = np.asarray(logits, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    require(logits.ndim == 3 and logits.shape[1:] == (3, 17)
            and rewards.shape == (len(logits), 24), "Incorrect logit/reward shape")
    finite(logits, "logits"); finite(rewards, "rewards")
    require(np.isin(rewards, (0.0, 0.5, 1.0)).all() and (rewards > 0).any(axis=1).all(),
            "Native reward support must contain a positive outcome in every state")
    probabilities, log_probabilities = base.policy_distribution(logits)
    J, native_gradient = base.expected_reward_and_logit_gradient(probabilities, rewards)
    positive = rewards > 0
    log_mass = np.full(rewards.shape, -np.inf, dtype=np.float64)
    log_mass[positive] = np.log(rewards[positive])
    for actor in range(3):
        log_mass += log_probabilities[:, actor, base.JOINT_ACTIONS[:, actor]]
    maximum = log_mass.max(axis=1, keepdims=True)
    finite(maximum, "largest log reward mass")
    shifted_mass = np.exp(log_mass - maximum)
    normalization = shifted_mass.sum(axis=1, keepdims=True)
    posterior = shifted_mass / normalization
    log_J = (maximum + np.log(normalization))[:, 0]
    finite(log_J, "logJ"); finite(posterior, "posterior")
    require((log_J <= 1e-12).all(), "Native expected reward cannot exceed1")
    require((posterior[~positive] == 0).all(), "Zero-reward contribution has posterior mass")
    log_gradient = -probabilities.copy()
    for t in range(24):
        for actor in range(3):
            log_gradient[:, actor, base.JOINT_ACTIONS[t, actor]] += posterior[:, t]
    finite(log_gradient, "logJ logit gradient")
    return {"probabilities": probabilities, "log_probabilities": log_probabilities,
            "J": J, "log_J": log_J, "mean_J_logit_gradient": native_gradient,
            "log_J_logit_gradient": log_gradient, "posterior_weights": posterior}


def loss_and_derivative(terms, objective, update):
    require(objective in OBJECTIVES, "Unknown objective")
    entropy, entropy_gradient = base.entropy_and_logit_gradient(terms["probabilities"], terms["log_probabilities"])
    beta = base.entropy_coefficient(update)
    if objective == "mean_J":
        selected_value, selected_gradient = terms["J"], terms["mean_J_logit_gradient"]
    else:
        selected_value, selected_gradient = terms["log_J"], terms["log_J_logit_gradient"]
    loss = -float(selected_value.mean() + beta * entropy.mean())
    derivative = -(selected_gradient + beta * entropy_gradient) / len(selected_value)
    finite(derivative, "loss derivative")
    require(np.isfinite(loss), "Nonfinite loss")
    return {"loss": loss, "derivative": derivative, "mean_actor_entropy": float(entropy.mean()),
            "entropy_coefficient": beta, "selected_objective_mean": float(selected_value.mean())}


def make_prepared():
    old = base.make_prepared()
    prepared = deepcopy(old)
    prepared.update(schema="triadic_coordination_v1", config=deepcopy(CONFIG),
        baseline_prepared_sha256=json_hash(old),
        baseline_partitions_sha256=json_hash(old["partitions"]),
        runs=[{"seed": seed, "objective": obj, "directory": f"seed_{seed}_{obj}"}
              for seed in SEEDS for obj in OBJECTIVES],
        training_state_samples_per_run=6000*256,
        training_state_samples_total=8*6000*256,
        weighted_structural_action_contributions=8*6000*256*24,
        offline_reward_table_entries=143424*24,
        complete_final_world_evaluations=8*143424)
    # Remove inherited wording whose denominator was a single-arm study.
    prepared.pop("training_state_samples_per_seed", None)
    require(prepared["partitions"] == old["partitions"] and prepared["actions"] == old["actions"],
            "Observation domain or full action support changed")
    return prepared


def sources():
    require(sha(base.__file__) == BASE_RUNNER_SHA, "Frozen baseline runner changed")
    paths = (Path(__file__), HERE / "plan.md", HERE / "tests/test_runner.py", HERE / "seed_selection_receipt.json",
             Path(base.__file__), Path(base.env.__file__))
    require(all(p.is_file() for p in paths), "Missing code/plan/tests/source before freezing")
    return {str(p.resolve().relative_to(ROOT)): sha(p) for p in paths}


def prepare(output):
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite preparation")
    hashes = sources()
    prepared = make_prepared()
    plan = {"schema": "triadic_coordination_v1", "prepared_at": now(), "config": deepcopy(CONFIG),
        "prepared_sha256": json_hash(prepared), "sources": hashes,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
        "no_training_executed_by_prepare": True,
        "contrast": "mean native expected reward versus mean log native expected reward",
        "seed_and_batch_pairing": "Neither initialization nor training RNG uses objective as an input."}
    output.mkdir(parents=True, exist_ok=False)
    for relative in hashes:
        target = output / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write_new(output / "prepared.json", prepared)
    write_new(output / "plan.json", plan)
    write_new(output / "freeze.json", {"plan_sha256": sha(output / "plan.json"), "prepared_sha256": sha(output / "prepared.json")})
    verify(output)
    return {"status": "prepared_not_trained", "output": str(output), "plan_sha256": sha(output / "plan.json"),
            "runs": prepared["runs"], "training_state_samples_total": prepared["training_state_samples_total"]}


def verify(output):
    output = Path(output).resolve()
    plan, prepared, freeze = [read(output / n) for n in ("plan.json", "prepared.json", "freeze.json")]
    require(sha(output / "plan.json") == freeze["plan_sha256"], "Plan changed")
    require(sha(output / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared input changed")
    require(json_hash(make_prepared()) == plan["prepared_sha256"], "Current prepared values differ")
    require(plan["config"] == CONFIG and plan["runtime"] == {"python": platform.python_version(), "numpy": np.__version__},
            "Configuration or runtime changed")
    require(plan["sources"] == sources(), "Current sources changed")
    for relative, digest in plan["sources"].items():
        require(sha(output / "source_snapshot" / relative) == digest, "Frozen source snapshot changed")
    return plan, prepared


def parameter_hash(actors):
    return json_hash({f"agent{i}_{key}": array_sha(value) for i, actor in enumerate(actors) for key, value in actor.items()})


def final_evaluations(actors, arrays, output):
    final, domain_counts = {}, {}
    for name in PARTITIONS:
        path = output / f"final_{name}.npz"
        score = base.evaluate(actors, arrays[name], save_path=path)
        # Only inspect saved real actions; never modify/reselect them.
        with np.load(path, allow_pickle=False) as data:
            actions, counts = np.unique(data["action_indices"], axis=0, return_counts=True)
        score["distinct_joint_argmax_actions"] = len(actions)
        score["joint_argmax_action_counts"] = [{"action_indices": a.tolist(), "worlds": int(n)} for a, n in zip(actions, counts)]
        for a, n in zip(actions, counts):
            key = tuple(int(x) for x in a)
            domain_counts[key] = domain_counts.get(key, 0) + int(n)
        final[name] = score
    domain = {"worlds": sum(domain_counts.values()), "distinct_joint_argmax_actions": len(domain_counts),
        "joint_argmax_action_counts": [{"action_indices": list(a), "worlds": n} for a, n in sorted(domain_counts.items())]}
    require(domain["worlds"] == 143424, "Incomplete full-domain action inventory")
    return final, domain


def train_run(seed, objective, prepared, arrays, output):
    output.mkdir(exist_ok=False)
    actors = [base.make_actor(np.random.SeedSequence([seed, i, 100])) for i in range(3)]
    for i, j in combinations(range(3), 2):
        require(all(not np.shares_memory(actors[i][k], actors[j][k]) for k in actors[i]), "Shared actor parameters")
    initial_sha = parameter_hash(actors)
    optimizer = base.make_adam(actors)
    batch_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
    monitor = []
    started = time.perf_counter()

    def checkpoint(update):
        path = output / f"checkpoint_{update:04d}.npz"
        digest = base.save_checkpoint(path, actors, optimizer, update, batch_rng)
        scores = {name: base.evaluate(actors, arrays[name], prepared["partitions"][name]["monitor_indices"]) for name in PARTITIONS}
        row = {"update": update, "checkpoint_sha256": digest, "monitor": scores, "elapsed_seconds": time.perf_counter()-started}
        monitor.append(row)
        with (output / "monitor.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json_bytes(row).decode())

    checkpoint(0)
    with (output / "training.jsonl").open("x", encoding="utf-8") as stream:
        for update in range(1, CONFIG["updates"]+1):
            ids = batch_rng.integers(0, len(arrays["train"]["states"]), size=CONFIG["batch_size"], dtype=np.int64)
            inputs = arrays["train"]["x"][ids]
            cached = [base.actor_forward(actors[i], inputs[:, i]) for i in range(3)]
            logits = np.stack([value[0] for value in cached], axis=1)
            terms = objective_terms(logits, arrays["train"]["rewards"][ids])
            selected = loss_and_derivative(terms, objective, update)
            gradients = [base.actor_backward(actors[i], cached[i][1], selected["derivative"][:, i]) for i in range(3)]
            norm, scale = base.adam_step(actors, gradients, optimizer, update)
            row = {"update": update, "objective": objective, "batch_indices_sha256": array_sha(ids),
                "batch_states_sha256": array_sha(arrays["train"]["packed_states"][ids]),
                "mean_J": float(terms["J"].mean()), "mean_log_J": float(terms["log_J"].mean()),
                "min_log_J": float(terms["log_J"].min()), "max_log_J": float(terms["log_J"].max()),
                "zero_float_J_states": int((terms["J"] == 0).sum()),
                "selected_objective_mean": selected["selected_objective_mean"], "loss": selected["loss"],
                "mean_actor_entropy": selected["mean_actor_entropy"], "entropy_coefficient": selected["entropy_coefficient"],
                "gradient_norm": norm, "gradient_clip_scale": scale, "elapsed_seconds": time.perf_counter()-started}
            stream.write(json_bytes(row).decode())
            if update in CHECKPOINTS:
                stream.flush()
                checkpoint(update)
    final, domain_actions = final_evaluations(actors, arrays, output)
    candidate = all(final[p]["greedy_full_success_rate"] >= .99 for p in PARTITIONS if p != "train")
    result = {"seed": seed, "objective": objective, "updates": 6000, "training_world_samples": 6000*256,
        "initial_parameter_sha256": initial_sha, "final": final, "full_domain_actions": domain_actions,
        "candidate_threshold_met": candidate,
        "monitor": monitor, "elapsed_seconds": time.perf_counter()-started,
        "training_log_sha256": sha(output / "training.jsonl"), "final_checkpoint_sha256": sha(output / "checkpoint_6000.npz"),
        "final_parameter_sha256": parameter_hash(actors)}
    write_new(output / "result.json", result)
    return result


def check_pairing(execution, results):
    index = {(r["seed"], r["objective"]): r for r in results}
    checks = []
    for seed in SEEDS:
        first, second = [index[(seed, obj)] for obj in OBJECTIVES]
        require(first["initial_parameter_sha256"] == second["initial_parameter_sha256"], "Paired initial parameters differ")
        require(first["monitor"][0]["monitor"] == second["monitor"][0]["monitor"], "Paired initial policies differ")
        paths = [execution / f"seed_{seed}_{obj}" / "training.jsonl" for obj in OBJECTIVES]
        count = 0
        with paths[0].open() as a, paths[1].open() as b:
            for line1, line2 in zip_longest(a, b):
                require(line1 is not None and line2 is not None, "Unequal paired training log length")
                x, y = json.loads(line1), json.loads(line2)
                require(all(x[k] == y[k] for k in ("update", "batch_indices_sha256", "batch_states_sha256", "entropy_coefficient")),
                        "Paired world stream or fixed schedule differs")
                count += 1
        require(count == 6000, "Missing paired updates")
        checks.append({"seed": seed, "same_initial_parameters": True, "same_initial_monitor": True,
            "paired_batch_and_state_hashes": count, "same_fixed_entropy_schedule": True})
    return checks


def primary_comparison(results):
    require(len(results) == 8, "All eight runs required for primary comparison")
    index = {(r["seed"], r["objective"]): r for r in results}
    require(set(index) == {(seed, obj) for seed in SEEDS for obj in OBJECTIVES}, "Missing or duplicate paired runs")
    rows = []
    for seed in SEEDS:
        first, second = [index[(seed, obj)]["final"]["new_needs_and_layouts"] for obj in OBJECTIVES]
        require(first["worlds"] == second["worlds"] == 8244, "Primary heldout denominator differs")
        rows.append({"seed": seed, "mean_J_full_success_rate": first["greedy_full_success_rate"],
            "mean_log_J_full_success_rate": second["greedy_full_success_rate"],
            "paired_difference_log_minus_mean": second["greedy_full_success_rate"]-first["greedy_full_success_rate"]})
    return {"partition": "new_needs_and_layouts", "endpoint_update": 6000, "metric": "greedy_full_success_rate",
        "expected_direction": "mean_log_J greater than mean_J", "seed_pairs": rows,
        "equal_weight_mean_paired_difference": sum(r["paired_difference_log_minus_mean"] for r in rows)/4,
        "independent_paired_initializations": 4, "significance_test": None}


def execute(output):
    output = Path(output).resolve()
    plan, prepared = verify(output)
    execution = output / "execution"
    require(not execution.exists(), "Never resume or overwrite started execution")
    execution.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_new(execution / "started.json", {"started_at": now(), "plan_sha256": sha(output / "plan.json"),
        "device": "cpu_numpy", "pid": os.getpid(), "thread_environment": {name: os.environ[name] for name in
            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}})
    try:
        arrays = {name: base.build_arrays(prepared["partitions"][name]) for name in PARTITIONS}
        write_new(execution / "input_arrays.json", {name: {"worlds": len(a["states"]), "features_sha256": array_sha(a["x"]),
            "native_rewards_sha256": array_sha(a["rewards"]), "states_sha256": array_sha(a["packed_states"])} for name, a in arrays.items()})
        results = []
        for run in prepared["runs"]:
            result = train_run(run["seed"], run["objective"], prepared, arrays, execution / run["directory"])
            results.append(result)
            print(json.dumps({"completed_run": run, "final_full_success_rates": {k: v["greedy_full_success_rate"] for k, v in result["final"].items()},
                "elapsed_seconds": time.perf_counter()-started}), flush=True)
        pairing = check_pairing(execution, results)
        verify(output)
        result = {"status": "completed", "completed_at": now(), "plan_sha256": sha(output / "plan.json"),
            "runs": results, "completed_run_count": 8, "paired_seed_count": 4, "pairing_checks": pairing,
            "primary_comparison": primary_comparison(results),
            "updates_total": 48000, "training_world_samples_total": prepared["training_state_samples_total"],
            "weighted_structural_action_contributions": prepared["weighted_structural_action_contributions"],
            "all_seed_candidates_by_objective": {obj: all(r["candidate_threshold_met"] for r in results if r["objective"] == obj) for obj in OBJECTIVES},
            "symbolic_phase_unlocked": False, "elapsed_seconds": time.perf_counter()-started,
            "scope": "A state-aggregation objective intervention with unchanged native reward/information/action support; not an exploration-only control or language experiment."}
        write_new(execution / "results.json", result)
        write_new(execution / "status.json", {"status": "completed", "completed_at": now(), "completed_runs": 8})
    except BaseException as error:
        write_new(execution / "failure.json", {"status": "failed", "failed_at": now(), "error_type": type(error).__name__,
            "error": str(error), "elapsed_seconds": time.perf_counter()-started})
        write_new(execution / "status.json", {"status": "failed", "failed_at": now()})
        raise
    return {"status": "completed", "output": str(execution), "all_seed_candidates_by_objective": result["all_seed_candidates_by_objective"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.out)
    elif args.command == "verify":
        verify(args.out); result = {"status": "verified_not_trained", "output": str(args.out.resolve())}
    else:
        result = execute(args.out)
    print(json.dumps(result, ensure_ascii=False))
