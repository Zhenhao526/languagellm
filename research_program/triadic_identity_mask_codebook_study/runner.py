"""Run the paired identity-ambiguity × codebook experiment."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import platform
import time
from pathlib import Path

import numpy as np

from research_program.triadic_message_study import runner as core
from . import design, wire

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARTITIONS = design.PARTITIONS
TARGET = design.TARGET
STEPS = design.STEPS
SEEDS = design.SEEDS
CONDITIONS = design.CONDITIONS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return design.sha(Path(path))


def read(path):
    return design.read(Path(path))


def write(path, value):
    design.write(Path(path), value)


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf8")


def parse(condition):
    return design.parse_condition(condition)


def source_paths():
    return design.source_hashes()


def make_route_permutations(rng, batch_size, route_mode, communication):
    identity = np.broadcast_to(np.arange(3, dtype=np.int8), (batch_size, 2, 3)).copy()
    if not communication or route_mode == "stable":
        return np.stack((identity, identity), axis=0)
    permutations = np.empty((batch_size, 2, 3), dtype=np.int8)
    for row in range(batch_size):
        for window in range(2):
            permutations[row, window] = rng.permutation(3)
    # The two paired message trajectories share the same exogenous route,
    # while their sampled messages remain independent.
    return np.stack((permutations, permutations), axis=0)


def evaluation_permutations(seed, partition_index, route_mode, batch_size, communication):
    rng = np.random.default_rng(np.random.SeedSequence([seed, 900, partition_index, int(route_mode == "masked")]))
    return make_route_permutations(rng, batch_size, route_mode, communication)[0]


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), "Never overwrite a frozen preparation")
    prepared = design.make_prepared()
    sources = source_paths()
    out.mkdir(parents=True)
    for relative in sources:
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    write(out / "prepared.json", prepared)
    write(out / "plan.json", dict(
        status="prepared_without_training",
        created_at=time.time(),
        source_sha256=sources,
        prepared_sha256=design.sha(out / "prepared.json"),
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        no_model_calls=True,
    ))
    write(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json")))
    verify(out)
    return dict(status="prepared_without_training", output=str(out), plan_sha256=sha(out / "plan.json"), budget=prepared["budget"])


def verify(out):
    out = Path(out).resolve()
    plan, prepared, freeze = (read(out / name) for name in ("plan.json", "prepared.json", "freeze.json"))
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan changed")
    require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared changed")
    require(design.json_hash(prepared) == design.json_hash(design.make_prepared()),
            "Current deterministic design differs")
    require(plan["source_sha256"] == source_paths(), "Source changed")
    require(plan["runtime"] == dict(python=platform.python_version(), numpy=np.__version__), "Runtime changed")
    for relative, digest in plan["source_sha256"].items():
        require(sha(out / "source_snapshot" / relative) == digest, f"Snapshot changed: {relative}")
    return plan, prepared


def evaluate(networks, arrays, condition, partition_index, route_mode, output):
    settings = parse(condition)
    states = arrays["states"]
    n = len(states)
    output = Path(output)
    require(not output.exists(), "Evaluation output already exists")
    observations = arrays["x_PI"]
    probabilities = np.empty((n, 3, 17), dtype=np.float64)
    choices = np.empty((n, 3), dtype=np.int16)
    messages = np.empty((n, 2, 3, 4), dtype=np.int8)
    wire_messages = np.empty((n, 2, 3, 4), dtype=np.int8)
    route_permutations = np.empty((n, 2, 3), dtype=np.int8)
    rewards = np.empty(n, dtype=np.float64)
    expected = np.empty(n, dtype=np.float64)
    full_probability = np.empty(n, dtype=np.float64)
    execution_probability = np.empty(n, dtype=np.float64)
    executed = np.empty((n, 3), dtype=bool)
    satisfied = np.empty((n, 3), dtype=bool)
    for start in range(0, n, 512):
        stop = min(start + 512, n)
        ids = np.arange(start, stop, dtype=np.int64)
        perms = evaluation_permutations(70101, partition_index, route_mode, len(ids), settings["communication"])
        trace = wire.rollout(networks, observations[ids], settings["communication"], settings["maps"], perms)
        probs, _ = core.base.policy_distribution(trace["action_logits"])
        greedy = np.argmax(probs, axis=-1)
        probabilities[ids] = probs
        choices[ids] = greedy
        messages[ids] = trace["messages"]
        wire_messages[ids] = trace["wire_messages"]
        route_permutations[ids] = perms
        expected[ids], full_probability[ids], execution_probability[ids] = core.base.exact_statistics(probs, arrays["rewards"][ids])
        for local, state_index in enumerate(ids):
            actions = {agent: core.base.ACTIONS[i][greedy[local, i]] for i, agent in enumerate(core.base.AGENTS)}
            outcome = core.base.env.settle(states[state_index], actions, require_match=True)
            rewards[state_index] = outcome["reward"]
            for i, agent in enumerate(core.base.AGENTS):
                executed[state_index, i] = outcome["individual_feedback"][agent]["executed"]
                satisfied[state_index, i] = outcome["individual_feedback"][agent]["own_need_satisfied"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, state_indices=np.arange(n), messages=messages, wire_messages=wire_messages,
                            route_permutations=route_permutations, action_indices=choices,
                            action_probabilities=probabilities, greedy_reward=rewards,
                            executed=executed, satisfied=satisfied,
                            conditional_exact_expected_reward=expected,
                            conditional_exact_full_success_probability=full_probability,
                            conditional_exact_execution_probability=execution_probability)
    return dict(
        worlds=n, condition=condition, route_mode=route_mode,
        communication=settings["communication"], map_name=settings["map_name"],
        data_sha256=sha(output), path=str(output),
        greedy_reward_mean=float(rewards.mean()),
        greedy_full_success_rate=float((rewards == 1).mean()),
        greedy_partial_success_rate=float((rewards == 0.5).mean()),
        physical_execution_rate=float(executed.any(axis=1).mean()),
        satisfied_agent_rate=float(satisfied.mean()),
        conditional_exact_expected_reward_mean=float(expected.mean()),
        conditional_exact_full_success_probability_mean=float(full_probability.mean()),
        conditional_exact_execution_probability_mean=float(execution_probability.mean()),
        messages_sha256=core.array_sha(messages),
        wire_messages_sha256=core.array_sha(wire_messages),
        route_permutations_sha256=core.array_sha(route_permutations),
        action_indices_sha256=core.array_sha(choices),
    )


def checkpoint(networks, optimizer, update, batch_rng, message_rngs, path):
    payload = {}
    for agent in range(3):
        for module, name in enumerate(core.MODULES):
            for key, value in networks[3 * agent + module].items():
                payload[f"agent{agent}_{name}_{key}"] = value
            for moment in ("m", "v"):
                for key, value in optimizer[3 * agent + module][moment].items():
                    payload[f"adam_agent{agent}_{name}_{moment}_{key}"] = value
    payload["update"] = np.array(update, dtype=np.int64)
    payload["batch_rng_json"] = np.array(json.dumps(batch_rng.bit_generator.state, sort_keys=True))
    payload["message_rngs_json"] = np.array(json.dumps({k: v.bit_generator.state for k, v in message_rngs.items()}, sort_keys=True))
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **payload)
    return sha(path)


def train_run(seed, condition, prepared, arrays, execution):
    settings = parse(condition)
    directory = Path(execution) / f"seed_{seed}_{condition}"
    directory.mkdir(parents=True, exist_ok=False)
    networks = core.make_networks(seed)
    optimizer = core.base.make_adam(networks)
    batch_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
    message_rngs = core.make_message_rngs(seed)
    route_rng = np.random.default_rng(np.random.SeedSequence([seed, 400]))
    initial = core.parameter_hash(networks)
    trajectory = []
    started = time.perf_counter()

    def save_checkpoint(update):
        checkpoint_path = directory / f"checkpoint_{update:04d}.npz"
        checkpoint_sha = checkpoint(networks, optimizer, update, batch_rng, message_rngs, checkpoint_path)
        evals = {}
        modes = [settings["route_mode"] if settings["communication"] else "stable"]
        if settings["communication"]:
            modes.append("masked" if settings["route_mode"] == "stable" else "stable")
        for mode in modes:
            name = f"target_{mode}_{update:04d}.npz"
            evals[mode] = evaluate(networks, arrays[TARGET], condition, list(PARTITIONS).index(TARGET), mode, directory / name)
        row = dict(update=update, checkpoint_sha256=checkpoint_sha, evaluations=evals,
                   elapsed_seconds=time.perf_counter() - started)
        trajectory.append(row)
        with (directory / "trajectory.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    save_checkpoint(0)
    with (directory / "training.jsonl").open("x", encoding="utf8") as stream:
        for update in range(1, design.UPDATES + 1):
            ids = batch_rng.integers(0, len(arrays["train"]["states"]), size=design.BATCH_SIZE, dtype=np.int64)
            uniforms = core.draw_uniforms(message_rngs, len(ids))
            route_perms = make_route_permutations(route_rng, len(ids), settings["route_mode"], settings["communication"])
            gradients, row = wire.training_gradients(
                networks, arrays["train"]["x_PI"][ids], arrays["train"]["rewards"][ids],
                settings["communication"], uniforms, route_perms, update, settings["maps"], "reciprocal",
            )
            norm, scale = core.base.adam_step(networks, gradients, optimizer, update)
            row.update(update=update, seed=seed, condition=condition,
                       batch_indices_sha256=core.array_sha(ids),
                       batch_states_sha256=core.array_sha(arrays["train"]["packed_states"][ids]),
                       sample_uniforms_sha256=core.array_sha(uniforms),
                       route_permutations_sha256=core.array_sha(route_perms),
                       gradient_norm=norm, gradient_clip_scale=scale)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if update in STEPS:
                stream.flush()
                save_checkpoint(update)
    result = dict(
        seed=seed, condition=condition, updates=design.UPDATES,
        initial_parameter_sha256=initial, final_parameter_sha256=core.parameter_hash(networks),
        training_log_sha256=sha(directory / "training.jsonl"),
        final_checkpoint_sha256=sha(directory / f"checkpoint_{design.UPDATES:04d}.npz"),
        trajectory=trajectory, elapsed_seconds=time.perf_counter() - started,
    )
    write(directory / "result.json", result)
    return result


def run_seed(seed, prepared, execution):
    arrays = {part: core.build_arrays(prepared["partitions"][part]) for part in PARTITIONS}
    hashes = {part: {key: core.array_sha(arrays[part][key]) for key in ("packed_states", "rewards", "x_PI")} for part in PARTITIONS}
    write(Path(execution) / f"seed_{seed}_arrays.json", hashes)
    return [train_run(seed, condition, prepared, arrays, execution) for condition in CONDITIONS]


def execute(out):
    out = Path(out).resolve()
    plan, prepared = verify(out)
    execution = out / "execution"
    require(not execution.exists(), "Never overwrite an execution")
    execution.mkdir()
    started = time.perf_counter()
    write(execution / "started.json", dict(started_at=time.time(), plan_sha256=sha(out / "plan.json"), seeds=list(SEEDS)))
    results = []
    try:
        for seed in SEEDS:
            results.extend(run_seed(seed, prepared, execution))
            print(json.dumps({"completed_seed": seed, "runs": len(results)}, ensure_ascii=False), flush=True)
        require([(r["seed"], r["condition"]) for r in results] == [(s, c) for s in SEEDS for c in CONDITIONS], "Noncanonical run grid")
        result = dict(status="completed", completed_at=time.time(), plan_sha256=sha(out / "plan.json"),
                      budget=prepared["budget"], runs=results, elapsed_seconds=time.perf_counter() - started)
        write(execution / "results.json", result)
        write(execution / "status.json", dict(status="completed", results_sha256=sha(execution / "results.json")))
        return dict(status="completed", runs=len(results), elapsed_seconds=result["elapsed_seconds"])
    except BaseException as error:
        write(execution / "failure.json", dict(status="failed", error_type=type(error).__name__, error=str(error)))
        raise


def summarize(out):
    out = Path(out).resolve()
    _, prepared = verify(out)
    result = read(out / "execution" / "results.json")
    index = {(row["seed"], row["condition"]): row for row in result["runs"]}
    rows = []
    for seed in SEEDS:
        def endpoint(condition, mode):
            return index[(seed, condition)]["trajectory"][-1]["evaluations"][mode]
        stable_public = endpoint("stable_public_live", "stable")["greedy_full_success_rate"]
        stable_private = endpoint("stable_private_live", "stable")["greedy_full_success_rate"]
        stable_silent = endpoint("stable_silent", "stable")["greedy_full_success_rate"]
        masked_public = endpoint("masked_public_live", "masked")["greedy_full_success_rate"]
        masked_private = endpoint("masked_private_live", "masked")["greedy_full_success_rate"]
        masked_silent = endpoint("masked_silent", "stable")["greedy_full_success_rate"]
        rows.append(dict(
            seed=seed,
            stable_public_live=stable_public, stable_private_live=stable_private, stable_silent=stable_silent,
            masked_public_live=masked_public, masked_private_live=masked_private, masked_silent=masked_silent,
            stable_public_minus_private=stable_public - stable_private,
            masked_live_gain_public=masked_public - masked_silent,
            masked_live_gain_private=masked_private - masked_silent,
            masked_public_private_interaction=(masked_public - masked_silent) - (masked_private - masked_silent),
            stable_route_public_gain=stable_public - stable_silent,
            stable_route_private_gain=stable_private - stable_silent,
        ))
    def mean(key):
        return float(np.mean([row[key] for row in rows]))
    summary = dict(
        schema="triadic_identity_mask_codebook_summary_v1",
        status="completed",
        endpoint_update=design.UPDATES,
        partition=TARGET,
        rows=rows,
        equal_weight_means={key: mean(key) for key in (
            "stable_public_minus_private", "masked_live_gain_public", "masked_live_gain_private",
            "masked_public_private_interaction", "stable_route_public_gain", "stable_route_private_gain",
        )},
        primary="masked_public_private_interaction",
        primary_interpretation=(
            "Positive values would indicate that public tokens retain more live-task "
            "gain than private tokens under randomized sender slots; this remains a "
            "convention-alignment diagnostic rather than a language score."
        ),
        design_sha256=sha(out / "prepared.json"),
        result_sha256=sha(out / "execution" / "results.json"),
    )
    write(out / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "execute", "summarize"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        answer = prepare(args.out)
    elif args.command == "verify":
        answer = verify(args.out)[0]
    elif args.command == "execute":
        answer = execute(args.out)
    else:
        answer = summarize(args.out)
    print(json.dumps(answer, ensure_ascii=False))
