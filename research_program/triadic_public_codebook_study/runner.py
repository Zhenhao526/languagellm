"""Train reciprocal PL policies under public/private wire codebooks."""
from __future__ import annotations

import argparse
from itertools import zip_longest
import json
import multiprocessing
import platform
from pathlib import Path
import shutil
import time

import numpy as np

from research_program.triadic_rule_formation_study import runner as formation
from research_program.triadic_reciprocal_execution_study import environment, kernel
from . import design, wire

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
core = formation.core
STEPS = design.STEPS
SEEDS = design.SEEDS
CONDITIONS = design.CONDITIONS
PARTS = design.PARTS
TARGET = design.TARGET


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return design.sha(Path(path))


def read(path):
    return design.read(Path(path))


def write(path, value):
    design.write(Path(path), value)


def process_status(path, **fields):
    write(path, dict(at=time.time(), **fields))


def make_arrays(spec):
    arrays = formation.make_arrays(spec)
    require(arrays["x_PL"].shape == (spec["world_count"], 3, 54), "Invalid PL feature shape")
    require(np.all(arrays["x_PL"][:, :, 53] == 0), "Full-information flag leaked")
    return arrays


def condition_name(condition):
    return condition


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, "Unknown seed/condition")
    return f"seed_{seed}_{condition}"


def maps_for(condition):
    map_name, live, maps = design.parse_condition(condition)
    return map_name, live, tuple(np.asarray(row, dtype=np.int8) for row in maps)


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), "Never overwrite preparation")
    static = design.make_prepared()
    source_hashes = design.sources()
    out.mkdir(parents=True)
    for relative in source_hashes:
        source = ROOT / relative
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    write(out / "prepared.json", static)
    input_paths = [
        formation.ORIGINAL / name for name in ("plan.json", "prepared.json", "freeze.json")
    ] + [formation.PHYSICS / name for name in ("plan.json", "freeze.json")]
    write(out / "plan.json", dict(
        status="prepared_without_training", created_at=time.time(),
        source_sha256=source_hashes,
        inputs_sha256={str(path.resolve().relative_to(ROOT)): sha(path) for path in input_paths},
        prepared_sha256=sha(out / "prepared.json"),
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        no_external_model=True,
    ))
    write(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json")))
    verify(out)
    return dict(status="prepared_without_training", output=str(out), plan_sha256=sha(out / "plan.json"), budget=static["budget"])


def verify(out):
    out = Path(out).resolve()
    plan = read(out / "plan.json")
    static = read(out / "prepared.json")
    freeze = read(out / "freeze.json")
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan changed")
    require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared changed")
    require(static == design.make_prepared(), "Static design changed")
    require(plan["source_sha256"] == design.sources(), "Source manifest changed")
    require(plan["runtime"] == dict(python=platform.python_version(), numpy=np.__version__), "Runtime changed")
    for relative, digest in plan["source_sha256"].items():
        require(sha(out / "source_snapshot" / relative) == digest, "Source snapshot changed: " + relative)
    for relative, digest in plan["inputs_sha256"].items():
        require(sha(ROOT / relative) == digest, "Frozen input changed: " + relative)
    return plan, static


def evaluate(networks, arrays, spec, case_spec, live, maps, path, map_name):
    path = Path(path)
    require(not path.exists(), "Evaluation output already exists")
    ids = np.arange(spec["world_count"], dtype=np.int64)
    states = arrays["packed_states"][ids]
    data = dict(
        states=states, state_indices=ids,
        messages=np.empty((len(ids), 2, 3, 4), dtype=np.int8),
        wire_messages=np.empty((len(ids), 2, 3, 4), dtype=np.int8),
        action_indices=np.empty((len(ids), 3), dtype=np.int16),
        action_probabilities=np.empty((len(ids), 3, 17), dtype=np.float64),
        conditional_exact_expected_reward=np.empty(len(ids)),
        conditional_exact_full_success_probability=np.empty(len(ids)),
        conditional_exact_execution_probability=np.empty(len(ids)),
        conditional_full_posterior_mass=np.empty(len(ids)),
    )
    for start in range(0, len(ids), 1024):
        stop = min(start + 1024, len(ids)); sl = slice(start, stop); ix = ids[sl]
        trace = wire.rollout(networks, arrays["x_PL"][ix], live, maps)
        terms = kernel.objective_terms(trace["action_logits"], arrays["rewards"][ix], design.RULE)
        data["messages"][sl] = trace["messages"]
        data["wire_messages"][sl] = trace["wire_messages"]
        data["action_probabilities"][sl] = terms["probabilities"]
        data["action_indices"][sl] = terms["probabilities"].argmax(-1)
        data["conditional_exact_expected_reward"][sl] = terms["native_expected_reward"]
        data["conditional_exact_full_success_probability"][sl] = terms["full_success_probability"]
        data["conditional_exact_execution_probability"][sl] = terms["execution_probability"]
        data["conditional_full_posterior_mass"][sl] = terms["full_success_posterior_mass"]
    actions = data["action_indices"]
    native = environment.settle(states, actions, design.RULE)
    strict = environment.settle(states, actions, "strict")
    common = environment.settle(states, actions, "reciprocal")
    truth = environment.truth_from_rewards(states, arrays["rewards"][ids])
    data.update(native)
    data.update({f"strict__{key}": value for key, value in strict.items()})
    data.update({f"common_reciprocal__{key}": value for key, value in common.items()})
    compact = {key: data[key] for key in (
        "state_indices", "messages", "wire_messages", "action_indices",
        "conditional_exact_expected_reward",
        "conditional_exact_full_success_probability",
        "conditional_exact_execution_probability",
        "conditional_full_posterior_mass",
    )}
    record = dict(
        native=environment.metrics(native, truth, actions, design.RULE),
        strict=environment.metrics(strict, truth, actions, "strict"),
        common_reciprocal=environment.metrics(common, truth, actions, "reciprocal"),
        need_response=dict(
            native=formation.cases.metrics(case_spec, native["actual_pair_index"]),
            common_reciprocal=formation.cases.metrics(case_spec, common["actual_pair_index"]),
        ),
    )
    with path.open("xb") as stream:
        np.savez_compressed(stream, **compact)
    record.update(
        path=str(path), data_sha256=sha(path), worlds=len(ids), partition=spec["partition"],
        live=bool(live), wire_maps=[list(map(int, row)) for row in maps], map_name=map_name,
        forward_module_samples=9 * len(ids), information="PL", scope="complete_partition",
    )
    return record, data["messages"]


def make_random_streams(seed):
    return np.random.default_rng(np.random.SeedSequence([seed, 200])), core.make_message_rngs(seed)


def train_run(seed, condition, static, arrays, execution):
    directory = Path(execution) / name(seed, condition)
    directory.mkdir(exist_ok=False)
    map_name, live, maps = maps_for(condition)
    networks = core.make_networks(seed)
    initial = core.parameter_hash(networks)
    optimizer = core.base.make_adam(networks)
    world_rng, message_rngs = make_random_streams(seed)
    trajectory = []; previous_messages = None; started = time.perf_counter()

    def checkpoint(step):
        nonlocal previous_messages
        checkpoint_path = directory / f"checkpoint_{step:04d}.npz"
        checkpoint_sha = core.save_checkpoint(checkpoint_path, networks, optimizer, step, world_rng, message_rngs)
        evaluation, messages = evaluate(
            networks, arrays[TARGET], static["partitions"][TARGET], static["need_response_cases"][TARGET], live, maps,
            directory / f"trajectory_{step:04d}_{TARGET}.npz", map_name
        )
        snapshot = formation.metrics.message_snapshot(messages, static["partitions"][TARGET])
        transition = None if previous_messages is None else formation.metrics.message_transition(previous_messages, messages, static["partitions"][TARGET])
        previous_messages = messages.copy()
        row = dict(update=step, checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
                   evaluation=evaluation, message_snapshot=snapshot, message_transition=transition,
                   map_name=map_name, wire_maps=[list(map(int, row)) for row in maps],
                   elapsed_seconds=time.perf_counter() - started)
        trajectory.append(row)
        with (directory / "trajectory.jsonl").open("a", encoding="utf8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        process_status(directory / "status.json", status="running", seed=seed, condition=condition, update=step,
                       checkpoint_sha256=checkpoint_sha)

    checkpoint(0)
    with (directory / "training.jsonl").open("x", encoding="utf8") as stream:
        for update in range(1, design.UPDATES + 1):
            uniforms = world_rng.random((design.BATCH_SIZE, 3))
            ids = formation.dataset.sample_indices(static["partitions"]["train"], uniforms)
            message_uniforms = core.draw_uniforms(message_rngs, design.BATCH_SIZE)
            gradients, row = wire.training_gradients(
                networks, arrays["train"]["x_PL"][ids], arrays["train"]["rewards"][ids], live,
                message_uniforms, update, design.RULE, maps
            )
            norm, scale = core.base.adam_step(networks, gradients, optimizer, update)
            row.update(update=update, seed=seed, condition=condition, map_name=map_name,
                       world_uniforms_sha256=core.array_sha(uniforms), batch_indices_sha256=core.array_sha(ids),
                       batch_states_sha256=core.array_sha(arrays["train"]["packed_states"][ids]),
                       sample_uniforms_sha256=core.array_sha(message_uniforms), gradient_norm=norm,
                       gradient_clip_scale=scale)
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            if update in STEPS:
                stream.flush(); checkpoint(update)
    finals = {}
    for part in PARTS:
        if part == TARGET:
            endpoint = trajectory[-1]["evaluation"]
            finals[part] = dict(endpoint, alias_of="trajectory_update_6000", additional_forward_module_samples=0)
        else:
            finals[part], _ = evaluate(networks, arrays[part], static["partitions"][part], static["need_response_cases"][part], live, maps,
                                        directory / f"final_{part}.npz", map_name)
    result = dict(seed=seed, condition=condition, map_name=map_name, live=live,
                  wire_maps=[list(map(int, row)) for row in maps], rule=design.RULE, updates=design.UPDATES,
                  initial_parameter_sha256=initial, final_parameter_sha256=core.parameter_hash(networks),
                  final_checkpoint_sha256=sha(directory / "checkpoint_6000.npz"),
                  training_log_sha256=sha(directory / "training.jsonl"), trajectory=trajectory, final=finals,
                  elapsed_seconds=time.perf_counter() - started)
    write(directory / "result.json", result)
    process_status(directory / "status.json", status="completed", seed=seed, condition=condition,
                   result_sha256=sha(directory / "result.json"))
    return result


def worker(payload):
    seed, static, execution = payload
    execution = Path(execution)
    arrays = {part: make_arrays(static["partitions"][part]) for part in PARTS}
    hashes = {part: {key: core.array_sha(arrays[part][key]) for key in ("packed_states", "rewards", "x_PL")}
              for part in PARTS}
    write(execution / f"seed_{seed}_arrays.json", {"array_hashes": hashes})
    runs = []
    for condition in CONDITIONS:
        runs.append(train_run(seed, condition, static, arrays, execution))
        print(json.dumps(dict(stage="run_completed", seed=seed, condition=condition), ensure_ascii=False), flush=True)
    require(hashes == {part: {key: core.array_sha(arrays[part][key]) for key in ("packed_states", "rewards", "x_PL")}
                       for part in PARTS}, "Worker arrays mutated")
    return runs


def verify_pairing(execution, runs):
    execution = Path(execution)
    for seed in SEEDS:
        cells = [run for run in runs if run["seed"] == seed]
        require(len(cells) == len(CONDITIONS), "Incomplete paired conditions")
        require(len({run["initial_parameter_sha256"] for run in cells}) == 1, "Unpaired initialization")
        streams = [(execution / name(seed, condition) / "training.jsonl").open(encoding="utf8") for condition in CONDITIONS]
        try:
            count = 0
            for lines in zip_longest(*streams):
                require(all(line is not None for line in lines), "Unequal training logs")
                rows = [json.loads(line) for line in lines]; count += 1
                require(all(row["update"] == count and row["seed"] == seed for row in rows), "Training identity mismatch")
                for key in ("world_uniforms_sha256", "batch_indices_sha256", "batch_states_sha256", "sample_uniforms_sha256", "entropy_coefficient"):
                    require(len({row[key] for row in rows}) == 1, "Paired stream differs: " + key)
            require(count == design.UPDATES, "Training update count mismatch")
        finally:
            for stream in streams:
                stream.close()


def measured_budget(runs):
    require(len(runs) == len(SEEDS) * len(CONDITIONS), "Incomplete run grid")
    actual = []
    for run in runs:
        require(run["updates"] == design.UPDATES and [row["update"] for row in run["trajectory"]] == list(STEPS), "Incomplete trajectory")
        endpoint = run["trajectory"][-1]["evaluation"]
        alias = run["final"][TARGET]
        require(alias["path"] == endpoint["path"] and alias["data_sha256"] == endpoint["data_sha256"], "Target alias mismatch")
        actual.extend(row["evaluation"] for row in run["trajectory"])
        actual.extend(run["final"][part] for part in PARTS if part != TARGET)
    require(len({row["path"] for row in actual}) == len(actual), "Repeated evaluation file")
    return dict(
        training_forward_module_samples=len(runs) * design.UPDATES * design.BATCH_SIZE * 2 * 9,
        evaluation_forward_module_samples=sum(row["forward_module_samples"] for row in actual),
        checkpoints=len(runs) * len(STEPS), evaluation_files=len(actual),
        evaluation_worlds=sum(row["worlds"] for row in actual), target_aliases=len(runs),
    )


def execute(out, workers=4):
    out = Path(out).resolve(); plan, static = verify(out); execution = out / "execution"
    require(not execution.exists(), "Never overwrite execution")
    execution.mkdir(); started = time.perf_counter()
    write(execution / "started.json", dict(at=time.time(), plan_sha256=sha(out / "plan.json"), workers=workers))
    try:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in SEEDS])
        runs = [run for group in groups for run in group]
        require([(run["seed"], run["condition"]) for run in runs] == [(seed, condition) for seed in SEEDS for condition in CONDITIONS], "Noncanonical grid")
        verify_pairing(execution, runs)
        arrays = [read(execution / f"seed_{seed}_arrays.json")["array_hashes"] for seed in SEEDS]
        require(all(value == arrays[0] for value in arrays), "Worker arrays differ")
        measured = measured_budget(runs)
        result = dict(status="completed", completed_at=time.time(), plan_sha256=sha(out / "plan.json"),
                      budget=static["budget"], measured_budget=measured, runs=runs, array_hashes=arrays[0],
                      elapsed_seconds=time.perf_counter() - started, no_external_model=True, optimizer_updates=len(runs) * design.UPDATES)
        write(execution / "results.json", result)
        write(execution / "status.json", dict(status="completed", results_sha256=sha(execution / "results.json")))
        return dict(status="completed", output=str(execution), elapsed_seconds=result["elapsed_seconds"], measured_budget=measured)
    except BaseException as error:
        write(execution / "failure.json", dict(status="failed", error=repr(error), elapsed_seconds=time.perf_counter() - started))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--out", required=True); parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    answer = prepare(args.out) if args.command == "prepare" else execute(args.out, args.workers) if args.command == "execute" else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
