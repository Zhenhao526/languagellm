"""Train a fresh receiver in each of the three role positions."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import shutil
import time
from copy import deepcopy
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment
from research_program.triadic_factorized_neutral_altpartner_study import (
    design as source_design,
    kernel as source_kernel,
    remap as source_remap,
    runner as source_runner,
)
from research_program.triadic_message_study import runner as core

from . import design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_ROOT = design.SOURCE_ROOT
MODULES = source_runner.MODULES
DIMS = source_runner.DIMS
CONFIG = dict(
    updates=design.UPDATES, batch_size=design.BATCH_SIZE, checkpoints=list(design.CHECKPOINTS),
    features=54, dtype="float64", learning_rate=0.001, global_gradient_clip=5.0,
    entropy_initial=0.001, entropy_zero_after_updates=1000, sender_entropy_coefficient=0,
    trajectories_per_state=2, sender_windows=2, sender_tokens_per_window=4,
    alphabet_size=8, proposal_count=16, intent_count=2, actions_per_actor=17, action_logits=18,
    evaluation_batch_size=8192, dimensions={k: list(v) for k, v in DIMS.items()},
    seeds=list(design.SEEDS), roles=list(design.ROLES), schedules=list(design.SCHEDULES), lives=["live", "silent"],
    conditions=list(design.CONDITIONS), task="receiver_role_crossover_on_altpair_coordination",
    source_observation="PL_own_need_and_public_layout_only",
    source_policy="factorized neutral/engage source checkpoint",
    adaptation_modules=["selected_role_sender1", "selected_role_sender2", "selected_role_action"],
    frozen_modules=["the other six modules"],
    optimizer="Adam; zero gradients supplied for the six frozen modules",
    pairing="same source checkpoint, fresh initialization, world uniforms, batch indices, message uniforms and rematch assignments for all roles and live/silent within each seed/schedule",
    evaluation="two fixed train layouts at checkpoints; all heldout layouts at final",
    no_model_calls=True, no_external_model=True, no_llm=True, no_vision_model=True,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return core.base.now()


def sha(path):
    return core.base.sha(path)


def json_hash(value):
    return core.base.json_hash(value)


def array_sha(value):
    return core.base.array_sha(value)


def parameter_hash(networks, indices=None):
    if indices is None:
        indices = range(len(networks))
    payload = {}
    for i in indices:
        for key, value in networks[i].items():
            payload[f"network_{i}_{key}"] = array_sha(value)
    return json_hash(payload)


def clone_network(network):
    return {key: value.copy() for key, value in network.items()}


def clone_networks(networks):
    return [clone_network(network) for network in networks]


def fresh_role_networks(seed, schedule_index):
    """One fresh three-module initialization reused across the three role arms."""
    return [
        source_runner.make_network(np.random.SeedSequence([seed, 9901, schedule_index, module_index]), DIMS[module])
        for module_index, module in enumerate(MODULES)
    ]


def source_run_path(seed, schedule):
    condition = design.source_condition(schedule)
    return SOURCE_ROOT / "execution" / f"seed_{seed}_{condition}"


def source_artifacts():
    """Hash all source inputs used by this experiment before freezing."""
    result = {}
    prepared = SOURCE_ROOT / "prepared.json"
    execution = SOURCE_ROOT / "execution" / "results.json"
    require(prepared.is_file() and execution.is_file(), "Completed source result is missing")
    result[str(prepared.resolve().relative_to(ROOT))] = sha(prepared)
    result[str(execution.resolve().relative_to(ROOT))] = sha(execution)
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            run_dir = source_run_path(seed, schedule)
            result_json = run_dir / "result.json"
            checkpoint = run_dir / "checkpoint_6000.npz"
            require(result_json.is_file() and checkpoint.is_file(), f"Missing source run for {seed}/{schedule}")
            result[str(result_json.resolve().relative_to(ROOT))] = sha(result_json)
            result[str(checkpoint.resolve().relative_to(ROOT))] = sha(checkpoint)
    return result


def source_code_paths():
    # Only execution and design inputs are frozen here.  Audit, aggregation,
    # and plotting scripts are post-hoc analysis code and may be repaired
    # without changing the trained policy or task definition.
    paths = [HERE / name for name in (
        "__init__.py", "design.py", "runner.py", "plan.md", "README.md",
        "tests/test_design.py", "tests/test_runner.py",
    )]
    paths += [
        Path(source_design.__file__), Path(source_kernel.__file__), Path(source_remap.__file__),
        Path(source_runner.__file__), Path(dataset.__file__), Path(environment.__file__),
        Path(core.__file__), Path(core.base.__file__),
    ]
    require(all(path.is_file() for path in paths), "Missing source code for freeze")
    return paths


def source_code_hashes():
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in source_code_paths()}


def load_source_prepared():
    path = SOURCE_ROOT / "prepared.json"
    require(path.is_file(), "Source prepared.json missing")
    prepared = json.loads(path.read_text())
    require(prepared.get("schema") == "triadic_factorized_neutral_altpartner_v1", "Unexpected source schema")
    require(prepared.get("seeds") == list(source_design.SEEDS), "Source seed grid changed")
    return prepared


def load_source_result(seed, schedule):
    path = source_run_path(seed, schedule) / "result.json"
    result = json.loads(path.read_text())
    condition = design.source_condition(schedule)
    require(result["seed"] == seed and result["condition"] == condition and result["live"] is True, "Source run identity mismatch")
    checkpoint = source_run_path(seed, schedule) / "checkpoint_6000.npz"
    require(result["final_checkpoint_sha256"] == sha(checkpoint), "Source checkpoint hash disagrees with result")
    require(result["updates"] == design.UPDATES, "Source update count changed")
    return result, checkpoint


def build_prepared():
    source_prepared = load_source_prepared()
    static = design.prepared(source_prepared)
    static["source_prepared_sha256"] = sha(SOURCE_ROOT / "prepared.json")
    static["source_code_sha256"] = source_code_hashes()
    static["source_artifacts_sha256"] = source_artifacts()
    static["runtime"] = dict(python=platform.python_version(), numpy=np.__version__)
    static["budget"] = budget(static)
    # A compact source-run index is included in the frozen input.
    static["source_runs"] = []
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            result, checkpoint = load_source_result(seed, schedule)
            static["source_runs"].append(dict(
                seed=seed, schedule=schedule, condition=design.source_condition(schedule),
                checkpoint=str(checkpoint.resolve().relative_to(ROOT)),
                checkpoint_sha256=sha(checkpoint), result_sha256=sha(source_run_path(seed, schedule) / "result.json"),
                source_final_parameter_sha256=result["final_parameter_sha256"],
            ))
    static["source_runs_sha256"] = json_hash(static["source_runs"])
    return static


def budget(static):
    runs = len(design.SEEDS) * len(design.CONDITIONS)
    monitor_worlds = static["monitor_spec"]["world_count"]
    final_worlds = static["final_spec"]["world_count"]
    monitor_records = runs * len(design.CHECKPOINTS)
    final_records = runs
    adaptation_train_forward = runs * design.UPDATES * design.BATCH_SIZE * 2 * 9
    monitor_forward = monitor_records * monitor_worlds * 9
    final_forward = final_records * final_worlds * 9
    baseline_forward = len(design.SEEDS) * len(design.SCHEDULES) * 2 * final_worlds * 9
    return dict(
        runs=runs, independent_seed_blocks=len(design.SEEDS), schedules=len(design.SCHEDULES), channels=2,
        training_updates=runs * design.UPDATES, training_world_samples=runs * design.UPDATES * design.BATCH_SIZE,
        training_forward_module_samples=adaptation_train_forward, monitor_records=monitor_records,
        monitor_worlds=monitor_records * monitor_worlds, monitor_forward_module_samples=monitor_forward,
        final_records=final_records, final_worlds=final_records * final_worlds,
        final_forward_module_samples=final_forward, inherited_baseline_forward_module_samples=baseline_forward,
        total_forward_module_samples=adaptation_train_forward + monitor_forward + final_forward + baseline_forward,
        frozen_source_checkpoint_count=len(design.SEEDS) * len(design.SCHEDULES),
    )


def prepared():
    return build_prepared()


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), "Never overwrite preparation")
    static = build_prepared()
    out.mkdir(parents=True)
    for relative in static["source_code_sha256"]:
        target = out / "source_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (out / "prepared.json").write_text(json.dumps(static, ensure_ascii=False, indent=2) + "\n")
    plan = dict(
        status="prepared_without_training", created_at=now(), config=CONFIG,
        source_prepared_sha256=static["source_prepared_sha256"], source_code_sha256=static["source_code_sha256"],
        source_artifacts_sha256=static["source_artifacts_sha256"], prepared_sha256=sha(out / "prepared.json"),
        runtime=static["runtime"], no_model_calls=True, no_training_updates=True,
        scientific_question="Can a freshly initialized receiver/participant recover a frozen pair's task protocol from task feedback, and does packet routing change that trajectory?",
    )
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}, ensure_ascii=False, indent=2) + "\n")
    verify(out)
    return dict(status="prepared_without_training", output=str(out), plan_sha256=sha(out / "plan.json"), budget=static["budget"])


def verify(out):
    out = Path(out).resolve()
    plan = json.loads((out / "plan.json").read_text())
    static = json.loads((out / "prepared.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan hash mismatch")
    require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared hash mismatch")
    current = build_prepared()
    require(static == current, "Current source/config differs from frozen preparation")
    require(plan["config"] == CONFIG and plan["runtime"] == current["runtime"], "Configuration/runtime changed")
    for relative, digest in plan["source_code_sha256"].items():
        require(sha(out / "source_snapshot" / relative) == digest, "Code snapshot changed: " + relative)
    for relative, digest in plan["source_artifacts_sha256"].items():
        require(sha(ROOT / relative) == digest, "Source artifact changed: " + relative)
    return plan, static


def make_arrays(spec):
    arrays = source_runner.make_arrays(spec)
    # Keep the source names explicit and avoid accidentally using FI/LL data.
    require(arrays["x_PL"].shape == (spec["world_count"], 3, 54), "Invalid PL array shape")
    require(np.all(arrays["x_PL"][:, :, 53] == 0), "Full-information flag leaked into PL")
    return arrays


def checkpoint(path, networks, optimizer, update, world_rng, message_rngs, rematch_rng):
    return source_runner.save_checkpoint(path, networks, optimizer, update, world_rng, message_rngs, rematch_rng)


def zero_frozen_gradients(gradients, networks, replaced_agent):
    require(len(gradients) == 9 and len(networks) == 9, "Expected nine source/new networks")
    output = [{key: np.zeros_like(value) for key, value in networks[i].items()} for i in range(9)]
    for module in range(3):
        index = 3 * replaced_agent + module
        output[index] = gradients[index]
    return output


def train_run(seed, role, schedule, live, source_networks, arrays, static, execution, baseline, fresh_networks):
    replaced_agent = design.role_index(role)
    condition = f"replace_{role}_new_receiver_PL_{schedule}_{'live' if live else 'silent'}"
    directory = Path(execution) / design.name(seed, condition)
    directory.mkdir(exist_ok=False)
    schedule_index = design.SCHEDULES.index(schedule)
    networks = clone_networks(source_networks)
    source_full_hash = parameter_hash(networks)
    frozen_indices = [index for index in range(9) if index // 3 != replaced_agent]
    frozen_hash = parameter_hash(networks, frozen_indices)
    source_role_hash = parameter_hash(networks, range(3 * replaced_agent, 3 * replaced_agent + 3))
    new_role_hash = parameter_hash(fresh_networks)
    require(new_role_hash != source_role_hash, "Fresh role initialization accidentally equals source role")
    networks[3 * replaced_agent:3 * replaced_agent + 3] = clone_networks(fresh_networks)
    initial_hash = parameter_hash(networks)
    optimizer = core.base.make_adam(networks)
    world_rng = np.random.default_rng(np.random.SeedSequence([seed, 2100, schedule_index]))
    message_seed = int(seed + 910000 + 1000 * schedule_index)
    message_rngs = core.make_message_rngs(message_seed)
    rematch_rng = np.random.default_rng(np.random.SeedSequence([seed, 2700, schedule_index]))
    train_spec = static["train_spec"]
    trajectory = []
    rematch_counts = np.zeros(6, dtype=np.int64)
    started = time.perf_counter()

    def save_and_evaluate(step):
        path = directory / f"checkpoint_{step:04d}.npz"
        digest = checkpoint(path, networks, optimizer, step, world_rng, message_rngs, rematch_rng)
        metrics = source_runner.evaluate_factorized(networks, arrays["monitor"], static["monitor_spec"], live)
        metrics["update"] = step
        row = dict(
            update=step, checkpoint_path=str(path), checkpoint_sha256=digest,
            target_trajectory=metrics, compact_worlds=static["monitor_spec"]["world_count"],
            forward_module_samples=9 * static["monitor_spec"]["world_count"],
            parameter_sha256=parameter_hash(networks), frozen_parameter_sha256=parameter_hash(networks, frozen_indices),
            new_agent_parameter_sha256=parameter_hash(networks, range(3 * replaced_agent, 3 * replaced_agent + 3)),
            elapsed_seconds=time.perf_counter() - started,
        )
        trajectory.append(row)
        with (directory / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(core.base.json_bytes(row).decode())

    save_and_evaluate(0)
    with (directory / "training.jsonl").open("w", encoding="utf-8") as stream:
        for update in range(1, design.UPDATES + 1):
            world_uniforms = world_rng.random((design.BATCH_SIZE, 3))
            ids = source_design.sample_indices(train_spec, world_uniforms)
            message_uniforms = core.draw_uniforms(message_rngs, design.BATCH_SIZE)
            canonical_x = arrays["train"]["x_PL"][ids]
            canonical_rewards = arrays["train"]["rewards"][ids]
            canonical_states = arrays["train"]["packed_states"][ids]
            if schedule == "rematched":
                permutation_ids = source_remap.permutation_indices(rematch_rng, design.BATCH_SIZE)
                x_batch, reward_batch, effective_states = source_remap.remap_batch(
                    canonical_x, canonical_rewards, canonical_states, permutation_ids
                )
                rematch_counts += np.bincount(permutation_ids, minlength=6)
            else:
                permutation_ids = np.zeros(design.BATCH_SIZE, dtype=np.int8)
                x_batch, reward_batch, effective_states = canonical_x, canonical_rewards, canonical_states
            gradients, row = source_kernel.training_gradients(
                networks, x_batch, reward_batch, live, message_uniforms, update
            )
            applied = zero_frozen_gradients(gradients, networks, replaced_agent)
            norm, scale = core.base.adam_step(networks, applied, optimizer, update)
            require(parameter_hash(networks, frozen_indices) == frozen_hash, "Frozen roles changed during update")
            row.update(
                update=update, seed=seed, schedule=schedule, live=bool(live), condition=condition,
                world_uniforms_sha256=array_sha(world_uniforms), batch_indices_sha256=array_sha(ids),
                canonical_batch_states_sha256=array_sha(canonical_states), effective_batch_states_sha256=array_sha(effective_states),
                sample_uniforms_sha256=array_sha(message_uniforms), permutation_indices_sha256=array_sha(permutation_ids),
                rematched=schedule == "rematched", gradient_norm=norm, gradient_clip_scale=scale,
                frozen_parameter_sha256=parameter_hash(networks, frozen_indices),
                new_agent_parameter_sha256=parameter_hash(networks, range(3 * replaced_agent, 3 * replaced_agent + 3)),
                elapsed_seconds=time.perf_counter() - started,
            )
            stream.write(core.base.json_bytes(row).decode())
            if update in design.CHECKPOINTS:
                stream.flush()
                save_and_evaluate(update)
    final = source_runner.evaluate_factorized(networks, arrays["final"], static["final_spec"], live)
    result = dict(
        seed=seed, role=role, replaced_agent=replaced_agent, schedule=schedule, condition=condition, live=bool(live), updates=design.UPDATES,
        source_checkpoint_sha256=sha(source_run_path(seed, schedule) / "checkpoint_6000.npz"),
        source_full_parameter_sha256=source_full_hash, source_role_parameter_sha256=source_role_hash,
        frozen_parameter_sha256=frozen_hash, new_agent_initial_parameter_sha256=new_role_hash,
        initial_parameter_sha256=initial_hash, final_parameter_sha256=parameter_hash(networks),
        final_checkpoint_sha256=sha(directory / "checkpoint_6000.npz"), training_log_sha256=sha(directory / "training.jsonl"),
        trajectory=trajectory, final={"new_layouts": final}, inherited_source={"new_layouts": baseline},
        rematch_histogram=rematch_counts.tolist(), rematch_assignments=int(rematch_counts.sum()),
        frozen_parameters_unchanged=True, no_external_model=True, elapsed_seconds=time.perf_counter() - started,
    )
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (directory / "status.json").write_text(json.dumps({
        "status": "completed", "seed": seed, "condition": condition,
        "result_sha256": sha(directory / "result.json"), "at": now(),
    }, ensure_ascii=False, indent=2) + "\n")
    return result


def worker(payload):
    seed, static, execution = payload
    arrays = {
        "train": make_arrays(static["train_spec"]),
        "monitor": make_arrays(static["monitor_spec"]),
        "final": make_arrays(static["final_spec"]),
    }
    key_names = ("packed_states", "rewards", "x_PL")
    array_hashes = {part: {key: array_sha(arrays[part][key]) for key in key_names} for part in arrays}
    (Path(execution) / f"seed_{seed}_arrays.json").write_text(json.dumps({"array_hashes": array_hashes}, indent=2) + "\n")
    baseline_cache = {}
    fresh_cache = {}
    runs = []
    for schedule in design.SCHEDULES:
        result, checkpoint_path = load_source_result(seed, schedule)
        source_networks = source_runner.load_networks(checkpoint_path)
        fresh_cache[schedule] = fresh_role_networks(seed, design.SCHEDULES.index(schedule))
        baseline_cache[schedule] = {}
        for live in design.LIVES:
            baseline_cache[schedule][bool(live)] = source_runner.evaluate_factorized(
                source_networks, arrays["final"], static["final_spec"], bool(live)
            )
        for role in design.ROLES:
            for live in design.LIVES:
                runs.append(train_run(seed, role, schedule, bool(live), source_networks, arrays, static, execution,
                                      baseline_cache[schedule][bool(live)], fresh_cache[schedule]))
    after = {part: {key: array_sha(arrays[part][key]) for key in key_names} for part in arrays}
    require(array_hashes == after, "Evaluation arrays mutated")
    return runs


def verify_pairing(execution, runs):
    execution = Path(execution)
    expected = {(seed, condition) for seed in design.SEEDS for condition in design.CONDITIONS}
    by = {(run["seed"], run["condition"]): run for run in runs}
    require(set(by) == expected and len(by) == len(expected), "Incomplete paired grid")
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            expected_rematch = design.UPDATES * design.BATCH_SIZE if schedule == "rematched" else 0
            role_runs = {}
            for role in design.ROLES:
                live = by[(seed, f"replace_{role}_new_receiver_PL_{schedule}_live")]
                silent = by[(seed, f"replace_{role}_new_receiver_PL_{schedule}_silent")]
                role_runs[role] = (live, silent)
                require(live["source_checkpoint_sha256"] == silent["source_checkpoint_sha256"], "Source checkpoint pairing failed")
                require(live["new_agent_initial_parameter_sha256"] == silent["new_agent_initial_parameter_sha256"], "Fresh role pairing failed")
                require(live["frozen_parameter_sha256"] == silent["frozen_parameter_sha256"], "Frozen role pairing failed")
                for run in (live, silent):
                    require(sum(run["rematch_histogram"]) == expected_rematch and run["rematch_assignments"] == expected_rematch, "Rematch count mismatch")
                live_path = execution / design.name(seed, live["condition"]) / "training.jsonl"
                silent_path = execution / design.name(seed, silent["condition"]) / "training.jsonl"
                with live_path.open() as ls, silent_path.open() as ss:
                    count = 0
                    for left, right in zip(ls, ss):
                        a, b = json.loads(left), json.loads(right); count += 1
                        require(a["update"] == b["update"] == count, "Paired update mismatch")
                        for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256", "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                            require(a[key] == b[key], "Live/silent paired stream differs: " + key)
                    require(count == design.UPDATES, "Training log count mismatch")
            require(len({live["new_agent_initial_parameter_sha256"] for live, _ in role_runs.values()}) == 1, "Role arms did not share fresh initialization")
            reference_live = role_runs[design.ROLES[0]][0]
            reference_rows = [json.loads(line) for line in (execution / design.name(seed, reference_live["condition"]) / "training.jsonl").read_text().splitlines()]
            for role in design.ROLES[1:]:
                candidate = role_runs[role][0]
                candidate_rows = [json.loads(line) for line in (execution / design.name(seed, candidate["condition"]) / "training.jsonl").read_text().splitlines()]
                for left, right in zip(reference_rows, candidate_rows):
                    for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256", "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                        require(left[key] == right[key], "Role-arm exogenous stream differs: " + key)
    return True


def measured_budget(runs):
    require(len(runs) == len(design.SEEDS) * len(design.CONDITIONS), "Unexpected run count")
    trajectories = [row for run in runs for row in run["trajectory"]]
    finals = [value for run in runs for value in run["final"].values()]
    require(all([row["update"] for row in run["trajectory"]] == list(design.CHECKPOINTS) for run in runs), "Incomplete trajectory")
    return dict(
        training_forward_module_samples=len(runs) * design.UPDATES * design.BATCH_SIZE * 2 * 9,
        monitor_forward_module_samples=sum(row["forward_module_samples"] for row in trajectories),
        final_forward_module_samples=sum(9 * row["worlds"] for row in finals),
        monitor_records=len(trajectories), final_records=len(finals),
        monitor_worlds=sum(row["compact_worlds"] for row in trajectories), final_worlds=sum(row["worlds"] for row in finals),
        rematched_assignments=sum(run["rematch_assignments"] for run in runs),
        optimizer_updates=len(runs) * design.UPDATES,
        frozen_agent_optimizer_updates=0,
    )


def execute(out, workers=4):
    out = Path(out).resolve()
    plan, static = verify(out)
    execution = out / "execution"
    require(not execution.exists(), "Never overwrite execution")
    execution.mkdir()
    started = time.perf_counter()
    (execution / "started.json").write_text(json.dumps({"started_at": now(), "pid": os.getpid(), "plan_sha256": sha(out / "plan.json")}, indent=2) + "\n")
    try:
        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in design.SEEDS])
        runs = [run for group in groups for run in group]
        runs.sort(key=lambda run: (run["seed"], design.CONDITIONS.index(run["condition"])))
        require([(run["seed"], run["condition"]) for run in runs] == [(seed, condition) for seed in design.SEEDS for condition in design.CONDITIONS], "Noncanonical run order")
        verify_pairing(execution, runs)
        measured = measured_budget(runs)
        result = dict(
            status="completed", completed_at=now(), elapsed_seconds=time.perf_counter() - started,
            plan_sha256=sha(out / "plan.json"), budget=static["budget"], measured_budget=measured, runs=runs,
            primary="live_minus_silent_new_receiver_transmission_gain", experiment_type="frozen_pair_new_receiver_adaptation",
            language_claim_automatically_supported=False, no_external_model=True, no_llm=True, no_vision_model=True,
        )
        (execution / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (execution / "status.json").write_text(json.dumps({"status": "completed", "at": now(), "results_sha256": sha(execution / "results.json")}, indent=2) + "\n")
        return dict(status="completed", output=str(execution), elapsed_seconds=result["elapsed_seconds"], measured_budget=measured)
    except BaseException as error:
        (execution / "failure.json").write_text(json.dumps({"status": "failed", "at": now(), "error": repr(error), "elapsed_seconds": time.perf_counter() - started}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    answer = prepare(args.out) if args.command == "prepare" else execute(args.out, args.workers) if args.command == "execute" else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
