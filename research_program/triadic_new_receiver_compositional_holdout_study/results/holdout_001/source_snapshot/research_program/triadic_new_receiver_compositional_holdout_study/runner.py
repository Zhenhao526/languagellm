"""Train a fresh receiver on seen joint combinations and audit a heldout set."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_factorized_neutral_altpartner_study import (
    design as source_design,
    kernel as source_kernel,
    remap as source_remap,
    runner as source_runner,
)
from research_program.triadic_message_study import runner as core
from research_program.triadic_new_receiver_transmission_study import runner as transmission_runner

from . import design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODULES = source_runner.MODULES
DIMS = source_runner.DIMS

CONFIG = dict(
    updates=design.UPDATES, batch_size=design.BATCH_SIZE, checkpoints=list(design.CHECKPOINTS),
    features=54, dtype="float64", learning_rate=0.001, global_gradient_clip=5.0,
    entropy_initial=0.001, entropy_zero_after_updates=1000, sender_entropy_coefficient=0,
    trajectories_per_state=2, sender_windows=2, sender_tokens_per_window=4,
    alphabet_size=8, proposal_count=16, intent_count=2, actions_per_actor=17, action_logits=18,
    evaluation_batch_size=8192, dimensions={k: list(v) for k, v in DIMS.items()},
    seeds=list(design.SEEDS), schedules=list(design.SCHEDULES), lives=["live", "silent"],
    conditions=list(design.CONDITIONS), arm=design.ARM,
    task="new_receiver_joint_combination_holdout",
    train_partition="1248_seen_semantic_orbits", heldout_partition="312_unseen_semantic_orbits",
    source_policy="audited factorized neutral/engage local NumPy tanh MLP",
    adaptation="replace A; update only A sender1/sender2/action modules on seen joint combinations",
    pairing="same parent, A initialization, world/message/rematch streams across live and silent; same streams as all-needs control",
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
    return json_hash({f"network_{i}_{key}": array_sha(value)
                      for i in indices for key, value in networks[i].items()})


def clone_networks(networks):
    return [{key: value.copy() for key, value in network.items()} for network in networks]


def full_run_path(seed, schedule):
    condition = f"{schedule}_new_receiver_PL_live"
    return design.TRANSMISSION_ROOT / "execution" / f"seed_{seed}_{condition}"


def load_generation1(seed, schedule):
    directory = full_run_path(seed, schedule)
    result_path, checkpoint = directory / "result.json", directory / "checkpoint_6000.npz"
    require(result_path.is_file() and checkpoint.is_file(), f"Missing generation-1 source for {seed}/{schedule}")
    result = json.loads(result_path.read_text())
    require(result["seed"] == seed and result["condition"] == f"{schedule}_new_receiver_PL_live" and result["live"] is True,
            "Generation-1 source identity mismatch")
    require(result["final_checkpoint_sha256"] == sha(checkpoint), "Generation-1 checkpoint hash mismatch")
    return result, checkpoint


def chain_control_path(seed, schedule, live):
    condition = f"generation2_replace_A_{schedule}_PL_{'live' if live else 'silent'}"
    return design.CHAIN_ROOT / "execution" / f"seed_{seed}_{condition}"


def load_chain_control(seed, schedule, live):
    directory = chain_control_path(seed, schedule, live)
    result_path, checkpoint = directory / "result.json", directory / "checkpoint_6000.npz"
    require(result_path.is_file() and checkpoint.is_file(), f"Missing all-needs control for {seed}/{schedule}/{live}")
    result = json.loads(result_path.read_text())
    require(result["seed"] == seed and result["generation"] == 2 and result["schedule"] == schedule and result["live"] is bool(live),
            "All-needs control identity mismatch")
    require(result["final_checkpoint_sha256"] == sha(checkpoint), "All-needs control checkpoint hash mismatch")
    return result, checkpoint


def source_artifacts():
    result = {}
    paths = (
        design.TRANSMISSION_ROOT / "plan.json", design.TRANSMISSION_ROOT / "prepared.json",
        design.TRANSMISSION_ROOT / "freeze.json", design.FULL_RESULT, design.FULL_AUDIT,
        design.CHAIN_ROOT / "plan.json", design.CHAIN_ROOT / "prepared.json",
        design.CHAIN_ROOT / "freeze.json", design.CHAIN_RESULT, design.CHAIN_AUDIT,
    )
    for path in paths:
        require(path.is_file(), f"Missing source artifact: {path}")
        result[str(path.resolve().relative_to(ROOT))] = sha(path)
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            for path in (full_run_path(seed, schedule) / "result.json", full_run_path(seed, schedule) / "checkpoint_6000.npz"):
                result[str(path.resolve().relative_to(ROOT))] = sha(path)
            for live in design.LIVES:
                control = chain_control_path(seed, schedule, bool(live))
                for path in (control / "result.json", control / "checkpoint_6000.npz"):
                    result[str(path.resolve().relative_to(ROOT))] = sha(path)
    return result


def source_code_paths():
    paths = [HERE / name for name in (
        "__init__.py", "design.py", "runner.py", "plan.md", "README.md",
        "tests/test_design.py", "tests/test_runner.py",
    )]
    paths += transmission_runner.source_code_paths()
    unique, seen = [], set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    require(all(path.is_file() for path in unique), "Missing frozen source code")
    return unique


def source_code_hashes():
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in source_code_paths()}


def build_prepared():
    full_prepared_path = design.TRANSMISSION_ROOT / "prepared.json"
    full_prepared = json.loads(full_prepared_path.read_text())
    static = design.prepared(full_prepared)
    static["full_reference_prepared_sha256"] = sha(full_prepared_path)
    static["full_reference_result_sha256"] = sha(design.FULL_RESULT)
    static["full_reference_audit_sha256"] = sha(design.FULL_AUDIT)
    static["chain_reference_result_sha256"] = sha(design.CHAIN_RESULT)
    static["chain_reference_audit_sha256"] = sha(design.CHAIN_AUDIT)
    static["source_code_sha256"] = source_code_hashes()
    static["source_artifacts_sha256"] = source_artifacts()
    static["runtime"] = dict(python=platform.python_version(), numpy=np.__version__)
    static["control_runs"] = []
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            for live in design.LIVES:
                control, checkpoint = load_chain_control(seed, schedule, bool(live))
                static["control_runs"].append(dict(
                    seed=seed, schedule=schedule, live=bool(live), condition=control["condition"],
                    checkpoint=str(checkpoint.resolve().relative_to(ROOT)), checkpoint_sha256=sha(checkpoint),
                    result_sha256=sha(checkpoint.parent / "result.json"), final_parameter_sha256=control["final_parameter_sha256"],
                ))
    static["control_runs_sha256"] = json_hash(static["control_runs"])
    static["budget"] = budget(static)
    return static


def budget(static):
    runs = len(design.SEEDS) * len(design.SCHEDULES) * 2
    monitor_worlds = static["monitor_spec"]["world_count"]
    heldout_worlds = static["heldout_spec"]["world_count"]
    seen_new_worlds = static["seen_new_spec"]["world_count"]
    trajectory_records = runs * len(design.CHECKPOINTS)
    final_records = runs * 2
    train_forward = runs * design.UPDATES * design.BATCH_SIZE * 2 * 9
    trajectory_forward = trajectory_records * (monitor_worlds + heldout_worlds) * 9
    final_forward = runs * (heldout_worlds + seen_new_worlds) * 9
    control_forward = trajectory_records * heldout_worlds * 9 + runs * (heldout_worlds + seen_new_worlds) * 9
    return dict(
        runs=runs, independent_seed_blocks=len(design.SEEDS), schedules=len(design.SCHEDULES), channels=2,
        training_updates=runs * design.UPDATES, training_world_samples=runs * design.UPDATES * design.BATCH_SIZE,
        training_forward_module_samples=train_forward, trajectory_records=trajectory_records,
        trajectory_monitor_worlds=trajectory_records * monitor_worlds, trajectory_heldout_worlds=trajectory_records * heldout_worlds,
        trajectory_forward_module_samples=trajectory_forward, final_records=final_records,
        final_worlds=runs * (heldout_worlds + seen_new_worlds), final_forward_module_samples=final_forward,
        all_needs_control_trajectory_forward_module_samples=control_forward,
        total_forward_module_samples=train_forward + trajectory_forward + final_forward + control_forward,
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
        source_code_sha256=static["source_code_sha256"], source_artifacts_sha256=static["source_artifacts_sha256"],
        full_reference_prepared_sha256=static["full_reference_prepared_sha256"],
        full_reference_result_sha256=static["full_reference_result_sha256"], full_reference_audit_sha256=static["full_reference_audit_sha256"],
        chain_reference_result_sha256=static["chain_reference_result_sha256"], chain_reference_audit_sha256=static["chain_reference_audit_sha256"],
        prepared_sha256=sha(out / "prepared.json"), runtime=static["runtime"], no_model_calls=True, no_training_updates=True,
        scientific_question="Does routed communication support task-protocol behavior on unseen joint combinations when all individual need values were seen during adaptation?",
    )
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n")
    (out / "freeze.json").write_text(json.dumps({"plan_sha256": sha(out / "plan.json"), "prepared_sha256": sha(out / "prepared.json")}, indent=2) + "\n")
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
    require(arrays["x_PL"].shape == (spec["world_count"], 3, 54), "Invalid PL array shape")
    require(np.all(arrays["x_PL"][:, :, 53] == 0), "Full-information flag leaked into PL")
    return arrays


def apply_replacement_gradients(networks, gradients):
    require(len(networks) == 9 and len(gradients) == 9, "Expected nine networks")
    return [
        ({key: value.copy() for key, value in gradients[i].items()} if i < 3
         else {key: np.zeros_like(value) for key, value in networks[i].items()})
        for i in range(9)
    ]


def train_run(seed, schedule, live, source_networks, arrays, static, execution, parent_result, parent_checkpoint):
    condition = f"{schedule}_new_receiver_compositional_{design.ARM}_PL_{'live' if live else 'silent'}"
    directory = Path(execution) / design.name(seed, condition)
    directory.mkdir(exist_ok=False)
    schedule_index = design.SCHEDULES.index(schedule)
    networks = clone_networks(source_networks)
    parent_hash = parameter_hash(networks)
    frozen_parent_hash = parameter_hash(networks, range(3, 9))
    fresh = [
        source_runner.make_network(np.random.SeedSequence([seed, 9901, 2, schedule_index, 0, module_index]), DIMS[module])
        for module_index, module in enumerate(MODULES)
    ]
    fresh_hash = parameter_hash(fresh)
    require(fresh_hash != parameter_hash(networks, range(3)), "Fresh A equals source A")
    networks[0:3] = fresh
    initial_hash = parameter_hash(networks)
    optimizer = core.base.make_adam(networks)
    world_rng = np.random.default_rng(np.random.SeedSequence([seed, 4100, 2, schedule_index]))
    message_rngs = core.make_message_rngs(int(seed + 930000 + 2 * 1000 + 100 * schedule_index))
    rematch_rng = np.random.default_rng(np.random.SeedSequence([seed, 4700, 2, schedule_index]))
    trajectory = []
    rematch_counts = np.zeros(6, dtype=np.int64)
    started = time.perf_counter()

    def checkpoint(step):
        path = directory / f"checkpoint_{step:04d}.npz"
        digest = source_runner.save_checkpoint(path, networks, optimizer, step, world_rng, message_rngs, rematch_rng)
        monitor = source_runner.evaluate_factorized(networks, arrays["monitor"], static["monitor_spec"], live)
        heldout = source_runner.evaluate_factorized(networks, arrays["heldout"], static["heldout_spec"], live)
        monitor["update"] = heldout["update"] = step
        row = dict(
            update=step, checkpoint_path=str(path), checkpoint_sha256=digest,
            target_trajectory={"monitor_seen": monitor, "heldout": heldout},
            compact_worlds=monitor["worlds"] + heldout["worlds"],
            forward_module_samples=9 * (monitor["worlds"] + heldout["worlds"]),
            parameter_sha256=parameter_hash(networks), frozen_parameter_sha256=parameter_hash(networks, range(3, 9)),
            replaced_agent_parameter_sha256=parameter_hash(networks, range(3)), elapsed_seconds=time.perf_counter() - started,
        )
        trajectory.append(row)
        with (directory / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(core.base.json_bytes(row).decode())

    checkpoint(0)
    with (directory / "training.jsonl").open("w", encoding="utf-8") as stream:
        for update in range(1, design.UPDATES + 1):
            world_uniforms = world_rng.random((design.BATCH_SIZE, 3))
            ids = source_design.sample_indices(static["train_spec"], world_uniforms)
            message_uniforms = core.draw_uniforms(message_rngs, design.BATCH_SIZE)
            canonical_x = arrays["train"]["x_PL"][ids]
            canonical_rewards = arrays["train"]["rewards"][ids]
            canonical_states = arrays["train"]["packed_states"][ids]
            if schedule == "rematched":
                permutation_ids = source_remap.permutation_indices(rematch_rng, design.BATCH_SIZE)
                x_batch, reward_batch, effective_states = source_remap.remap_batch(canonical_x, canonical_rewards, canonical_states, permutation_ids)
                rematch_counts += np.bincount(permutation_ids, minlength=6)
            else:
                permutation_ids = np.zeros(design.BATCH_SIZE, dtype=np.int8)
                x_batch, reward_batch, effective_states = canonical_x, canonical_rewards, canonical_states
            gradients, row = source_kernel.training_gradients(networks, x_batch, reward_batch, live, message_uniforms, update)
            norm, scale = core.base.adam_step(networks, apply_replacement_gradients(networks, gradients), optimizer, update)
            require(parameter_hash(networks, range(3, 9)) == frozen_parent_hash, "Frozen B/C changed")
            row.update(
                update=update, seed=seed, schedule=schedule, live=bool(live), condition=condition,
                world_uniforms_sha256=array_sha(world_uniforms), batch_indices_sha256=array_sha(ids),
                canonical_batch_states_sha256=array_sha(canonical_states), effective_batch_states_sha256=array_sha(effective_states),
                sample_uniforms_sha256=array_sha(message_uniforms), permutation_indices_sha256=array_sha(permutation_ids),
                rematched=schedule == "rematched", gradient_norm=norm, gradient_clip_scale=scale,
                frozen_parameter_sha256=parameter_hash(networks, range(3, 9)), replaced_agent_parameter_sha256=parameter_hash(networks, range(3)),
                elapsed_seconds=time.perf_counter() - started,
            )
            stream.write(core.base.json_bytes(row).decode())
            if update in design.CHECKPOINTS:
                stream.flush(); checkpoint(update)
    heldout_final = source_runner.evaluate_factorized(networks, arrays["heldout"], static["heldout_spec"], live)
    seen_new_final = source_runner.evaluate_factorized(networks, arrays["seen_new"], static["seen_new_spec"], live)
    result = dict(
        seed=seed, schedule=schedule, arm=design.ARM, condition=condition, live=bool(live), updates=design.UPDATES,
        parent_checkpoint_sha256=sha(parent_checkpoint), parent_parameter_sha256=parent_hash,
        frozen_parent_parameter_sha256=frozen_parent_hash, new_agent_initial_parameter_sha256=fresh_hash,
        initial_parameter_sha256=initial_hash, final_parameter_sha256=parameter_hash(networks),
        final_checkpoint_sha256=sha(directory / "checkpoint_6000.npz"), training_log_sha256=sha(directory / "training.jsonl"),
        trajectory=trajectory, final={"heldout": heldout_final, "seen_new": seen_new_final},
        rematch_histogram=rematch_counts.tolist(), rematch_assignments=int(rematch_counts.sum()),
        frozen_parameter_unchanged=True, parent_result_sha256=sha(Path(parent_checkpoint).parent / "result.json"), no_external_model=True,
        elapsed_seconds=time.perf_counter() - started,
    )
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (directory / "status.json").write_text(json.dumps({"status": "completed", "seed": seed, "condition": condition,
        "result_sha256": sha(directory / "result.json"), "at": now()}, ensure_ascii=False, indent=2) + "\n")
    return result


def evaluate_control(seed, schedule, live, arrays, static, execution):
    source_result, source_checkpoint = load_chain_control(seed, schedule, live)
    source_result = dict(source_result, _path=str(chain_control_path(seed, schedule, live) / "result.json"))
    directory = chain_control_path(seed, schedule, live)
    trajectory = []
    for step in design.CHECKPOINTS:
        checkpoint = directory / f"checkpoint_{step:04d}.npz"
        networks = source_runner.load_networks(checkpoint)
        heldout = source_runner.evaluate_factorized(networks, arrays["heldout"], static["heldout_spec"], bool(live)); heldout["update"] = step
        monitor = source_runner.evaluate_factorized(networks, arrays["monitor"], static["monitor_spec"], bool(live)); monitor["update"] = step
        trajectory.append(dict(update=step, checkpoint_path=str(checkpoint), checkpoint_sha256=sha(checkpoint),
                               target_trajectory={"monitor_seen": monitor, "heldout": heldout},
                               compact_worlds=monitor["worlds"] + heldout["worlds"],
                               forward_module_samples=9 * (monitor["worlds"] + heldout["worlds"]),
                               parameter_sha256=parameter_hash(networks)))
    checkpoint = directory / "checkpoint_6000.npz"
    networks = source_runner.load_networks(checkpoint)
    return dict(seed=seed, schedule=schedule, arm="all_needs_control", condition=source_result["condition"], live=bool(live),
                parent_checkpoint_sha256=source_result["parent_checkpoint_sha256"], final_parameter_sha256=source_result["final_parameter_sha256"],
                final_checkpoint_sha256=sha(checkpoint), training_log_sha256=source_result["training_log_sha256"], trajectory=trajectory,
                final={"heldout": source_runner.evaluate_factorized(networks, arrays["heldout"], static["heldout_spec"], bool(live)),
                       "seen_new": source_runner.evaluate_factorized(networks, arrays["seen_new"], static["seen_new_spec"], bool(live))},
                source_result_sha256=sha(directory / "result.json"), no_new_training=True)


def worker(payload):
    seed, static, execution = payload
    arrays = {part: make_arrays(static[f"{part}_spec"]) for part in ("train", "monitor", "heldout", "seen_new")}
    keys = ("packed_states", "rewards", "x_PL")
    before = {part: {key: array_sha(arrays[part][key]) for key in keys} for part in arrays}
    (Path(execution) / f"seed_{seed}_arrays.json").write_text(json.dumps({"array_hashes": before}, indent=2) + "\n")
    seen_runs, control_runs = [], []
    for schedule in design.SCHEDULES:
        source_result, source_checkpoint = load_generation1(seed, schedule)
        source_networks = source_runner.load_networks(source_checkpoint)
        require(parameter_hash(source_networks) == source_result["final_parameter_sha256"], "Generation-1 parameter mismatch")
        for live in design.LIVES:
            seen_runs.append(train_run(seed, schedule, bool(live), source_networks, arrays, static, execution,
                                       source_result, source_checkpoint))
            control_runs.append(evaluate_control(seed, schedule, bool(live), arrays, static, execution))
    after = {part: {key: array_sha(arrays[part][key]) for key in keys} for part in arrays}
    require(before == after, "Evaluation arrays mutated")
    return seen_runs, control_runs


def verify_pairing(execution, seen_runs, control_runs):
    execution = Path(execution)
    expected = {(seed, schedule, bool(live)) for seed in design.SEEDS for schedule in design.SCHEDULES for live in design.LIVES}
    seen_by = {(r["seed"], r["schedule"], bool(r["live"])): r for r in seen_runs}
    control_by = {(r["seed"], r["schedule"], bool(r["live"])): r for r in control_runs}
    require(set(seen_by) == expected and set(control_by) == expected, "Incomplete seen/control grid")
    for seed, schedule, live in sorted(expected):
        other = seen_by[seed, schedule, not live]
        run = seen_by[seed, schedule, live]
        for key in ("parent_checkpoint_sha256", "parent_parameter_sha256", "new_agent_initial_parameter_sha256", "frozen_parent_parameter_sha256"):
            require(run[key] == other[key], "Live/silent parent mismatch: " + key)
        expected_rematch = design.UPDATES * design.BATCH_SIZE if schedule == "rematched" else 0
        require(run["rematch_assignments"] == other["rematch_assignments"] == expected_rematch, "Rematch count mismatch")
        left = execution / design.name(seed, run["condition"]) / "training.jsonl"
        right = execution / design.name(seed, other["condition"]) / "training.jsonl"
        with left.open() as ls, right.open() as rs:
            count = 0
            for a_line, b_line in zip(ls, rs):
                a, b = json.loads(a_line), json.loads(b_line); count += 1
                require(a["update"] == b["update"] == count, "Paired update mismatch")
                for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256",
                            "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                    require(a[key] == b[key], "Paired stream differs: " + key)
            require(count == design.UPDATES, "Training log count mismatch")
        control = control_by[seed, schedule, live]
        require(control["source_result_sha256"] == sha(chain_control_path(seed, schedule, live) / "result.json"), "Control result hash mismatch")
    return True


def measured_budget(seen_runs, control_runs):
    require(len(seen_runs) == len(control_runs) == len(design.SEEDS) * len(design.SCHEDULES) * 2, "Unexpected run count")
    rows = [row for run in seen_runs + control_runs for row in run["trajectory"]]
    finals = [value for run in seen_runs + control_runs for value in run["final"].values()]
    require(all([row["update"] for row in run["trajectory"]] == list(design.CHECKPOINTS) for run in seen_runs + control_runs), "Incomplete trajectories")
    return dict(
        seen_training_forward_module_samples=len(seen_runs) * design.UPDATES * design.BATCH_SIZE * 2 * 9,
        seen_trajectory_forward_module_samples=sum(row["forward_module_samples"] for run in seen_runs for row in run["trajectory"]),
        seen_final_forward_module_samples=sum(9 * value["worlds"] for run in seen_runs for value in run["final"].values()),
        control_trajectory_forward_module_samples=sum(row["forward_module_samples"] for run in control_runs for row in run["trajectory"]),
        control_final_forward_module_samples=sum(9 * value["worlds"] for run in control_runs for value in run["final"].values()),
        trajectory_records=len(rows), final_records=len(finals), optimizer_updates=len(seen_runs) * design.UPDATES,
        rematched_assignments=sum(run.get("rematch_assignments", 0) for run in seen_runs),
    )


def execute(out):
    out = Path(out).resolve(); plan, static = verify(out)
    execution = out / "execution"; require(not execution.exists(), "Never overwrite execution"); execution.mkdir()
    started = time.perf_counter()
    (execution / "started.json").write_text(json.dumps({"started_at": now(), "pid": os.getpid(), "plan_sha256": sha(out / "plan.json")}, indent=2) + "\n")
    try:
        with multiprocessing.get_context("spawn").Pool(4) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in design.SEEDS])
        seen_runs = [run for group in groups for run in group[0]]
        control_runs = [run for group in groups for run in group[1]]
        expected_order = [(seed, schedule, bool(live)) for seed in design.SEEDS for schedule in design.SCHEDULES for live in design.LIVES]
        require([(r["seed"], r["schedule"], bool(r["live"])) for r in seen_runs] == expected_order, "Noncanonical seen order")
        require([(r["seed"], r["schedule"], bool(r["live"])) for r in control_runs] == expected_order, "Noncanonical control order")
        verify_pairing(execution, seen_runs, control_runs)
        measured = measured_budget(seen_runs, control_runs)
        result = dict(status="completed", completed_at=now(), elapsed_seconds=time.perf_counter() - started,
                      plan_sha256=sha(out / "plan.json"), budget=static["budget"], measured_budget=measured,
                      runs=seen_runs, all_needs_control_runs=control_runs,
                      primary="heldout joint-combination live-minus-silent and seen-to-heldout transfer gap",
                      experiment_type="new_receiver_compositional_joint_holdout", language_claim_automatically_supported=False,
                      no_external_model=True, no_llm=True, no_vision_model=True)
        (execution / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (execution / "status.json").write_text(json.dumps({"status": "completed", "at": now(), "results_sha256": sha(execution / "results.json")}, indent=2) + "\n")
        return dict(status="completed", output=str(execution), elapsed_seconds=result["elapsed_seconds"], measured_budget=measured)
    except BaseException as error:
        (execution / "failure.json").write_text(json.dumps({"status": "failed", "at": now(), "error": repr(error), "elapsed_seconds": time.perf_counter() - started}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--out", required=True)
    args = parser.parse_args(); answer = prepare(args.out) if args.command == "prepare" else execute(args.out) if args.command == "execute" else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
