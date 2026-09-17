"""Execute a short learner-replacement chain over the audited PL protocol.

Generation 1 is the completed live transmission checkpoint.  Generation 2
replaces A and adapts only A; generation 3 replaces B and adapts only B from
the generation-2 live endpoint.  Live and silent runs share every stochastic
stream within a generation so that the packet-routing contrast remains
paired.
"""
from __future__ import annotations

import argparse
import json
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
    seeds=list(design.SEEDS), schedules=list(design.SCHEDULES), generations=list(design.GENERATIONS),
    replacement_roles={"generation2": "A", "generation3": "B"},
    lives=["live", "silent"], conditions=list(design.CONDITIONS),
    task="two_generation_frozen_partner_protocol_chain",
    observation="PL_own_need_and_public_layout_only",
    source_policy="audited factorized neutral/engage local NumPy tanh MLP",
    adaptation="only the replaced agent's three modules are updated; other agents are frozen",
    optimizer="Adam; exact zero gradients for frozen modules",
    pairing="same parent endpoint, fresh learner initialization, worlds, batches, packet uniforms and rematching assignments within each generation/schedule",
    evaluation="same two-layout monitor and six-layout heldout final evaluation as transmission reference",
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
    """Hash this package's network list with an unambiguous index namespace."""
    if indices is None:
        indices = range(len(networks))
    return json_hash({f"network_{i}_{key}": array_sha(value)
                      for i in indices for key, value in networks[i].items()})


def clone_network(network):
    return {key: value.copy() for key, value in network.items()}


def clone_networks(networks):
    return [clone_network(network) for network in networks]


def full_run_path(seed, schedule, live=True):
    suffix = "live" if live else "silent"
    condition = f"{schedule}_new_receiver_PL_{suffix}"
    return design.TRANSMISSION_ROOT / "execution" / f"seed_{seed}_{condition}"


def load_generation1_result(seed, schedule):
    path = full_run_path(seed, schedule, live=True)
    result_path = path / "result.json"
    checkpoint_path = path / "checkpoint_6000.npz"
    require(result_path.is_file() and checkpoint_path.is_file(), f"Missing generation-1 reference for {seed}/{schedule}")
    result = json.loads(result_path.read_text())
    expected = f"{schedule}_new_receiver_PL_live"
    require(result.get("seed") == seed and result.get("condition") == expected and result.get("live") is True,
            "Generation-1 reference identity mismatch")
    require(result.get("updates") == design.UPDATES and result.get("final_checkpoint_sha256") == sha(checkpoint_path),
            "Generation-1 reference checkpoint mismatch")
    return result, checkpoint_path


def fresh_replaced_networks(seed, generation, schedule_index):
    """Fresh learner for the replaced role; identical in paired live/silent runs."""
    replaced = design.replaced_agent(generation)
    return [
        source_runner.make_network(
            np.random.SeedSequence([seed, 9901, generation, schedule_index, replaced, module_index]),
            DIMS[module],
        )
        for module_index, module in enumerate(MODULES)
    ]


def replaced_indices(generation):
    agent = design.replaced_agent(generation)
    return tuple(3 * agent + module_index for module_index in range(3))


def source_artifacts():
    """Hash every completed input checkpoint used to seed the chain."""
    result = {}
    for path in (
        design.TRANSMISSION_ROOT / "plan.json",
        design.TRANSMISSION_ROOT / "prepared.json",
        design.TRANSMISSION_ROOT / "freeze.json",
        design.FULL_RESULT,
        design.FULL_AUDIT,
    ):
        require(path.is_file(), f"Missing generation-1 artifact: {path}")
        result[str(path.resolve().relative_to(ROOT))] = sha(path)
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            run_dir = full_run_path(seed, schedule, live=True)
            for filename in ("result.json", "checkpoint_6000.npz"):
                path = run_dir / filename
                require(path.is_file(), f"Missing generation-1 source file: {path}")
                result[str(path.resolve().relative_to(ROOT))] = sha(path)
    return result


def source_code_paths():
    paths = [HERE / name for name in (
        "__init__.py", "design.py", "runner.py", "plan.md", "README.md",
        "tests/test_design.py", "tests/test_runner.py",
    )]
    paths += transmission_runner.source_code_paths()
    unique = []
    seen = set()
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
    require(full_prepared_path.is_file(), "Generation-1 prepared input is missing")
    full_prepared = json.loads(full_prepared_path.read_text())
    static = design.prepared(full_prepared)
    static["full_reference_prepared_sha256"] = sha(full_prepared_path)
    static["full_reference_result_sha256"] = sha(design.FULL_RESULT)
    static["full_reference_audit_sha256"] = sha(design.FULL_AUDIT)
    static["source_code_sha256"] = source_code_hashes()
    static["source_artifacts_sha256"] = source_artifacts()
    static["runtime"] = dict(python=platform.python_version(), numpy=np.__version__)
    static["generation1_runs"] = []
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            result, checkpoint_path = load_generation1_result(seed, schedule)
            static["generation1_runs"].append(dict(
                seed=seed, schedule=schedule,
                condition=f"{schedule}_new_receiver_PL_live",
                checkpoint=str(checkpoint_path.resolve().relative_to(ROOT)),
                checkpoint_sha256=sha(checkpoint_path),
                result_sha256=sha(checkpoint_path.parent / "result.json"),
                final_parameter_sha256=result["final_parameter_sha256"],
            ))
    static["generation1_runs_sha256"] = json_hash(static["generation1_runs"])
    static["budget"] = budget(static)
    return static


def budget(static):
    runs = len(design.SEEDS) * len(design.GENERATIONS) * len(design.SCHEDULES) * 2
    monitor_worlds = static["monitor_spec"]["world_count"]
    final_worlds = static["final_spec"]["world_count"]
    monitor_records = runs * len(design.CHECKPOINTS)
    final_records = runs
    train_forward = runs * design.UPDATES * design.BATCH_SIZE * 2 * 9
    monitor_forward = monitor_records * monitor_worlds * 9
    final_forward = final_records * final_worlds * 9
    parent_baseline_forward = runs * final_worlds * 9
    return dict(
        runs=runs, independent_seed_blocks=len(design.SEEDS), generations=len(design.GENERATIONS),
        schedules=len(design.SCHEDULES), channels=2,
        training_updates=runs * design.UPDATES,
        training_world_samples=runs * design.UPDATES * design.BATCH_SIZE,
        training_forward_module_samples=train_forward,
        monitor_records=monitor_records, monitor_worlds=monitor_records * monitor_worlds,
        monitor_forward_module_samples=monitor_forward,
        final_records=final_records, final_worlds=final_records * final_worlds,
        final_forward_module_samples=final_forward,
        parent_baseline_forward_module_samples=parent_baseline_forward,
        total_forward_module_samples=train_forward + monitor_forward + final_forward + parent_baseline_forward,
        generation1_source_checkpoints=len(design.SEEDS) * len(design.SCHEDULES),
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
        full_reference_result_sha256=static["full_reference_result_sha256"],
        full_reference_audit_sha256=static["full_reference_audit_sha256"],
        prepared_sha256=sha(out / "prepared.json"), runtime=static["runtime"],
        no_model_calls=True, no_training_updates=True,
        scientific_question="Does a short learner-replacement chain retain a task protocol, and does packet routing change adaptation and intergenerational retention?",
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


def apply_replacement_gradients(networks, gradients, generation):
    allowed = set(replaced_indices(generation))
    require(len(networks) == 9 and len(gradients) == 9, "Expected nine networks and gradients")
    return [
        ({key: value.copy() for key, value in gradients[i].items()} if i in allowed
         else {key: np.zeros_like(value) for key, value in networks[i].items()})
        for i in range(9)
    ]


def train_run(seed, generation, schedule, live, parent_networks, arrays, static, execution,
              parent_meta, inherited_parent):
    arm = "replace_A" if generation == 2 else "replace_B"
    condition = f"generation{generation}_{arm}_{schedule}_PL_{'live' if live else 'silent'}"
    directory = Path(execution) / design.name(seed, condition)
    directory.mkdir(exist_ok=False)
    schedule_index = design.SCHEDULES.index(schedule)
    networks = clone_networks(parent_networks)
    parent_hash = parameter_hash(networks)
    frozen_indices = tuple(i for i in range(9) if i not in replaced_indices(generation))
    frozen_parent_hash = parameter_hash(networks, frozen_indices)
    fresh = fresh_replaced_networks(seed, generation, schedule_index)
    fresh_hash = parameter_hash(fresh)
    parent_replaced_hash = parameter_hash(networks, replaced_indices(generation))
    require(fresh_hash != parent_replaced_hash, "Fresh learner accidentally equals parent role")
    for local, index in enumerate(replaced_indices(generation)):
        networks[index] = fresh[local]
    initial_hash = parameter_hash(networks)
    optimizer = core.base.make_adam(networks)
    world_rng = np.random.default_rng(np.random.SeedSequence([seed, 4100, generation, schedule_index]))
    message_rngs = core.make_message_rngs(int(seed + 930000 + generation * 1000 + 100 * schedule_index))
    rematch_rng = np.random.default_rng(np.random.SeedSequence([seed, 4700, generation, schedule_index]))
    trajectory = []
    rematch_counts = np.zeros(6, dtype=np.int64)
    started = time.perf_counter()

    def save_and_evaluate(step):
        path = directory / f"checkpoint_{step:04d}.npz"
        digest = source_runner.save_checkpoint(path, networks, optimizer, step, world_rng, message_rngs, rematch_rng)
        metrics = source_runner.evaluate_factorized(networks, arrays["monitor"], static["monitor_spec"], live)
        metrics["update"] = step
        row = dict(
            update=step, checkpoint_path=str(path), checkpoint_sha256=digest,
            target_trajectory=metrics, compact_worlds=metrics["worlds"],
            forward_module_samples=9 * metrics["worlds"], parameter_sha256=parameter_hash(networks),
            frozen_parameter_sha256=parameter_hash(networks, frozen_indices),
            replaced_agent_parameter_sha256=parameter_hash(networks, replaced_indices(generation)),
            elapsed_seconds=time.perf_counter() - started,
        )
        trajectory.append(row)
        with (directory / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(core.base.json_bytes(row).decode())

    save_and_evaluate(0)
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
            norm, scale = core.base.adam_step(
                networks, apply_replacement_gradients(networks, gradients, generation), optimizer, update
            )
            require(parameter_hash(networks, frozen_indices) == frozen_parent_hash,
                    "Frozen parent modules changed during update")
            row.update(
                update=update, seed=seed, generation=generation, arm=arm, schedule=schedule,
                live=bool(live), condition=condition,
                world_uniforms_sha256=array_sha(world_uniforms), batch_indices_sha256=array_sha(ids),
                canonical_batch_states_sha256=array_sha(canonical_states), effective_batch_states_sha256=array_sha(effective_states),
                sample_uniforms_sha256=array_sha(message_uniforms), permutation_indices_sha256=array_sha(permutation_ids),
                rematched=schedule == "rematched", gradient_norm=norm, gradient_clip_scale=scale,
                frozen_parameter_sha256=parameter_hash(networks, frozen_indices),
                replaced_agent_parameter_sha256=parameter_hash(networks, replaced_indices(generation)),
                elapsed_seconds=time.perf_counter() - started,
            )
            stream.write(core.base.json_bytes(row).decode())
            if update in design.CHECKPOINTS:
                stream.flush()
                save_and_evaluate(update)
    final = source_runner.evaluate_factorized(networks, arrays["final"], static["final_spec"], live)
    result = dict(
        seed=seed, generation=generation, arm=arm, replaced_agent=design.replaced_agent(generation),
        schedule=schedule, condition=condition, live=bool(live), updates=design.UPDATES,
        parent_generation=parent_meta["generation"], parent_condition=parent_meta["condition"],
        parent_checkpoint_sha256=parent_meta["checkpoint_sha256"], parent_parameter_sha256=parent_hash,
        parent_replaced_agent_parameter_sha256=parent_replaced_hash,
        frozen_parent_parameter_sha256=frozen_parent_hash,
        new_agent_initial_parameter_sha256=fresh_hash, initial_parameter_sha256=initial_hash,
        final_parameter_sha256=parameter_hash(networks),
        final_checkpoint_sha256=sha(directory / "checkpoint_6000.npz"),
        training_log_sha256=sha(directory / "training.jsonl"), trajectory=trajectory,
        final={"new_layouts": final}, inherited_parent={"new_layouts": inherited_parent},
        rematch_histogram=rematch_counts.tolist(), rematch_assignments=int(rematch_counts.sum()),
        frozen_parameter_unchanged=True, no_external_model=True,
        elapsed_seconds=time.perf_counter() - started,
    )
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (directory / "status.json").write_text(json.dumps({
        "status": "completed", "seed": seed, "generation": generation, "condition": condition,
        "result_sha256": sha(directory / "result.json"), "at": now(),
    }, ensure_ascii=False, indent=2) + "\n")
    return result


def worker(payload):
    seed, static, execution = payload
    arrays = {part: make_arrays(static[f"{part}_spec"]) for part in ("train", "monitor", "final")}
    key_names = ("packed_states", "rewards", "x_PL")
    before = {part: {key: array_sha(arrays[part][key]) for key in key_names} for part in arrays}
    (Path(execution) / f"seed_{seed}_arrays.json").write_text(json.dumps({"array_hashes": before}, indent=2) + "\n")
    runs = []
    generation2_live = {}
    generation2_meta = {}

    # Run the complete generation-2 block first, preserving the canonical grid.
    for schedule in design.SCHEDULES:
        source_result, source_checkpoint = load_generation1_result(seed, schedule)
        source_networks = source_runner.load_networks(source_checkpoint)
        source_hash = parameter_hash(source_networks)
        require(source_hash == source_result["final_parameter_sha256"], "Generation-1 parameter hash mismatch")
        parent_meta = dict(generation=1, condition=f"{schedule}_new_receiver_PL_live",
                           checkpoint_sha256=sha(source_checkpoint))
        baselines = {
            bool(live): source_runner.evaluate_factorized(source_networks, arrays["final"], static["final_spec"], bool(live))
            for live in design.LIVES
        }
        for live in design.LIVES:
            result = train_run(seed, 2, schedule, bool(live), source_networks, arrays, static, execution,
                                parent_meta, baselines[bool(live)])
            runs.append(result)
            if live:
                generation2_live[schedule] = result

    # Generation 3 inherits only the live generation-2 endpoint.  Its silent
    # run is a paired counterfactual from that same endpoint.
    for schedule in design.SCHEDULES:
        parent_result = generation2_live[schedule]
        parent_checkpoint = Path(execution) / design.name(seed, parent_result["condition"]) / "checkpoint_6000.npz"
        require(sha(parent_checkpoint) == parent_result["final_checkpoint_sha256"], "Generation-2 live checkpoint mismatch")
        parent_networks = source_runner.load_networks(parent_checkpoint)
        require(parameter_hash(parent_networks) == parent_result["final_parameter_sha256"], "Generation-2 endpoint hash mismatch")
        generation2_meta[schedule] = dict(
            generation=2, condition=parent_result["condition"], checkpoint_sha256=sha(parent_checkpoint),
        )
        baselines = {
            bool(live): source_runner.evaluate_factorized(parent_networks, arrays["final"], static["final_spec"], bool(live))
            for live in design.LIVES
        }
        for live in design.LIVES:
            runs.append(train_run(seed, 3, schedule, bool(live), parent_networks, arrays, static, execution,
                                  generation2_meta[schedule], baselines[bool(live)]))
    after = {part: {key: array_sha(arrays[part][key]) for key in key_names} for part in arrays}
    require(before == after, "Evaluation arrays mutated")
    return runs


def verify_pairing(execution, runs):
    execution = Path(execution)
    expected = {(seed, condition) for seed in design.SEEDS for condition in design.CONDITIONS}
    by = {(run["seed"], run["condition"]): run for run in runs}
    require(set(by) == expected and len(by) == len(expected), "Incomplete generation grid")
    for seed in design.SEEDS:
        for generation in design.GENERATIONS:
            arm = "replace_A" if generation == 2 else "replace_B"
            for schedule in design.SCHEDULES:
                live = by[(seed, f"generation{generation}_{arm}_{schedule}_PL_live")]
                silent = by[(seed, f"generation{generation}_{arm}_{schedule}_PL_silent")]
                for key in ("parent_checkpoint_sha256", "parent_parameter_sha256", "new_agent_initial_parameter_sha256",
                            "frozen_parent_parameter_sha256"):
                    require(live[key] == silent[key], f"Live/silent parent pairing failed: {key}")
                require(live["frozen_parameter_unchanged"] and silent["frozen_parameter_unchanged"], "Frozen modules changed")
                expected_rematch = design.UPDATES * design.BATCH_SIZE if schedule == "rematched" else 0
                for run in (live, silent):
                    require(sum(run["rematch_histogram"]) == expected_rematch and run["rematch_assignments"] == expected_rematch,
                            "Rematch count mismatch")
                live_path = execution / design.name(seed, live["condition"]) / "training.jsonl"
                silent_path = execution / design.name(seed, silent["condition"]) / "training.jsonl"
                with live_path.open() as ls, silent_path.open() as ss:
                    count = 0
                    for left, right in zip(ls, ss):
                        a, b = json.loads(left), json.loads(right)
                        count += 1
                        require(a["update"] == b["update"] == count, "Paired update mismatch")
                        for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256",
                                    "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                            require(a[key] == b[key], "Live/silent paired stream differs: " + key)
                    require(count == design.UPDATES, "Training log count mismatch")
                if generation == 3:
                    parent = by[(seed, f"generation2_replace_A_{schedule}_PL_live")]
                    require(live["parent_checkpoint_sha256"] == parent["final_checkpoint_sha256"], "Generation chain link failed")
                    require(silent["parent_checkpoint_sha256"] == parent["final_checkpoint_sha256"], "Generation chain pairing failed")
                else:
                    source = next(item for item in json.loads((Path(execution).parent / "prepared.json").read_text())["generation1_runs"]
                                  if item["seed"] == seed and item["schedule"] == schedule)
                    require(live["parent_checkpoint_sha256"] == source["checkpoint_sha256"], "Generation-1 link failed")
    return True


def measured_budget(runs):
    expected = len(design.SEEDS) * len(design.GENERATIONS) * len(design.SCHEDULES) * 2
    require(len(runs) == expected, "Unexpected chain run count")
    trajectories = [row for run in runs for row in run["trajectory"]]
    finals = [value for run in runs for value in run["final"].values()]
    require(all([row["update"] for row in run["trajectory"]] == list(design.CHECKPOINTS) for run in runs), "Incomplete trajectory")
    return dict(
        training_forward_module_samples=len(runs) * design.UPDATES * design.BATCH_SIZE * 2 * 9,
        monitor_forward_module_samples=sum(row["forward_module_samples"] for row in trajectories),
        final_forward_module_samples=sum(9 * row["worlds"] for row in finals),
        parent_baseline_forward_module_samples=sum(9 * value["worlds"]
                                                  for run in runs
                                                  for value in run["inherited_parent"].values()),
        monitor_records=len(trajectories), final_records=len(finals),
        monitor_worlds=sum(row["compact_worlds"] for row in trajectories), final_worlds=sum(row["worlds"] for row in finals),
        rematched_assignments=sum(run["rematch_assignments"] for run in runs),
        optimizer_updates=len(runs) * design.UPDATES, frozen_optimizer_updates=0,
    )


def execute(out):
    out = Path(out).resolve()
    plan, static = verify(out)
    execution = out / "execution"
    require(not execution.exists(), "Never overwrite execution")
    execution.mkdir()
    started = time.perf_counter()
    (execution / "started.json").write_text(json.dumps({"started_at": now(), "pid": os.getpid(), "plan_sha256": sha(out / "plan.json")}, indent=2) + "\n")
    try:
        with multiprocessing.get_context("spawn").Pool(4) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in design.SEEDS])
        runs = [run for group in groups for run in group]
        require([(run["seed"], run["condition"]) for run in runs] ==
                [(seed, condition) for seed in design.SEEDS for condition in design.CONDITIONS], "Noncanonical run order")
        verify_pairing(execution, runs)
        measured = measured_budget(runs)
        result = dict(
            status="completed", completed_at=now(), elapsed_seconds=time.perf_counter() - started,
            plan_sha256=sha(out / "plan.json"), budget=static["budget"], measured_budget=measured, runs=runs,
            primary="generation-wise live-minus-silent adaptation and parent-to-child retention",
            experiment_type="two_generation_frozen_partner_protocol_chain",
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
    args = parser.parse_args()
    answer = prepare(args.out) if args.command == "prepare" else execute(args.out) if args.command == "execute" else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
