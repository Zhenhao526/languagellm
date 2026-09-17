"""Run action-only and sender-only new-receiver adaptation arms."""
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
    kernel as source_kernel,
    remap as source_remap,
    runner as source_runner,
)
from research_program.triadic_message_study import runner as core
from research_program.triadic_new_receiver_transmission_study import runner as transmission_runner

from . import design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONFIG = dict(
    updates=design.UPDATES, batch_size=design.BATCH_SIZE, checkpoints=list(design.CHECKPOINTS),
    features=54, dtype="float64", learning_rate=0.001, global_gradient_clip=5.0,
    entropy_initial=0.001, entropy_zero_after_updates=1000, sender_entropy_coefficient=0,
    trajectories_per_state=2, sender_windows=2, sender_tokens_per_window=4,
    alphabet_size=8, proposal_count=16, intent_count=2, actions_per_actor=17, action_logits=18,
    evaluation_batch_size=8192, dimensions={k: list(v) for k, v in source_runner.DIMS.items()},
    seeds=list(design.SEEDS), schedules=list(design.SCHEDULES), arms=list(design.ARMS),
    lives=["live", "silent"], conditions=list(design.CONDITIONS),
    task="new_receiver_module_selective_adaptation",
    source_observation="PL_own_need_and_public_layout_only",
    source_policy="factorized neutral/engage source checkpoint",
    arm_modules={arm: design.allowed_modules(arm) for arm in design.ARMS},
    frozen_modules="all A/B modules and non-listed C modules",
    optimizer="Adam; exact zero gradients for frozen modules",
    pairing="same source checkpoint, fresh C initialization, world uniforms, batch indices, packet uniforms and rematching assignments across arms/channels",
    no_model_calls=True, no_external_model=True, no_llm=True, no_vision_model=True,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return core.base.sha(path)


def json_hash(value):
    return core.base.json_hash(value)


def array_sha(value):
    return core.base.array_sha(value)


def own_parameter_hash(networks, indices=None):
    if indices is None:
        indices = range(len(networks))
    payload = {}
    for i in indices:
        for key, value in networks[i].items():
            payload[f"network_{i}_{key}"] = array_sha(value)
    return json_hash(payload)


def clone_networks(networks):
    return [{key: value.copy() for key, value in network.items()} for network in networks]


def fresh_c_networks(seed, schedule_index):
    return transmission_runner.fresh_c_networks(seed, schedule_index)


def source_run_path(seed, schedule):
    return transmission_runner.source_run_path(seed, schedule)


def load_source_result(seed, schedule):
    return transmission_runner.load_source_result(seed, schedule)


def load_full_reference():
    path = design.FULL_RESULT
    require(path.is_file() and design.FULL_AUDIT.is_file(), "Full-C transmission reference missing")
    full = json.loads(path.read_text())
    audit = json.loads(design.FULL_AUDIT.read_text())
    require(audit.get("status") == "passed" and len(full.get("runs", [])) == 32, "Full-C reference is not audited")
    return full


def source_artifacts():
    result = transmission_runner.source_artifacts()
    for path in (design.TRANSMISSION_ROOT / "plan.json", design.TRANSMISSION_ROOT / "prepared.json",
                 design.FULL_RESULT, design.FULL_AUDIT):
        require(path.is_file(), "Missing full-C reference artifact")
        result[str(path.resolve().relative_to(ROOT))] = sha(path)
    return result


def source_code_paths():
    paths = [HERE / name for name in (
        "__init__.py", "design.py", "runner.py", "plan.md", "README.md",
        "tests/test_design.py", "tests/test_runner.py",
    )]
    paths += transmission_runner.source_code_paths()
    require(all(path.is_file() for path in paths), "Missing source code for freeze")
    unique = []
    seen = set()
    for path in paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    return unique


def source_code_hashes():
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in source_code_paths()}


def build_prepared():
    full_prepared = json.loads((design.TRANSMISSION_ROOT / "prepared.json").read_text())
    static = design.prepared(full_prepared)
    static["full_reference_prepared_sha256"] = sha(design.TRANSMISSION_ROOT / "prepared.json")
    static["full_reference_result_sha256"] = sha(design.FULL_RESULT)
    static["full_reference_audit_sha256"] = sha(design.FULL_AUDIT)
    static["source_code_sha256"] = source_code_hashes()
    static["source_artifacts_sha256"] = source_artifacts()
    static["runtime"] = dict(python=platform.python_version(), numpy=np.__version__)
    static["budget"] = budget(static)
    return static


def budget(static):
    runs = len(design.SEEDS) * len(design.SCHEDULES) * len(design.ARMS) * 2
    monitor_worlds = static["monitor_spec"]["world_count"]
    final_worlds = static["final_spec"]["world_count"]
    monitor_records = runs * len(design.CHECKPOINTS)
    final_records = runs
    train_forward = runs * design.UPDATES * design.BATCH_SIZE * 2 * 9
    monitor_forward = monitor_records * monitor_worlds * 9
    final_forward = final_records * final_worlds * 9
    return dict(
        runs=runs, seed_blocks=len(design.SEEDS), schedules=len(design.SCHEDULES), arms=len(design.ARMS), channels=2,
        training_updates=runs * design.UPDATES, training_world_samples=runs * design.UPDATES * design.BATCH_SIZE,
        training_forward_module_samples=train_forward, monitor_records=monitor_records,
        monitor_worlds=monitor_records * monitor_worlds, monitor_forward_module_samples=monitor_forward,
        final_records=final_records, final_worlds=final_records * final_worlds,
        final_forward_module_samples=final_forward, total_forward_module_samples=train_forward + monitor_forward + final_forward,
        full_reference_runs=32,
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
        status="prepared_without_training", created_at=core.base.now(), config=CONFIG,
        source_code_sha256=static["source_code_sha256"], source_artifacts_sha256=static["source_artifacts_sha256"],
        full_reference_prepared_sha256=static["full_reference_prepared_sha256"],
        full_reference_result_sha256=static["full_reference_result_sha256"],
        full_reference_audit_sha256=static["full_reference_audit_sha256"],
        prepared_sha256=sha(out / "prepared.json"), runtime=static["runtime"],
        no_model_calls=True, no_training_updates=True,
        scientific_question="Which C modules are needed for a new receiver to benefit from and recover a frozen pair's task protocol?",
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
    require(static == build_prepared(), "Current source/config differs from frozen preparation")
    require(plan["config"] == CONFIG and plan["runtime"] == static["runtime"], "Configuration/runtime changed")
    for relative, digest in plan["source_code_sha256"].items():
        require(sha(out / "source_snapshot" / relative) == digest, "Code snapshot changed: " + relative)
    for relative, digest in plan["source_artifacts_sha256"].items():
        require(sha(ROOT / relative) == digest, "Source artifact changed: " + relative)
    return plan, static


def make_arrays(spec):
    return transmission_runner.make_arrays(spec)


def trainable_indices(arm):
    return {"action_only": (8,), "sender_only": (6, 7)}[arm]


def apply_gradients(networks, gradients, arm):
    allowed = set(trainable_indices(arm))
    return [
        ({key: value.copy() for key, value in gradients[i].items()} if i in allowed
         else {key: np.zeros_like(value) for key, value in networks[i].items()})
        for i in range(9)
    ]


def train_run(seed, schedule, arm, live, source_networks, arrays, static, execution, baseline):
    condition = f"{schedule}_new_receiver_{arm}_PL_{'live' if live else 'silent'}"
    directory = Path(execution) / design.name(seed, condition)
    directory.mkdir(exist_ok=False)
    schedule_index = design.SCHEDULES.index(schedule)
    networks = clone_networks(source_networks)
    source_ab_hash = own_parameter_hash(networks, range(6))
    source_c_hash = own_parameter_hash(networks, range(6, 9))
    new_c = fresh_c_networks(seed, schedule_index)
    new_c_hash = own_parameter_hash(new_c, range(3))
    require(new_c_hash != source_c_hash, "Fresh C equals source C")
    networks[6:9] = new_c
    initial_hash = own_parameter_hash(networks)
    optimizer = core.base.make_adam(networks)
    # Match the full-C reference streams exactly so module contrasts are
    # paired with the already completed transmission run.
    world_rng = np.random.default_rng(np.random.SeedSequence([seed, 2100, schedule_index]))
    message_rngs = core.make_message_rngs(int(seed + 910000 + 1000 * schedule_index))
    rematch_rng = np.random.default_rng(np.random.SeedSequence([seed, 2700, schedule_index]))
    trajectory = []
    rematch_counts = np.zeros(6, dtype=np.int64)
    started = time.perf_counter()

    def checkpoint(step):
        path = directory / f"checkpoint_{step:04d}.npz"
        digest = source_runner.save_checkpoint(path, networks, optimizer, step, world_rng, message_rngs, rematch_rng)
        metrics = source_runner.evaluate_factorized(networks, arrays["monitor"], static["monitor_spec"], live)
        metrics["update"] = step
        row = dict(
            update=step, checkpoint_path=str(path), checkpoint_sha256=digest,
            target_trajectory=metrics, compact_worlds=metrics["worlds"], forward_module_samples=9 * metrics["worlds"],
            parameter_sha256=own_parameter_hash(networks), frozen_ab_parameter_sha256=own_parameter_hash(networks, range(6)),
            new_agent_parameter_sha256=own_parameter_hash(networks, range(6, 9)), elapsed_seconds=time.perf_counter() - started,
        )
        trajectory.append(row)
        with (directory / "trajectory.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(core.base.json_bytes(row).decode())

    checkpoint(0)
    with (directory / "training.jsonl").open("w", encoding="utf-8") as stream:
        for update in range(1, design.UPDATES + 1):
            world_uniforms = world_rng.random((design.BATCH_SIZE, 3))
            ids = transmission_runner.source_design.sample_indices(static["train_spec"], world_uniforms)
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
            norm, scale = core.base.adam_step(networks, apply_gradients(networks, gradients, arm), optimizer, update)
            require(own_parameter_hash(networks, range(6)) == source_ab_hash, "Frozen A/B changed")
            row.update(
                update=update, seed=seed, schedule=schedule, arm=arm, live=bool(live), condition=condition,
                world_uniforms_sha256=array_sha(world_uniforms), batch_indices_sha256=array_sha(ids),
                canonical_batch_states_sha256=array_sha(canonical_states), effective_batch_states_sha256=array_sha(effective_states),
                sample_uniforms_sha256=array_sha(message_uniforms), permutation_indices_sha256=array_sha(permutation_ids),
                rematched=schedule == "rematched", gradient_norm=norm, gradient_clip_scale=scale,
                frozen_ab_parameter_sha256=own_parameter_hash(networks, range(6)),
                new_agent_parameter_sha256=own_parameter_hash(networks, range(6, 9)),
                trainable_modules=design.allowed_modules(arm), elapsed_seconds=time.perf_counter() - started,
            )
            stream.write(core.base.json_bytes(row).decode())
            if update in design.CHECKPOINTS:
                stream.flush()
                checkpoint(update)
    final = source_runner.evaluate_factorized(networks, arrays["final"], static["final_spec"], live)
    result = dict(
        seed=seed, schedule=schedule, arm=arm, condition=condition, live=bool(live), updates=design.UPDATES,
        source_checkpoint_sha256=sha(source_run_path(seed, schedule) / "checkpoint_6000.npz"),
        source_c_parameter_sha256=source_c_hash, frozen_ab_parameter_sha256=source_ab_hash,
        new_agent_initial_parameter_sha256=new_c_hash, initial_parameter_sha256=initial_hash,
        final_parameter_sha256=own_parameter_hash(networks),
        final_checkpoint_sha256=sha(directory / "checkpoint_6000.npz"), training_log_sha256=sha(directory / "training.jsonl"),
        trajectory=trajectory, final={"new_layouts": final}, inherited_source={"new_layouts": baseline},
        rematch_histogram=rematch_counts.tolist(), rematch_assignments=int(rematch_counts.sum()),
        trainable_modules=design.allowed_modules(arm), frozen_ab_parameter_unchanged=True,
        no_external_model=True, elapsed_seconds=time.perf_counter() - started,
    )
    (directory / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (directory / "status.json").write_text(json.dumps({"status": "completed", "seed": seed, "condition": condition, "result_sha256": sha(directory / "result.json"), "at": core.base.now()}, ensure_ascii=False, indent=2) + "\n")
    return result


def worker(payload):
    seed, static, execution = payload
    arrays = {part: make_arrays(static[f"{part}_spec"]) for part in ("train", "monitor", "final")}
    keys = ("packed_states", "rewards", "x_PL")
    hashes = {part: {key: array_sha(arrays[part][key]) for key in keys} for part in arrays}
    (Path(execution) / f"seed_{seed}_arrays.json").write_text(json.dumps({"array_hashes": hashes}, indent=2) + "\n")
    runs = []
    for schedule in design.SCHEDULES:
        _, checkpoint_path = load_source_result(seed, schedule)
        source_networks = source_runner.load_networks(checkpoint_path)
        baselines = {
            bool(live): source_runner.evaluate_factorized(source_networks, arrays["final"], static["final_spec"], bool(live))
            for live in design.LIVES
        }
        for arm in design.ARMS:
            for live in design.LIVES:
                runs.append(train_run(seed, schedule, arm, bool(live), source_networks, arrays, static, execution, baselines[bool(live)]))
    after = {part: {key: array_sha(arrays[part][key]) for key in keys} for part in arrays}
    require(hashes == after, "Arrays mutated")
    return runs


def verify_pairing(execution, runs):
    execution = Path(execution)
    expected = {(seed, condition) for seed in design.SEEDS for condition in design.CONDITIONS}
    by = {(r["seed"], r["condition"]): r for r in runs}
    require(set(by) == expected and len(by) == len(expected), "Incomplete run grid")
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            for arm in design.ARMS:
                live = by[(seed, f"{schedule}_new_receiver_{arm}_PL_live")]
                silent = by[(seed, f"{schedule}_new_receiver_{arm}_PL_silent")]
                require(live["new_agent_initial_parameter_sha256"] == silent["new_agent_initial_parameter_sha256"], "C init not paired")
                require(live["frozen_ab_parameter_sha256"] == silent["frozen_ab_parameter_sha256"], "A/B not paired")
                expected_rematch = design.UPDATES * design.BATCH_SIZE if schedule == "rematched" else 0
                for row in (live, silent):
                    require(sum(row["rematch_histogram"]) == expected_rematch and row["rematch_assignments"] == expected_rematch, "Rematch count mismatch")
                paths = [execution / design.name(seed, live["condition"]) / "training.jsonl",
                         execution / design.name(seed, silent["condition"]) / "training.jsonl"]
                with paths[0].open() as left, paths[1].open() as right:
                    n = 0
                    for line_left, line_right in zip(left, right):
                        a, b = json.loads(line_left), json.loads(line_right)
                        n += 1
                        require(a["update"] == b["update"] == n, "Paired update mismatch")
                        for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256",
                                    "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                            require(a[key] == b[key], "Paired stream differs: " + key)
                    require(n == design.UPDATES, "Training log count mismatch")
            # Cross-arm streams are also fixed, so arm contrasts share the same worlds.
            for live in design.LIVES:
                rows = []
                for arm in design.ARMS:
                    path = execution / design.name(seed, f"{schedule}_new_receiver_{arm}_PL_{'live' if live else 'silent'}") / "training.jsonl"
                    rows.append(path.open())
                try:
                    for lines in zip(*rows):
                        decoded = [json.loads(line) for line in lines]
                        for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256",
                                    "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                            require(len({row[key] for row in decoded}) == 1, "Cross-arm stream differs: " + key)
                finally:
                    for stream in rows:
                        stream.close()
    return True


def measured_budget(runs):
    require(len(runs) == 64, "Expected 64 selective runs")
    trajectory = [row for run in runs for row in run["trajectory"]]
    finals = [x for run in runs for x in run["final"].values()]
    return dict(
        training_forward_module_samples=len(runs) * design.UPDATES * design.BATCH_SIZE * 2 * 9,
        monitor_forward_module_samples=sum(row["forward_module_samples"] for row in trajectory),
        final_forward_module_samples=sum(9 * row["worlds"] for row in finals),
        monitor_records=len(trajectory), final_records=len(finals),
        optimizer_updates=len(runs) * design.UPDATES,
        frozen_agent_optimizer_updates=0,
    )


def execute(out):
    out = Path(out).resolve()
    plan, static = verify(out)
    execution = out / "execution"
    require(not execution.exists(), "Never overwrite execution")
    execution.mkdir()
    started = time.perf_counter()
    (execution / "started.json").write_text(json.dumps({"started_at": core.base.now(), "pid": os.getpid(), "plan_sha256": sha(out / "plan.json")}, indent=2) + "\n")
    try:
        with multiprocessing.get_context("spawn").Pool(4) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in design.SEEDS])
        runs = [run for group in groups for run in group]
        require([(r["seed"], r["condition"]) for r in runs] == [(seed, condition) for seed in design.SEEDS for condition in design.CONDITIONS], "Noncanonical run order")
        verify_pairing(execution, runs)
        measured = measured_budget(runs)
        result = dict(status="completed", completed_at=core.base.now(), elapsed_seconds=time.perf_counter() - started,
                      plan_sha256=sha(out / "plan.json"), budget=static["budget"], measured_budget=measured, runs=runs,
                      primary="module_selective_live_minus_silent_transmission", experiment_type="new_receiver_module_ablation",
                      full_reference_result_sha256=static["full_reference_result_sha256"],
                      language_claim_automatically_supported=False, no_external_model=True, no_llm=True, no_vision_model=True)
        (execution / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        (execution / "status.json").write_text(json.dumps({"status": "completed", "at": core.base.now(), "results_sha256": sha(execution / "results.json")}, indent=2) + "\n")
        return dict(status="completed", output=str(execution), elapsed_seconds=result["elapsed_seconds"], measured_budget=measured)
    except BaseException as error:
        (execution / "failure.json").write_text(json.dumps({"status": "failed", "at": core.base.now(), "error": repr(error), "elapsed_seconds": time.perf_counter() - started}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    answer = prepare(args.out) if args.command == "prepare" else execute(args.out) if args.command == "execute" else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
