"""Independent replay audit for the joint-combination holdout experiment.

The execution process writes metrics while training.  This module never trusts
those metrics: it reloads every saved checkpoint, regenerates the evaluation
arrays, recomputes both seen and heldout partitions, and checks the paired
training streams and the frozen all-needs controls.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from itertools import zip_longest
from pathlib import Path

import numpy as np

from . import design, runner
from research_program.triadic_factorized_neutral_altpartner_study import runner as source_runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


METRIC_FLOATS = (
    "value", "q_rate", "conditional_q_rate", "target_pair_legal_rate",
    "proposal_legal_rate", "physical_execution_rate", "engagement_rate",
    "neutral_rate", "third_agent_neutral_rate", "exact_expected_reward_mean",
    "exact_full_success_probability_mean", "exact_partial_success_probability_mean",
)
METRIC_INTS = (
    "worlds", "physical_worlds", "proposal_legal_denominator",
    "conditional_q_denominator_worlds", "conditional_q_numerator_worlds",
    "target_pair_denominator_worlds", "target_pair_numerator_worlds",
    "legal_plan_count", "legal_pair_count",
)
METRIC_VECTORS = ("actor_engagement_rates", "legal_plan_selection_counts", "legal_pair_selection_counts")


def compare_metric(actual, saved):
    """Return the largest numeric discrepancy between two evaluator outputs."""
    require(actual["partition"] == saved["partition"], "Metric partition mismatch")
    require(bool(actual["live"]) == bool(saved["live"]), "Metric channel mismatch")
    error = 0.0
    for key in METRIC_FLOATS:
        error = max(error, abs(float(actual[key]) - float(saved[key])))
    for key in METRIC_INTS:
        require(int(actual[key]) == int(saved[key]), "Metric integer mismatch: " + key)
    for key in METRIC_VECTORS:
        require(len(actual[key]) == len(saved[key]), "Metric vector length mismatch: " + key)
        for x, y in zip(actual[key], saved[key]):
            error = max(error, abs(float(x) - float(y)))
    require(error <= 1e-12, f"Metric replay discrepancy {error}")
    return error


def fresh_networks(seed, schedule):
    schedule_index = design.SCHEDULES.index(schedule)
    return [
        source_runner.make_network(
            np.random.SeedSequence([seed, 9901, 2, schedule_index, 0, module_index]),
            runner.DIMS[module],
        )
        for module_index, module in enumerate(runner.MODULES)
    ]


def metric_path(row, key):
    require(key in row["target_trajectory"], "Missing target trajectory: " + key)
    return row["target_trajectory"][key]


def verify_arrays(execution, static):
    """Regenerate deterministic arrays and check every worker's array receipt."""
    expected = {}
    for part in ("train", "monitor", "heldout", "seen_new"):
        arrays = runner.make_arrays(static[f"{part}_spec"])
        expected[part] = {key: runner.array_sha(arrays[key]) for key in ("packed_states", "rewards", "x_PL")}
    for seed in design.SEEDS:
        receipt = execution / f"seed_{seed}_arrays.json"
        require(receipt.is_file(), "Missing array receipt: " + receipt.name)
        saved = json.loads(receipt.read_text()).get("array_hashes")
        require(saved == expected, "Array receipt mismatch: " + receipt.name)
    return expected


def check_trajectory_metric(run, trajectory, arrays, static, max_error):
    update = int(trajectory["update"])
    require(update in design.CHECKPOINTS, "Unexpected checkpoint update")
    require(trajectory["target_trajectory"]["monitor_seen"]["update"] == update,
            "Monitor update marker mismatch")
    require(trajectory["target_trajectory"]["heldout"]["update"] == update,
            "Heldout update marker mismatch")
    monitor = source_runner.evaluate_factorized(
        arrays["networks"], arrays["monitor"], static["monitor_spec"], bool(run["live"])
    )
    heldout = source_runner.evaluate_factorized(
        arrays["networks"], arrays["heldout"], static["heldout_spec"], bool(run["live"])
    )
    max_error = max(max_error, compare_metric(monitor, metric_path(trajectory, "monitor_seen")))
    max_error = max(max_error, compare_metric(heldout, metric_path(trajectory, "heldout")))
    require(trajectory["compact_worlds"] == monitor["worlds"] + heldout["worlds"], "Trajectory world count mismatch")
    require(trajectory["forward_module_samples"] == 9 * trajectory["compact_worlds"],
            "Trajectory forward count mismatch")
    return max_error, monitor["worlds"] + heldout["worlds"]


def check_final_seen(run, networks, arrays, static, max_error):
    heldout = source_runner.evaluate_factorized(networks, arrays["heldout"], static["heldout_spec"], bool(run["live"]))
    seen_new = source_runner.evaluate_factorized(networks, arrays["seen_new"], static["seen_new_spec"], bool(run["live"]))
    max_error = max(max_error, compare_metric(heldout, run["final"]["heldout"]))
    max_error = max(max_error, compare_metric(seen_new, run["final"]["seen_new"]))
    return max_error, heldout["worlds"] + seen_new["worlds"]


def check_log_matches_result(path, trajectory):
    with path.open(encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream]
    require(len(lines) == design.UPDATES, "Training log length mismatch")
    require(all(int(row["update"]) == i for i, row in enumerate(lines, 1)), "Training update sequence mismatch")
    return lines


def audit_seen_run(source, execution, static, row, arrays_by_part, source_result, source_checkpoint):
    seed, schedule, live = int(row["seed"]), row["schedule"], bool(row["live"])
    condition = f"{schedule}_new_receiver_compositional_{design.ARM}_PL_{'live' if live else 'silent'}"
    require(row["condition"] == condition and row["arm"] == design.ARM, "Seen run identity mismatch")
    require(row["updates"] == design.UPDATES and row["no_external_model"] is True, "Seen run config mismatch")
    source_networks = source_runner.load_networks(source_checkpoint)
    source_full_hash = runner.parameter_hash(source_networks)
    source_frozen_hash = runner.parameter_hash(source_networks, range(3, 9))
    require(row["parent_checkpoint_sha256"] == runner.sha(source_checkpoint), "Seen parent checkpoint mismatch")
    require(row["parent_parameter_sha256"] == source_full_hash == source_result["final_parameter_sha256"],
            "Seen parent parameter mismatch")
    require(row["frozen_parent_parameter_sha256"] == source_frozen_hash, "Seen frozen parent mismatch")
    require(row["parent_result_sha256"] == runner.sha(source_checkpoint.parent / "result.json"), "Seen parent result mismatch")

    fresh = fresh_networks(seed, schedule)
    fresh_hash = runner.parameter_hash(fresh)
    require(row["new_agent_initial_parameter_sha256"] == fresh_hash, "Fresh initialization hash mismatch")
    initial = runner.clone_networks(source_networks)
    initial[0:3] = runner.clone_networks(fresh[0:3])
    require(row["initial_parameter_sha256"] == runner.parameter_hash(initial), "Initial parameter hash mismatch")

    directory = execution / design.name(seed, condition)
    require(directory.is_dir(), "Missing seen run directory")
    trajectories = sorted(row["trajectory"], key=lambda value: int(value["update"]))
    require([int(value["update"]) for value in trajectories] == list(design.CHECKPOINTS), "Seen trajectory grid mismatch")
    max_error = 0.0
    trajectory_worlds = 0
    frozen_checks = 0
    for trajectory in trajectories:
        update = int(trajectory["update"])
        checkpoint = directory / f"checkpoint_{update:04d}.npz"
        require(checkpoint.is_file() and runner.sha(checkpoint) == trajectory["checkpoint_sha256"],
                "Seen checkpoint hash mismatch")
        networks = source_runner.load_networks(checkpoint)
        require(runner.parameter_hash(networks) == trajectory["parameter_sha256"], "Seen checkpoint parameter mismatch")
        require(runner.parameter_hash(networks, range(3, 9)) == trajectory["frozen_parameter_sha256"] == source_frozen_hash,
                "Frozen B/C changed")
        require(trajectory["replaced_agent_parameter_sha256"] == runner.parameter_hash(networks, range(3)),
                "Fresh A checkpoint hash mismatch")
        if update == 0:
            require(trajectory["parameter_sha256"] == row["initial_parameter_sha256"], "Initial checkpoint mismatch")
            require(runner.parameter_hash(networks, range(3)) == runner.parameter_hash(fresh, range(3)),
                    "Fresh A weights differ at update zero")
        max_error, worlds = check_trajectory_metric(row, dict(trajectory, _unused=True),
                                                    {**arrays_by_part, "networks": networks}, static, max_error)
        trajectory_worlds += worlds
        frozen_checks += 1

    final_checkpoint = directory / "checkpoint_6000.npz"
    require(final_checkpoint.is_file() and runner.sha(final_checkpoint) == row["final_checkpoint_sha256"],
            "Seen final checkpoint hash mismatch")
    networks = source_runner.load_networks(final_checkpoint)
    require(runner.parameter_hash(networks) == row["final_parameter_sha256"], "Seen final parameter mismatch")
    max_error, final_worlds = check_final_seen(row, networks, arrays_by_part, static, max_error)
    log_path = directory / "training.jsonl"
    require(log_path.is_file() and runner.sha(log_path) == row["training_log_sha256"], "Seen training log hash mismatch")
    log_rows = check_log_matches_result(log_path, row["trajectory"])
    expected_assignments = design.UPDATES * design.BATCH_SIZE if schedule == "rematched" else 0
    require(row["rematch_assignments"] == expected_assignments, "Seen rematch count mismatch")
    if schedule == "static":
        require(row["rematch_histogram"] == [0] * 6, "Static run has rematches")
    require(row["frozen_parameter_unchanged"] is True, "Frozen invariant missing")
    return dict(max_error=max_error, trajectory_worlds=trajectory_worlds, final_worlds=final_worlds,
                frozen_checks=frozen_checks, log_rows=len(log_rows))


def audit_control_run(execution, static, row, arrays_by_part, max_error):
    seed, schedule, live = int(row["seed"]), row["schedule"], bool(row["live"])
    source_result, source_checkpoint = runner.load_chain_control(seed, schedule, live)
    directory = runner.chain_control_path(seed, schedule, live)
    require(row["arm"] == "all_needs_control" and row["condition"] == source_result["condition"],
            "Control identity mismatch")
    require(row["no_new_training"] is True, "Control is marked as trained")
    require(row["source_result_sha256"] == runner.sha(directory / "result.json"), "Control result hash mismatch")
    require(row["final_checkpoint_sha256"] == runner.sha(source_checkpoint), "Control checkpoint hash mismatch")
    require(row["final_parameter_sha256"] == source_result["final_parameter_sha256"], "Control final parameter mismatch")
    require(row["parent_checkpoint_sha256"] == source_result["parent_checkpoint_sha256"], "Control parent mismatch")
    require(row["training_log_sha256"] == source_result["training_log_sha256"], "Control log source mismatch")
    trajectories = sorted(row["trajectory"], key=lambda value: int(value["update"]))
    require([int(value["update"]) for value in trajectories] == list(design.CHECKPOINTS), "Control trajectory grid mismatch")
    trajectory_worlds = 0
    for trajectory in trajectories:
        update = int(trajectory["update"])
        checkpoint = directory / f"checkpoint_{update:04d}.npz"
        require(checkpoint.is_file() and runner.sha(checkpoint) == trajectory["checkpoint_sha256"],
                "Control checkpoint hash mismatch")
        networks = source_runner.load_networks(checkpoint)
        require(runner.parameter_hash(networks) == trajectory["parameter_sha256"], "Control parameter hash mismatch")
        monitor = source_runner.evaluate_factorized(networks, arrays_by_part["monitor"], static["monitor_spec"], live)
        heldout = source_runner.evaluate_factorized(networks, arrays_by_part["heldout"], static["heldout_spec"], live)
        require(trajectory["target_trajectory"]["monitor_seen"]["update"] == update, "Control monitor update mismatch")
        require(trajectory["target_trajectory"]["heldout"]["update"] == update, "Control heldout update mismatch")
        max_error = max(max_error, compare_metric(monitor, trajectory["target_trajectory"]["monitor_seen"]))
        max_error = max(max_error, compare_metric(heldout, trajectory["target_trajectory"]["heldout"]))
        require(trajectory["compact_worlds"] == monitor["worlds"] + heldout["worlds"], "Control world count mismatch")
        require(trajectory["forward_module_samples"] == 9 * trajectory["compact_worlds"], "Control forward count mismatch")
        trajectory_worlds += monitor["worlds"] + heldout["worlds"]
    final_checkpoint = directory / "checkpoint_6000.npz"
    networks = source_runner.load_networks(final_checkpoint)
    heldout = source_runner.evaluate_factorized(networks, arrays_by_part["heldout"], static["heldout_spec"], live)
    seen_new = source_runner.evaluate_factorized(networks, arrays_by_part["seen_new"], static["seen_new_spec"], live)
    max_error = max(max_error, compare_metric(heldout, row["final"]["heldout"]))
    max_error = max(max_error, compare_metric(seen_new, row["final"]["seen_new"]))
    return dict(max_error=max_error, trajectory_worlds=trajectory_worlds,
                final_worlds=heldout["worlds"] + seen_new["worlds"])


def audit_pairing(execution):
    paired_rows = 0
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            left_path = execution / design.name(seed, f"{schedule}_new_receiver_compositional_{design.ARM}_PL_live") / "training.jsonl"
            right_path = execution / design.name(seed, f"{schedule}_new_receiver_compositional_{design.ARM}_PL_silent") / "training.jsonl"
            with left_path.open(encoding="utf-8") as left, right_path.open(encoding="utf-8") as right:
                count = 0
                for a_line, b_line in zip_longest(left, right):
                    require(a_line is not None and b_line is not None, "Paired logs have unequal length")
                    a, b = json.loads(a_line), json.loads(b_line)
                    count += 1
                    require(a["update"] == b["update"] == count, "Paired update mismatch")
                    require(a["seed"] == b["seed"] == seed and a["schedule"] == b["schedule"] == schedule,
                            "Paired log identity mismatch")
                    for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256",
                                "canonical_batch_states_sha256", "effective_batch_states_sha256",
                                "permutation_indices_sha256"):
                        require(a[key] == b[key], "Paired stream differs: " + key)
                require(count == design.UPDATES, "Paired log row count mismatch")
                paired_rows += count
    return paired_rows


def main(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    plan, static = runner.verify(source)
    execution = source / "execution"
    result_path = execution / "results.json"
    status_path = execution / "status.json"
    require(result_path.is_file() and status_path.is_file(), "Execution result/status missing")
    result = json.loads(result_path.read_text())
    status = json.loads(status_path.read_text())
    require(status.get("status") == "completed" and status.get("results_sha256") == runner.sha(result_path),
            "Execution status/hash mismatch")
    expected = len(design.SEEDS) * len(design.SCHEDULES) * 2
    seen_runs, control_runs = result.get("runs", []), result.get("all_needs_control_runs", [])
    require(len(seen_runs) == expected and len(control_runs) == expected, "Incomplete seen/control grid")
    seen_by = {(r["seed"], r["schedule"], bool(r["live"])): r for r in seen_runs}
    control_by = {(r["seed"], r["schedule"], bool(r["live"])): r for r in control_runs}
    require(len(seen_by) == expected and len(control_by) == expected, "Duplicate seen/control identity")
    expected_keys = {(seed, schedule, bool(live)) for seed in design.SEEDS for schedule in design.SCHEDULES for live in design.LIVES}
    require(set(seen_by) == expected_keys and set(control_by) == expected_keys, "Run grid differs from frozen design")
    require(result["plan_sha256"] == runner.sha(source / "plan.json"), "Execution plan hash mismatch")
    require(result["measured_budget"] == runner.measured_budget(seen_runs, control_runs), "Measured budget mismatch")

    split = design.split_needs()
    require(static["split"] == split, "Frozen joint holdout split changed")
    arrays = {part: runner.make_arrays(static[f"{part}_spec"]) for part in ("train", "monitor", "heldout", "seen_new")}
    verify_arrays(execution, static)

    max_error = 0.0
    seen_trajectory_worlds = seen_final_worlds = control_trajectory_worlds = control_final_worlds = 0
    seen_frozen_checks = 0
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            source_result, source_checkpoint = runner.load_generation1(seed, schedule)
            for live in design.LIVES:
                seen_row = seen_by[seed, schedule, bool(live)]
                control_row = control_by[seed, schedule, bool(live)]
                seen_dir = execution / design.name(seed, seen_row["condition"])
                saved_seen = json.loads((seen_dir / "result.json").read_text())
                require(saved_seen == seen_row, "Per-run seen result differs from aggregate")
                report = audit_seen_run(source, execution, static, seen_row, arrays, source_result, source_checkpoint)
                max_error = max(max_error, report["max_error"])
                seen_trajectory_worlds += report["trajectory_worlds"]
                seen_final_worlds += report["final_worlds"]
                seen_frozen_checks += report["frozen_checks"]
                report = audit_control_run(execution, static, control_row, arrays, max_error)
                max_error = max(max_error, report["max_error"])
                control_trajectory_worlds += report["trajectory_worlds"]
                control_final_worlds += report["final_worlds"]

    paired_rows = audit_pairing(execution)
    verification = dict(
        status="passed", source=str(source), plan_sha256=runner.sha(source / "plan.json"),
        prepared_sha256=runner.sha(source / "prepared.json"),
        seen_runs=len(seen_runs), all_needs_control_runs=len(control_runs),
        checkpoint_evaluations=expected * len(design.CHECKPOINTS) * 2,
        final_evaluations=expected * 2,
        seen_trajectory_worlds=seen_trajectory_worlds, seen_final_worlds=seen_final_worlds,
        control_trajectory_worlds=control_trajectory_worlds, control_final_worlds=control_final_worlds,
        model_forwards=9 * (seen_trajectory_worlds + seen_final_worlds + control_trajectory_worlds + control_final_worlds),
        adaptation_optimizer_updates=len(seen_runs) * design.UPDATES,
        frozen_agent_optimizer_updates=0, frozen_seen_checkpoints=seen_frozen_checks,
        paired_training_log_rows=paired_rows, max_abs_error=float(max_error),
        checkpoint_replayed=True, final_replayed=True, control_replayed=True,
        joint_holdout_split_verified=True, marginal_need_coverage_verified=True,
        paired_streams_verified=True, all_needs_controls_verified=True,
        no_external_model=True, no_llm=True, no_vision_model=True,
    )
    path = output / "verification.json"
    path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n")
    receipt = dict(status="passed", verification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   model_forwards=verification["model_forwards"], optimizer_updates=verification["adaptation_optimizer_updates"],
                   frozen_agent_optimizer_updates=0)
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main(args.source, args.output)
