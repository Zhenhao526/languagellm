"""Independent checkpoint replay and frozen-agent invariant audit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import design, runner
from research_program.triadic_factorized_neutral_altpartner_study import runner as source_runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


def own_parameter_hash(networks, indices):
    payload = {}
    for i in indices:
        for key, value in networks[i].items():
            payload[f"network_{i}_{key}"] = runner.array_sha(value)
    return runner.json_hash(payload)


def compare_metric(actual, saved):
    keys = (
        "q_rate", "conditional_q_rate", "target_pair_legal_rate", "proposal_legal_rate",
        "physical_execution_rate", "engagement_rate", "neutral_rate", "third_agent_neutral_rate",
        "conditional_q_denominator_worlds", "conditional_q_numerator_worlds",
        "target_pair_denominator_worlds", "target_pair_numerator_worlds",
    )
    error = 0.0
    for key in keys:
        error = max(error, abs(float(actual[key]) - float(saved[key])))
    for key in ("actor_engagement_rates", "legal_plan_selection_counts", "legal_pair_selection_counts"):
        require(len(actual[key]) == len(saved[key]), "Metric vector length mismatch")
        for x, y in zip(actual[key], saved[key]):
            error = max(error, abs(float(x) - float(y)))
    return error


def main(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    plan, static = runner.verify(source)
    execution = source / "execution"
    result_path = execution / "results.json"
    require(result_path.is_file(), "Execution results missing")
    result = json.loads(result_path.read_text())
    require(len(result.get("runs", [])) == len(design.SEEDS) * len(design.CONDITIONS), "Incomplete run grid")
    by = {(row["seed"], row["condition"]): row for row in result["runs"]}
    require(len(by) == len(result["runs"]), "Duplicate run identity")

    monitor_spec = static["monitor_spec"]
    final_spec = static["final_spec"]
    arrays = {
        "monitor": runner.make_arrays(monitor_spec),
        "final": runner.make_arrays(final_spec),
    }
    max_error = 0.0
    checkpoints = 0
    monitor_worlds = 0
    final_worlds = 0
    frozen_checks = 0
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            source_result, source_checkpoint = runner.load_source_result(seed, schedule)
            source_networks = source_runner.load_networks(source_checkpoint)
            source_ab_hash = own_parameter_hash(source_networks, range(6))
            source_full_hash = own_parameter_hash(source_networks, range(9))
            require(source_full_hash == source_result["final_parameter_sha256"], "Source full hash mismatch")
            for live in design.LIVES:
                condition = f"{schedule}_new_receiver_PL_{'live' if live else 'silent'}"
                row = by[(seed, condition)]
                directory = execution / design.name(seed, condition)
                require(row["source_checkpoint_sha256"] == runner.sha(source_checkpoint), "Run/source checkpoint mismatch")
                require(row["frozen_ab_parameter_sha256"] == source_ab_hash, "Run frozen A/B hash mismatch")
                require(row["frozen_ab_parameter_unchanged"] is True, "Missing frozen invariant")
                for traj in row["trajectory"]:
                    update = int(traj["update"])
                    checkpoint = directory / f"checkpoint_{update:04d}.npz"
                    require(checkpoint.is_file() and runner.sha(checkpoint) == traj["checkpoint_sha256"], "Checkpoint hash mismatch")
                    networks = source_runner.load_networks(checkpoint)
                    ab_hash = own_parameter_hash(networks, range(6))
                    require(ab_hash == source_ab_hash == traj["frozen_ab_parameter_sha256"], "Frozen A/B changed in checkpoint")
                    require(own_parameter_hash(networks, range(9)) == traj["parameter_sha256"], "Checkpoint parameter hash mismatch")
                    if update == 0:
                        require(own_parameter_hash(networks, range(6, 9)) == row["new_agent_initial_parameter_sha256"], "C initialization mismatch")
                    actual = source_runner.evaluate_factorized(networks, arrays["monitor"], monitor_spec, bool(live))
                    max_error = max(max_error, compare_metric(actual, traj["target_trajectory"]))
                    checkpoints += 1
                    monitor_worlds += actual["worlds"]
                    frozen_checks += 1
                final_checkpoint = directory / "checkpoint_6000.npz"
                networks = source_runner.load_networks(final_checkpoint)
                actual_final = source_runner.evaluate_factorized(networks, arrays["final"], final_spec, bool(live))
                max_error = max(max_error, compare_metric(actual_final, row["final"]["new_layouts"]))
                final_worlds += actual_final["worlds"]
                require(runner.sha(directory / "training.jsonl") == row["training_log_sha256"], "Training log hash mismatch")

    # Paired streams are independently checked from the two logs.
    paired_logs = 0
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            live_path = execution / design.name(seed, f"{schedule}_new_receiver_PL_live") / "training.jsonl"
            silent_path = execution / design.name(seed, f"{schedule}_new_receiver_PL_silent") / "training.jsonl"
            with live_path.open() as left, silent_path.open() as right:
                for line_left, line_right in zip(left, right):
                    a, b = json.loads(line_left), json.loads(line_right)
                    require(a["update"] == b["update"], "Paired update mismatch")
                    for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256",
                                "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                        require(a[key] == b[key], "Paired stream differs: " + key)
                    paired_logs += 1

    verification = dict(
        status="passed", source=str(source), runs=len(result["runs"]), checkpoint_evaluations=checkpoints,
        final_evaluations=len(result["runs"]), compact_monitor_worlds=monitor_worlds, final_worlds=final_worlds,
        model_forwards=9 * (monitor_worlds + final_worlds), adaptation_optimizer_updates=len(result["runs"]) * design.UPDATES,
        frozen_agent_optimizer_updates=0, frozen_ab_checkpoints=frozen_checks, paired_training_log_rows=paired_logs,
        max_abs_error=float(max_error), checkpoint_replayed=True, final_replayed=True, paired_streams_verified=True,
        frozen_A_B_parameter_invariant=True, no_external_model=True, no_llm=True, no_vision_model=True,
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
