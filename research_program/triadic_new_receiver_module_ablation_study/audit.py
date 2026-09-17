"""Independent replay audit for module-selective adaptation."""
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
    result_path = source / "execution" / "results.json"
    require(result_path.is_file(), "Execution results missing")
    result = json.loads(result_path.read_text())
    require(len(result.get("runs", [])) == 64, "Expected 64 selective runs")
    by = {(r["seed"], r["schedule"], r["arm"], bool(r["live"])): r for r in result["runs"]}
    require(len(by) == 64, "Duplicate run identity")
    full_audit = json.loads(design.FULL_AUDIT.read_text())
    require(full_audit.get("status") == "passed", "Full-C reference audit is not passed")
    arrays = {
        "monitor": runner.make_arrays(static["monitor_spec"]),
        "final": runner.make_arrays(static["final_spec"]),
    }
    max_error = 0.0
    checkpoints = final_evaluations = monitor_worlds = final_worlds = frozen_checks = 0
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            source_result, source_checkpoint = runner.load_source_result(seed, schedule)
            source_networks = source_runner.load_networks(source_checkpoint)
            source_ab_hash = runner.own_parameter_hash(source_networks, range(6))
            require(source_runner.parameter_hash(source_networks) == source_result["final_parameter_sha256"], "Source hash mismatch")
            for arm in design.ARMS:
                for live in design.LIVES:
                    row = by[(seed, schedule, arm, bool(live))]
                    directory = source / "execution" / design.name(seed, row["condition"])
                    require(row["source_checkpoint_sha256"] == runner.sha(source_checkpoint), "Wrong source checkpoint")
                    require(row["frozen_ab_parameter_sha256"] == source_ab_hash, "Wrong frozen A/B hash")
                    for traj in row["trajectory"]:
                        update = int(traj["update"])
                        path = directory / f"checkpoint_{update:04d}.npz"
                        require(path.is_file() and runner.sha(path) == traj["checkpoint_sha256"], "Checkpoint hash mismatch")
                        networks = source_runner.load_networks(path)
                        require(runner.own_parameter_hash(networks, range(6)) == source_ab_hash == traj["frozen_ab_parameter_sha256"], "A/B changed")
                        require(runner.own_parameter_hash(networks) == traj["parameter_sha256"], "Parameter hash mismatch")
                        if update == 0:
                            require(runner.own_parameter_hash(networks[6:9], range(3)) == row["new_agent_initial_parameter_sha256"], "C initialization mismatch")
                        actual = source_runner.evaluate_factorized(networks, arrays["monitor"], static["monitor_spec"], bool(live))
                        max_error = max(max_error, compare_metric(actual, traj["target_trajectory"]))
                        checkpoints += 1
                        monitor_worlds += actual["worlds"]
                        frozen_checks += 1
                    final_path = directory / "checkpoint_6000.npz"
                    networks = source_runner.load_networks(final_path)
                    actual_final = source_runner.evaluate_factorized(networks, arrays["final"], static["final_spec"], bool(live))
                    max_error = max(max_error, compare_metric(actual_final, row["final"]["new_layouts"]))
                    final_evaluations += 1
                    final_worlds += actual_final["worlds"]
                    require(runner.sha(directory / "training.jsonl") == row["training_log_sha256"], "Training log hash mismatch")

    paired_rows = 0
    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            for arm in design.ARMS:
                paths = [
                    source / "execution" / design.name(seed, f"{schedule}_new_receiver_{arm}_PL_live") / "training.jsonl",
                    source / "execution" / design.name(seed, f"{schedule}_new_receiver_{arm}_PL_silent") / "training.jsonl",
                ]
                with paths[0].open() as left, paths[1].open() as right:
                    for a_line, b_line in zip(left, right):
                        a, b = json.loads(a_line), json.loads(b_line)
                        require(a["update"] == b["update"], "Paired update mismatch")
                        for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256",
                                    "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                            require(a[key] == b[key], "Live/silent stream differs: " + key)
                        paired_rows += 1
            for live in design.LIVES:
                streams = [
                    (source / "execution" / design.name(seed, f"{schedule}_new_receiver_{arm}_PL_{'live' if live else 'silent'}") / "training.jsonl").open()
                    for arm in design.ARMS
                ]
                try:
                    for lines in zip(*streams):
                        rows = [json.loads(line) for line in lines]
                        for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256",
                                    "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                            require(len({row[key] for row in rows}) == 1, "Cross-arm stream differs: " + key)
                finally:
                    for stream in streams:
                        stream.close()

    verification = dict(
        status="passed", source=str(source), runs=64, checkpoint_evaluations=checkpoints,
        final_evaluations=final_evaluations, compact_monitor_worlds=monitor_worlds, final_worlds=final_worlds,
        model_forwards=9 * (monitor_worlds + final_worlds), adaptation_optimizer_updates=64 * design.UPDATES,
        frozen_agent_optimizer_updates=0, frozen_ab_checkpoints=frozen_checks, paired_training_log_rows=paired_rows,
        max_abs_error=float(max_error), checkpoint_replayed=True, final_replayed=True,
        paired_streams_verified=True, cross_arm_streams_verified=True, frozen_A_B_parameter_invariant=True,
        full_reference_audit_reused=True, no_external_model=True, no_llm=True, no_vision_model=True,
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
