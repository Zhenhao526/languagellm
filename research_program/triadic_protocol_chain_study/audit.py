"""Independent replay audit for the two-generation protocol chain."""
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


def condition(generation, schedule, live):
    arm = "replace_A" if generation == 2 else "replace_B"
    return f"generation{generation}_{arm}_{schedule}_PL_{'live' if live else 'silent'}"


def main(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    plan, static = runner.verify(source)
    result_path = source / "execution" / "results.json"
    require(result_path.is_file(), "Execution results missing")
    result = json.loads(result_path.read_text())
    expected_count = len(design.SEEDS) * len(design.GENERATIONS) * len(design.SCHEDULES) * 2
    require(len(result.get("runs", [])) == expected_count, "Incomplete chain run grid")
    by = {(row["seed"], row["generation"], row["schedule"], bool(row["live"])): row for row in result["runs"]}
    require(len(by) == expected_count, "Duplicate chain run identity")
    source_runs = {(item["seed"], item["schedule"]): item for item in static["generation1_runs"]}

    arrays = {
        "monitor": runner.make_arrays(static["monitor_spec"]),
        "final": runner.make_arrays(static["final_spec"]),
    }
    max_error = 0.0
    checkpoints = final_evaluations = parent_evaluations = 0
    monitor_worlds = final_worlds = parent_worlds = 0
    frozen_checks = 0

    for seed in design.SEEDS:
        for schedule in design.SCHEDULES:
            # Independently recover generation-1 parent and generation-2 live
            # endpoint.  Only generation-2 live is allowed to seed generation 3.
            source_result, source_checkpoint = runner.load_generation1_result(seed, schedule)
            source_networks = source_runner.load_networks(source_checkpoint)
            require(runner.parameter_hash(source_networks) == source_result["final_parameter_sha256"],
                    "Generation-1 source hash mismatch")
            g2_live = by[(seed, 2, schedule, True)]
            g2_live_checkpoint = source / "execution" / runner.design.name(seed, g2_live["condition"]) / "checkpoint_6000.npz"
            require(runner.sha(g2_live_checkpoint) == g2_live["final_checkpoint_sha256"], "Generation-2 live checkpoint hash mismatch")
            g2_networks = source_runner.load_networks(g2_live_checkpoint)
            require(runner.parameter_hash(g2_networks) == g2_live["final_parameter_sha256"], "Generation-2 live endpoint hash mismatch")
            parent_networks = {2: source_networks, 3: g2_networks}
            parent_meta = {
                2: dict(generation=1, checkpoint_sha256=runner.sha(source_checkpoint), network_hash=runner.parameter_hash(source_networks)),
                3: dict(generation=2, checkpoint_sha256=runner.sha(g2_live_checkpoint), network_hash=runner.parameter_hash(g2_networks)),
            }
            parent_conditions = {2: f"{schedule}_new_receiver_PL_live", 3: g2_live["condition"]}

            for generation in design.GENERATIONS:
                frozen_indices = tuple(i for i in range(9) if i not in runner.replaced_indices(generation))
                for live in design.LIVES:
                    row = by[(seed, generation, schedule, bool(live))]
                    expected_condition = condition(generation, schedule, bool(live))
                    require(row["condition"] == expected_condition, "Run condition mismatch")
                    parent = parent_networks[generation]
                    pmeta = parent_meta[generation]
                    require(row["parent_generation"] == pmeta["generation"], "Parent generation mismatch")
                    require(row["parent_condition"] == parent_conditions[generation], "Parent condition mismatch")
                    require(row["parent_checkpoint_sha256"] == pmeta["checkpoint_sha256"], "Parent checkpoint mismatch")
                    require(row["parent_parameter_sha256"] == pmeta["network_hash"], "Parent parameter mismatch")
                    parent_eval = source_runner.evaluate_factorized(parent, arrays["final"], static["final_spec"], bool(live))
                    max_error = max(max_error, compare_metric(parent_eval, row["inherited_parent"]["new_layouts"]))
                    parent_evaluations += 1
                    parent_worlds += parent_eval["worlds"]

                    directory = source / "execution" / runner.design.name(seed, row["condition"])
                    require(row["frozen_parameter_unchanged"] is True, "Missing frozen invariant")
                    for traj in row["trajectory"]:
                        update = int(traj["update"])
                        checkpoint = directory / f"checkpoint_{update:04d}.npz"
                        require(checkpoint.is_file() and runner.sha(checkpoint) == traj["checkpoint_sha256"], "Checkpoint hash mismatch")
                        networks = source_runner.load_networks(checkpoint)
                        require(runner.parameter_hash(networks) == traj["parameter_sha256"], "Checkpoint parameter hash mismatch")
                        require(runner.parameter_hash(networks, frozen_indices) == traj["frozen_parameter_sha256"], "Frozen checkpoint hash mismatch")
                        require(runner.parameter_hash(networks, frozen_indices) == row["frozen_parent_parameter_sha256"], "Frozen parent changed")
                        if update == 0:
                            fresh = runner.fresh_replaced_networks(seed, generation, design.SCHEDULES.index(schedule))
                            require(runner.parameter_hash(fresh) == row["new_agent_initial_parameter_sha256"],
                                    "Fresh learner initialization mismatch")
                        actual = source_runner.evaluate_factorized(networks, arrays["monitor"], static["monitor_spec"], bool(live))
                        max_error = max(max_error, compare_metric(actual, traj["target_trajectory"]))
                        checkpoints += 1
                        monitor_worlds += actual["worlds"]
                        frozen_checks += 1

                    final_checkpoint = directory / "checkpoint_6000.npz"
                    require(final_checkpoint.is_file() and runner.sha(final_checkpoint) == row["final_checkpoint_sha256"], "Final checkpoint hash mismatch")
                    networks = source_runner.load_networks(final_checkpoint)
                    actual_final = source_runner.evaluate_factorized(networks, arrays["final"], static["final_spec"], bool(live))
                    max_error = max(max_error, compare_metric(actual_final, row["final"]["new_layouts"]))
                    final_evaluations += 1
                    final_worlds += actual_final["worlds"]
                    require(runner.sha(directory / "training.jsonl") == row["training_log_sha256"], "Training log hash mismatch")

            # Independent chain link for the generation-3 pair.
            for live in design.LIVES:
                g3 = by[(seed, 3, schedule, bool(live))]
                require(g3["parent_checkpoint_sha256"] == g2_live["final_checkpoint_sha256"], "Generation-3 does not inherit generation-2 live")
            require(by[(seed, 3, schedule, True)]["parent_checkpoint_sha256"] ==
                    by[(seed, 3, schedule, False)]["parent_checkpoint_sha256"], "Generation-3 pair has different parent")

    paired_rows = 0
    for seed in design.SEEDS:
        for generation in design.GENERATIONS:
            for schedule in design.SCHEDULES:
                left = source / "execution" / design.name(seed, condition(generation, schedule, True)) / "training.jsonl"
                right = source / "execution" / design.name(seed, condition(generation, schedule, False)) / "training.jsonl"
                with left.open() as ls, right.open() as rs:
                    count = 0
                    for line_left, line_right in zip(ls, rs):
                        a, b = json.loads(line_left), json.loads(line_right)
                        require(a["update"] == b["update"] == count + 1, "Paired update mismatch")
                        for key in ("world_uniforms_sha256", "sample_uniforms_sha256", "batch_indices_sha256",
                                    "canonical_batch_states_sha256", "effective_batch_states_sha256", "permutation_indices_sha256"):
                            require(a[key] == b[key], "Paired stream differs: " + key)
                        count += 1
                    require(count == design.UPDATES, "Paired training log length mismatch")
                    paired_rows += count

    verification = dict(
        status="passed", source=str(source), runs=len(result["runs"]), checkpoint_evaluations=checkpoints,
        final_evaluations=final_evaluations, parent_evaluations=parent_evaluations,
        compact_monitor_worlds=monitor_worlds, final_worlds=final_worlds, parent_worlds=parent_worlds,
        model_forwards=9 * (monitor_worlds + final_worlds + parent_worlds),
        adaptation_optimizer_updates=len(result["runs"]) * design.UPDATES,
        frozen_agent_optimizer_updates=0, frozen_parent_checkpoints=frozen_checks,
        paired_training_log_rows=paired_rows, max_abs_error=float(max_error),
        checkpoint_replayed=True, final_replayed=True, parent_replayed=True,
        paired_streams_verified=True, generation_links_verified=True,
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
