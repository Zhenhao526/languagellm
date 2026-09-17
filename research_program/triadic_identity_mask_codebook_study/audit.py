"""Independent numerical and file-level audit for identity_mask_001."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from research_program.triadic_message_study import runner as core
from . import design, runner, wire


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def close(a, b, atol=1e-12):
    return abs(float(a) - float(b)) <= atol


def metric_check(meta, saved):
    rewards = saved["greedy_reward"]
    executed = saved["executed"]
    satisfied = saved["satisfied"]
    expected = saved["conditional_exact_expected_reward"]
    full_probability = saved["conditional_exact_full_success_probability"]
    execution_probability = saved["conditional_exact_execution_probability"]
    values = {
        "greedy_reward_mean": float(rewards.mean()),
        "greedy_full_success_rate": float((rewards == 1).mean()),
        "greedy_partial_success_rate": float((rewards == 0.5).mean()),
        "physical_execution_rate": float(executed.any(axis=1).mean()),
        "satisfied_agent_rate": float(satisfied.mean()),
        "conditional_exact_expected_reward_mean": float(expected.mean()),
        "conditional_exact_full_success_probability_mean": float(full_probability.mean()),
        "conditional_exact_execution_probability_mean": float(execution_probability.mean()),
    }
    max_error = 0.0
    for key, value in values.items():
        max_error = max(max_error, abs(float(meta[key]) - value))
        require(close(meta[key], value), f"Metric mismatch: {key}")
    return max_error


def audit(output: str | Path) -> dict:
    output = Path(output).resolve()
    plan, prepared = runner.verify(output)
    execution = output / "execution"
    results = design.read(execution / "results.json")
    require(results["status"] == "completed", "Execution is not complete")
    require(len(results["runs"]) == len(design.SEEDS) * len(design.CONDITIONS), "Run count mismatch")
    expected_grid = [(seed, condition) for seed in design.SEEDS for condition in design.CONDITIONS]
    observed_grid = [(row["seed"], row["condition"]) for row in results["runs"]]
    require(observed_grid == expected_grid, "Run grid/order mismatch")
    evaluation_count = 0
    training_count = 0
    max_metric_error = 0.0
    for row in results["runs"]:
        seed, condition = row["seed"], row["condition"]
        directory = execution / f"seed_{seed}_{condition}"
        require(directory.is_dir(), f"Missing run directory: {directory}")
        require(design.read(directory / "result.json") == row, f"Result copy mismatch: {directory.name}")
        trajectory = row["trajectory"]
        require([item["update"] for item in trajectory] == list(design.STEPS), f"Checkpoint schedule mismatch: {directory.name}")
        training_lines = (directory / "training.jsonl").read_text(encoding="utf8").splitlines()
        require(len(training_lines) == design.UPDATES, f"Training log length mismatch: {directory.name}")
        training_count += len(training_lines)
        for expected_update, line in enumerate(training_lines, 1):
            record = json.loads(line)
            require(record["update"] == expected_update, f"Training update mismatch: {directory.name}:{expected_update}")
            require(np.isfinite(record["gradient_norm"]) and np.isfinite(record["gradient_clip_scale"]),
                    f"Non-finite optimizer diagnostic: {directory.name}:{expected_update}")
        settings = design.parse_condition(condition)
        for checkpoint in trajectory:
            checkpoint_path = directory / f"checkpoint_{checkpoint['update']:04d}.npz"
            require(checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}")
            require(runner.sha(checkpoint_path) == checkpoint["checkpoint_sha256"], f"Checkpoint hash mismatch: {checkpoint_path}")
            expected_modes = [settings["route_mode"] if settings["communication"] else "stable"]
            if settings["communication"]:
                expected_modes.append("masked" if settings["route_mode"] == "stable" else "stable")
            require(set(checkpoint["evaluations"]) == set(expected_modes), f"Evaluation mode mismatch: {directory.name}")
            for mode, meta in checkpoint["evaluations"].items():
                path = directory / f"target_{mode}_{checkpoint['update']:04d}.npz"
                require(path.is_file(), f"Missing evaluation: {path}")
                require(runner.sha(path) == meta["data_sha256"], f"Evaluation hash mismatch: {path}")
                with np.load(path, allow_pickle=False) as saved:
                    require(saved["state_indices"].ndim == 1 and np.array_equal(saved["state_indices"], np.arange(len(saved["state_indices"]))),
                            f"State index mismatch: {path}")
                    require(saved["messages"].shape[1:] == (2, 3, 4), f"Message shape mismatch: {path}")
                    require(saved["wire_messages"].shape == saved["messages"].shape, f"Wire shape mismatch: {path}")
                    require(saved["route_permutations"].shape[1:] == (2, 3), f"Route shape mismatch: {path}")
                    for key in saved.files:
                        if saved[key].dtype.kind in "fc":
                            require(np.isfinite(saved[key]).all(), f"Non-finite array {key}: {path}")
                    message_values = saved["messages"].reshape(-1, 3, 4)
                    recomputed_wire = wire.recode_tokens(message_values, settings["maps"]).reshape(saved["messages"].shape)
                    require(np.array_equal(recomputed_wire, saved["wire_messages"]), f"Wire map mismatch: {path}")
                    permutations = saved["route_permutations"]
                    if mode == "stable" or not settings["communication"]:
                        require(np.array_equal(permutations, np.broadcast_to(np.arange(3, dtype=np.int8), permutations.shape)),
                                f"Stable/silent route is not identity: {path}")
                    else:
                        require(np.all(np.sort(permutations, axis=2) == np.arange(3)), f"Invalid masked permutation: {path}")
                    max_metric_error = max(max_metric_error, metric_check(meta, saved))
                    require(meta["messages_sha256"] == core.array_sha(saved["messages"]), f"Message hash mismatch: {path}")
                    require(meta["wire_messages_sha256"] == core.array_sha(saved["wire_messages"]), f"Wire hash mismatch: {path}")
                    require(meta["route_permutations_sha256"] == core.array_sha(saved["route_permutations"]), f"Route hash mismatch: {path}")
                    require(meta["action_indices_sha256"] == core.array_sha(saved["action_indices"]), f"Action hash mismatch: {path}")
                evaluation_count += 1
    summary = design.read(output / "summary.json")
    require(summary["result_sha256"] == runner.sha(execution / "results.json"), "Summary/result hash mismatch")
    return dict(
        schema="triadic_identity_mask_audit_v1",
        status="passed",
        plan_sha256=runner.sha(output / "plan.json"),
        prepared_sha256=runner.sha(output / "prepared.json"),
        result_sha256=runner.sha(execution / "results.json"),
        run_count=len(results["runs"]),
        evaluation_count=evaluation_count,
        training_log_rows=training_count,
        nonfinite_arrays=0,
        max_metric_recompute_error=max_metric_error,
        route_and_wire_checks="passed",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    answer = audit(args.out)
    print(json.dumps(answer, ensure_ascii=False, sort_keys=True))
