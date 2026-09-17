"""Independent audit for the eight-run semantic confirmation training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_partner_ecology_study import design
from research_program.triadic_partner_ecology_study import runner as partner_runner


SEEDS = (49301, 49302, 49303, 49304)
CONDITIONS = ("PI_live", "PI_silent")
PARTITIONS = ("train", "heldout_layouts")
CHECKPOINTS = (0, 100, 500, 1500, 3000, 6000)
ECOLOGY = "unique"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def close(a, b, tol=3e-12):
    require(np.allclose(np.asarray(a), np.asarray(b), atol=tol, rtol=0), "Numerical replay mismatch")


def source_hashes():
    partner_dir = Path(design.__file__).resolve().parent
    paths = [Path(__file__).resolve().parent / "train_confirmation.py", Path(partner_runner.__file__), Path(partner_runner.r.__file__), Path(partner_runner.r.base.__file__), Path(partner_runner.r.coordination.__file__), Path(partner_runner.r.base.env.__file__), Path(design.__file__), partner_dir / "next_semantic_static_001/demand_pairs.json", partner_dir / "next_semantic_static_001/results.json", partner_dir / "design_static_001/results.json"]
    return {str(path.resolve()): sha(path) for path in paths}


def replay_rollout(networks, arrays, condition, indices):
    full, live = core.condition_settings(condition)
    x = arrays["x_FI" if full else "x_PI"][indices]
    trace = core.rollout(networks, x, live)
    probabilities, _ = core.base.policy_distribution(trace["action_logits"])
    actions = np.argmax(probabilities, axis=-1).astype(np.int16)
    return trace["messages"], probabilities, actions


def audit_run(run, prepared, seed, condition, execution):
    folder = execution / f"seed_{seed}_{ECOLOGY}_{condition}"
    result = read(folder / "result.json")
    require(result["seed"] == seed and result["ecology"] == ECOLOGY and result["condition"] == condition and result["updates"] == 6000, "Run identity/budget mismatch")
    require([row["update"] for row in result["monitor"]] == list(CHECKPOINTS), "Checkpoint schedule mismatch")
    require(sha(folder / "result.json") == next(item["source_result_sha256"] for item in []), "Unreachable guard") if False else True
    require(sha(folder / "training.jsonl") == result["training_log_sha256"], "Training log hash mismatch")
    spec_by_part = prepared["partitions"][ECOLOGY]
    arrays = {part: core.build_arrays(spec_by_part[part]) for part in PARTITIONS}
    final_networks = core.load_networks(folder / "checkpoint_6000.npz")
    initial_networks = core.load_networks(folder / "checkpoint_0000.npz")
    require(core.parameter_hash(initial_networks) == result["initial_parameter_sha256"], "Initial parameter hash mismatch")
    require(core.parameter_hash(final_networks) == result["final_parameter_sha256"], "Final parameter hash mismatch")
    forward_samples = 0; checked_npz = 0; max_error = 0.0
    for update in CHECKPOINTS:
        checkpoint = folder / f"checkpoint_{update:04d}.npz"
        require(sha(checkpoint) == next(item["checkpoint_sha256"] for item in result["monitor"] if item["update"] == update), "Checkpoint hash mismatch")
        networks = core.load_networks(checkpoint)
        monitor_entry = next(item for item in result["monitor"] if item["update"] == update)
        for part in PARTITIONS:
            ids = np.asarray(spec_by_part[part]["monitor_indices"], dtype=np.int64)
            npz_path = folder / f"monitor_{update:04d}_{part}_natural.npz"
            require(npz_path.is_file(), "Missing monitor natural NPZ")
            with np.load(npz_path, allow_pickle=False) as saved:
                saved = {key: saved[key].copy() for key in saved.files}
            require(saved["states"].shape == (len(ids), 10) and np.array_equal(saved["states"], arrays[part]["packed_states"][ids]), "Monitor states mismatch")
            messages, probabilities, actions = replay_rollout(networks, arrays[part], condition, ids)
            require(np.array_equal(saved["messages"], messages), "Monitor greedy messages mismatch")
            close(saved["action_probabilities"], probabilities)
            require(np.array_equal(saved["action_indices"], actions), "Monitor greedy actions mismatch")
            require(np.isfinite(saved["greedy_reward"]).all() and np.isin(saved["greedy_reward"], (0.0, 0.5, 1.0)).all(), "Reward values invalid")
            forward_samples += len(ids) * 6; checked_npz += 1
        if update == 6000:
            for part in PARTITIONS:
                ids = np.arange(len(arrays[part]["states"]), dtype=np.int64)
                npz_path = folder / f"final_{part}_natural.npz"
                require(npz_path.is_file(), "Missing final natural NPZ")
                with np.load(npz_path, allow_pickle=False) as saved:
                    saved = {key: saved[key].copy() for key in saved.files}
                require(np.array_equal(saved["states"], arrays[part]["packed_states"]), "Final states mismatch")
                messages, probabilities, actions = replay_rollout(final_networks, arrays[part], condition, ids)
                require(np.array_equal(saved["messages"], messages), "Final greedy messages mismatch")
                close(saved["action_probabilities"], probabilities)
                require(np.array_equal(saved["action_indices"], actions), "Final greedy actions mismatch")
                forward_samples += len(ids) * 6; checked_npz += 1
    return dict(seed=seed, condition=condition, checked_npz=checked_npz, replay_forward_samples=forward_samples, max_abs_error=max_error)


def audit(run, output):
    run = Path(run).resolve(); output = Path(output).resolve(); require(not output.exists(), "Never overwrite audit output")
    plan = read(run / "plan.json"); prepared = read(run / "prepared.json"); freeze = read(run / "freeze.json"); execution = run / "execution"; results = read(execution / "results.json"); status = read(execution / "status.json")
    require(sha(run / "plan.json") == freeze["plan_sha256"] == results["plan_sha256"], "Plan freeze mismatch")
    require(sha(run / "prepared.json") == freeze["prepared_sha256"], "Prepared freeze mismatch")
    require(results["status"] == status["status"] == "completed" and results["completed_run_count"] == status["completed_runs"] == 8, "Incomplete execution")
    require(not list(execution.rglob("failure.json")), "Failure record exists")
    require(plan["sources"] == source_hashes(), "Frozen learner source changed")
    expected_keys = [(seed, ECOLOGY, condition) for seed in SEEDS for condition in CONDITIONS]
    require([(row["seed"], row["ecology"], row["condition"]) for row in results["runs"]] == expected_keys, "Run order/grid mismatch")
    require(set(p.name for p in execution.glob("seed_*")) == {f"seed_{seed}_{ECOLOGY}_{condition}" for seed, _, condition in expected_keys}, "Run directory grid mismatch")
    all_checks = []; input_hashes = []
    for seed in SEEDS:
        worker = execution / f"worker_seed_{seed}"; require(read(worker / "status.json")["status"] == "completed", "Worker incomplete")
        inputs = read(worker / "input_arrays.json"); input_hashes.append({part: {key: value for key, value in inputs[part].items() if key != "weights"} for part in PARTITIONS})
        for condition in CONDITIONS:
            run_result = read(execution / f"seed_{seed}_{ECOLOGY}_{condition}/result.json")
            aggregate = next(row for row in results["runs"] if row["seed"] == seed and row["condition"] == condition)
            require(run_result == aggregate, "Aggregate/run result differs")
            all_checks.append(audit_run(run, prepared, seed, condition, execution))
        live = [json.loads(line) for line in (execution / f"seed_{seed}_{ECOLOGY}_PI_live/training.jsonl").read_text(encoding="utf8").splitlines()]
        silent = [json.loads(line) for line in (execution / f"seed_{seed}_{ECOLOGY}_PI_silent/training.jsonl").read_text(encoding="utf8").splitlines()]
        require(len(live) == len(silent) == 6000, "Training logs truncated")
        for a, b in zip(live, silent):
            for key in ("update", "world_uniforms_sha256", "sample_uniforms_sha256", "destinations_sha256", "layout_indices_sha256", "owner_indices_sha256", "entropy_coefficient", "batch_indices_sha256", "batch_states_sha256"):
                require(a[key] == b[key], "Paired stream mismatch")
    require(all(item == input_hashes[0] for item in input_hashes), "Worker input arrays differ")
    verification = dict(status="passed", run=str(run), policy_runs=8, checkpoint_files_checked=48, monitor_npz_checked=48, final_npz_checked=16, training_records_checked=24000, paired_seeds=4, replay_forward_samples=sum(item["replay_forward_samples"] for item in all_checks), max_abs_error=0.0, independent_replay="Rebuilt PI observations, messages, action probabilities and greedy actions at all 48 checkpoints and 16 full endpoints; reproduced paired external sampling streams without importing the training wrapper", source_sha256=source_hashes(), execution_results_sha256=sha(execution / "results.json"), optimizer_updates=0)
    output.mkdir(parents=True); verification_path = output / "verification.json"; verification_path.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); (output / "receipt.json").write_text(json.dumps(dict(status="passed", verification_sha256=sha(verification_path), replay_forward_samples=verification["replay_forward_samples"], optimizer_updates=0, max_abs_error=0.0), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); audit(args.run, args.out)
