"""Independent PI-live/PI-silent training for the semantic-transfer readout.

The task ecology and learner are frozen imports from the audited partner
ecology study.  This wrapper changes only the seed set and the pre-registered
two-arm confirmation scope; it does not alter the learner implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import time
from copy import deepcopy
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_partner_ecology_study import design as partner_design
from research_program.triadic_partner_ecology_study import runner as partner_runner


SEEDS = (49301, 49302, 49303, 49304)
CONDITIONS = ("PI_live", "PI_silent")
ECOLOGY = "unique"
PARTITIONS = ("train", "heldout_layouts")
PARTNER_DIR = Path(partner_design.__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def source_paths():
    paths = [
        Path(__file__),
        Path(partner_design.__file__),
        Path(partner_runner.__file__),
        Path(partner_runner.r.__file__),
        Path(partner_runner.r.base.__file__),
        Path(partner_runner.r.coordination.__file__),
        Path(partner_runner.r.base.env.__file__),
        PARTNER_DIR / "next_semantic_static_001/demand_pairs.json",
        PARTNER_DIR / "next_semantic_static_001/results.json",
        PARTNER_DIR / "design_static_001/results.json",
    ]
    require(all(path.is_file() for path in paths), "Missing frozen source")
    return {str(path.resolve()): sha(path) for path in paths}


def make_prepared():
    prepared = deepcopy(partner_design.make_prepared())
    prepared.update(
        schema="triadic_semantic_formation_confirmation_prepared_v1",
        seeds=list(SEEDS),
        ecologies=[ECOLOGY],
        conditions=list(CONDITIONS),
        partition_names=list(PARTITIONS),
        runs=[dict(seed=seed, ecology=ECOLOGY, condition=condition, directory=f"seed_{seed}_{ECOLOGY}_{condition}") for seed in SEEDS for condition in CONDITIONS],
        confirmation_scope="new independent seeds; unique ecology; PI-live versus PI-silent; same 18/6 layout split; all unique demands train",
        semantic_readout="aligned source-message transfer minus stratum-misaligned placebo, pre-registered before execution",
        no_new_task_rules=True,
        automatic_followon_experiment=False,
    )
    return prepared


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), "Never overwrite preparation")
    sources = source_paths()
    prepared = make_prepared()
    manifest = dict(
        schema="triadic_semantic_formation_confirmation_manifest_v1",
        seeds=list(SEEDS),
        ecology=ECOLOGY,
        conditions=list(CONDITIONS),
        partitions=list(PARTITIONS),
        updates=6000,
        batch_size=256,
        candidate=str((PARTNER_DIR / "next_semantic_static_001/demand_pairs.json").resolve()),
        candidate_sha256=sources[str((PARTNER_DIR / "next_semantic_static_001/demand_pairs.json").resolve())],
        semantic_readout="aligned minus cyclic stratum-misaligned source-message placebo on fixed-role disjoint-action cases",
        axis_weighting="equal kind, length and destination axes after case-level aggregation",
        no_training_by_prepare=True,
        no_optimizer_updates=True,
    )
    out.mkdir(parents=True)
    write_new(out / "manifest.json", manifest)
    prepared["manifest_sha256"] = sha(out / "manifest.json")
    prepared["sources_sha256"] = sources
    write_new(out / "prepared.json", prepared)
    plan = dict(
        schema="triadic_semantic_formation_confirmation_plan_v1",
        status="prepared_without_training",
        created_at=core.base.now(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        seeds=list(SEEDS),
        conditions=list(CONDITIONS),
        ecology=ECOLOGY,
        sources=sources,
        candidate_sha256=manifest["candidate_sha256"],
        manifest_sha256=sha(out / "manifest.json"),
        prepared_sha256=sha(out / "prepared.json"),
        no_training=True,
        no_optimizer_updates=True,
    )
    write_new(out / "plan.json", plan)
    write_new(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json")))
    write_new(out / "receipt.json", dict(status="prepared_not_trained", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), seeds=list(SEEDS), runs=len(SEEDS) * len(CONDITIONS), optimizer_updates=0, model_calls=0))
    return dict(status="prepared_not_trained", output=str(out), runs=len(SEEDS) * len(CONDITIONS))


def verify(out):
    out = Path(out).resolve()
    plan = read(out / "plan.json"); prepared = read(out / "prepared.json"); freeze = read(out / "freeze.json")
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan freeze mismatch")
    require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared freeze mismatch")
    require(prepared["manifest_sha256"] == sha(out / "manifest.json") == plan["manifest_sha256"], "Manifest mismatch")
    require(plan["sources"] == source_paths(), "Frozen source changed")
    require(prepared["sources_sha256"] == plan["sources"], "Prepared source map mismatch")
    return plan, prepared


def seed_worker(out, seed):
    out = Path(out).resolve(); execution = out / "execution"; prepared = read(out / "prepared.json")
    worker = execution / f"worker_seed_{seed}"; worker.mkdir(exist_ok=False)
    started = time.perf_counter()
    write_new(worker / "started.json", dict(seed=seed, pid=os.getpid(), parent_pid=os.getppid(), started_at=core.base.now(), plan_sha256=sha(out / "plan.json"), thread_environment={key: os.environ[key] for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}))
    try:
        arrays = {part: core.build_arrays(prepared["partitions"][ECOLOGY][part]) for part in PARTITIONS}
        weight_receipts = {part: partner_runner.save_weights(worker / f"weights_{ECOLOGY}_{part}.npz", prepared["partitions"][ECOLOGY][part], execution) for part in PARTITIONS}
        inputs = {part: dict(worlds=len(arrays[part]["states"]), states_sha256=core.array_sha(arrays[part]["packed_states"]), private_information_features_sha256=core.array_sha(arrays[part]["x_PI"]), native_rewards_sha256=core.array_sha(arrays[part]["rewards"]), weights=weight_receipts[part]) for part in PARTITIONS}
        write_new(worker / "input_arrays.json", inputs)
        results = []
        for condition in CONDITIONS:
            run_dir = execution / f"seed_{seed}_{ECOLOGY}_{condition}"
            result = partner_runner.train_run(seed, ECOLOGY, condition, prepared, arrays, weight_receipts, run_dir, execution)
            results.append(result)
        write_new(worker / "results.json", dict(status="completed", seed=seed, runs=results, elapsed_seconds=time.perf_counter() - started))
        write_new(worker / "status.json", dict(status="completed", completed_at=core.base.now()))
    except BaseException as error:
        write_new(worker / "failure.json", dict(status="failed", seed=seed, error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started))
        write_new(worker / "status.json", dict(status="failed", failed_at=core.base.now()))
        raise


def paired_training_check(execution, seed):
    live_path = execution / f"seed_{seed}_{ECOLOGY}_PI_live/training.jsonl"
    silent_path = execution / f"seed_{seed}_{ECOLOGY}_PI_silent/training.jsonl"
    live = [json.loads(line) for line in live_path.read_text(encoding="utf8").splitlines()]
    silent = [json.loads(line) for line in silent_path.read_text(encoding="utf8").splitlines()]
    require(len(live) == len(silent) == 6000, "Training log length mismatch")
    common = ("update", "world_uniforms_sha256", "sample_uniforms_sha256", "destinations_sha256", "layout_indices_sha256", "owner_indices_sha256", "entropy_coefficient")
    within = ("batch_indices_sha256", "batch_states_sha256")
    for a, b in zip(live, silent):
        require(all(a[key] == b[key] for key in common), "Paired common stream mismatch")
        require(all(a[key] == b[key] for key in within), "Paired unique-state stream mismatch")
    live_result = read(execution / f"seed_{seed}_{ECOLOGY}_PI_live/result.json")
    silent_result = read(execution / f"seed_{seed}_{ECOLOGY}_PI_silent/result.json")
    require(live_result["initial_parameter_sha256"] == silent_result["initial_parameter_sha256"], "Initial parameters differ")
    return dict(seed=seed, updates=6000, common_stream_updates=6000, actual_states_matched=True, initial_parameters_matched=True)


def execute(out, workers=4):
    out = Path(out).resolve(); _, prepared = verify(out); execution = out / "execution"; require(not execution.exists(), "Never resume execution")
    execution.mkdir(); started = time.perf_counter()
    write_new(execution / "started.json", dict(started_at=core.base.now(), plan_sha256=sha(out / "plan.json"), device="cpu_numpy", worker_count=int(workers)))
    context = multiprocessing.get_context("spawn"); processes = []
    try:
        for seed in SEEDS:
            process = context.Process(target=seed_worker, args=(str(out), seed), name=f"semantic_confirmation_seed_{seed}")
            processes.append(process); process.start()
        for process in processes: process.join()
        require(all(process.exitcode == 0 for process in processes), "A seed worker failed")
        runs = []
        for seed in SEEDS:
            worker = read(execution / f"worker_seed_{seed}/results.json"); require(worker["status"] == "completed" and worker["seed"] == seed, "Worker result mismatch")
            require([(row["ecology"], row["condition"]) for row in worker["runs"]] == [(ECOLOGY, condition) for condition in CONDITIONS], "Worker arm order mismatch")
            for row in worker["runs"]:
                run = read(execution / f"seed_{seed}_{ECOLOGY}_{row['condition']}/result.json")
                require(run == row, "Worker aggregate differs from run JSON")
                runs.append(run)
        pairing = [paired_training_check(execution, seed) for seed in SEEDS]
        result = dict(status="completed", completed_at=core.base.now(), plan_sha256=sha(out / "plan.json"), runs=runs, seeds=list(SEEDS), conditions=list(CONDITIONS), ecology=ECOLOGY, completed_run_count=len(runs), paired_seed_count=len(SEEDS), pairing_checks=pairing, updates_total=len(runs) * 6000, optimizer_updates=len(runs) * 6000, model_calls=0, elapsed_seconds=time.perf_counter() - started, scope="Independent seeds for pre-registered aligned-minus-placebo semantic-transfer readout; training task itself remains the audited unique PI ecology")
        write_new(execution / "results.json", result); write_new(execution / "status.json", dict(status="completed", completed_runs=len(runs), completed_at=core.base.now())); write_new(execution / "receipt.json", dict(status="passed", results_sha256=sha(execution / "results.json"), completed_runs=len(runs), optimizer_updates=result["optimizer_updates"], elapsed_seconds=result["elapsed_seconds"]))
        return dict(status="completed", output=str(execution), runs=len(runs), results_sha256=sha(execution / "results.json"))
    except BaseException as error:
        for process in processes:
            if process.is_alive(): process.terminate()
        write_new(execution / "failure.json", dict(status="failed", error_type=type(error).__name__, error=str(error), elapsed_seconds=time.perf_counter() - started))
        write_new(execution / "status.json", dict(status="failed", failed_at=core.base.now()))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--out", required=True); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args()
    if args.command == "prepare": value = prepare(args.out)
    elif args.command == "verify": verify(args.out); value = dict(status="verified", output=str(Path(args.out).resolve()))
    else: value = execute(args.out, args.workers)
    print(json.dumps(value, ensure_ascii=False))
