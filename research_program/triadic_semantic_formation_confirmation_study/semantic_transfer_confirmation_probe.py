"""Aligned source-message transfer probe on the independent confirmation runs."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_partner_ecology_study import semantic_transfer_probe as primary


SEEDS = (49301, 49302, 49303, 49304)
CONDITIONS = ("PI_live", "PI_silent")
AXES = primary.AXES
PARTITION = "heldout_layouts"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha(value):
    return hashlib.sha256(np.asarray(value).tobytes(order="C")).hexdigest()


def write_new(path, value):
    path = Path(path); require(not path.exists(), "Refuse to overwrite " + str(path)); path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def prepare(out, source, candidate):
    out = Path(out).resolve(); source = Path(source).resolve(); candidate = Path(candidate).resolve(); require(not out.exists(), "Never overwrite preparation")
    cases = primary._eligible_cases(candidate); counts = {axis: sum(case["axis"] == axis for case in cases) for axis in AXES}; require(counts == {"kind_wood_fiber": 48, "length_short_long": 48, "destination_L_R": 144}, "Candidate axis balance")
    manifest = dict(schema="triadic_semantic_formation_confirmation_transfer_manifest_v1", source=str(source), candidate=str(candidate), partition=PARTITION, conditions=list(CONDITIONS), seeds=list(SEEDS), cases=len(cases), backgrounds_per_case=36, axis_case_counts=counts, intervention="aligned source endpoint first-window message shown to target cross-viewers; W2/action recomputed for PI-live", no_training_by_probe=True)
    out.mkdir(parents=True); write_new(out / "manifest.json", manifest)
    prepared = dict(schema="triadic_semantic_formation_confirmation_transfer_prepared_v1", source=str(source), candidate=str(candidate), source_prepared_sha256=sha(source / "prepared.json"), source_freeze_sha256=sha(source / "freeze.json"), source_results_sha256=sha(source / "execution/results.json"), candidate_sha256=sha(candidate), manifest_sha256=sha(out / "manifest.json"), partition=PARTITION, conditions=list(CONDITIONS), seeds=list(SEEDS), cases=len(cases), backgrounds_per_case=36, axis_case_counts=counts, no_training=True, no_optimizer_updates=True)
    write_new(out / "prepared.json", prepared); plan = dict(status="prepared_without_policy_reads", created_at=core.base.now(), runtime=dict(python=platform.python_version(), numpy=np.__version__), source_sha256={"source_prepared.json": prepared["source_prepared_sha256"], "source_freeze.json": prepared["source_freeze_sha256"], "source_execution_results.json": prepared["source_results_sha256"], "candidate.json": prepared["candidate_sha256"], "manifest.json": sha(out / "manifest.json"), "prepared.json": sha(out / "prepared.json")}, prepared_sha256=sha(out / "prepared.json"), no_training=True, no_optimizer_updates=True)
    write_new(out / "plan.json", plan); write_new(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"))); write_new(out / "receipt.json", dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), cases=len(cases), model_calls=0, optimizer_updates=0))
    return dict(status="prepared_static_only", output=str(out), cases=len(cases))


def verify(out):
    out = Path(out).resolve(); plan = json.loads((out / "plan.json").read_text()); prepared = json.loads((out / "prepared.json").read_text()); freeze = json.loads((out / "freeze.json").read_text())
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan mismatch"); require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared mismatch"); require(prepared["manifest_sha256"] == sha(out / "manifest.json"), "Manifest mismatch")
    source = Path(prepared["source"]); candidate = Path(prepared["candidate"]); require(sha(source / "prepared.json") == prepared["source_prepared_sha256"] and sha(source / "freeze.json") == prepared["source_freeze_sha256"] and sha(source / "execution/results.json") == prepared["source_results_sha256"], "Source changed"); require(sha(candidate) == prepared["candidate_sha256"], "Candidate changed")
    return plan, prepared


def run_policy(source, prepared, seed, condition, cases):
    source = Path(source); folder = source / "execution" / f"seed_{seed}_unique_{condition}"; checkpoint = folder / "checkpoint_6000.npz"; source_result = folder / "result.json"; data_path = folder / f"final_{PARTITION}_natural.npz"; require(checkpoint.is_file() and source_result.is_file() and data_path.is_file(), "Missing policy files")
    networks = core.load_networks(checkpoint); static = json.loads((source / "prepared.json").read_text()); spec = static["partitions"]["unique"][PARTITION]; arrays = core.build_arrays(spec); lookup = {tuple(need): i for i, need in enumerate(spec["needs"])}; nphysical = len(spec["layouts"]) * len(spec["private_sites"])
    with np.load(data_path, allow_pickle=False) as archive: saved = {key: archive[key].copy() for key in archive.files}
    n = int(spec["world_count"]); require(saved["states"].shape[0] == n and saved["messages"].shape == (n, 2, 3, 4), "NPZ shape mismatch"); first_all = saved["messages"][:, 0]; natural_probs = saved["action_probabilities"]; live = condition.endswith("_live"); rows = []
    for case in cases:
        source_ids = lookup[tuple(case["source_needs"])] * nphysical + np.arange(nphysical, dtype=np.int64); target_ids = lookup[tuple(case["target_needs"])] * nphysical + np.arange(nphysical, dtype=np.int64); require(np.array_equal(saved["states"][source_ids, 3:], saved["states"][target_ids, 3:]), "Background mismatch")
        target_first = first_all[target_ids]; source_sender = first_all[source_ids, case["changed_person"]]; natural = natural_probs[target_ids, case["listener"]]
        if live:
            all_intervened = primary._action_probs_after_transfer(networks, arrays["x_PI"][target_ids], target_first, source_sender, case["changed_person"], True); intervened = all_intervened[:, case["listener"], :]; forwards = 6 * nphysical; alias = False
        else:
            intervened = natural.copy(); forwards = 0; alias = True
        values = primary._case_metrics(intervened, natural, case["source_action_set"], case["target_action_set"])
        rows.append(dict(**case, checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(source_result), data_sha256=sha(data_path), target_indices_sha256=array_sha(target_ids), source_indices_sha256=array_sha(source_ids), target_first_messages_sha256=array_sha(target_first), source_sender_messages_sha256=array_sha(source_sender), trained_channel=condition, alias_of_natural=alias, model_forward_samples=forwards, intervention="aligned source endpoint first-window message visible to target cross-viewers; target self-view retained; W2/action recomputed for PI-live", **values))
    return dict(seed=seed, condition=condition, trained_channel=condition, checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(source_result), data_sha256=sha(data_path), target_first_messages_sha256=array_sha(first_all), policy_rows=rows, alias_of_natural=not live)


def worker(payload):
    source, prepared, seed, condition = payload; return run_policy(source, prepared, int(seed), condition, primary._eligible_cases(prepared["candidate"]))


def execute(out, workers=4):
    out = Path(out).resolve(); _, prepared = verify(out); execution = out / "execution"; require(not execution.exists(), "Never overwrite execution"); execution.mkdir(); started = time.perf_counter(); tasks = [(prepared["source"], prepared, seed, condition) for seed in SEEDS for condition in CONDITIONS]
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool: policies = pool.map(worker, tasks)
    require(len(policies) == 8 and all(len(policy["policy_rows"]) == 240 for policy in policies), "Incomplete policy grid"); rows = [row for policy in policies for row in policy["policy_rows"]]; require(len(rows) == 1920 and sum(not row["alias_of_natural"] for row in rows) == 960, "Incomplete rows")
    result = dict(status="completed_json_only_semantic_transfer_confirmation_probe", completed_at=core.base.now(), elapsed_seconds=time.perf_counter() - started, source=prepared["source"], candidate=prepared["candidate"], partition=PARTITION, seeds=list(SEEDS), conditions=list(CONDITIONS), policy_blocks=8, cases_per_policy=240, intervention_rows=1920, live_rows=960, alias_rows=960, backgrounds_per_case=36, worlds=69120, model_forward_samples=sum(row["model_forward_samples"] for row in rows), optimizer_updates=0, training_updates=0, model_calls=0, axis_case_counts={axis: sum(row["axis"] == axis for row in rows) for axis in AXES}, policies=policies, interpretation_boundary="Aligned message-transfer readout for independent confirmation policies; not evidence of lexical meaning or language origin.")
    write_new(execution / "results.json", result); result_sha = sha(execution / "results.json"); write_new(execution / "status.json", dict(status="completed", results_sha256=result_sha, completed_at=core.base.now(), elapsed_seconds=result["elapsed_seconds"])); write_new(execution / "receipt.json", dict(status="passed", results_sha256=result_sha, policy_blocks=8, intervention_rows=1920, live_rows=960, alias_rows=960, worlds=69120, model_forward_samples=result["model_forward_samples"], optimizer_updates=0)); return dict(status=result["status"], results_sha256=result_sha, rows=1920)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("command", choices=("prepare", "verify", "execute")); parser.add_argument("--out", required=True); parser.add_argument("--source"); parser.add_argument("--candidate"); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args()
    if args.command == "prepare": require(args.source and args.candidate, "prepare requires --source and --candidate"); value = prepare(args.out, args.source, args.candidate)
    elif args.command == "verify": verify(args.out); value = dict(status="verified", output=str(Path(args.out).resolve()))
    else: value = execute(args.out, args.workers)
    print(json.dumps(value, ensure_ascii=False))
