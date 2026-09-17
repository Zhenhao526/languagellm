"""Paired misaligned-message placebo for the independent confirmation probe.

Each placebo donor has the same sender, listener, semantic axis and direction
as the target case, but its source message is cyclically assigned from another
case in that stratum.  The producer recomputes both the aligned intervention
and the misaligned intervention from the frozen policy; the independent audit
replays both without importing this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import time
from collections import defaultdict
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_partner_ecology_study import semantic_transfer_probe as primary


SEEDS = (49301, 49302, 49303, 49304)
CONDITIONS = primary.CONDITIONS
AXES = primary.AXES
PARTITION = primary.PARTITION
METRICS = primary.METRICS


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
    value = np.asarray(value)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def placebo_mapping(cases):
    groups = defaultdict(list)
    for index, case in enumerate(cases):
        groups[(case["axis"], case["changed_person"], case["listener"], case["direction"])].append(index)
    mapping = list(range(len(cases)))
    for indices in groups.values():
        require(len(indices) >= 2, "Placebo stratum too small")
        for position, index in enumerate(indices):
            mapping[index] = indices[(position + 1) % len(indices)]
    require(all(mapping[index] != index for index in range(len(cases))), "Placebo must be misaligned")
    return mapping, {"groups": len(groups), "group_sizes": {str(key): len(value) for key, value in sorted(groups.items(), key=lambda item: str(item[0]))}}


def prepare(out, primary_probe):
    out = Path(out).resolve()
    primary_probe = Path(primary_probe).resolve()
    require(not out.exists(), "Never overwrite preparation")
    require((primary_probe / "prepared.json").is_file() and (primary_probe / "execution/results.json").is_file(), "Missing primary probe")
    primary_prepared = json.loads((primary_probe / "prepared.json").read_text(encoding="utf8"))
    cases = primary._eligible_cases(primary_prepared["candidate"])
    mapping, mapping_meta = placebo_mapping(cases)
    manifest = dict(
        schema="triadic_unique_fixed_role_semantic_transfer_placebo_manifest_v1",
        primary_probe=str(primary_probe),
        source=primary_prepared["source"],
        candidate=primary_prepared["candidate"],
        partition=PARTITION,
        conditions=list(CONDITIONS),
        cases=len(cases),
        directed_cases=len(cases),
        axis_case_counts={axis: sum(case["axis"] == axis for case in cases) for axis in AXES},
        backgrounds_per_case=36,
        placebo="cyclically misalign source messages within axis×sender×listener×direction strata; no donor case is itself",
        placebo_mapping=mapping,
        placebo_mapping_meta=mapping_meta,
        intervention="aligned source endpoint first-window message and stratum-misaligned source endpoint first-window message, each shown to target cross-viewers",
        controls="PI-silent rows are exact natural aliases; no new training",
    )
    out.mkdir(parents=True)
    write_new(out / "manifest.json", manifest)
    prepared = dict(
        schema="triadic_unique_fixed_role_semantic_transfer_placebo_probe_v1",
        primary_probe=str(primary_probe),
        source=primary_prepared["source"],
        candidate=primary_prepared["candidate"],
        primary_results_sha256=sha(primary_probe / "execution/results.json"),
        source_prepared_sha256=sha(Path(primary_prepared["source"]) / "prepared.json"),
        source_freeze_sha256=sha(Path(primary_prepared["source"]) / "freeze.json"),
        source_results_sha256=sha(Path(primary_prepared["source"]) / "execution/results.json"),
        candidate_sha256=sha(primary_prepared["candidate"]),
        manifest_sha256=sha(out / "manifest.json"),
        partition=PARTITION,
        conditions=list(CONDITIONS),
        cases=len(cases),
        backgrounds_per_case=36,
        axis_case_counts=manifest["axis_case_counts"],
        placebo_mapping_sha256=array_sha(np.asarray(mapping, dtype=np.int64)),
        no_training=True,
        no_optimizer_updates=True,
        no_external_model=True,
    )
    write_new(out / "prepared.json", prepared)
    plan = dict(
        status="prepared_without_policy_reads",
        created_at=core.base.now(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        source_sha256={
            "primary_probe_results.json": prepared["primary_results_sha256"],
            "source_prepared.json": prepared["source_prepared_sha256"],
            "source_freeze.json": prepared["source_freeze_sha256"],
            "source_execution_results.json": prepared["source_results_sha256"],
            "candidate.json": prepared["candidate_sha256"],
            "manifest.json": sha(out / "manifest.json"),
            "prepared.json": sha(out / "prepared.json"),
        },
        prepared_sha256=sha(out / "prepared.json"),
        placebo_mapping_sha256=prepared["placebo_mapping_sha256"],
        no_training=True,
        no_optimizer_updates=True,
    )
    write_new(out / "plan.json", plan)
    write_new(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json")))
    write_new(out / "receipt.json", dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), cases=len(cases), model_calls=0, optimizer_updates=0))
    return dict(status="prepared_static_only", output=str(out), cases=len(cases))


def verify(out):
    out = Path(out).resolve()
    plan = json.loads((out / "plan.json").read_text(encoding="utf8"))
    prepared = json.loads((out / "prepared.json").read_text(encoding="utf8"))
    freeze = json.loads((out / "freeze.json").read_text(encoding="utf8"))
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan hash mismatch")
    require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared hash mismatch")
    require(prepared["manifest_sha256"] == sha(out / "manifest.json"), "Manifest hash mismatch")
    require(plan["source_sha256"]["manifest.json"] == sha(out / "manifest.json"), "Manifest changed")
    require(plan["source_sha256"]["prepared.json"] == sha(out / "prepared.json"), "Prepared changed")
    source = Path(prepared["source"])
    primary_probe = Path(prepared["primary_probe"])
    require(sha(primary_probe / "execution/results.json") == prepared["primary_results_sha256"], "Primary results changed")
    require(sha(source / "prepared.json") == prepared["source_prepared_sha256"], "Source prepared changed")
    require(sha(source / "freeze.json") == prepared["source_freeze_sha256"], "Source freeze changed")
    require(sha(source / "execution/results.json") == prepared["source_results_sha256"], "Source results changed")
    require(sha(prepared["candidate"]) == prepared["candidate_sha256"], "Candidate changed")
    return plan, prepared


def run_policy(source, prepared, seed, condition, cases, mapping):
    source = Path(source)
    condition_dir = source / "execution" / f"seed_{seed}_unique_{condition}"
    checkpoint = condition_dir / "checkpoint_6000.npz"
    result_path = condition_dir / "result.json"
    data_path = condition_dir / f"final_{PARTITION}_natural.npz"
    require(checkpoint.is_file() and result_path.is_file() and data_path.is_file(), "Missing source policy files")
    networks = core.load_networks(checkpoint)
    static = json.loads((source / "prepared.json").read_text(encoding="utf8"))
    spec = static["partitions"]["unique"][PARTITION]
    arrays = core.build_arrays(spec)
    need_lookup = {tuple(need): index for index, need in enumerate(spec["needs"])}
    nphysical = len(spec["layouts"]) * len(spec["private_sites"])
    with np.load(data_path, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    n = int(spec["world_count"])
    require(saved["states"].shape[0] == n and saved["messages"].shape[1:] == (2, 3, 4), "Source NPZ shape mismatch")
    first_all = saved["messages"][:, 0]
    natural_probs_all = saved["action_probabilities"]
    live = condition.endswith("_live")
    rows = []
    for index, case in enumerate(cases):
        donor_case = cases[mapping[index]]
        source_need_index = need_lookup[tuple(case["source_needs"])]
        target_need_index = need_lookup[tuple(case["target_needs"])]
        donor_need_index = need_lookup[tuple(donor_case["source_needs"])]
        backgrounds = np.arange(nphysical, dtype=np.int64)
        source_indices = source_need_index * nphysical + backgrounds
        target_indices = target_need_index * nphysical + backgrounds
        donor_indices = donor_need_index * nphysical + backgrounds
        require(np.array_equal(saved["states"][source_indices, 3:], saved["states"][target_indices, 3:]), "Endpoint backgrounds differ")
        target_first = first_all[target_indices]
        source_sender = first_all[source_indices, case["changed_person"], :]
        placebo_sender = first_all[donor_indices, case["changed_person"], :]
        natural_listener = natural_probs_all[target_indices, case["listener"], :]
        if live:
            aligned_all = primary._action_probs_after_transfer(networks, arrays["x_PI"][target_indices], target_first, source_sender, case["changed_person"], True)
            placebo_all = primary._action_probs_after_transfer(networks, arrays["x_PI"][target_indices], target_first, placebo_sender, case["changed_person"], True)
            aligned = aligned_all[:, case["listener"], :]
            placebo = placebo_all[:, case["listener"], :]
            forwards = 12 * nphysical
        else:
            aligned = natural_listener.copy()
            placebo = natural_listener.copy()
            forwards = 0
        aligned_values = primary._case_metrics(aligned, natural_listener, case["source_action_set"], case["target_action_set"])
        placebo_values = primary._case_metrics(placebo, natural_listener, case["source_action_set"], case["target_action_set"])
        rows.append(dict(**case,
            placebo_donor_case_id=donor_case["case_id"], placebo_donor_source_needs=donor_case["source_needs"],
            checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(result_path), data_sha256=sha(data_path),
            target_indices_sha256=array_sha(target_indices), source_indices_sha256=array_sha(source_indices), donor_indices_sha256=array_sha(donor_indices),
            target_first_messages_sha256=array_sha(target_first), source_sender_messages_sha256=array_sha(source_sender), placebo_sender_messages_sha256=array_sha(placebo_sender),
            trained_channel=condition, alias_of_natural=not live, model_forward_samples=forwards,
            intervention="aligned source endpoint first-window message or cyclically misaligned stratum donor, visible to target cross-viewers; target self-view retained",
            aligned=aligned_values, placebo=placebo_values,
        ))
    return dict(seed=seed, condition=condition, trained_channel=condition, checkpoint_sha256=sha(checkpoint), source_result_sha256=sha(result_path), data_sha256=sha(data_path), target_first_messages_sha256=array_sha(first_all), policy_rows=rows, alias_of_natural=not live)


def worker(payload):
    source, prepared, seed, condition = payload
    cases = primary._eligible_cases(prepared["candidate"])
    mapping, _ = placebo_mapping(cases)
    return run_policy(source, prepared, int(seed), condition, cases, mapping)


def execute(out, workers=4):
    out = Path(out).resolve()
    _, prepared = verify(out)
    execution = out / "execution"
    require(not execution.exists(), "Never overwrite execution")
    execution.mkdir()
    started = time.perf_counter()
    tasks = [(prepared["source"], prepared, seed, condition) for seed in SEEDS for condition in CONDITIONS]
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool:
        policies = pool.map(worker, tasks)
    require(len(policies) == 8 and all(len(policy["policy_rows"]) == 240 for policy in policies), "Incomplete policy grid")
    rows = [row for policy in policies for row in policy["policy_rows"]]
    require(len(rows) == 1920 and sum(not row["alias_of_natural"] for row in rows) == 960, "Incomplete row grid")
    result = dict(
        status="completed_json_only_semantic_transfer_confirmation_placebo_probe",
        completed_at=core.base.now(), elapsed_seconds=time.perf_counter() - started,
        source=prepared["source"], primary_probe=prepared["primary_probe"], candidate=prepared["candidate"], partition=PARTITION,
        conditions=list(CONDITIONS), seeds=list(SEEDS), policy_blocks=8, cases_per_policy=240, intervention_rows=len(rows), live_rows=960, alias_rows=960,
        backgrounds_per_case=36, worlds=len(rows) * 36, model_forward_samples=sum(row["model_forward_samples"] for row in rows), optimizer_updates=0, training_updates=0, model_calls=0,
        axis_case_counts={axis: sum(row["axis"] == axis for row in rows) for axis in AXES}, policies=policies,
        interpretation_boundary="Aligned-versus-stratum-misaligned source-message contrast for fixed task policies; not evidence of lexical meaning or language origin.",
    )
    write_new(execution / "results.json", result)
    result_sha = sha(execution / "results.json")
    write_new(execution / "status.json", dict(status="completed", results_sha256=result_sha, completed_at=core.base.now(), elapsed_seconds=result["elapsed_seconds"]))
    write_new(execution / "receipt.json", dict(status="passed", results_sha256=result_sha, policy_blocks=8, intervention_rows=1920, live_rows=960, alias_rows=960, worlds=result["worlds"], model_forward_samples=result["model_forward_samples"], optimizer_updates=0))
    return dict(status=result["status"], results_sha256=result_sha, rows=len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--primary-probe")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.primary_probe, "prepare requires --primary-probe")
        value = prepare(args.out, args.primary_probe)
    elif args.command == "verify":
        verify(args.out)
        value = dict(status="verified", output=str(Path(args.out).resolve()))
    else:
        value = execute(args.out, args.workers)
    print(json.dumps(value, ensure_ascii=False))
