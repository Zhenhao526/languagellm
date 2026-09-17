"""Post-hoc directional message-transfer probe on fixed-role unique cases.

The source and target worlds differ in exactly one sender's private need.  The
listener's researcher-defined full-success action sets are disjoint while both
endpoints keep the same participation role.  A source first-window message is
therefore transplanted to the target world for cross-viewers only; the sender's
self-view remains the target message.  The probe measures movement toward the
source action set, with PI-silent aliases as a closed-route control.
"""
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

from research_program.triadic_message_study import runner as message_runner
from research_program.triadic_partner_ecology_study import runner as partner_runner


SEEDS = (49101, 49102, 49103, 49104)
CONDITIONS = ("PI_live", "PI_silent")
AXES = ("kind_wood_fiber", "length_short_long", "destination_L_R")
PARTITION = "heldout_layouts"
METRICS = ("target_set_mass", "source_set_mass", "target_set_hit", "source_set_hit", "action_change_rate")


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


def _eligible_cases(candidate_path):
    rows = json.loads(Path(candidate_path).read_text(encoding="utf8"))
    result = []
    for row in rows:
        if row["ecology"] != "unique":
            continue
        for listener_row in row["listeners"]:
            if not listener_row["full_success_action_sets_disjoint"]:
                continue
            if listener_row["unique_participation_wait_switch"]:
                continue
            listener = ("ABC").index(listener_row["listener"])
            sender = ("ABC").index(row["changed_person"])
            require(listener != sender, "Listener cannot be changed sender")
            for direction, source_key, target_key, source_set_key, target_set_key in (
                ("before_to_after", "needs_before", "needs_after", "successful_actions_before", "successful_actions_after"),
                ("after_to_before", "needs_after", "needs_before", "successful_actions_after", "successful_actions_before"),
            ):
                result.append(
                    dict(
                        case_id=f"{row['pair_id']}__{direction}__listener_{listener}",
                        pair_id=row["pair_id"],
                        axis=row["axis"],
                        changed_person=sender,
                        listener=listener,
                        direction=direction,
                        source_needs=list(row[source_key]),
                        target_needs=list(row[target_key]),
                        source_action_set=list(listener_row[source_set_key]),
                        target_action_set=list(listener_row[target_set_key]),
                        source_role=listener_row["role_before"] if direction == "before_to_after" else listener_row["role_after"],
                        target_role=listener_row["role_after"] if direction == "before_to_after" else listener_row["role_before"],
                    )
                )
    result.sort(key=lambda x: (AXES.index(x["axis"]), tuple(x["source_needs"]), tuple(x["target_needs"]), x["changed_person"], x["listener"], x["direction"]))
    require(len(result) == 240, "Expected 120 fixed-role cases in both directions")
    require({x["axis"] for x in result} == set(AXES), "All semantic axes required")
    require(all(x["source_role"] == x["target_role"] == "always_participate" for x in result), "Role confound in candidate set")
    require(all(set(x["source_action_set"]).isdisjoint(x["target_action_set"]) for x in result), "Action sets must be disjoint")
    return result


def prepare(out, source, candidate):
    out = Path(out).resolve()
    source = Path(source).resolve()
    candidate = Path(candidate).resolve()
    require(not out.exists(), "Never overwrite preparation")
    require((source / "prepared.json").is_file() and (source / "freeze.json").is_file(), "Missing source freeze")
    require((source / "execution/results.json").is_file(), "Missing completed source results")
    cases = _eligible_cases(candidate)
    counts = {axis: sum(x["axis"] == axis for x in cases) for axis in AXES}
    require(counts == {"kind_wood_fiber": 48, "length_short_long": 48, "destination_L_R": 144}, "Unexpected axis balance")
    manifest = dict(
        schema="triadic_unique_fixed_role_semantic_transfer_manifest_v1",
        source=str(source),
        candidate=str(candidate),
        partition=PARTITION,
        conditions=list(CONDITIONS),
        cases=len(cases),
        directed_cases=len(cases),
        axis_case_counts=counts,
        backgrounds_per_case=36,
        case_weighting="equal axis, then equal directed case, then equal heldout layout×owner background",
        eligibility="unique ecology; listener full-success action sets disjoint; no participation/wait switch; exactly one sender need axis changes",
        intervention="transplant selected sender's source endpoint first-window message to target cross-viewers; target self-view and all other sender messages remain natural",
        controls="PI-silent rows are exact closed-route aliases; no new training",
    )
    # Write the manifest first, then bind the prepared record to its exact
    # file hash. This keeps the static preparation chain auditable.
    out.mkdir(parents=True)
    write_new(out / "manifest.json", manifest)
    prepared = dict(
        schema="triadic_unique_fixed_role_semantic_transfer_probe_v1",
        source=str(source),
        candidate=str(candidate),
        source_prepared_sha256=sha(source / "prepared.json"),
        source_freeze_sha256=sha(source / "freeze.json"),
        source_results_sha256=sha(source / "execution/results.json"),
        candidate_sha256=sha(candidate),
        manifest_sha256=sha(out / "manifest.json"),
        partition=PARTITION,
        conditions=list(CONDITIONS),
        cases=len(cases),
        backgrounds_per_case=36,
        axis_case_counts=counts,
        no_training=True,
        no_optimizer_updates=True,
        no_external_model=True,
    )
    write_new(out / "prepared.json", prepared)
    plan = dict(
        status="prepared_without_policy_reads",
        created_at=message_runner.base.now(),
        runtime=dict(python=platform.python_version(), numpy=np.__version__),
        source_sha256={
            "source_prepared.json": prepared["source_prepared_sha256"],
            "source_freeze.json": prepared["source_freeze_sha256"],
            "source_execution_results.json": prepared["source_results_sha256"],
            "candidate.json": prepared["candidate_sha256"],
            "manifest.json": sha(out / "manifest.json"),
            "prepared.json": sha(out / "prepared.json"),
        },
        prepared_sha256=sha(out / "prepared.json"),
        no_training=True,
        no_optimizer_updates=True,
    )
    write_new(out / "plan.json", plan)
    write_new(out / "freeze.json", dict(plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json")))
    write_new(out / "receipt.json", dict(status="prepared_static_only", plan_sha256=sha(out / "plan.json"), prepared_sha256=sha(out / "prepared.json"), cases=len(cases), model_calls=0, optimizer_updates=0))
    return dict(status="prepared_static_only", output=str(out), cases=len(cases))


def verify(out):
    out = Path(out).resolve()
    plan = json.loads((out / "plan.json").read_text())
    prepared = json.loads((out / "prepared.json").read_text())
    freeze = json.loads((out / "freeze.json").read_text())
    require(sha(out / "plan.json") == freeze["plan_sha256"], "Plan hash mismatch")
    require(sha(out / "prepared.json") == freeze["prepared_sha256"] == plan["prepared_sha256"], "Prepared hash mismatch")
    require(plan["source_sha256"]["manifest.json"] == sha(out / "manifest.json"), "Manifest changed")
    require(prepared["manifest_sha256"] == sha(out / "manifest.json"), "Prepared manifest hash mismatch")
    require(plan["source_sha256"]["prepared.json"] == sha(out / "prepared.json"), "Prepared source hash changed")
    source = Path(prepared["source"])
    candidate = Path(prepared["candidate"])
    require(sha(source / "prepared.json") == prepared["source_prepared_sha256"], "Source prepared changed")
    require(sha(source / "freeze.json") == prepared["source_freeze_sha256"], "Source freeze changed")
    require(sha(source / "execution/results.json") == prepared["source_results_sha256"], "Source results changed")
    require(sha(candidate) == prepared["candidate_sha256"], "Candidate set changed")
    return plan, prepared


def _route_transfer(target_tokens, source_sender_tokens, sender, live):
    target_tokens = np.asarray(target_tokens)
    source_sender_tokens = np.asarray(source_sender_tokens)
    require(target_tokens.ndim == 3 and target_tokens.shape[1:] == (3, 4), "Invalid target tokens")
    require(source_sender_tokens.shape == (len(target_tokens), 4), "Invalid source tokens")
    visibility = np.ones((3, 3), dtype=np.float64) if live else np.eye(3, dtype=np.float64)
    onehot = np.eye(8, dtype=np.float64)[target_tokens]
    visible = onehot[:, None, :, :, :] * visibility[None, :, :, None, None]
    if live:
        source_onehot = np.eye(8, dtype=np.float64)[source_sender_tokens]
        for viewer in range(3):
            if viewer != int(sender):
                visible[:, viewer, int(sender)] = source_onehot
    bits = np.broadcast_to(visibility[None], (len(target_tokens), 3, 3)).copy()
    return np.concatenate((visible.reshape(len(target_tokens), 3, 96), bits), axis=-1)


def _action_probs_after_transfer(networks, x, target_first, source_sender_tokens, sender, live):
    routed_first = _route_transfer(target_first, source_sender_tokens, sender, live)
    second_input = np.concatenate((x, routed_first), axis=-1)
    second_logits = []
    for actor in range(3):
        z, _ = message_runner.base.actor_forward(networks[3 * actor + 1], second_input[:, actor])
        second_logits.append(z.reshape(len(x), 4, 8))
    second_probabilities, _ = message_runner.base.policy_distribution(np.stack(second_logits, axis=1))
    second_tokens = message_runner.categorical_tokens(second_probabilities)
    action_input = np.concatenate((x, routed_first, message_runner.routed_window(second_tokens, live)), axis=-1)
    action_logits = []
    for actor in range(3):
        z, _ = message_runner.base.actor_forward(networks[3 * actor + 2], action_input[:, actor])
        action_logits.append(z)
    action_probabilities, _ = message_runner.base.policy_distribution(np.stack(action_logits, axis=1))
    return action_probabilities


def _case_metrics(probabilities, natural_probabilities, source_set, target_set, natural_actions=None):
    source_set = np.asarray(source_set, dtype=np.int64)
    target_set = np.asarray(target_set, dtype=np.int64)
    natural_target = natural_probabilities[:, target_set].sum(axis=1)
    natural_source = natural_probabilities[:, source_set].sum(axis=1)
    intervention_target = probabilities[:, target_set].sum(axis=1)
    intervention_source = probabilities[:, source_set].sum(axis=1)
    natural_argmax = natural_probabilities.argmax(axis=1)
    intervention_argmax = probabilities.argmax(axis=1)
    n = len(probabilities)
    natural = dict(
        target_set_mass=float(natural_target.mean()), source_set_mass=float(natural_source.mean()),
        target_set_hit=float(np.isin(natural_argmax, target_set).mean()), source_set_hit=float(np.isin(natural_argmax, source_set).mean()),
        action_change_rate=0.0,
    )
    intervened = dict(
        target_set_mass=float(intervention_target.mean()), source_set_mass=float(intervention_source.mean()),
        target_set_hit=float(np.isin(intervention_argmax, target_set).mean()), source_set_hit=float(np.isin(intervention_argmax, source_set).mean()),
        action_change_rate=float((intervention_argmax != natural_argmax).mean()),
    )
    delta = {key: float(intervened[key] - natural[key]) for key in METRICS}
    source_pull = float((intervened["source_set_mass"] - natural["source_set_mass"]) - (intervened["target_set_mass"] - natural["target_set_mass"]))
    greedy_source_pull = float((intervened["source_set_hit"] - natural["source_set_hit"]) - (intervened["target_set_hit"] - natural["target_set_hit"]))
    return dict(natural=natural, intervened=intervened, delta=delta, source_pull=source_pull, greedy_source_pull=greedy_source_pull, worlds=n)


def run_policy(source, prepared, seed, condition, cases):
    source = Path(source)
    condition_dir = source / "execution" / f"seed_{seed}_unique_{condition}"
    checkpoint = condition_dir / "checkpoint_6000.npz"
    result_path = condition_dir / "result.json"
    data_path = condition_dir / f"final_{PARTITION}_natural.npz"
    require(checkpoint.is_file() and result_path.is_file() and data_path.is_file(), "Missing source policy files")
    networks = message_runner.load_networks(checkpoint)
    static = json.loads((source / "prepared.json").read_text())
    spec = static["partitions"]["unique"][PARTITION]
    arrays = message_runner.build_arrays(spec)
    need_lookup = {tuple(need): index for index, need in enumerate(spec["needs"])}
    nphysical = len(spec["layouts"]) * len(spec["private_sites"])
    with np.load(data_path, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    require(saved["states"].shape[0] == int(spec["world_count"]) and saved["messages"].shape[1:] == (2, 3, 4), "Source NPZ shape mismatch")
    target_first_all = saved["messages"][:, 0]
    natural_probs_all = saved["action_probabilities"]
    rows = []
    live = condition.endswith("_live")
    for case in cases:
        source_need_index = need_lookup[tuple(case["source_needs"])]
        target_need_index = need_lookup[tuple(case["target_needs"])]
        backgrounds = np.arange(nphysical, dtype=np.int64)
        source_indices = source_need_index * nphysical + backgrounds
        target_indices = target_need_index * nphysical + backgrounds
        require(np.array_equal(saved["states"][source_indices, 3:], saved["states"][target_indices, 3:]), "Endpoint backgrounds differ")
        target_first = target_first_all[target_indices]
        source_sender = target_first_all[source_indices, case["changed_person"], :]
        natural_listener = natural_probs_all[target_indices, case["listener"], :]
        if live:
            intervened_all = _action_probs_after_transfer(
                networks,
                arrays["x_PI"][target_indices],
                target_first,
                source_sender,
                case["changed_person"],
                True,
            )
            # The rollout returns all actors; select the listener column.
            intervened_listener = intervened_all[:, case["listener"], :]
            alias = False
            forwards = 6 * nphysical
        else:
            # The transfer targets cross-viewer blocks only.  PI-silent has no
            # such block, so the listener's natural action distribution is an
            # exact alias and requires no forward pass.
            intervened_listener = natural_listener.copy()
            alias = True
            forwards = 0
        values = _case_metrics(intervened_listener, natural_listener, case["source_action_set"], case["target_action_set"])
        rows.append(
            dict(
                **case,
                checkpoint_sha256=sha(checkpoint),
                source_result_sha256=sha(result_path),
                data_sha256=sha(data_path),
                target_indices_sha256=array_sha(target_indices),
                source_indices_sha256=array_sha(source_indices),
                target_first_messages_sha256=array_sha(target_first),
                source_sender_messages_sha256=array_sha(source_sender),
                trained_channel=condition,
                alias_of_natural=alias,
                model_forward_samples=forwards,
                intervention="source endpoint first-window message visible to target cross-viewers; target self-view retained; W2/action recomputed for PI-live",
                **values,
            )
        )
    return dict(
        seed=seed,
        condition=condition,
        trained_channel=condition,
        checkpoint_sha256=sha(checkpoint),
        source_result_sha256=sha(result_path),
        data_sha256=sha(data_path),
        target_first_messages_sha256=array_sha(target_first_all),
        policy_rows=rows,
        alias_of_natural=not live,
    )


def worker(payload):
    source, prepared, seed, condition = payload
    cases = _eligible_cases(prepared["candidate"])
    return run_policy(source, prepared, int(seed), condition, cases)


def execute(out, workers=4):
    out = Path(out).resolve()
    plan, prepared = verify(out)
    execution = out / "execution"
    require(not execution.exists(), "Never overwrite execution")
    execution.mkdir()
    tasks = [(prepared["source"], prepared, seed, condition) for seed in SEEDS for condition in CONDITIONS]
    started = time.perf_counter()
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool:
        policies = pool.map(worker, tasks)
    require(len(policies) == 8 and all(len(p["policy_rows"]) == 240 for p in policies), "Complete policy grid required")
    rows = [row for policy in policies for row in policy["policy_rows"]]
    require(len(rows) == 1920 and sum(not row["alias_of_natural"] for row in rows) == 960, "Complete row grid required")
    result = dict(
        status="completed_json_only_semantic_transfer_probe",
        completed_at=message_runner.base.now(),
        elapsed_seconds=time.perf_counter() - started,
        source=prepared["source"],
        candidate=prepared["candidate"],
        partition=PARTITION,
        conditions=list(CONDITIONS),
        seeds=list(SEEDS),
        policy_blocks=len(policies),
        cases_per_policy=240,
        intervention_rows=len(rows),
        live_rows=sum(not row["alias_of_natural"] for row in rows),
        alias_rows=sum(row["alias_of_natural"] for row in rows),
        backgrounds_per_case=36,
        worlds=len(rows) * 36,
        model_forward_samples=sum(row["model_forward_samples"] for row in rows),
        optimizer_updates=0,
        training_updates=0,
        model_calls=0,
        axis_case_counts={axis: sum(row["axis"] == axis for row in rows) for axis in AXES},
        policies=policies,
        interpretation_boundary="Directional content-transfer sensitivity of fixed task policies; not evidence of lexical meaning or language origin.",
    )
    write_new(execution / "results.json", result)
    results_sha = sha(execution / "results.json")
    write_new(execution / "status.json", dict(status="completed", results_sha256=results_sha, completed_at=message_runner.base.now(), elapsed_seconds=result["elapsed_seconds"]))
    write_new(execution / "receipt.json", dict(status="passed", results_sha256=results_sha, policy_blocks=8, intervention_rows=1920, live_rows=960, alias_rows=960, worlds=result["worlds"], model_forward_samples=result["model_forward_samples"], optimizer_updates=0))
    return dict(status="completed_json_only_semantic_transfer_probe", results_sha256=results_sha, rows=len(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "verify", "execute"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--source")
    parser.add_argument("--candidate")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.command == "prepare":
        require(args.source and args.candidate, "prepare requires --source and --candidate")
        value = prepare(args.out, args.source, args.candidate)
    elif args.command == "verify":
        verify(args.out)
        value = dict(status="verified", output=str(Path(args.out).resolve()))
    else:
        value = execute(args.out, args.workers)
    print(json.dumps(value, ensure_ascii=False))
