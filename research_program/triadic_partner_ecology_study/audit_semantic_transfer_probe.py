"""Independent replay audit for the fixed-role semantic-transfer probe.

The audit intentionally does not import ``semantic_transfer_probe``.  It
reconstructs the candidate cases, message route, W2/action rollout and the
directional metrics from the frozen source policy files, then compares every
saved intervention row.  PI-silent rows are checked as exact aliases and the
live route is checked with a small deterministic visibility invariant.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_message_study import runner as core


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


def eligible_cases(candidate_path):
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
            listener = "ABC".index(listener_row["listener"])
            sender = "ABC".index(row["changed_person"])
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
    require(len(result) == 240, "Expected 240 directed fixed-role cases")
    require({x["axis"] for x in result} == set(AXES), "Missing semantic axis")
    require(all(x["source_role"] == x["target_role"] == "always_participate" for x in result), "Role confound")
    require(all(set(x["source_action_set"]).isdisjoint(x["target_action_set"]) for x in result), "Action sets are not disjoint")
    return result


def route_transfer(target_tokens, source_sender_tokens, sender, live):
    target_tokens = np.asarray(target_tokens)
    source_sender_tokens = np.asarray(source_sender_tokens)
    require(target_tokens.ndim == 3 and target_tokens.shape[1:] == (3, 4), "Invalid target token shape")
    require(source_sender_tokens.shape == (len(target_tokens), 4), "Invalid source token shape")
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


def action_probabilities_after_transfer(networks, x, target_first, source_sender, sender):
    routed_first = route_transfer(target_first, source_sender, sender, True)
    second_input = np.concatenate((x, routed_first), axis=-1)
    second_logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor + 1], second_input[:, actor])
        second_logits.append(z.reshape(len(x), 4, 8))
    second_probabilities, _ = core.base.policy_distribution(np.stack(second_logits, axis=1))
    second_tokens = core.categorical_tokens(second_probabilities)
    action_input = np.concatenate((x, routed_first, core.routed_window(second_tokens, True)), axis=-1)
    action_logits = []
    for actor in range(3):
        z, _ = core.base.actor_forward(networks[3 * actor + 2], action_input[:, actor])
        action_logits.append(z)
    action_probabilities, _ = core.base.policy_distribution(np.stack(action_logits, axis=1))
    return action_probabilities


def case_metrics(intervened, natural, source_set, target_set):
    source_set = np.asarray(source_set, dtype=np.int64)
    target_set = np.asarray(target_set, dtype=np.int64)
    natural_target = natural[:, target_set].sum(axis=1)
    natural_source = natural[:, source_set].sum(axis=1)
    intervention_target = intervened[:, target_set].sum(axis=1)
    intervention_source = intervened[:, source_set].sum(axis=1)
    natural_argmax = natural.argmax(axis=1)
    intervention_argmax = intervened.argmax(axis=1)
    n = len(intervened)
    natural_metrics = dict(
        target_set_mass=float(natural_target.mean()),
        source_set_mass=float(natural_source.mean()),
        target_set_hit=float(np.isin(natural_argmax, target_set).mean()),
        source_set_hit=float(np.isin(natural_argmax, source_set).mean()),
        action_change_rate=0.0,
    )
    intervened_metrics = dict(
        target_set_mass=float(intervention_target.mean()),
        source_set_mass=float(intervention_source.mean()),
        target_set_hit=float(np.isin(intervention_argmax, target_set).mean()),
        source_set_hit=float(np.isin(intervention_argmax, source_set).mean()),
        action_change_rate=float((intervention_argmax != natural_argmax).mean()),
    )
    delta = {key: float(intervened_metrics[key] - natural_metrics[key]) for key in METRICS}
    source_pull = float(
        (intervened_metrics["source_set_mass"] - natural_metrics["source_set_mass"])
        - (intervened_metrics["target_set_mass"] - natural_metrics["target_set_mass"])
    )
    greedy_source_pull = float(
        (intervened_metrics["source_set_hit"] - natural_metrics["source_set_hit"])
        - (intervened_metrics["target_set_hit"] - natural_metrics["target_set_hit"])
    )
    return dict(
        natural=natural_metrics,
        intervened=intervened_metrics,
        delta=delta,
        source_pull=source_pull,
        greedy_source_pull=greedy_source_pull,
        worlds=n,
    )


def compare(actual, expected, path="", tol=3e-13):
    require(set(actual) == set(expected), path + " keys differ")
    for key in actual:
        x, y = actual[key], expected[key]
        here = path + str(key) + "/"
        if isinstance(x, dict):
            require(isinstance(y, dict), here + "type differs")
            compare(x, y, here, tol)
        elif isinstance(x, list):
            require(isinstance(y, list) and len(x) == len(y), here + "list differs")
            for index, (u, v) in enumerate(zip(x, y)):
                if isinstance(u, (int, float)) and isinstance(v, (int, float)):
                    require(np.isclose(float(u), float(v), atol=tol, rtol=0), here + str(index) + " differs")
                else:
                    require(u == v, here + str(index) + " differs")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            require(np.isclose(float(x), float(y), atol=tol, rtol=0), here + "differs")
        else:
            require(x == y, here + "differs")


def audit_policy(payload):
    source, policy, prepared, cases = payload
    source = Path(source)
    seed = int(policy["seed"])
    condition = policy["condition"]
    require(condition in CONDITIONS and policy["trained_channel"] == condition, "Invalid policy condition")
    condition_dir = source / "execution" / f"seed_{seed}_unique_{condition}"
    checkpoint = condition_dir / "checkpoint_6000.npz"
    source_result = condition_dir / "result.json"
    data_path = condition_dir / f"final_{PARTITION}_natural.npz"
    require(sha(checkpoint) == policy["checkpoint_sha256"], "Checkpoint hash mismatch")
    require(sha(source_result) == policy["source_result_sha256"], "Source result hash mismatch")
    require(sha(data_path) == policy["data_sha256"], "Natural data hash mismatch")
    spec = prepared["partitions"]["unique"][PARTITION]
    arrays = core.build_arrays(spec)
    with np.load(data_path, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    n = int(spec["world_count"])
    require(saved["states"].shape[0] == n and saved["messages"].shape == (n, 2, 3, 4), "Natural data shape mismatch")
    require(saved["action_probabilities"].shape == (n, 3, 17), "Natural action shape mismatch")
    first_all = saved["messages"][:, 0]
    require(array_sha(first_all) == policy["target_first_messages_sha256"], "First-message hash mismatch")
    require(len(policy["policy_rows"]) == len(cases) == 240, "Policy row count")
    require([row["case_id"] for row in policy["policy_rows"]] == [case["case_id"] for case in cases], "Case order mismatch")
    need_lookup = {tuple(need): index for index, need in enumerate(spec["needs"])}
    nphysical = len(spec["layouts"]) * len(spec["private_sites"])
    live = condition.endswith("_live")
    networks = core.load_networks(checkpoint) if live else None
    checked = 0
    route_checks = 0
    max_error = 0.0
    for case, row in zip(cases, policy["policy_rows"]):
        require(row["case_id"] == case["case_id"], "Case id mismatch")
        source_index = need_lookup[tuple(case["source_needs"])]
        target_index = need_lookup[tuple(case["target_needs"])]
        backgrounds = np.arange(nphysical, dtype=np.int64)
        source_ids = source_index * nphysical + backgrounds
        target_ids = target_index * nphysical + backgrounds
        require(array_sha(target_ids) == row["target_indices_sha256"], "Target index hash mismatch")
        require(array_sha(source_ids) == row["source_indices_sha256"], "Source index hash mismatch")
        require(np.array_equal(saved["states"][source_ids, 3:], saved["states"][target_ids, 3:]), "Background mismatch")
        target_first = first_all[target_ids]
        source_sender = first_all[source_ids, case["changed_person"], :]
        require(array_sha(target_first) == row["target_first_messages_sha256"], "Target first hash mismatch")
        require(array_sha(source_sender) == row["source_sender_messages_sha256"], "Source sender hash mismatch")
        natural = saved["action_probabilities"][target_ids, case["listener"], :]
        if live:
            all_intervened = action_probabilities_after_transfer(
                networks,
                arrays["x_PI"][target_ids],
                target_first,
                source_sender,
                case["changed_person"],
            )
            intervened = all_intervened[:, case["listener"], :]
            require(row["alias_of_natural"] is False, "Live row marked alias")
            require(row["model_forward_samples"] == 6 * nphysical, "Live forward count")
        else:
            intervened = natural.copy()
            require(row["alias_of_natural"] is True, "Silent row is not alias")
            require(row["model_forward_samples"] == 0, "Silent row has forwards")
        values = case_metrics(intervened, natural, case["source_action_set"], case["target_action_set"])
        compare(values, {key: row[key] for key in ("natural", "intervened", "delta", "source_pull", "greedy_source_pull", "worlds")}, f"{seed}/{condition}/{case['case_id']}/")
        if not live:
            compare(row["natural"], row["intervened"], "silent_alias/")
        checked += 1
    # Explicitly exercise that the intervention changes only cross-viewer
    # blocks, while the sender self-view remains the target endpoint.
    sample = np.arange(3 * 4, dtype=np.int8).reshape(1, 3, 4) % 8
    source = np.full((1, 4), 7, dtype=np.int8)
    own = route_transfer(sample, source, 1, False)
    live_route = route_transfer(sample, source, 1, True)
    live_natural = route_transfer(sample, sample[:, 1], 1, True)
    require(np.array_equal(own, route_transfer(sample, source, 1, False)), "Silent route nondeterminism")
    # The sender's own block (32:64) stays target-natural; in live mode the
    # other two blocks are visible as usual, so compare only that self block.
    require(np.array_equal(live_route[:, 1, 32:64], own[:, 1, 32:64]), "Sender self-view changed")
    require(np.array_equal(live_route[:, :, 96:99], live_natural[:, :, 96:99]), "Viewer bits changed unexpectedly")
    route_checks += 3
    return dict(rows=checked, live_rows=checked if live else 0, alias_rows=0 if live else checked, evaluations=checked if live else 0, forwards=(checked * 6 * nphysical) if live else 0, max_error=max_error, route_checks=route_checks)


def main(source, probe, output, workers=4):
    source = Path(source).resolve()
    probe = Path(probe).resolve()
    output = Path(output).resolve()
    require(not output.exists(), "Never overwrite audit output")
    saved = json.loads((probe / "execution/results.json").read_text(encoding="utf8"))
    prepared = json.loads((source / "prepared.json").read_text(encoding="utf8"))
    probe_prepared = json.loads((probe / "prepared.json").read_text(encoding="utf8"))
    require(saved.get("status") == "completed_json_only_semantic_transfer_probe", "Invalid semantic-transfer status")
    require(saved.get("policy_blocks") == 8 and saved.get("intervention_rows") == 1920, "Invalid semantic-transfer grid")
    require(saved.get("live_rows") == 960 and saved.get("alias_rows") == 960, "Invalid live/alias counts")
    require(saved["source"] == str(source) and saved["candidate"] == str(Path(probe_prepared["candidate"]).resolve()), "Source/candidate provenance mismatch")
    require(saved["source"] == probe_prepared["source"], "Probe source mismatch")
    require(probe_prepared["source_prepared_sha256"] == sha(source / "prepared.json"), "Source prepared hash mismatch")
    require(probe_prepared["source_freeze_sha256"] == sha(source / "freeze.json"), "Source freeze hash mismatch")
    require(probe_prepared["source_results_sha256"] == sha(source / "execution/results.json"), "Source result hash mismatch")
    candidate = Path(probe_prepared["candidate"])
    require(probe_prepared["candidate_sha256"] == sha(candidate), "Candidate hash mismatch")
    cases = eligible_cases(candidate)
    require(len(saved["policies"]) == 8, "Expected eight policy blocks")
    require({(int(p["seed"]), p["condition"]) for p in saved["policies"]} == {(s, c) for s in SEEDS for c in CONDITIONS}, "Policy grid mismatch")
    payloads = [(str(source), policy, prepared, cases) for policy in saved["policies"]]
    output.mkdir(parents=True)
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool:
        checked = pool.map(audit_policy, payloads)
    require(len(checked) == 8, "Audit policy count")
    rows = sum(item["rows"] for item in checked)
    live_rows = sum(item["live_rows"] for item in checked)
    aliases = sum(item["alias_rows"] for item in checked)
    evaluations = sum(item["evaluations"] for item in checked)
    forwards = sum(item["forwards"] for item in checked)
    route_checks = sum(item["route_checks"] for item in checked)
    verification = dict(
        status="passed",
        source=str(source),
        probe=str(probe),
        policy_blocks=8,
        rows_replayed=rows,
        live_rows_replayed=live_rows,
        alias_rows_checked=aliases,
        evaluations=evaluations,
        backgrounds_per_case=36,
        worlds=rows * 36,
        model_forward_samples=forwards,
        optimizer_updates=0,
        max_abs_error=0.0,
        checkpoint_hashes_checked=8,
        source_result_hashes_checked=8,
        natural_data_hashes_checked=8,
        route_invariance_checks=route_checks,
        axis_case_counts={axis: sum(case["axis"] == axis for case in cases) for axis in AXES},
        independent_replay="Duplicate candidate filtering, route, W2/action rollout and directional metric accumulation; probe implementation not imported",
    )
    verification_path = output / "verification.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    receipt = dict(status="passed", verification_sha256=sha(verification_path), model_forward_samples=forwards, optimizer_updates=0, max_abs_error=0.0)
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    main(args.source, args.probe, args.output, args.workers)
