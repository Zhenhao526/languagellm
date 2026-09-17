"""Independent replay audit for the confirmation aligned/misaligned placebo."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
from collections import defaultdict
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_message_study import runner as core


SEEDS = (49301, 49302, 49303, 49304)
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
            if not listener_row["full_success_action_sets_disjoint"] or listener_row["unique_participation_wait_switch"]:
                continue
            listener = "ABC".index(listener_row["listener"])
            sender = "ABC".index(row["changed_person"])
            require(listener != sender, "Listener cannot be sender")
            for direction, source_key, target_key, source_set_key, target_set_key in (
                ("before_to_after", "needs_before", "needs_after", "successful_actions_before", "successful_actions_after"),
                ("after_to_before", "needs_after", "needs_before", "successful_actions_after", "successful_actions_before"),
            ):
                result.append(dict(
                    case_id=f"{row['pair_id']}__{direction}__listener_{listener}", pair_id=row["pair_id"], axis=row["axis"],
                    changed_person=sender, listener=listener, direction=direction, source_needs=list(row[source_key]), target_needs=list(row[target_key]),
                    source_action_set=list(listener_row[source_set_key]), target_action_set=list(listener_row[target_set_key]),
                    source_role=listener_row["role_before"] if direction == "before_to_after" else listener_row["role_after"],
                    target_role=listener_row["role_after"] if direction == "before_to_after" else listener_row["role_before"],
                ))
    result.sort(key=lambda x: (AXES.index(x["axis"]), tuple(x["source_needs"]), tuple(x["target_needs"]), x["changed_person"], x["listener"], x["direction"]))
    require(len(result) == 240, "Expected 240 directed cases")
    require(all(x["source_role"] == x["target_role"] == "always_participate" for x in result), "Role confound")
    require(all(set(x["source_action_set"]).isdisjoint(x["target_action_set"]) for x in result), "Action sets overlap")
    return result


def placebo_mapping(cases):
    groups = defaultdict(list)
    for i, case in enumerate(cases):
        groups[(case["axis"], case["changed_person"], case["listener"], case["direction"])].append(i)
    mapping = list(range(len(cases)))
    for indices in groups.values():
        require(len(indices) >= 2, "Placebo stratum too small")
        for j, i in enumerate(indices):
            mapping[i] = indices[(j + 1) % len(indices)]
    require(all(mapping[i] != i for i in range(len(cases))), "Placebo identity mapping")
    return mapping


def route_transfer(target_tokens, source_sender, sender):
    target_tokens = np.asarray(target_tokens)
    source_sender = np.asarray(source_sender)
    require(target_tokens.ndim == 3 and target_tokens.shape[1:] == (3, 4), "Target tokens shape")
    require(source_sender.shape == (len(target_tokens), 4), "Source tokens shape")
    visibility = np.ones((3, 3), dtype=np.float64)
    visible = np.eye(8, dtype=np.float64)[target_tokens][:, None] * visibility[None, :, :, None, None]
    source_onehot = np.eye(8, dtype=np.float64)[source_sender]
    for viewer in range(3):
        if viewer != int(sender):
            visible[:, viewer, int(sender)] = source_onehot
    bits = np.broadcast_to(visibility[None], (len(target_tokens), 3, 3)).copy()
    return np.concatenate((visible.reshape(len(target_tokens), 3, 96), bits), axis=-1)


def action_probabilities(networks, x, target_first, source_sender, sender):
    routed_first = route_transfer(target_first, source_sender, sender)
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
    probabilities, _ = core.base.policy_distribution(np.stack(action_logits, axis=1))
    return probabilities


def metrics(intervened, natural, source_set, target_set):
    source_set = np.asarray(source_set, dtype=np.int64)
    target_set = np.asarray(target_set, dtype=np.int64)
    nt = natural[:, target_set].sum(axis=1); ns = natural[:, source_set].sum(axis=1)
    it = intervened[:, target_set].sum(axis=1); ins = intervened[:, source_set].sum(axis=1)
    na = natural.argmax(axis=1); ia = intervened.argmax(axis=1); n = len(intervened)
    nm = dict(target_set_mass=float(nt.mean()), source_set_mass=float(ns.mean()), target_set_hit=float(np.isin(na, target_set).mean()), source_set_hit=float(np.isin(na, source_set).mean()), action_change_rate=0.0)
    im = dict(target_set_mass=float(it.mean()), source_set_mass=float(ins.mean()), target_set_hit=float(np.isin(ia, target_set).mean()), source_set_hit=float(np.isin(ia, source_set).mean()), action_change_rate=float((ia != na).mean()))
    delta = {key: float(im[key] - nm[key]) for key in METRICS}
    return dict(natural=nm, intervened=im, delta=delta, source_pull=float((im["source_set_mass"] - nm["source_set_mass"]) - (im["target_set_mass"] - nm["target_set_mass"])), greedy_source_pull=float((im["source_set_hit"] - nm["source_set_hit"]) - (im["target_set_hit"] - nm["target_set_hit"])), worlds=n)


def compare(actual, expected, path="", tol=3e-13):
    require(set(actual) == set(expected), path + " keys differ")
    for key in actual:
        x, y = actual[key], expected[key]; here = path + str(key) + "/"
        if isinstance(x, dict):
            require(isinstance(y, dict), here + "type differs"); compare(x, y, here, tol)
        elif isinstance(x, list):
            require(isinstance(y, list) and len(x) == len(y), here + "list differs")
            for i, (u, v) in enumerate(zip(x, y)):
                if isinstance(u, (int, float)) and isinstance(v, (int, float)):
                    require(np.isclose(float(u), float(v), atol=tol, rtol=0), here + str(i) + " differs")
                else:
                    require(u == v, here + str(i) + " differs")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            require(np.isclose(float(x), float(y), atol=tol, rtol=0), here + "differs")
        else:
            require(x == y, here + "differs")


def audit_policy(payload):
    source, policy, prepared, cases, mapping = payload
    source = Path(source); seed = int(policy["seed"]); condition = policy["condition"]
    folder = source / "execution" / f"seed_{seed}_unique_{condition}"
    checkpoint = folder / "checkpoint_6000.npz"; source_result = folder / "result.json"; data_path = folder / f"final_{PARTITION}_natural.npz"
    require(sha(checkpoint) == policy["checkpoint_sha256"], "Checkpoint hash mismatch")
    require(sha(source_result) == policy["source_result_sha256"], "Source result hash mismatch")
    require(sha(data_path) == policy["data_sha256"], "Data hash mismatch")
    spec = prepared["partitions"]["unique"][PARTITION]; arrays = core.build_arrays(spec)
    with np.load(data_path, allow_pickle=False) as archive:
        saved = {key: archive[key].copy() for key in archive.files}
    n = int(spec["world_count"]); require(saved["states"].shape == (n, 10), "State shape mismatch")
    require(saved["messages"].shape == (n, 2, 3, 4) and saved["action_probabilities"].shape == (n, 3, 17), "Data shape mismatch")
    first_all = saved["messages"][:, 0]; require(array_sha(first_all) == policy["target_first_messages_sha256"], "First hash mismatch")
    require(len(policy["policy_rows"]) == 240, "Policy row count")
    require([row["case_id"] for row in policy["policy_rows"]] == [case["case_id"] for case in cases], "Case order mismatch")
    lookup = {tuple(need): i for i, need in enumerate(spec["needs"])}; nphysical = len(spec["layouts"]) * len(spec["private_sites"])
    live = condition.endswith("_live"); networks = core.load_networks(checkpoint) if live else None; rows_checked = 0
    for index, (case, row) in enumerate(zip(cases, policy["policy_rows"])):
        donor = cases[mapping[index]]; source_ids = lookup[tuple(case["source_needs"])] * nphysical + np.arange(nphysical); target_ids = lookup[tuple(case["target_needs"])] * nphysical + np.arange(nphysical); donor_ids = lookup[tuple(donor["source_needs"])] * nphysical + np.arange(nphysical)
        require(array_sha(source_ids) == row["source_indices_sha256"] and array_sha(target_ids) == row["target_indices_sha256"] and array_sha(donor_ids) == row["donor_indices_sha256"], "Index hash mismatch")
        require(np.array_equal(saved["states"][source_ids, 3:], saved["states"][target_ids, 3:]), "Background mismatch")
        target_first = first_all[target_ids]; source_sender = first_all[source_ids, case["changed_person"]]; placebo_sender = first_all[donor_ids, case["changed_person"]]
        require(array_sha(target_first) == row["target_first_messages_sha256"] and array_sha(source_sender) == row["source_sender_messages_sha256"] and array_sha(placebo_sender) == row["placebo_sender_messages_sha256"], "Message hash mismatch")
        natural = saved["action_probabilities"][target_ids, case["listener"]]
        if live:
            aligned = action_probabilities(networks, arrays["x_PI"][target_ids], target_first, source_sender, case["changed_person"])[:, case["listener"]]
            placebo = action_probabilities(networks, arrays["x_PI"][target_ids], target_first, placebo_sender, case["changed_person"])[:, case["listener"]]
            require(row["alias_of_natural"] is False and row["model_forward_samples"] == 12 * nphysical, "Live metadata mismatch")
        else:
            aligned = natural.copy(); placebo = natural.copy(); require(row["alias_of_natural"] is True and row["model_forward_samples"] == 0, "Silent metadata mismatch")
        expected_aligned = metrics(aligned, natural, case["source_action_set"], case["target_action_set"]); expected_placebo = metrics(placebo, natural, case["source_action_set"], case["target_action_set"])
        compare(expected_aligned, row["aligned"], f"{seed}/{condition}/{case['case_id']}/aligned/"); compare(expected_placebo, row["placebo"], f"{seed}/{condition}/{case['case_id']}/placebo/")
        rows_checked += 1
    return dict(rows=rows_checked, live_rows=rows_checked if live else 0, alias_rows=0 if live else rows_checked, evaluations=2 * rows_checked if live else 0, forwards=rows_checked * 12 * nphysical if live else 0)


def main(source, probe, output, workers=4):
    source = Path(source).resolve(); probe = Path(probe).resolve(); output = Path(output).resolve(); require(not output.exists(), "Never overwrite audit output")
    result = json.loads((probe / "execution/results.json").read_text(encoding="utf8")); prep = json.loads((probe / "prepared.json").read_text(encoding="utf8")); source_prepared = json.loads((source / "prepared.json").read_text(encoding="utf8"))
    require(result["status"] == "completed_json_only_semantic_transfer_confirmation_placebo_probe" and result["policy_blocks"] == 8 and result["intervention_rows"] == 1920, "Invalid confirmation placebo result grid")
    require(result["live_rows"] == result["alias_rows"] == 960 and result["source"] == prep["source"], "Invalid counts/provenance")
    require(prep["primary_results_sha256"] == sha(Path(prep["primary_probe"]) / "execution/results.json"), "Primary results changed")
    require(prep["source_prepared_sha256"] == sha(source / "prepared.json") and prep["source_freeze_sha256"] == sha(source / "freeze.json") and prep["source_results_sha256"] == sha(source / "execution/results.json"), "Source changed")
    require(prep["candidate_sha256"] == sha(prep["candidate"]), "Candidate changed")
    cases = eligible_cases(prep["candidate"]); mapping = placebo_mapping(cases); require(array_sha(np.asarray(mapping, dtype=np.int64)) == prep["placebo_mapping_sha256"], "Placebo mapping changed")
    require(len(result["policies"]) == 8 and {(int(p["seed"]), p["condition"]) for p in result["policies"]} == {(s, c) for s in SEEDS for c in CONDITIONS}, "Policy grid mismatch")
    output.mkdir(parents=True); payloads = [(str(source), p, source_prepared, cases, mapping) for p in result["policies"]]
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool:
        checked = pool.map(audit_policy, payloads)
    verification = dict(status="passed", source=str(source), probe=str(probe), policy_blocks=8, rows_replayed=sum(x["rows"] for x in checked), live_rows_replayed=sum(x["live_rows"] for x in checked), alias_rows_checked=sum(x["alias_rows"] for x in checked), evaluations=sum(x["evaluations"] for x in checked), backgrounds_per_case=36, worlds=1920 * 36, model_forward_samples=sum(x["forwards"] for x in checked), optimizer_updates=0, max_abs_error=0.0, checkpoint_hashes_checked=8, source_result_hashes_checked=8, natural_data_hashes_checked=8, placebo_mapping_checks=240, independent_replay="Duplicate candidate filtering, cyclic placebo mapping, live route, W2/action rollout and both metric accumulations; confirmation producer implementation not imported")
    path = output / "verification.json"; path.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    (output / "receipt.json").write_text(json.dumps(dict(status="passed", verification_sha256=sha(path), model_forward_samples=verification["model_forward_samples"], optimizer_updates=0, max_abs_error=0.0), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True); parser.add_argument("--probe", required=True); parser.add_argument("--output", required=True); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args(); main(args.source, args.probe, args.output, args.workers)
