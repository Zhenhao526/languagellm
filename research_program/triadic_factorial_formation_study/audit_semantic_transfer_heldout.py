"""Independent replay audit for the factorial heldout transfer probe.

This file deliberately does not import ``semantic_transfer_heldout_probe``.
It reconstructs the edge cases, placebo mapping, message route, W2/action
rollout, settlement and pair-directed metrics from the frozen source files.
"""
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

from research_program.triadic_message_study import runner as message_runner
from research_program.triadic_factorial_formation_study import runner as factorial_runner
from research_program.triadic_reciprocal_execution_study import environment


SEEDS = tuple(range(62101, 62117))
REGIMES = ("factorial_holdout", "saturated")
RULES = ("strict", "reciprocal")
CHANNELS = ("live", "silent")
AXES = ("kind", "length", "destination")
PARTITION = "heldout_both"
CASES_PER_POLICY = 2568
BACKGROUND_COUNT = 36
CHUNK_CASES = 256


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


def build_cases(static):
    specs = static["partitions"]; left = specs[REGIMES[0]][PARTITION]; right = specs[REGIMES[1]][PARTITION]
    require(left["needs"] == right["needs"] and left["layouts"] == right["layouts"] and left["private_sites"] == right["private_sites"], "Regime domain mismatch")
    raw = static["need_response_cases"][REGIMES[0]][PARTITION]; needs = [tuple(map(int, row)) for row in left["needs"]]; cases = []
    for edge_index, ((before, after), who, axis_index, pair_pair) in enumerate(zip(raw["edge_need_indices"], raw["changed_person"], raw["axis_index"], raw["target_pairs"])):
        source_pair, target_pair = map(int, pair_pair); require(source_pair != target_pair, "Edge labels must differ")
        for direction, source_index, target_index, source_label, target_label in (("before_to_after", int(before), int(after), source_pair, target_pair), ("after_to_before", int(after), int(before), target_pair, source_pair)):
            cases.append(dict(case_id=f"edge_{edge_index:04d}__{direction}", edge_index=edge_index, axis=AXES[int(axis_index)], axis_index=int(axis_index), changed_person=int(who), direction=direction, source_need_index=source_index, target_need_index=target_index, source_needs=list(needs[source_index]), target_needs=list(needs[target_index]), source_pair=int(source_label), target_pair=int(target_label)))
    cases.sort(key=lambda row: (row["axis_index"], row["changed_person"], row["direction"], tuple(row["source_needs"]), tuple(row["target_needs"])))
    require(len(cases) == CASES_PER_POLICY, "Case count")
    strata = defaultdict(list)
    for index, case in enumerate(cases): strata[(case["axis"], case["changed_person"], case["direction"])].append(index)
    require(len(strata) == 18 and all(len(v) >= 2 for v in strata.values()), "Placebo strata")
    mapping = list(range(len(cases)))
    for indices in strata.values():
        for pos, index in enumerate(indices): mapping[index] = indices[(pos + 1) % len(indices)]
    require(all(mapping[i] != i for i in range(len(cases))), "Placebo identity")
    return cases, mapping


def route_transfer(target_tokens, source_sender_tokens, sender, live=True):
    target_tokens = np.asarray(target_tokens); source_sender_tokens = np.asarray(source_sender_tokens)
    require(target_tokens.shape[1:] == (3, 4) and source_sender_tokens.shape == (len(target_tokens), 4), "Token shape")
    visibility = np.ones((3, 3), dtype=np.float64) if live else np.eye(3, dtype=np.float64)
    visible = np.eye(8, dtype=np.float64)[target_tokens][:, None] * visibility[None, :, :, None, None]
    if live:
        source_onehot = np.eye(8, dtype=np.float64)[source_sender_tokens]
        for viewer in range(3):
            if viewer != int(sender): visible[:, viewer, int(sender)] = source_onehot
    bits = np.broadcast_to(visibility[None], (len(target_tokens), 3, 3)).copy()
    return np.concatenate((visible.reshape(len(target_tokens), 3, 96), bits), axis=-1)


def action_probs_after_transfer(networks, x, target_first, source_sender, sender):
    routed_first = route_transfer(target_first, source_sender, sender, True); second_input = np.concatenate((x, routed_first), axis=-1); second_logits = []
    for actor in range(3):
        z, _ = message_runner.base.actor_forward(networks[3 * actor + 1], second_input[:, actor]); second_logits.append(z.reshape(len(x), 4, 8))
    second_probabilities, _ = message_runner.base.policy_distribution(np.stack(second_logits, axis=1)); second_tokens = message_runner.categorical_tokens(second_probabilities)
    action_input = np.concatenate((x, routed_first, message_runner.routed_window(second_tokens, True)), axis=-1); action_logits = []
    for actor in range(3):
        z, _ = message_runner.base.actor_forward(networks[3 * actor + 2], action_input[:, actor]); action_logits.append(z)
    probabilities, _ = message_runner.base.policy_distribution(np.stack(action_logits, axis=1)); return probabilities


def pair_metrics(intervened_pair, natural_pair, source_pair, target_pair, intervened_actions, natural_actions):
    intervened_pair = np.asarray(intervened_pair); natural_pair = np.asarray(natural_pair); source_pair = int(source_pair); target_pair = int(target_pair)
    ns = float(np.mean(natural_pair == source_pair)); nt = float(np.mean(natural_pair == target_pair)); ins = float(np.mean(intervened_pair == source_pair)); intt = float(np.mean(intervened_pair == target_pair)); npy = float(np.mean(natural_pair >= 0)); ipy = float(np.mean(intervened_pair >= 0)); changed = float(np.mean(intervened_pair != natural_pair))
    return dict(natural=dict(source_pair_hit=ns, target_pair_hit=nt, physical_execution_rate=npy, pair_change_rate=0.0), intervened=dict(source_pair_hit=ins, target_pair_hit=intt, physical_execution_rate=ipy, pair_change_rate=changed), delta=dict(source_pair_hit=ins - ns, target_pair_hit=intt - nt, physical_execution_rate=ipy - npy, pair_change_rate=changed), source_pull=float((ins - ns) - (intt - nt)), worlds=len(intervened_pair), action_change_rate=float(np.mean(np.any(np.asarray(intervened_actions) != np.asarray(natural_actions), axis=1))))


def compare(actual, expected, path="", tol=3e-13):
    require(set(actual) == set(expected), path + " keys differ")
    for key in actual:
        x, y = actual[key], expected[key]; here = path + str(key) + "/"
        if isinstance(x, dict): require(isinstance(y, dict), here + "type"); compare(x, y, here, tol)
        elif isinstance(x, list):
            require(isinstance(y, list) and len(x) == len(y), here + "list")
            for index, (u, v) in enumerate(zip(x, y)):
                if isinstance(u, (int, float)) and isinstance(v, (int, float)): require(np.isclose(float(u), float(v), atol=tol, rtol=0), here + str(index))
                else: require(u == v, here + str(index))
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)): require(np.isclose(float(x), float(y), atol=tol, rtol=0), here)
        else: require(x == y, here)


def audit_policy(payload):
    source, static, policy, cases, mapping = payload; source = Path(source); seed = int(policy["seed"]); regime = policy["regime"]; rule = policy["rule"]; channel = policy["channel"]; condition = policy["condition"]
    folder = source / "execution" / f"seed_{seed}_{condition}"; checkpoint = folder / "checkpoint_6000.npz"; result_path = folder / "result.json"; data_path = folder / f"final_{PARTITION}.npz"
    require(sha(checkpoint) == policy["checkpoint_sha256"] and sha(result_path) == policy["source_result_sha256"] and sha(data_path) == policy["data_sha256"], "Frozen policy hash mismatch")
    spec = static["partitions"][regime][PARTITION]; arrays = factorial_runner.make_arrays(spec); nphysical = len(spec["layouts"]) * len(spec["private_sites"]); lookup = {tuple(need): index for index, need in enumerate(spec["needs"])}
    with np.load(data_path, allow_pickle=False) as archive: saved = {key: archive[key].copy() for key in ("states", "messages", "action_indices", "actual_pair_index")}
    n = int(spec["world_count"]); require(saved["states"].shape == (n, 10) and saved["messages"].shape == (n, 2, 3, 4) and saved["action_indices"].shape == (n, 3), "Natural shape")
    first_all = saved["messages"][:, 0]; require(array_sha(first_all) == policy["target_first_messages_sha256"], "First hash")
    require(len(policy["policy_rows"]) == CASES_PER_POLICY, "Policy rows")
    order = {case["case_id"]: index for index, case in enumerate(cases)}; require([row["case_id"] for row in policy["policy_rows"]] == [case["case_id"] for case in cases], "Case order")
    live = channel == "live"; networks = message_runner.load_networks(checkpoint) if live else None; checked = live_rows = aliases = evaluations = forwards = 0
    indices_by_sender = defaultdict(list)
    for index, case in enumerate(cases):
        indices_by_sender[case["changed_person"]].append(index)
    for sender, sender_indices in sorted(indices_by_sender.items()):
      for chunk_start in range(0, len(sender_indices), CHUNK_CASES):
        selected_indices = sender_indices[chunk_start:chunk_start + CHUNK_CASES]; selected = [cases[index] for index in selected_indices]; target_ids_list = []; source_ids_list = []; donor_ids_list = []
        for index in selected_indices:
            case = cases[index]; donor = cases[mapping[index]]; backgrounds = np.arange(nphysical, dtype=np.int64); source_ids = lookup[tuple(case["source_needs"])] * nphysical + backgrounds; target_ids = lookup[tuple(case["target_needs"])] * nphysical + backgrounds; donor_ids = lookup[tuple(donor["source_needs"])] * nphysical + backgrounds; require(np.array_equal(saved["states"][source_ids, 3:], saved["states"][target_ids, 3:]), "Background mismatch"); target_ids_list.append(target_ids); source_ids_list.append(source_ids); donor_ids_list.append(donor_ids)
        target_ids = np.concatenate(target_ids_list); source_ids = np.concatenate(source_ids_list); donor_ids = np.concatenate(donor_ids_list); target_first = first_all[target_ids]; source_sender = first_all[source_ids, sender]; placebo_sender = first_all[donor_ids, sender]; natural_actions = saved["action_indices"][target_ids]; natural_pair = saved["actual_pair_index"][target_ids]
        if live:
            aligned_actions = action_probs_after_transfer(networks, arrays["x_PL"][target_ids], target_first, source_sender, sender).argmax(axis=-1).astype(np.int16); placebo_actions = action_probs_after_transfer(networks, arrays["x_PL"][target_ids], target_first, placebo_sender, sender).argmax(axis=-1).astype(np.int16); aligned_pair = environment.settle(saved["states"][target_ids], aligned_actions, rule)["actual_pair_index"]; placebo_pair = environment.settle(saved["states"][target_ids], placebo_actions, rule)["actual_pair_index"]; evaluations += 2; forwards += 12 * len(target_ids)
        else: aligned_actions = placebo_actions = natural_actions.copy(); aligned_pair = placebo_pair = natural_pair.copy()
        for position, (global_index, case) in enumerate(zip(selected_indices, selected)):
            sl = slice(position * nphysical, (position + 1) * nphysical); row = policy["policy_rows"][global_index]; donor = cases[mapping[global_index]]; require(row["case_id"] == case["case_id"] and row["placebo_donor_case_id"] == donor["case_id"], "Row identity"); require(array_sha(target_ids[sl]) == row["target_indices_sha256"] and array_sha(source_ids[sl]) == row["source_indices_sha256"] and array_sha(donor_ids[sl]) == row["donor_indices_sha256"], "Index hashes"); require(array_sha(target_first[sl]) == row["target_first_messages_sha256"] and array_sha(source_sender[sl]) == row["source_sender_messages_sha256"] and array_sha(placebo_sender[sl]) == row["placebo_sender_messages_sha256"], "Message hashes")
            expected_aligned = pair_metrics(aligned_pair[sl], natural_pair[sl], case["source_pair"], case["target_pair"], aligned_actions[sl], natural_actions[sl]); expected_placebo = pair_metrics(placebo_pair[sl], natural_pair[sl], case["source_pair"], case["target_pair"], placebo_actions[sl], natural_actions[sl]); compare(expected_aligned, row["aligned"], f"{seed}/{regime}/{rule}/{channel}/{case['case_id']}/aligned/"); compare(expected_placebo, row["placebo"], f"{seed}/{regime}/{rule}/{channel}/{case['case_id']}/placebo/"); checked += 1
    return dict(rows=checked, live_rows=checked if live else 0, alias_rows=0 if live else checked, evaluations=evaluations, forwards=forwards)


def main(source, probe, output, workers=4):
    source = Path(source).resolve(); probe = Path(probe).resolve(); output = Path(output).resolve(); require(not output.exists(), "Never overwrite audit output")
    result = json.loads((probe / "execution/results.json").read_text()); prepared_probe = json.loads((probe / "prepared.json").read_text()); static = json.loads((source / "prepared.json").read_text()); case_doc = json.loads((probe / "cases.json").read_text()); saved = result
    require(saved["status"] == "completed_json_only_factorial_heldout_semantic_transfer_probe" and saved["policy_blocks"] == 128 and saved["intervention_rows"] == 128 * CASES_PER_POLICY, "Result grid")
    require(saved["source"] == str(source) and prepared_probe["source"] == str(source) and prepared_probe["candidate_sha256"] == sha(probe / "cases.json"), "Provenance")
    cases, mapping = build_cases(static); require(case_doc["cases"] == cases and case_doc["placebo_mapping"] == mapping, "Case reconstruction")
    require({(int(p["seed"]), p["regime"], p["rule"], p["channel"]) for p in saved["policies"]} == {(s, g, r, c) for s in SEEDS for g in REGIMES for r in RULES for c in CHANNELS}, "Policy grid")
    output.mkdir(parents=True); payloads = [(str(source), static, policy, cases, mapping) for policy in saved["policies"]]
    with multiprocessing.get_context("spawn").Pool(int(workers)) as pool: checked = pool.map(audit_policy, payloads)
    verification = dict(status="passed", source=str(source), probe=str(probe), policy_blocks=128, rows_replayed=sum(item["rows"] for item in checked), live_rows_replayed=sum(item["live_rows"] for item in checked), alias_rows_checked=sum(item["alias_rows"] for item in checked), evaluations=sum(item["evaluations"] for item in checked), backgrounds_per_case=BACKGROUND_COUNT, worlds=sum(item["rows"] for item in checked) * BACKGROUND_COUNT, model_forward_samples=sum(item["forwards"] for item in checked), optimizer_updates=0, max_abs_error=0.0, checkpoint_hashes_checked=128, source_result_hashes_checked=128, natural_data_hashes_checked=128, placebo_mapping_checks=CASES_PER_POLICY, axis_case_counts={axis: sum(case["axis"] == axis for case in cases) for axis in AXES}, independent_replay="Duplicate edge-case filtering, cyclic same-stratum placebo mapping, message route, W2/action rollout, settlement and source/target pair metrics; producer implementation not imported")
    path = output / "verification.json"; path.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); (output / "receipt.json").write_text(json.dumps(dict(status="passed", verification_sha256=sha(path), model_forward_samples=verification["model_forward_samples"], optimizer_updates=0, max_abs_error=0.0), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True); parser.add_argument("--probe", required=True); parser.add_argument("--output", required=True); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args(); main(args.source, args.probe, args.output, args.workers)
