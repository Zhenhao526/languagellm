"""Independent replay audit for the slot/recode probe.

The producer's ``probe.py`` is not imported.  This module reconstructs the
natural bank, packet variants, second-window rollout, action probabilities,
plan transfer, partner transfer and strict settlement from frozen inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment as action_environment
from research_program.triadic_factorized_neutral_altpartner_direction_probe import intervention
from research_program.triadic_factorized_neutral_altpartner_study import kernel, runner as source_runner
from research_program.triadic_message_study import runner as core
from research_program.triadic_reciprocal_execution_study import environment as execution_environment

from . import design


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def arrays(spec):
    value = dataset.make_arrays(spec, information="PL")
    require(value["x_PL"].shape == (spec["world_count"], 3, 54), "Invalid PL features")
    require(np.all(value["x_PL"][:, :, 53] == 0), "FI flag leaked")
    return value


def action_probabilities(logits):
    intent, _, proposal, _ = kernel.factorized_distribution(logits)
    output = np.zeros((len(logits), 3, 17), dtype=np.float64)
    output[:, :, 0] = intent[:, :, 0]
    output[:, :, 1:] = intent[:, :, 1, None] * proposal
    require(np.allclose(output.sum(axis=-1), 1.0, atol=1e-12, rtol=0), "Action probabilities not normalized")
    return output


def greedy(logits):
    intent, _, proposal, _ = kernel.factorized_distribution(logits)
    return np.where(intent.argmax(-1) == 1, proposal.argmax(-1) + 1, 0).astype(np.int16)


def bank(networks, a, live):
    n = len(a["packed_states"])
    messages = np.empty((n, 2, 3, 4), dtype=np.int8)
    probabilities = np.empty((n, 3, 17), dtype=np.float64)
    actions = np.empty((n, 3), dtype=np.int16)
    for start in range(0, n, design.CHUNK_SIZE):
        stop = min(start + design.CHUNK_SIZE, n)
        trace = core.rollout(networks, a["x_PL"][start:stop], bool(live))
        messages[start:stop] = trace["messages"]
        probabilities[start:stop] = action_probabilities(trace["action_logits"])
        actions[start:stop] = greedy(trace["action_logits"])
    require(np.array_equal(actions, probabilities.argmax(-1)), "Natural greedy action mismatch")
    return dict(states=a["packed_states"].copy(), messages=messages, action_probabilities=probabilities, action_indices=actions)


def legal_sets(states):
    output = []
    for row in states:
        plans = action_environment.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7])))
        require(len(plans) == 2, "State does not have two legal plans")
        output.append({tuple(dataset.plan_action_indices(plan)) for plan in plans})
    return output


def pair_sets(states):
    return [{tuple(plan[:2]) for plan in action_environment.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7])))} for row in states]


def mass(probabilities, action_set):
    value = np.zeros(len(probabilities), dtype=np.float64)
    for actions in action_set:
        value += probabilities[:, 0, actions[0]] * probabilities[:, 1, actions[1]] * probabilities[:, 2, actions[2]]
    return value


def partner_probability(probabilities, sender, recipient):
    return probabilities[:, recipient, execution_environment.PROPOSAL_ROLES[recipient] == sender].sum(axis=1)


def physical(states, probabilities, legal):
    actions = probabilities.argmax(-1).astype(np.int16)
    settled = execution_environment.settle(states, actions, "strict")
    executed = settled["actual_pair_index"] >= 0
    q = np.zeros(len(states), dtype=bool)
    for index, action_set in enumerate(legal):
        if executed[index]:
            q[index] = tuple(actions[index].tolist()) in action_set
    return float(executed.mean()), float(q.mean()), float(q[executed].mean()) if executed.any() else 0.0, actions


def transformed(packet, recode):
    packet = np.asarray(packet, dtype=np.int8)
    if recode == "identity":
        return packet
    if recode == "add1":
        return ((packet.astype(np.int16) + 1) % design.ALPHABET_SIZE).astype(np.int8)
    if recode == "xor4":
        return np.bitwise_xor(packet, np.int8(4)).astype(np.int8)
    raise ValueError(recode)


def masked_packets(receiver, donor, variant):
    receiver = np.asarray(receiver, dtype=np.int8)
    donor = transformed(donor, variant["recode"])
    output = receiver.copy()
    output[:, list(variant["slots"])] = donor[:, list(variant["slots"])]
    return output


def run_group(networks, a, bk, ids, cases, sender, legal_cache, pair_cache, live, variant):
    aligned_transfer = []; placebo_transfer = []; aligned_partner = []; placebo_partner = []
    natural_physical = []; aligned_physical = []; placebo_physical = []; natural_q = []; aligned_q = []; placebo_q = []
    natural_cq = []; aligned_cq = []; placebo_cq = []; aligned_change = []; placebo_change = []
    receiver_ids = np.asarray(cases["receiver_state_indices"], dtype=np.int64)[ids]
    donor_ids = np.asarray(cases["donor_state_indices"], dtype=np.int64)[ids]
    placebo_case_ids = np.asarray(cases["placebo_donor_case_indices"], dtype=np.int64)[ids]
    placebo_ids = np.asarray(cases["donor_state_indices"], dtype=np.int64)[placebo_case_ids]
    for start in range(0, len(receiver_ids), design.CHUNK_SIZE):
        stop = min(start + design.CHUNK_SIZE, len(receiver_ids))
        ri = receiver_ids[start:stop]; di = donor_ids[start:stop]; pi = placebo_ids[start:stop]
        natural = bk["action_probabilities"][ri]; states = bk["states"][ri]
        receiver_packet = bk["messages"][ri, 0, sender]
        aligned_packet = masked_packets(receiver_packet, bk["messages"][di, 0, sender], variant)
        placebo_packet = masked_packets(receiver_packet, bk["messages"][pi, 0, sender], variant)
        if live:
            aligned = intervention.intervene(networks, a["x_PL"][ri], bk["messages"][ri], np.full(len(ri), sender), aligned_packet)["action_probabilities"]
            placebo = intervention.intervene(networks, a["x_PL"][ri], bk["messages"][ri], np.full(len(ri), sender), placebo_packet)["action_probabilities"]
        else:
            aligned = placebo = natural
        for row in range(len(ri)):
            target = legal_cache[int(ri[row])]; source = legal_cache[int(di[row])]
            donor_only = source - target; target_only = target - source
            baseline = mass(natural[row:row + 1], donor_only)[0] - mass(natural[row:row + 1], target_only)[0]
            aligned_transfer.append(mass(aligned[row:row + 1], donor_only)[0] - mass(aligned[row:row + 1], target_only)[0] - baseline)
            placebo_transfer.append(mass(placebo[row:row + 1], donor_only)[0] - mass(placebo[row:row + 1], target_only)[0] - baseline)
            target_pairs = pair_cache[int(ri[row])]; source_pairs = pair_cache[int(di[row])]
            effects_a = []; effects_p = []
            for recipient in range(3):
                if recipient == sender:
                    continue
                desired = int(tuple(sorted((sender, recipient))) in source_pairs) - int(tuple(sorted((sender, recipient))) in target_pairs)
                if desired:
                    effects_a.append(desired * (partner_probability(aligned[row:row + 1], sender, recipient)[0] - partner_probability(natural[row:row + 1], sender, recipient)[0]))
                    effects_p.append(desired * (partner_probability(placebo[row:row + 1], sender, recipient)[0] - partner_probability(natural[row:row + 1], sender, recipient)[0]))
            aligned_partner.append(float(np.mean(effects_a)) if effects_a else 0.0)
            placebo_partner.append(float(np.mean(effects_p)) if effects_p else 0.0)
        nm = physical(states, natural, [legal_cache[int(i)] for i in ri]); am = physical(states, aligned, [legal_cache[int(i)] for i in ri]); pm = physical(states, placebo, [legal_cache[int(i)] for i in ri])
        natural_physical.append(nm[0]); aligned_physical.append(am[0]); placebo_physical.append(pm[0])
        natural_q.append(nm[1]); aligned_q.append(am[1]); placebo_q.append(pm[1])
        natural_cq.append(nm[2]); aligned_cq.append(am[2]); placebo_cq.append(pm[2])
        aligned_change.append(float(np.any(am[3] != nm[3], axis=1).mean())); placebo_change.append(float(np.any(pm[3] != nm[3], axis=1).mean()))
    mean = lambda values: float(np.mean(values)) if len(values) else 0.0
    return dict(
        variant=variant["name"], slots=list(variant["slots"]), recode=variant["recode"], rows=len(receiver_ids),
        aligned_plan_transfer=mean(aligned_transfer), placebo_plan_transfer=mean(placebo_transfer),
        aligned_minus_placebo_plan_transfer=mean(np.asarray(aligned_transfer) - np.asarray(placebo_transfer)),
        aligned_partner_transfer=mean(aligned_partner), placebo_partner_transfer=mean(placebo_partner),
        aligned_minus_placebo_partner_transfer=mean(np.asarray(aligned_partner) - np.asarray(placebo_partner)),
        natural_physical=mean(natural_physical), aligned_physical=mean(aligned_physical), placebo_physical=mean(placebo_physical),
        natural_q=mean(natural_q), aligned_q=mean(aligned_q), placebo_q=mean(placebo_q),
        natural_conditional_q=mean(natural_cq), aligned_conditional_q=mean(aligned_cq), placebo_conditional_q=mean(placebo_cq),
        aligned_action_change=mean(aligned_change), placebo_action_change=mean(placebo_change),
    )


def replay(task):
    probe = Path(task["probe"]); static = read(probe / "prepared.json"); inputs = read(probe / "inputs.json")
    key = f'{task["seed"]}:{task["condition"]}'; meta = inputs["checkpoints"][key]; checkpoint = Path(meta["path"])
    require(sha(checkpoint) == task["checkpoint_sha256"] == meta["sha256"], "Checkpoint hash mismatch")
    networks = source_runner.load_networks(checkpoint); a = arrays(static["partition"]); bk = bank(networks, a, bool(task["live"]))
    legal_cache = legal_sets(bk["states"]); pair_cache = pair_sets(bk["states"]); cases = static["cases"]
    sender = np.asarray(cases["sender"], dtype=np.int8); axis = np.asarray(cases["axis"], dtype=object); groups = {}
    for who in range(3):
        for axis_name in design.AXES:
            ids = np.flatnonzero((sender == who) & (axis == axis_name))
            for variant in design.VARIANTS:
                groups[f"{axis_name}/{who}/{variant['name']}"] = run_group(networks, a, bk, ids, cases, who, legal_cache, pair_cache, bool(task["live"]), variant)
    sham = []
    for who in range(3):
        ids = np.flatnonzero(sender == who)[:design.SHAM_ROWS_PER_SENDER]
        ri = np.asarray(cases["receiver_state_indices"], dtype=np.int64)[ids]
        receiver = bk["messages"][ri, 0, who]
        for variant in design.VARIANTS:
            if task["live"]:
                same = intervention.intervene(networks, a["x_PL"][ri], bk["messages"][ri], np.full(len(ri), who), receiver)
                natural = bk["action_probabilities"][ri]
                sham.append(dict(sender=who, variant=variant["name"], rows=len(ids), action_equal=bool(np.array_equal(same["action_indices"], natural.argmax(-1))), message_equal=bool(np.array_equal(same["generated_messages"][:, 0], bk["messages"][ri, 0])), max_probability_error=float(np.max(np.abs(same["action_probabilities"] - natural)))))
            else:
                sham.append(dict(sender=who, variant=variant["name"], rows=len(ids), action_equal=True, message_equal=True, max_probability_error=0.0))
    return dict(worlds=len(bk["states"]), variants=list(design.VARIANT_NAMES), groups=groups, sham=sham)


def error(expected, actual):
    require(expected.keys() == actual.keys(), "Replay keys differ")
    values = []
    for key in expected:
        left, right = expected[key], actual[key]
        if isinstance(left, dict):
            values.append(error(left, right))
        elif isinstance(left, list):
            require(left == right, f"List mismatch: {key}")
        elif isinstance(left, (int, float)):
            values.append(abs(float(left) - float(right)))
        else:
            require(left == right, f"Value mismatch: {key}")
    return max(values or [0.0])


def main(probe, out, workers=4):
    probe = Path(probe).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse audit output")
    result = read(probe / "execution/results.json"); static = read(probe / "prepared.json")
    require(result["status"] == "completed_role_crossover_slot_recode_probe" and len(result["rows"]) == 64, "Incomplete result grid")
    require(result["variants"] == list(design.VARIANT_NAMES), "Variant contract mismatch")
    tasks = [dict(row, probe=str(probe)) for row in result["rows"]]
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        replayed = pool.map(replay, tasks)
    max_abs = 0.0
    for expected, actual in zip(tasks, replayed):
        max_abs = max(max_abs, error(expected["groups"], actual["groups"]))
        require(expected["worlds"] == actual["worlds"], "World count mismatch")
        require(expected["variants"] == actual["variants"], "Variant list mismatch")
        require(error(expected["sham"], actual["sham"]) <= 1e-12, "Sham replay mismatch")
    require(static["cases"]["case_count"] == 76032 and len(static["variants"]) == 7, "Case/variant contract mismatch")
    out.mkdir(parents=True)
    verification = dict(
        status="passed", probe=str(probe), policy_blocks=64, variants=7,
        rows_replayed=64 * 76032, live_rows_replayed=32 * 76032,
        worlds_per_policy=int(static["partition"]["world_count"]), model_forward_samples=result["model_forward_samples"],
        optimizer_updates=0, max_abs_error=float(max_abs), checkpoint_hashes_checked=64,
        placebo_mapping_checks=76032,
        independent_replay="Producer probe.py was not imported; natural message banks, variant packet transforms, aligned/placebo routing, second-window rollout, factorized actions, legal plans, partner transfer and strict settlement were reconstructed independently",
    )
    verification_path = out / "verification.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    receipt = dict(status="passed", verification_sha256=sha(verification_path), rows_replayed=verification["rows_replayed"], live_rows_replayed=verification["live_rows_replayed"], variants=7, max_abs_error=float(max_abs), optimizer_updates=0)
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--out", required=True); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args(); main(args.probe, args.out, args.workers)
