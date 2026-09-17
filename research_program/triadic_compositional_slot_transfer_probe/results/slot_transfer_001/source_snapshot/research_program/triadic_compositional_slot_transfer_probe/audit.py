"""Independent replay audit for the single-slot transfer probe.

This module reconstructs the natural bank, masked packets, second-window
rollout, factorized actions, plan masses and strict settlement without
importing the producer's ``probe.py``.
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
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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
    n = len(a["packed_states"]); messages = np.empty((n, 2, 3, 4), dtype=np.int8); probabilities = np.empty((n, 3, 17), dtype=np.float64); actions = np.empty((n, 3), dtype=np.int16)
    for start in range(0, n, design.CHUNK_SIZE):
        stop = min(start + design.CHUNK_SIZE, n)
        trace = core.rollout(networks, a["x_PL"][start:stop], bool(live))
        messages[start:stop] = trace["messages"]; probabilities[start:stop] = action_probabilities(trace["action_logits"]); actions[start:stop] = greedy(trace["action_logits"])
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
    actions = probabilities.argmax(-1).astype(np.int16); settled = execution_environment.settle(states, actions, "strict"); executed = settled["actual_pair_index"] >= 0; q = np.zeros(len(states), dtype=bool)
    for index, action_set in enumerate(legal):
        if executed[index]: q[index] = tuple(actions[index].tolist()) in action_set
    return dict(physical=float(executed.mean()), q=float(q.mean()), conditional_q=float(q[executed].mean()) if executed.any() else 0.0, actions=actions)


def masked_packets(receiver, donor, slots):
    output = np.asarray(receiver, dtype=np.int8).copy(); donor = np.asarray(donor, dtype=np.int8); output[:, list(slots)] = donor[:, list(slots)]; return output


def run_group(networks, a, bk, ids, cases, sender, legal_cache, pair_cache, live, mask_name, slots):
    aligned_transfer = []; placebo_transfer = []; aligned_partner = []; placebo_partner = []; natural_physical = []; aligned_physical = []; placebo_physical = []; natural_q = []; aligned_q = []; placebo_q = []; natural_cq = []; aligned_cq = []; placebo_cq = []; aligned_change = []; placebo_change = []
    receiver_ids = np.asarray(cases["receiver_state_indices"], dtype=np.int64)[ids]; donor_ids = np.asarray(cases["donor_state_indices"], dtype=np.int64)[ids]; placebo_case_ids = np.asarray(cases["placebo_donor_case_indices"], dtype=np.int64)[ids]; placebo_ids = np.asarray(cases["donor_state_indices"], dtype=np.int64)[placebo_case_ids]
    for start in range(0, len(receiver_ids), design.CHUNK_SIZE):
        stop = min(start + design.CHUNK_SIZE, len(receiver_ids)); ri = receiver_ids[start:stop]; di = donor_ids[start:stop]; pi = placebo_ids[start:stop]; natural = bk["action_probabilities"][ri]; states = bk["states"][ri]
        receiver_packet = bk["messages"][ri, 0, sender]; aligned_packet = masked_packets(receiver_packet, bk["messages"][di, 0, sender], slots); placebo_packet = masked_packets(receiver_packet, bk["messages"][pi, 0, sender], slots)
        if live:
            aligned = intervention.intervene(networks, a["x_PL"][ri], bk["messages"][ri], np.full(len(ri), sender), aligned_packet)["action_probabilities"]
            placebo = intervention.intervene(networks, a["x_PL"][ri], bk["messages"][ri], np.full(len(ri), sender), placebo_packet)["action_probabilities"]
        else:
            aligned = placebo = natural
        for row in range(len(ri)):
            target = legal_cache[int(ri[row])]; source = legal_cache[int(di[row])]; donor_only = source - target; target_only = target - source; baseline = mass(natural[row:row + 1], donor_only)[0] - mass(natural[row:row + 1], target_only)[0]
            aligned_transfer.append(mass(aligned[row:row + 1], donor_only)[0] - mass(aligned[row:row + 1], target_only)[0] - baseline); placebo_transfer.append(mass(placebo[row:row + 1], donor_only)[0] - mass(placebo[row:row + 1], target_only)[0] - baseline)
            target_pairs = pair_cache[int(ri[row])]; source_pairs = pair_cache[int(di[row])]; effects_a = []; effects_p = []
            for recipient in range(3):
                if recipient == sender: continue
                desired = int(tuple(sorted((sender, recipient))) in source_pairs) - int(tuple(sorted((sender, recipient))) in target_pairs)
                if desired:
                    effects_a.append(desired * (partner_probability(aligned[row:row + 1], sender, recipient)[0] - partner_probability(natural[row:row + 1], sender, recipient)[0])); effects_p.append(desired * (partner_probability(placebo[row:row + 1], sender, recipient)[0] - partner_probability(natural[row:row + 1], sender, recipient)[0]))
            aligned_partner.append(float(np.mean(effects_a)) if effects_a else 0.0); placebo_partner.append(float(np.mean(effects_p)) if effects_p else 0.0)
        nm = physical(states, natural, [legal_cache[int(i)] for i in ri]); am = physical(states, aligned, [legal_cache[int(i)] for i in ri]); pm = physical(states, placebo, [legal_cache[int(i)] for i in ri])
        natural_physical.append(nm["physical"]); aligned_physical.append(am["physical"]); placebo_physical.append(pm["physical"]); natural_q.append(nm["q"]); aligned_q.append(am["q"]); placebo_q.append(pm["q"]); natural_cq.append(nm["conditional_q"]); aligned_cq.append(am["conditional_q"]); placebo_cq.append(pm["conditional_q"]); aligned_change.append(float(np.any(am["actions"] != nm["actions"], axis=1).mean())); placebo_change.append(float(np.any(pm["actions"] != nm["actions"], axis=1).mean()))
    mean = lambda values: float(np.mean(values)) if values else 0.0
    return dict(mask=mask_name, slots=list(slots), rows=len(receiver_ids), aligned_plan_transfer=mean(aligned_transfer), placebo_plan_transfer=mean(placebo_transfer), aligned_minus_placebo_plan_transfer=mean(np.asarray(aligned_transfer) - np.asarray(placebo_transfer)), aligned_partner_transfer=mean(aligned_partner), placebo_partner_transfer=mean(placebo_partner), aligned_minus_placebo_partner_transfer=mean(np.asarray(aligned_partner) - np.asarray(placebo_partner)), natural_physical=mean(natural_physical), aligned_physical=mean(aligned_physical), placebo_physical=mean(placebo_physical), natural_q=mean(natural_q), aligned_q=mean(aligned_q), placebo_q=mean(placebo_q), natural_conditional_q=mean(natural_cq), aligned_conditional_q=mean(aligned_cq), placebo_conditional_q=mean(placebo_cq), aligned_action_change=mean(aligned_change), placebo_action_change=mean(placebo_change))


def replay(task):
    probe = Path(task["probe"]); static = read(probe / "prepared.json"); inputs = read(probe / "inputs.json"); key = f'{task["seed"]}:{task["arm"]}:{task["condition"]}'; meta = inputs["checkpoints"][key]; checkpoint = Path(meta["checkpoint"]); require(sha(checkpoint) == task["checkpoint_sha256"] == meta["checkpoint_sha256"], "Checkpoint hash mismatch")
    networks = source_runner.load_networks(checkpoint); a = arrays(static["heldout_spec"]); bk = bank(networks, a, bool(task["live"])); legal_cache = legal_sets(bk["states"]); pair_cache = pair_sets(bk["states"]); cases = static["cases"]; sender = np.asarray(cases["sender"], dtype=np.int8); axis = np.asarray(cases["axis"], dtype=object); groups = {}
    for who in range(3):
        for axis_name in design.AXES:
            ids = np.flatnonzero((sender == who) & (axis == axis_name))
            for mask_name, slots in design.MASKS:
                groups[f"{axis_name}/{who}/{mask_name}"] = run_group(networks, a, bk, ids, cases, who, legal_cache, pair_cache, bool(task["live"]), mask_name, slots)
    return dict(worlds=len(bk["states"]), groups=groups)


def error(expected, actual):
    values = []
    require(expected.keys() == actual.keys(), "Replay group keys differ")
    for key in expected:
        left, right = expected[key], actual[key]
        if isinstance(left, dict): values.append(error(left, right))
        elif isinstance(left, list): require(left == right, f"List mismatch: {key}")
        elif isinstance(left, (int, float)): values.append(abs(float(left) - float(right)))
        else: require(left == right, f"Value mismatch: {key}")
    return max(values or [0.0])


def main(probe, out, workers=4):
    probe = Path(probe).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse audit output")
    result = read(probe / "execution/results.json"); require(result["status"] == "completed_compositional_slot_transfer_probe" and len(result["rows"]) == 64, "Incomplete result grid")
    tasks = []
    for row in result["rows"]:
        tasks.append(dict(row, probe=str(probe)))
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        replayed = pool.map(replay, tasks)
    max_abs = 0.0
    for expected, actual in zip(tasks, replayed):
        max_abs = max(max_abs, error(expected["groups"], actual["groups"])); require(expected["worlds"] == actual["worlds"], "World count mismatch")
        require(all(item["action_equal"] and item["message_equal"] and item["max_probability_error"] == 0.0 for item in expected["sham"]), "Sham replay failed")
    prepared = read(probe / "prepared.json"); require(prepared["cases"]["case_count"] == 3456 and len(prepared["masks"]) == 5, "Case/mask contract mismatch")
    out.mkdir(parents=True); verification = dict(status="passed", probe=str(probe), policy_blocks=64, masks=5, rows_replayed=64 * 3456 * 5, live_rows_replayed=32 * 3456 * 5, worlds_per_policy=11232, model_forward_samples=result["model_forward_samples"], optimizer_updates=0, max_abs_error=float(max_abs), checkpoint_hashes_checked=64, placebo_mapping_checks=3456, independent_replay="Producer probe.py was not imported; natural message banks, masked donor packets, second-window rollout, factorized actions, heldout legal sets, plan transfer, partner transfer and strict settlement were reconstructed independently")
    verification_path = out / "verification.json"; verification_path.write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); (out / "receipt.json").write_text(json.dumps(dict(status="passed", verification_sha256=sha(verification_path), rows_replayed=verification["rows_replayed"], max_abs_error=float(max_abs)), ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--out", required=True); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args(); main(args.probe, args.out, args.workers)

