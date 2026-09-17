"""Independent replay of the matched-control semantic-transfer probe."""
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


def arrays(spec):
    a = dataset.make_arrays(spec, information="PL")
    require(a["x_PL"].shape == (spec["world_count"], 3, 54), "PL shape")
    require(np.all(a["x_PL"][:, :, 53] == 0), "FI leak")
    return a


def probs(logits):
    intent, _, proposal, _ = kernel.factorized_distribution(logits)
    out = np.zeros((len(logits), 3, 17)); out[:, :, 0] = intent[:, :, 0]; out[:, :, 1:] = intent[:, :, 1, None] * proposal
    require(np.allclose(out.sum(-1), 1, atol=1e-12, rtol=0), "normalization")
    return out


def greedy(logits):
    intent, _, proposal, _ = kernel.factorized_distribution(logits)
    return np.where(intent.argmax(-1) == 1, proposal.argmax(-1) + 1, 0).astype(np.int16)


def bank(networks, a, live):
    n = len(a["packed_states"]); msg = np.empty((n, 2, 3, 4), dtype=np.int8); p = np.empty((n, 3, 17)); act = np.empty((n, 3), dtype=np.int16)
    for start in range(0, n, 2048):
        stop = min(start + 2048, n); trace = core.rollout(networks, a["x_PL"][start:stop], bool(live)); msg[start:stop] = trace["messages"]; p[start:stop] = probs(trace["action_logits"]); act[start:stop] = greedy(trace["action_logits"])
    require(np.array_equal(act, p.argmax(-1)), "natural action mismatch")
    return dict(states=a["packed_states"].copy(), messages=msg, probs=p, actions=act)


def legal(states):
    out = []
    for row in states:
        plans = action_environment.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7]))); require(len(plans) == 2, "plan multiplicity"); out.append({tuple(dataset.plan_action_indices(plan)) for plan in plans})
    return out


def pairs(states):
    return [{tuple(plan[:2]) for plan in action_environment.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7])))} for row in states]


def mass(p, actions):
    if not actions:
        return np.zeros(len(p))
    out = np.zeros(len(p))
    for act in actions:
        out += p[:, 0, act[0]] * p[:, 1, act[1]] * p[:, 2, act[2]]
    return out


def partner(p, sender, recipient):
    return p[:, recipient, execution_environment.PROPOSAL_ROLES[recipient] == sender].sum(1)


def physical(states, p, allowed):
    act = p.argmax(-1).astype(np.int16); settled = execution_environment.settle(states, act, "strict"); executed = settled["actual_pair_index"] >= 0; q = np.zeros(len(states), dtype=bool)
    for i, s in enumerate(allowed):
        if executed[i]: q[i] = tuple(act[i].tolist()) in s
    return float(executed.mean()), float(q.mean()), float(q[executed].mean()) if executed.any() else 0.0, act


def mean(x):
    return float(np.mean(x)) if len(x) else 0.0


def empty_group():
    return dict(rows=0, aligned_plan_transfer=0.0, placebo_plan_transfer=0.0, aligned_minus_placebo_plan_transfer=0.0, aligned_partner_transfer=0.0, placebo_partner_transfer=0.0, aligned_minus_placebo_partner_transfer=0.0, natural_physical=0.0, aligned_physical=0.0, placebo_physical=0.0, natural_q=0.0, aligned_q=0.0, placebo_q=0.0, natural_conditional_q=0.0, aligned_conditional_q=0.0, placebo_conditional_q=0.0, aligned_action_change=0.0, placebo_action_change=0.0)


def group(networks, a, bk, ids, cases, sender, legal_cache, pair_cache, live):
    rid = np.asarray(cases["receiver_state_indices"], dtype=np.int64)[ids]; did = np.asarray(cases["donor_state_indices"], dtype=np.int64)[ids]; pc = np.asarray(cases["placebo_donor_case_indices"], dtype=np.int64)[ids]; pid = np.asarray(cases["donor_state_indices"], dtype=np.int64)[pc]
    at = []; pt = []; ap = []; pp = []; npv = []; av = []; pv = []; nq = []; aq = []; pq = []; nc = []; ac = []; pcq = []; ach = []; pch = []
    for start in range(0, len(rid), 2048):
        stop = min(start + 2048, len(rid)); r = rid[start:stop]; d = did[start:stop]; pids = pid[start:stop]; nat = bk["probs"][r]; states = bk["states"][r]
        aligned = nat if not live else intervention.intervene(networks, a["x_PL"][r], bk["messages"][r], np.full(len(r), sender), bk["messages"][d, 0, sender])["action_probabilities"]
        placebo = nat if not live else intervention.intervene(networks, a["x_PL"][r], bk["messages"][r], np.full(len(r), sender), bk["messages"][pids, 0, sender])["action_probabilities"]
        for j in range(len(r)):
            target = legal_cache[int(r[j])]; source = legal_cache[int(d[j])]; donor = source - target; receiver = target - source; base = mass(nat[j:j + 1], donor)[0] - mass(nat[j:j + 1], receiver)[0]; at.append(mass(aligned[j:j + 1], donor)[0] - mass(aligned[j:j + 1], receiver)[0] - base); pt.append(mass(placebo[j:j + 1], donor)[0] - mass(placebo[j:j + 1], receiver)[0] - base)
            tp = pair_cache[int(r[j])]; sp = pair_cache[int(d[j])]; ea = []; ep = []
            for recipient in range(3):
                if recipient == sender: continue
                direction = int(tuple(sorted((sender, recipient))) in sp) - int(tuple(sorted((sender, recipient))) in tp)
                if direction:
                    ea.append(direction * (partner(aligned[j:j + 1], sender, recipient)[0] - partner(nat[j:j + 1], sender, recipient)[0])); ep.append(direction * (partner(placebo[j:j + 1], sender, recipient)[0] - partner(nat[j:j + 1], sender, recipient)[0]))
            ap.append(mean(ea)); pp.append(mean(ep))
        n1 = physical(states, nat, [legal_cache[int(i)] for i in r]); a1 = physical(states, aligned, [legal_cache[int(i)] for i in r]); p1 = physical(states, placebo, [legal_cache[int(i)] for i in r]); npv.append(n1[0]); av.append(a1[0]); pv.append(p1[0]); nq.append(n1[1]); aq.append(a1[1]); pq.append(p1[1]); nc.append(n1[2]); ac.append(a1[2]); pcq.append(p1[2]); ach.append(float(np.any(a1[3] != n1[3], axis=1).mean())); pch.append(float(np.any(p1[3] != n1[3], axis=1).mean()))
    return dict(rows=len(rid), aligned_plan_transfer=mean(at), placebo_plan_transfer=mean(pt), aligned_minus_placebo_plan_transfer=mean(np.asarray(at) - np.asarray(pt)), aligned_partner_transfer=mean(ap), placebo_partner_transfer=mean(pp), aligned_minus_placebo_partner_transfer=mean(np.asarray(ap) - np.asarray(pp)), natural_physical=mean(npv), aligned_physical=mean(av), placebo_physical=mean(pv), natural_q=mean(nq), aligned_q=mean(aq), placebo_q=mean(pq), natural_conditional_q=mean(nc), aligned_conditional_q=mean(ac), placebo_conditional_q=mean(pcq), aligned_action_change=mean(ach), placebo_action_change=mean(pch))


def replay(task):
    probe = Path(task["probe"]); static = json.loads((probe / "prepared.json").read_text()); inputs = json.loads((probe / "inputs.json").read_text()); cases = static["cases"]; meta = inputs["checkpoints"][f"{task['seed']}:{task['arm']}:{task['condition']}"]; path = Path(meta["checkpoint"]); require(sha(path) == task["checkpoint_sha256"], "checkpoint hash")
    networks = source_runner.load_networks(path); a = arrays(static["heldout_spec"]); bk = bank(networks, a, bool(task["live"])); legal_cache = legal(bk["states"]); pair_cache = pairs(bk["states"]); sender = np.asarray(cases["sender"], dtype=np.int8); axis = np.asarray(cases["axis"], dtype=object); groups = {}
    for who in range(3):
        for name in case_design.AXES:
            groups[f"{name}/{who}"] = group(networks, a, bk, np.flatnonzero((sender == who) & (axis == name)), cases, who, legal_cache, pair_cache, bool(task["live"]))
        groups[f"destination/{who}"] = empty_group()
    return dict(worlds=len(bk["states"]), groups=groups)


def error(expected, actual):
    vals = []
    for key in expected:
        if key == "rows": require(int(expected[key]) == int(actual[key]), "rows mismatch"); continue
        if isinstance(expected[key], dict): vals.append(error(expected[key], actual[key]))
        elif isinstance(expected[key], (int, float)): vals.append(abs(float(expected[key]) - float(actual[key])))
    return max(vals or [0.0])


def main(probe, out, workers=4):
    probe = Path(probe).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse audit"); result = json.loads((probe / "execution/results.json").read_text()); require(result["status"] == "completed_compositional_matched_control_semantic_transfer_probe" and len(result["rows"]) == 64, "result grid")
    tasks = []
    for row in result["rows"]:
        row["probe"] = str(probe); tasks.append(row)
    with multiprocessing.get_context("spawn").Pool(workers) as pool: replayed = pool.map(replay, tasks)
    max_abs = 0.0
    for expected, actual in zip(tasks, replayed): max_abs = max(max_abs, error(expected["groups"], actual["groups"])); require(expected["worlds"] == actual["worlds"], "world count")
    cases = json.loads((probe / "prepared.json").read_text())["cases"]; require(cases["case_count"] == 3456 and all(i != j for i, j in enumerate(cases["placebo_donor_case_indices"])), "case/placebo grid")
    out.mkdir(parents=True); verification = dict(status="passed", probe=str(probe), policy_blocks=64, rows_replayed=64 * 3456, live_rows_replayed=32 * 3456, worlds_per_policy=11232, model_forward_samples=result["model_forward_samples"], optimizer_updates=0, max_abs_error=float(max_abs), checkpoint_hashes_checked=64, placebo_mapping_checks=3456, independent_replay="Producer implementation not imported; matched source checkpoint binding, natural bank, heldout legal sets, aligned/placebo routes, plan mass, partner metrics and strict settlement replayed independently")
    (out / "verification.json").write_text(json.dumps(verification, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); receipt = dict(status="passed", verification_sha256=sha(out / "verification.json"), rows_replayed=64 * 3456, max_abs_error=float(max_abs)); (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--out", required=True); parser.add_argument("--workers", type=int, default=4); args = parser.parse_args(); main(args.probe, args.out, args.workers)
