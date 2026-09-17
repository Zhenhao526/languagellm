"""Post-hoc aligned/placebo message transfer over frozen factorized policies."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment as action_environment
from research_program.triadic_factorized_neutral_altpartner_study import kernel, runner as source_runner
from research_program.triadic_factorized_neutral_altpartner_direction_probe import intervention
from research_program.triadic_message_study import runner as core
from research_program.triadic_reciprocal_execution_study import environment as execution_environment

from . import design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_RUN = ROOT / "research_program/triadic_factorized_neutral_altpartner_study/results/altpair_001"
SOURCE_AGGREGATE = SOURCE_RUN.parent / "aggregation_altpair_001/results.json"
SOURCE_AUDIT = SOURCE_RUN.parent / "audit_altpair_001/verification.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf8")


def source_manifest():
    paths = [HERE / name for name in ("__init__.py", "design.py", "probe.py", "audit.py", "aggregate.py", "plan.md")]
    paths += [
        Path(source_runner.__file__), Path(source_runner.design.__file__), Path(source_runner.kernel.__file__),
        Path(source_runner.remap.__file__), Path(core.__file__), Path(core.base.__file__), Path(dataset.__file__),
        Path(action_environment.__file__), Path(execution_environment.__file__), SOURCE_RUN / "plan.json",
        SOURCE_RUN / "prepared.json", SOURCE_RUN / "freeze.json", SOURCE_AGGREGATE, SOURCE_AUDIT,
    ]
    require(all(path.is_file() for path in paths), "Missing source or frozen input")
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}


def frozen_inputs():
    source_runner.verify(SOURCE_RUN)
    aggregate = json.loads(SOURCE_AGGREGATE.read_text(encoding="utf8")); audit = json.loads(SOURCE_AUDIT.read_text(encoding="utf8"))
    require(str(aggregate.get("status", "")).startswith("completed"), "Source aggregate incomplete")
    require(audit.get("status") == "passed", "Source audit incomplete")
    checkpoints = {}
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            directory = SOURCE_RUN / "execution" / f"seed_{seed}_{condition}"
            result_path = directory / "result.json"; checkpoint = directory / "checkpoint_6000.npz"
            require(result_path.is_file() and checkpoint.is_file(), f"Missing source policy {seed}:{condition}")
            result = json.loads(result_path.read_text(encoding="utf8")); digest = sha(checkpoint)
            require(digest == result["final_checkpoint_sha256"], "Checkpoint hash mismatch")
            checkpoints[f"{seed}:{condition}"] = dict(path=str(checkpoint.resolve()), sha256=digest, seed=seed, condition=condition)
    return dict(source_run=str(SOURCE_RUN), source_plan_sha256=sha(SOURCE_RUN / "plan.json"),
                source_prepared_sha256=sha(SOURCE_RUN / "prepared.json"), source_freeze_sha256=sha(SOURCE_RUN / "freeze.json"),
                aggregate_sha256=sha(SOURCE_AGGREGATE), audit_sha256=sha(SOURCE_AUDIT), checkpoints=checkpoints)


def prepared():
    static = design.make_prepared(); static["source_sha256"] = source_manifest()
    static["runtime"] = dict(python=platform.python_version(), numpy=np.__version__)
    policy_count = len(design.SEEDS) * len(design.CONDITIONS); natural_worlds = int(static["partition"]["world_count"])
    case_rows = int(static["cases"]["case_count"]); live_policy_count = policy_count // 2
    natural_samples = policy_count * natural_worlds * 9
    intervention_samples = live_policy_count * (case_rows + 3 * design.SHAM_ROWS_PER_SENDER) * 12
    static["budget"] = dict(policy_rows=policy_count, natural_worlds=policy_count * natural_worlds,
                             natural_module_samples=natural_samples, case_rows=policy_count * case_rows,
                             sham_rows=policy_count * 3 * design.SHAM_ROWS_PER_SENDER,
                             intervention_module_samples=intervention_samples,
                             total_new_module_samples=natural_samples + intervention_samples,
                             aligned_and_placebo=True)
    return static


def make_arrays(spec):
    arrays = dataset.make_arrays(spec, information="PL")
    arrays["native_rewards"] = arrays["rewards"].copy()
    require(arrays["x_PL"].shape == (spec["world_count"], 3, 54), "Invalid PL shape")
    require(np.all(arrays["x_PL"][:, :, 53] == 0), "Full-information flag leaked")
    require(np.all((arrays["native_rewards"] == 1).sum(axis=1) == 2), "Probe world lacks two legal plans")
    return arrays


def joint_action_probabilities(logits):
    intent_p, _, proposal_p, _ = kernel.factorized_distribution(logits)
    result = np.zeros((len(logits), 3, 17), dtype=np.float64)
    result[:, :, 0] = intent_p[:, :, 0]; result[:, :, 1:] = intent_p[:, :, 1, None] * proposal_p
    require(np.allclose(result.sum(axis=-1), 1.0, atol=1e-12, rtol=0), "Action probabilities do not normalize")
    return result


def greedy_actions(logits):
    intent_p, _, proposal_p, _ = kernel.factorized_distribution(logits)
    return np.where(intent_p.argmax(-1) == 1, proposal_p.argmax(-1) + 1, 0).astype(np.int16)


def native_bank(networks, arrays, live):
    n = len(arrays["packed_states"]); bank = dict(states=arrays["packed_states"].copy(),
        messages=np.empty((n, 2, 3, 4), dtype=np.int8), action_probabilities=np.empty((n, 3, 17), dtype=np.float64),
        action_indices=np.empty((n, 3), dtype=np.int16))
    for start in range(0, n, design.CHUNK_SIZE):
        stop = min(start + design.CHUNK_SIZE, n); trace = core.rollout(networks, arrays["x_PL"][start:stop], bool(live))
        bank["messages"][start:stop] = trace["messages"]; bank["action_probabilities"][start:stop] = joint_action_probabilities(trace["action_logits"])
        bank["action_indices"][start:stop] = greedy_actions(trace["action_logits"])
    require(np.array_equal(bank["action_indices"], bank["action_probabilities"].argmax(-1)), "Natural greedy mismatch")
    return bank


def plan_actions(plan):
    return tuple(dataset.plan_action_indices(plan))


def legal_sets(states):
    output = []
    for row in states:
        plans = action_environment.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7])))
        require(len(plans) == 2, "State is not a two-plan world")
        output.append({plan_actions(plan) for plan in plans})
    return output


def pair_sets(states):
    output = []
    for row in states:
        plans = action_environment.full_success_plans(tuple(map(int, row[:3])), tuple(map(int, row[3:7])))
        output.append({tuple(plan[:2]) for plan in plans})
    return output


def mass(probabilities, action_set):
    if not action_set:
        return 0.0
    value = np.zeros(len(probabilities), dtype=np.float64)
    for actions in action_set:
        value += (probabilities[:, 0, actions[0]] * probabilities[:, 1, actions[1]] * probabilities[:, 2, actions[2]])
    return value


def partner_probability(probabilities, sender, recipient):
    mask = execution_environment.PROPOSAL_ROLES[recipient] == sender
    return probabilities[:, recipient, mask].sum(axis=1)


def physical_metrics(states, probabilities, legal):
    actions = probabilities.argmax(-1).astype(np.int16); settled = execution_environment.settle(states, actions, "strict")
    physical = settled["actual_pair_index"] >= 0; q = np.zeros(len(states), dtype=bool)
    for i, action_set in enumerate(legal):
        if physical[i]: q[i] = tuple(actions[i].tolist()) in action_set
    return dict(physical=float(physical.mean()), q=float(q.mean()), conditional_q=float(q[physical].mean()) if physical.any() else 0.0,
                action_change=None, actions=actions)


def _mean(values):
    return float(np.mean(values)) if values else 0.0


def run_group(networks, arrays, bank, case_indices, cases, sender, legal_cache, pair_cache, live):
    aligned_transfer=[]; placebo_transfer=[]; aligned_partner=[]; placebo_partner=[]
    natural_physical=[]; aligned_physical=[]; placebo_physical=[]; natural_q=[]; aligned_q=[]; placebo_q=[]
    natural_cq=[]; aligned_cq=[]; placebo_cq=[]; aligned_change=[]; placebo_change=[]
    aligned_donor_ids = np.asarray(cases["donor_state_indices"], dtype=np.int64)[case_indices]
    placebo_case_ids = np.asarray(cases["placebo_donor_case_indices"], dtype=np.int64)[case_indices]
    placebo_donor_ids = np.asarray(cases["donor_state_indices"], dtype=np.int64)[placebo_case_ids]
    receiver_ids = np.asarray(cases["receiver_state_indices"], dtype=np.int64)[case_indices]
    for start in range(0, len(receiver_ids), design.CHUNK_SIZE):
        stop = min(start + design.CHUNK_SIZE, len(receiver_ids)); ri=receiver_ids[start:stop]; di=aligned_donor_ids[start:stop]; pi=placebo_donor_ids[start:stop]
        natural=bank["action_probabilities"][ri]; states=bank["states"][ri]
        aligned = natural if not live else intervention.intervene(networks, arrays["x_PL"][ri], bank["messages"][ri], np.full(len(ri), sender), bank["messages"][di,0,sender])["action_probabilities"]
        placebo = natural if not live else intervention.intervene(networks, arrays["x_PL"][ri], bank["messages"][ri], np.full(len(ri), sender), bank["messages"][pi,0,sender])["action_probabilities"]
        for row in range(len(ri)):
            target_legal=legal_cache[int(ri[row])]; source_legal=legal_cache[int(di[row])]
            donor_only=source_legal-target_legal; target_only=target_legal-source_legal
            natural_signed=mass(natural[row:row+1], donor_only)[0]-mass(natural[row:row+1], target_only)[0]
            aligned_transfer.append((mass(aligned[row:row+1], donor_only)[0]-mass(aligned[row:row+1], target_only)[0])-natural_signed)
            placebo_transfer.append((mass(placebo[row:row+1], donor_only)[0]-mass(placebo[row:row+1], target_only)[0])-natural_signed)
            target_pairs=pair_cache[int(ri[row])]; source_pairs=pair_cache[int(di[row])]; effects_a=[]; effects_p=[]
            for recipient in range(3):
                if recipient==sender: continue
                desired=int(tuple(sorted((sender,recipient))) in source_pairs)-int(tuple(sorted((sender,recipient))) in target_pairs)
                if desired:
                    effects_a.append(desired*(partner_probability(aligned[row:row+1],sender,recipient)[0]-partner_probability(natural[row:row+1],sender,recipient)[0]))
                    effects_p.append(desired*(partner_probability(placebo[row:row+1],sender,recipient)[0]-partner_probability(natural[row:row+1],sender,recipient)[0]))
            aligned_partner.append(_mean(effects_a)); placebo_partner.append(_mean(effects_p))
        nm=physical_metrics(states,natural,[legal_cache[int(i)] for i in ri]); am=physical_metrics(states,aligned,[legal_cache[int(i)] for i in ri]); pm=physical_metrics(states,placebo,[legal_cache[int(i)] for i in ri])
        natural_physical.append(nm["physical"]); aligned_physical.append(am["physical"]); placebo_physical.append(pm["physical"])
        natural_q.append(nm["q"]); aligned_q.append(am["q"]); placebo_q.append(pm["q"]); natural_cq.append(nm["conditional_q"]); aligned_cq.append(am["conditional_q"]); placebo_cq.append(pm["conditional_q"])
        aligned_change.append(float(np.any(am["actions"] != nm["actions"], axis=1).mean())); placebo_change.append(float(np.any(pm["actions"] != nm["actions"], axis=1).mean()))
    return dict(rows=len(receiver_ids), aligned_plan_transfer=_mean(aligned_transfer), placebo_plan_transfer=_mean(placebo_transfer), aligned_minus_placebo_plan_transfer=_mean(np.asarray(aligned_transfer)-np.asarray(placebo_transfer)),
        aligned_partner_transfer=_mean(aligned_partner), placebo_partner_transfer=_mean(placebo_partner), aligned_minus_placebo_partner_transfer=_mean(np.asarray(aligned_partner)-np.asarray(placebo_partner)),
        natural_physical=_mean(natural_physical), aligned_physical=_mean(aligned_physical), placebo_physical=_mean(placebo_physical), natural_q=_mean(natural_q), aligned_q=_mean(aligned_q), placebo_q=_mean(placebo_q), natural_conditional_q=_mean(natural_cq), aligned_conditional_q=_mean(aligned_cq), placebo_conditional_q=_mean(placebo_cq), aligned_action_change=_mean(aligned_change), placebo_action_change=_mean(placebo_change))


def run_policy(networks, arrays, cases, live):
    bank=native_bank(networks, arrays, live); legal_cache=legal_sets(bank["states"]); pair_cache=pair_sets(bank["states"])
    case_sender=np.asarray(cases["sender"],dtype=np.int8); case_axis=np.asarray(cases["axis"],dtype=object); groups={}
    for sender in range(3):
        for axis in design.AXES:
            groups[f"{axis}/{sender}"]=run_group(networks,arrays,bank,np.flatnonzero((case_sender==sender)&(case_axis==axis)),cases,sender,legal_cache,pair_cache,live)
    sham=[]
    for sender in range(3):
        ids=np.flatnonzero(case_sender==sender)[:design.SHAM_ROWS_PER_SENDER]
        if live:
            same=intervention.intervene(networks,arrays["x_PL"][np.asarray(cases["receiver_state_indices"])[ids]],bank["messages"][np.asarray(cases["receiver_state_indices"])[ids]],np.full(len(ids),sender),bank["messages"][np.asarray(cases["receiver_state_indices"])[ids],0,sender])
            nat=bank["action_probabilities"][np.asarray(cases["receiver_state_indices"])[ids]]
            sham.append(dict(sender=sender, rows=len(ids),
                             action_equal=bool(np.array_equal(same["action_indices"], nat.argmax(-1))),
                             message_equal=bool(np.array_equal(same["generated_messages"][:, 0], bank["messages"][np.asarray(cases["receiver_state_indices"])[ids], 0])),
                             max_probability_error=float(np.max(np.abs(same["action_probabilities"] - nat)))))
        else: sham.append(dict(sender=sender,rows=len(ids),action_equal=True,message_equal=True,max_probability_error=0.0))
    return dict(worlds=len(bank["states"]), natural_module_samples=9*len(bank["states"]), intervention_module_samples=(len(cases["receiver_state_indices"])+3*design.SHAM_ROWS_PER_SENDER)*12 if live else 0, groups=groups, sham=sham)


def prepare(out):
    out=Path(out).resolve(); require(not out.exists(),"Never overwrite preparation"); static=prepared(); inputs=frozen_inputs(); out.mkdir(parents=True)
    for relative in static["source_sha256"]:
        source=ROOT/relative; target=out/"source_snapshot"/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(source,target)
    write(out/"prepared.json",static); write(out/"inputs.json",inputs); write(out/"plan.json",dict(status="prepared_without_probe_forward",created_at=core.base.now(),config=static.get("config",{}),prepared_sha256=sha(out/"prepared.json"),inputs_sha256=sha(out/"inputs.json"),source_sha256=static["source_sha256"])); write(out/"freeze.json",dict(plan_sha256=sha(out/"plan.json"),prepared_sha256=sha(out/"prepared.json"),inputs_sha256=sha(out/"inputs.json"))); verify(out); return dict(status="prepared_without_probe_forward",output=str(out),case_rows=static["cases"]["case_count"])


def verify(out):
    out=Path(out).resolve(); static=json.loads((out/"prepared.json").read_text()); plan=json.loads((out/"plan.json").read_text()); freeze=json.loads((out/"freeze.json").read_text()); inputs=json.loads((out/"inputs.json").read_text())
    require(sha(out/"prepared.json")==freeze["prepared_sha256"],"Prepared hash mismatch"); require(sha(out/"plan.json")==freeze["plan_sha256"],"Plan hash mismatch"); require(sha(out/"inputs.json")==freeze["inputs_sha256"],"Inputs hash mismatch"); require(static==prepared(),"Static preparation changed"); require(inputs==frozen_inputs(),"Frozen source changed")
    for relative,digest in plan["source_sha256"].items(): require(sha(out/"source_snapshot"/relative)==digest,"Snapshot changed: "+relative)
    return plan,static,inputs


def worker(payload):
    seed,static,inputs,execution=payload; execution=Path(execution); out=execution/f"seed_{seed}"; out.mkdir(parents=True,exist_ok=False); spec=static["partition"]; cases=static["cases"]; arrays=make_arrays(spec); rows=[]
    for condition in design.CONDITIONS:
        meta=inputs["checkpoints"][f"{seed}:{condition}"]; path=Path(meta["path"]); require(sha(path)==meta["sha256"],"Checkpoint binding mismatch"); networks=source_runner.load_networks(path); started=time.perf_counter(); summary=run_policy(networks,arrays,cases,condition.endswith("_live")); rows.append(dict(seed=seed,condition=condition,schedule=condition.split("_",1)[0],live=condition.endswith("_live"),checkpoint_sha256=sha(path),parameter_sha256=source_runner.parameter_hash(networks),elapsed_seconds=time.perf_counter()-started,**summary)); write(out/f"{condition}.json",rows[-1])
    return rows


def execute(out, workers=4):
    out=Path(out).resolve(); verify(out); static=json.loads((out/"prepared.json").read_text()); inputs=json.loads((out/"inputs.json").read_text()); execution=out/"execution"; require(not execution.exists(),"Never overwrite execution"); execution.mkdir(); started=time.perf_counter(); write(execution/"started.json",dict(started_at=core.base.now(),plan_sha256=sha(out/"plan.json")))
    try:
        payloads=[(seed,static,inputs,str(execution)) for seed in design.SEEDS]
        with multiprocessing.get_context("spawn").Pool(workers) as pool: groups=pool.map(worker,payloads)
        rows=[row for group in groups for row in group]; require(len(rows)==len(design.SEEDS)*len(design.CONDITIONS),"Incomplete policy grid")
        result=dict(status="completed_factorized_semantic_transfer_probe",created_at=core.base.now(),elapsed_seconds=time.perf_counter()-started,plan_sha256=sha(out/"plan.json"),rows=rows,policy_blocks=len(rows),case_count=static["cases"]["case_count"],worlds=static["partition"]["world_count"],model_forward_samples=static["budget"]["total_new_module_samples"],optimizer_updates=0,no_training_updates=True,posthoc=True)
        write(execution/"results.json",result); result_sha=sha(execution/"results.json"); write(execution/"status.json",dict(status="completed",results_sha256=result_sha)); write(execution/"receipt.json",dict(status="passed",results_sha256=result_sha,policy_blocks=len(rows),case_count=static["cases"]["case_count"],optimizer_updates=0)); return result
    except BaseException as error:
        write(execution/"failure.json",dict(status="failed",error=repr(error),elapsed_seconds=time.perf_counter()-started)); raise


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=("prepare","verify","execute")); parser.add_argument("--out",required=True); parser.add_argument("--workers",type=int,default=4); args=parser.parse_args(); answer=prepare(args.out) if args.command=="prepare" else execute(args.out,args.workers) if args.command=="execute" else verify(args.out)[0]; print(json.dumps(answer,ensure_ascii=False))
