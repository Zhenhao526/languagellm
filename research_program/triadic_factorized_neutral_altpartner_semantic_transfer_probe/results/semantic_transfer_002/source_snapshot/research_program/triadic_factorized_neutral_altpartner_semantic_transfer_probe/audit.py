"""Independent replay audit for the aligned/placebo factorized probe.

This module deliberately does not import ``probe``. It reconstructs the natural
bank, message interventions, legal plan sets, settlement metrics and group
summaries from the frozen checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment as action_environment
from research_program.triadic_factorized_neutral_altpartner_study import kernel, runner as source_runner
from research_program.triadic_factorized_neutral_altpartner_direction_probe import intervention
from research_program.triadic_message_study import runner as core
from research_program.triadic_reciprocal_execution_study import environment as execution_environment

from . import design


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_arrays(spec):
    arrays = dataset.make_arrays(spec, information="PL")
    require(arrays["x_PL"].shape == (spec["world_count"], 3, 54), "Invalid PL shape")
    require(np.all(arrays["x_PL"][:, :, 53] == 0), "Full-information flag leaked")
    require(np.all((arrays["rewards"] == 1).sum(axis=1) == 2), "Two-plan support changed")
    return arrays


def action_probabilities(logits):
    intent, _, proposal, _ = kernel.factorized_distribution(logits)
    output = np.zeros((len(logits), 3, 17), dtype=np.float64)
    output[:, :, 0] = intent[:, :, 0]; output[:, :, 1:] = intent[:, :, 1, None] * proposal
    require(np.allclose(output.sum(-1), 1.0, atol=1e-12, rtol=0), "Action normalization")
    return output


def greedy(logits):
    intent, _, proposal, _ = kernel.factorized_distribution(logits)
    return np.where(intent.argmax(-1) == 1, proposal.argmax(-1) + 1, 0).astype(np.int16)


def bank(networks, arrays, live):
    n=len(arrays["packed_states"]); messages=np.empty((n,2,3,4),dtype=np.int8); probs=np.empty((n,3,17)); actions=np.empty((n,3),dtype=np.int16)
    for start in range(0,n,design.CHUNK_SIZE):
        stop=min(start+design.CHUNK_SIZE,n); trace=core.rollout(networks,arrays["x_PL"][start:stop],bool(live)); messages[start:stop]=trace["messages"]; probs[start:stop]=action_probabilities(trace["action_logits"]); actions[start:stop]=greedy(trace["action_logits"])
    require(np.array_equal(actions,probs.argmax(-1)),"Natural action mismatch")
    return dict(states=arrays["packed_states"].copy(),messages=messages,probs=probs,actions=actions)


def legal(states):
    output=[]
    for row in states:
        plans=action_environment.full_success_plans(tuple(map(int,row[:3])),tuple(map(int,row[3:7]))); require(len(plans)==2,"Legal plan count"); output.append({tuple(dataset.plan_action_indices(p)) for p in plans})
    return output


def pairs(states):
    output=[]
    for row in states:
        plans=action_environment.full_success_plans(tuple(map(int,row[:3])),tuple(map(int,row[3:7]))); output.append({tuple(p[:2]) for p in plans})
    return output


def mass(probabilities, actions):
    if not actions: return np.zeros(len(probabilities),dtype=np.float64)
    value=np.zeros(len(probabilities),dtype=np.float64)
    for action in actions: value += probabilities[:,0,action[0]]*probabilities[:,1,action[1]]*probabilities[:,2,action[2]]
    return value


def partner(probabilities,sender,recipient):
    return probabilities[:,recipient,execution_environment.PROPOSAL_ROLES[recipient]==sender].sum(1)


def physical(states,probabilities,legal_sets):
    actions=probabilities.argmax(-1).astype(np.int16); settled=execution_environment.settle(states,actions,"strict"); executed=settled["actual_pair_index"]>=0; q=np.zeros(len(states),dtype=bool)
    for i,allowed in enumerate(legal_sets):
        if executed[i]: q[i]=tuple(actions[i].tolist()) in allowed
    return float(executed.mean()),float(q.mean()),float(q[executed].mean()) if executed.any() else 0.0,actions


def mean(values): return float(np.mean(values)) if len(values) else 0.0


def run_group(networks,arrays,bk,indices,cases,sender,legal_cache,pair_cache,live):
    rid=np.asarray(cases["receiver_state_indices"],dtype=np.int64)[indices]; did=np.asarray(cases["donor_state_indices"],dtype=np.int64)[indices]; pc=np.asarray(cases["placebo_donor_case_indices"],dtype=np.int64)[indices]; pid=np.asarray(cases["donor_state_indices"],dtype=np.int64)[pc]
    at=[];pt=[];ap=[];pp=[];npv=[];av=[];pv=[];nq=[];aq=[];pq=[];nc=[];ac=[];pcq=[];ach=[];pch=[]
    for start in range(0,len(rid),design.CHUNK_SIZE):
        stop=min(start+design.CHUNK_SIZE,len(rid)); r=rid[start:stop]; d=did[start:stop]; p=pid[start:stop]; nat=bk["probs"][r]; states=bk["states"][r]
        aligned=nat if not live else intervention.intervene(networks,arrays["x_PL"][r],bk["messages"][r],np.full(len(r),sender),bk["messages"][d,0,sender])["action_probabilities"]
        placebo=nat if not live else intervention.intervene(networks,arrays["x_PL"][r],bk["messages"][r],np.full(len(r),sender),bk["messages"][p,0,sender])["action_probabilities"]
        for j in range(len(r)):
            target=legal_cache[int(r[j])]; source=legal_cache[int(d[j])]; donor=source-target; receiver=target-source; base=mass(nat[j:j+1],donor)[0]-mass(nat[j:j+1],receiver)[0]
            at.append(mass(aligned[j:j+1],donor)[0]-mass(aligned[j:j+1],receiver)[0]-base); pt.append(mass(placebo[j:j+1],donor)[0]-mass(placebo[j:j+1],receiver)[0]-base)
            te=pair_cache[int(r[j])]; se=pair_cache[int(d[j])]; aa=[]; ppj=[]
            for recipient in range(3):
                if recipient==sender: continue
                direction=int(tuple(sorted((sender,recipient))) in se)-int(tuple(sorted((sender,recipient))) in te)
                if direction:
                    aa.append(direction*(partner(aligned[j:j+1],sender,recipient)[0]-partner(nat[j:j+1],sender,recipient)[0])); ppj.append(direction*(partner(placebo[j:j+1],sender,recipient)[0]-partner(nat[j:j+1],sender,recipient)[0]))
            ap.append(mean(aa)); pp.append(mean(ppj))
        n1=physical(states,nat,[legal_cache[int(i)] for i in r]); a1=physical(states,aligned,[legal_cache[int(i)] for i in r]); p1=physical(states,placebo,[legal_cache[int(i)] for i in r])
        npv.append(n1[0]); av.append(a1[0]); pv.append(p1[0]); nq.append(n1[1]); aq.append(a1[1]); pq.append(p1[1]); nc.append(n1[2]); ac.append(a1[2]); pcq.append(p1[2]); ach.append(float(np.any(a1[3]!=n1[3],axis=1).mean())); pch.append(float(np.any(p1[3]!=n1[3],axis=1).mean()))
    return dict(rows=len(rid),aligned_plan_transfer=mean(at),placebo_plan_transfer=mean(pt),aligned_minus_placebo_plan_transfer=mean(np.asarray(at)-np.asarray(pt)),aligned_partner_transfer=mean(ap),placebo_partner_transfer=mean(pp),aligned_minus_placebo_partner_transfer=mean(np.asarray(ap)-np.asarray(pp)),natural_physical=mean(npv),aligned_physical=mean(av),placebo_physical=mean(pv),natural_q=mean(nq),aligned_q=mean(aq),placebo_q=mean(pq),natural_conditional_q=mean(nc),aligned_conditional_q=mean(ac),placebo_conditional_q=mean(pcq),aligned_action_change=mean(ach),placebo_action_change=mean(pch))


def replay(policy):
    probe=Path(policy["probe"]); static=json.loads((probe/"prepared.json").read_text()); cases=json.loads((probe/"prepared.json").read_text())["cases"]; inputs=json.loads((probe/"inputs.json").read_text()); path=Path(inputs["checkpoints"][f"{policy['seed']}:{policy['condition']}"]["path"]); require(sha(path)==policy["checkpoint_sha256"],"Checkpoint hash"); networks=source_runner.load_networks(path); arrays=make_arrays(static["partition"]); bk=bank(networks,arrays,bool(policy["live"])); legal_cache=legal(bk["states"]); pair_cache=pairs(bk["states"]); sender=np.asarray(cases["sender"],dtype=np.int8); axis=np.asarray(cases["axis"],dtype=object); groups={}
    for who in range(3):
        for name in design.AXES:
            ids=np.flatnonzero((sender==who)&(axis==name)); groups[f"{name}/{who}"]=run_group(networks,arrays,bk,ids,cases,who,legal_cache,pair_cache,bool(policy["live"]))
    return dict(worlds=len(bk["states"]),groups=groups)


def max_error(expected,actual):
    errors=[]
    for key in expected:
        if key in {"rows"}: require(int(expected[key])==int(actual[key]),"Row count mismatch"); continue
        if isinstance(expected[key],dict): errors.append(max_error(expected[key],actual[key]))
        elif isinstance(expected[key],(int,float)): errors.append(abs(float(expected[key])-float(actual[key])))
    return max(errors or [0.0])


def main(probe,out,workers=4):
    probe=Path(probe).resolve(); out=Path(out).resolve(); require(not out.exists(),"Refuse to overwrite audit"); result=json.loads((probe/"execution/results.json").read_text()); static=json.loads((probe/"prepared.json").read_text()); require(result["status"]=="completed_factorized_semantic_transfer_probe" and len(result["rows"])==64,"Incomplete result")
    for index,row in enumerate(result["rows"]): row["probe"]=str(probe)
    with multiprocessing.get_context("spawn").Pool(workers) as pool: replayed=pool.map(replay,result["rows"])
    max_abs=0.0
    for expected,actual in zip(result["rows"],replayed): max_abs=max(max_abs,max_error(expected["groups"],actual["groups"])); require(expected["worlds"]==actual["worlds"],"World count")
    cases=static["cases"]; require(cases["case_count"]==76032 and len(cases["placebo_donor_case_indices"])==76032,"Case grid"); mapping=cases["placebo_donor_case_indices"]; require(all(i!=j for i,j in enumerate(mapping)),"Placebo identity")
    require(all(int(row["intervention_module_samples"])==(76032+3*design.SHAM_ROWS_PER_SENDER)*12 if row["live"] else int(row["intervention_module_samples"])==0 for row in result["rows"]),"Forward budget")
    out.mkdir(parents=True); verification=dict(status="passed",probe=str(probe),policy_blocks=64,rows_replayed=64*76032,live_rows_replayed=32*76032,worlds_per_policy=56160,model_forward_samples=result["model_forward_samples"],optimizer_updates=0,max_abs_error=float(max_abs),checkpoint_hashes_checked=64,placebo_mapping_checks=76032,independent_replay="Producer implementation not imported; natural bank, aligned/placebo routing, legal plan mass, partner transfer, strict settlement and group summaries recomputed independently")
    (out/"verification.json").write_text(json.dumps(verification,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8"); receipt=dict(status="passed",verification_sha256=sha(out/"verification.json"),policy_blocks=64,rows_replayed=64*76032,max_abs_error=float(max_abs),optimizer_updates=0); (out/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8"); print(json.dumps(verification,ensure_ascii=False))


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--probe",required=True); parser.add_argument("--out",required=True); parser.add_argument("--workers",type=int,default=4); args=parser.parse_args(); main(args.probe,args.out,args.workers)
