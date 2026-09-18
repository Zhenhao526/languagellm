"""Independent replay audit for structured factor-sharing executions."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from . import design,runner,environment,policy

def _read_rows(path): return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
def _check_stream(row,ep):
    for k,v in runner._stream_hashes(ep).items():
        if row[k]!=v: raise AssertionError(f"stream mismatch {k}")
def _parent_replay(execution,result):
    seed=int(result["seed"]); arch=result["architecture"]; pop=result["population"]; community=int(result["community"]); updates=int(result["updates"]); run=Path(execution)/"parents"/f"seed_{seed}_{arch}_{pop}_community_{community}"; rows=_read_rows(run/"training.jsonl"); assert len(rows)==updates
    communities,_=runner.load_checkpoint(run/"checkpoint_0000.npz",arch); params=communities[0]; max_error=0.0; cps=0
    for update,row in enumerate(rows,1):
        ep=design.episode_stream(seed,design.BATCH_SIZE,support="full",update=update); ep["partner_id"]=np.full(len(ep["goal"]),community*design.PARTNERS_PER_COMMUNITY,dtype=np.int8); ep["community_id"]=np.full(len(ep["goal"]),community,dtype=np.int8); _check_stream(row,ep); tr=environment.rollout([params],None,ep,arch,pop,"hidden"); g=runner.parent_gradient(params,ep,tr); norm,scale=runner.update_params(params,g); max_error=max(max_error,abs(row["return_mean"]-float(tr["team_return"].mean())),abs(row["gradient_norm"]-norm),abs(row["gradient_clip_scale"]-scale)); assert row["parameter_sha256"]==policy.parameter_hash(params)
        if update in result["checkpoints"]:
            assert row.get("checkpoint_sha256")==runner.sha(run/f"checkpoint_{update:04d}.npz"); cps+=1
    assert result["final_parameter_sha256"]==policy.parameter_hash(params); assert result["final_checkpoint_sha256"]==runner.sha(run/f"checkpoint_{updates:04d}.npz"); return len(rows),cps,max_error

def _child_replay(execution,result):
    seed=int(result["seed"]); cond=result["condition"]; arch,pop,vis,support=design.parse_child_condition(cond); run=Path(execution)/"children"/f"seed_{seed}_{cond}"; rows=_read_rows(run/"training.jsonl"); updates=int(result.get("updates",len(rows))); assert len(rows)==updates; communities,fresh=runner.load_checkpoint(run/"checkpoint_0000.npz",arch); assert fresh is not None; max_error=0.0;cps=0
    for update,row in enumerate(rows,1):
        ep=design.episode_stream(seed,design.BATCH_SIZE,support=support,update=update); goals=np.asarray(ep["goal_meaning"]); assert not np.any(goals==design.heldout_combo(seed)) if support=="heldout_combo" else True; assert not np.any((goals//design.VALUES)==design.heldout_value(seed)) if support=="heldout_value" else True; _check_stream(row,ep); tr=environment.rollout(communities,fresh,ep,arch,pop,vis); g=runner.child_gradient(fresh,ep,tr,vis); norm,scale=runner.update_params(fresh,g); max_error=max(max_error,abs(row["return_mean"]-float(tr["team_return"].mean())),abs(row["gradient_norm"]-norm),abs(row["gradient_clip_scale"]-scale)); assert row["parameter_sha256"]==policy.parameter_hash(fresh); assert row["community_parameter_sha256"]==[policy.parameter_hash(p) for p in communities]
        if update in result["checkpoints"]: assert row.get("checkpoint_sha256")==runner.sha(run/f"checkpoint_{update:04d}.npz"); cps+=1
    assert result["final_parameter_sha256"]==policy.parameter_hash(fresh); assert result["final_community_parameter_sha256"]==[policy.parameter_hash(p) for p in communities]; assert result["final_checkpoint_sha256"]==runner.sha(run/f"checkpoint_{updates:04d}.npz"); return len(rows),cps,max_error

def _pair(a,b,fields):
    if a is None or b is None:return 0
    aa={(int(r["seed"]),int(r["update"])):r for r in a}; bb={(int(r["seed"]),int(r["update"])):r for r in b}; keys=sorted(set(aa)&set(bb))
    for k in keys:
        for f in fields:
            if aa[k].get(f)!=bb[k].get(f): raise AssertionError(f"pair mismatch {f} {k}")
    return len(keys)

def audit(prepared,execution):
    _,cfg=runner.verify(prepared); payload=json.loads((Path(execution)/"results.json").read_text()); pl={};cl={}; pr=cr=pc=cc=0; err=0.0
    for r in payload["parents"]:
        rows,x,e=_parent_replay(execution,r); pr+=rows;pc+=x;err=max(err,e); key=(int(r["seed"]),r["architecture"],r["population"],int(r["community"])); pl[key]=_read_rows(Path(execution)/"parents"/f"seed_{r['seed']}_{r['architecture']}_{r['population']}_community_{r['community']}"/"training.jsonl")
    for r in payload["children"]:
        rows,x,e=_child_replay(execution,r); cr+=rows;cc+=x;err=max(err,e); cl[(int(r["seed"]),r["condition"])]=_read_rows(Path(execution)/"children"/f"seed_{r['seed']}_{r['condition']}"/"training.jsonl")
    def child(s,a,p,v,su): return cl.get((int(s),f"{a}_{p}_{v}_{su}"))
    fields=["scene_sha256","goal_sha256","partner_sha256","role_sha256","message_uniform_sha256","action_uniform_sha256"]; arch_pairs=vis_pairs=pop_pairs=support_pairs=0
    for s in design.SEEDS:
      for p in design.POPULATIONS:
       for v in design.VISIBILITIES:
        for su in design.SUPPORTS:
         arch_pairs+=_pair(child(s,"holistic",p,v,su),child(s,"factorized",p,v,su),fields)
       for a in design.ARCHITECTURES:
        for su in design.SUPPORTS: vis_pairs+=_pair(child(s,a,p,"hidden",su),child(s,a,p,"visible",su),fields)
      for a in design.ARCHITECTURES:
       for v in design.VISIBILITIES:
        for su in design.SUPPORTS: pop_pairs+=_pair(child(s,a,"aligned",v,su),child(s,a,"conflict",v,su),fields)
      for a in design.ARCHITECTURES:
       for p in design.POPULATIONS:
        for v in design.VISIBILITIES:
         support_pairs+=_pair(child(s,a,p,v,"full"),child(s,a,p,v,"heldout_combo"),[x for x in fields if x!="goal_sha256"]); support_pairs+=_pair(child(s,a,p,v,"full"),child(s,a,p,v,"heldout_value"),[x for x in fields if x!="goal_sha256"])
    expected_parent=len(design.SEEDS)*len(design.PARENT_CONDITIONS)*design.COMMUNITIES; expected_child=len(design.SEEDS)*len(design.CHILD_CONDITIONS); complete=len(payload["parents"])==expected_parent and len(payload["children"])==expected_child
    return {"schema":"structured_factorization_audit_v1","status":"passed" if complete and err<=1e-12 else ("partial" if err<=1e-12 else "failed"),"prepared_schema":cfg["schema"],"parent_runs":len(payload["parents"]),"child_runs":len(payload["children"]),"expected_parent_runs":expected_parent,"expected_child_runs":expected_child,"parent_training_log_rows":pr,"child_training_log_rows":cr,"parent_checkpoints":pc,"child_checkpoints":cc,"paired_architecture_rows":arch_pairs,"paired_visibility_rows":vis_pairs,"paired_population_rows":pop_pairs,"paired_support_rows":support_pairs,"max_abs_replay_error":float(err),"raw_execution_tree":str(execution)}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--prepared",required=True);p.add_argument("--execution",required=True);p.add_argument("--out",required=True);a=p.parse_args(); d=audit(a.prepared,a.execution);Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n");print(json.dumps(d,ensure_ascii=False))
