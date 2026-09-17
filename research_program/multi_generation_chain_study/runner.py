"""Train, audit and evaluate an alternating multi-generation signaling chain."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import shutil
import time
from pathlib import Path
import numpy as np
from . import design, environment, policy
from research_program.action_dependent_signaling_study import design as action_design

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def json_bytes(x): return (json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))+"\n").encode()
def finite(x):
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,list): return all(finite(v) for v in x)
    if isinstance(x,(float,int,np.number)): return bool(np.isfinite(float(x)))
    return True

def source_hashes():
    rels=("__init__.py","design.py","policy.py","environment.py","runner.py","aggregate.py","audit.py","plan.md","tests/test_game.py")
    return {str((HERE/x).relative_to(ROOT)):sha(HERE/x) for x in rels}

def dependency_hashes():
    rels=("research_program/action_dependent_signaling_study/design.py","research_program/action_dependent_signaling_study/policy.py","research_program/action_dependent_signaling_study/environment.py","research_program/action_dependent_signaling_study/policy.py","research_program/heldout_composition_study/design.py")
    return {x:sha(ROOT/x) for x in dict.fromkeys(rels)}

def parent_path(parent_root,seed):
    return Path(parent_root)/f"seed_{seed}_{design.PARENT_CONDITION}"/f"checkpoint_{design.CHECKPOINTS[-1]:04d}.npz"

def parent_hashes(parent_root):
    out={}
    for seed in design.SEEDS:
        p=parent_path(parent_root,seed); design.require(p.is_file(),f"missing parent checkpoint {p}"); out[str(seed)]=sha(p)
    return out

def parent_composable_from_analysis(path):
    payload=json.loads(Path(path).read_text()); out={}
    for row in payload["rows"]:
        if row.get("role")!="worker" or row.get("channel")!="live": continue
        seed=int(row["seed"]); status=bool(row["parent_composable"])
        if seed in out and out[seed]!=status: raise ValueError(f"inconsistent parent status {seed}")
        out[seed]=status
    design.require(set(out)==set(design.SEEDS),"parent analysis does not cover all seeds")
    return out

def prepare(out,parent_root,parent_analysis):
    out=Path(out).resolve(); design.require(not out.exists(),"refuse overwrite")
    parent_root=Path(parent_root).resolve(); cfg=design.prepare(parent_root,parent_hashes(parent_root),parent_composable_from_analysis(parent_analysis))
    cfg.update({"runs":len(design.SEEDS)*len(design.CONDITIONS),"chain_events":len(design.GENERATIONS),"training_rows_per_chain":len(design.GENERATIONS)*design.UPDATES,"evaluation_episodes_per_goal":1024})
    src=source_hashes(); deps=dependency_hashes(); out.mkdir(parents=True)
    for rel in src:
        q=out/"source_snapshot"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/rel,q)
    for rel in deps:
        q=out/"dependency_snapshot"/rel; q.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/rel,q)
    shutil.copy2(parent_analysis,out/"parent_analysis.json")
    (out/"prepared.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2)+"\n")
    plan={"schema":"multi_generation_chain_study_v1","prepared_sha256":sha(out/"prepared.json"),"parent_analysis_sha256":sha(out/"parent_analysis.json"),"sources":src,"dependencies":deps,"runtime":{"python":platform.python_version(),"numpy":np.__version__},"config":cfg}
    (out/"plan.json").write_text(json.dumps(plan,ensure_ascii=False,indent=2)+"\n")
    (out/"freeze.json").write_text(json.dumps({"plan_sha256":sha(out/"plan.json"),"prepared_sha256":sha(out/"prepared.json"),"parent_analysis_sha256":sha(out/"parent_analysis.json")},indent=2)+"\n")
    verify(out); return {"status":"prepared","out":str(out),"plan_sha256":sha(out/"plan.json"),"prepared_sha256":sha(out/"prepared.json")}

def verify(out):
    out=Path(out); plan=json.loads((out/"plan.json").read_text()); cfg=json.loads((out/"prepared.json").read_text()); fr=json.loads((out/"freeze.json").read_text())
    design.require(sha(out/"plan.json")==fr["plan_sha256"] and sha(out/"prepared.json")==fr["prepared_sha256"]==plan["prepared_sha256"],"freeze hash mismatch")
    design.require(sha(out/"parent_analysis.json")==fr["parent_analysis_sha256"]==plan["parent_analysis_sha256"],"parent analysis changed")
    design.require(plan["config"]==cfg and plan["sources"]==source_hashes() and plan["dependencies"]==dependency_hashes(),"source or config changed")
    for rel,d in plan["sources"].items(): design.require(sha(out/"source_snapshot"/rel)==d,"source snapshot changed "+rel)
    for rel,d in plan["dependencies"].items(): design.require(sha(out/"dependency_snapshot"/rel)==d,"dependency snapshot changed "+rel)
    design.require(parent_hashes(cfg["parent_root"])==cfg["parent_checkpoint_sha256"],"parent checkpoint changed")
    return plan,cfg

def future_returns(rewards): return np.flip(np.cumsum(np.flip(rewards,axis=1),axis=1),axis=1)
def center(values,keys):
    values=np.asarray(values,dtype=np.float64); keys=np.asarray(keys); out=np.zeros_like(values)
    for key in np.unique(keys):
        idx=np.flatnonzero(keys==key)
        if len(idx)>1: out[idx]=values[idx]-(values[idx].sum()-values[idx])/(len(idx)-1)
    return out

def entropy_grad(prob):
    lp=np.log(np.maximum(prob,1e-300)); ent=-(prob*lp).sum(axis=-1,keepdims=True); return -prob*(lp+ent)

def rollout(p,ep,representation,channel,*,sample,message_mode="natural",partner_filter=None,message_override=None):
    return environment.rollout(p,ep,representation,channel,sample=sample,message_mode=message_mode,partner_filter=partner_filter,message_override=message_override)

def gradient(p,ep,tr,role,representation,beta):
    g={k:np.zeros_like(v) for k,v in p.items()}
    future=future_returns(tr["rewards"])
    if role=="worker":
        idx=np.flatnonzero(tr["active"])
        if len(idx)==0: return g
        for t in range(design.ACTION_START,design.HORIZON):
            ap=tr["action_probs"][t][idx]; state=tr["state"][t][idx]; local=tr["local"][t][idx]; inv=tr["inventory"][t][idx]
            adv=center(future[idx,t],state*1000+t*100+local*10+inv)
            one=np.zeros_like(ap); one[np.arange(len(idx)),tr["actions"][idx,t]]=1.0
            d=-(adv[:,None]*(one-ap))/len(idx)-beta*entropy_grad(ap)/len(idx)
            np.add.at(g["worker_logits"],(np.full(len(idx),design.TARGET_WORKER,dtype=np.int64),state,np.full(len(idx),t,dtype=np.int64),local,inv),d)
    else:
        if role!="sender" or tr["active"].sum()==0: return g
        context=action_design.goal_index(ep["goal"]); arrivals=action_design.message_arrival_times("staged","dual2")
        n=len(ep["goal"])
        for slot,arrival in enumerate(arrivals):
            sp=tr["message_probs"][slot]; one=np.zeros_like(sp); one[np.arange(n),tr["selected_messages"][:,slot]]=1.0
            adv=center(future[:,arrival+1:].sum(axis=1),context)
            d=-(adv[:,None]*(one-sp))/n-beta*entropy_grad(sp)/n
            np.add.at(g["sender_logits_hidden"][:,slot,:],context,d)
    return g

def sequence(p,goal):
    idx=int(goal[0])*2+int(goal[1]); return np.asarray([policy.softmax(p["sender_logits_hidden"][idx,slot]).argmax() for slot in range(design.MESSAGE_SLOTS)],dtype=np.int64)

def recombination_override(p,ep):
    donors=np.asarray([[0,0],[0,1],[1,0],[1,1]],dtype=np.int8); out=np.zeros((len(ep["goal"]),design.MESSAGE_SLOTS),dtype=np.int64)
    for i,goal in enumerate(ep["goal"]):
        target=np.asarray(goal,dtype=np.int8)
        d0=donors[np.flatnonzero(donors[:,0]==target[0])[0]]; d1=donors[np.flatnonzero(donors[:,1]==target[1])[0]]
        out[i,0]=sequence(p,d0)[0]; out[i,1]=sequence(p,d1)[1]
    return out

def semantic_probe(p, seed, representation):
    """Probe the sender/receiver mapping on every goal and local layout.

    This is a descriptive readout.  It is evaluated after training and never
    enters the gradient or the held-out composability rule.
    """
    patterns = np.asarray([[0,0],[0,1],[1,0],[1,1]], dtype=np.int8)
    by_goal = []
    for goal_idx in range(4):
        goal = np.asarray([goal_idx // 2, goal_idx % 2], dtype=np.int8)
        scores = []
        for local_pattern in patterns:
            ep = {
                "site_type": np.stack([local_pattern, 1 - local_pattern], axis=-1)[None, ...],
                "goal": goal[None, :].copy(),
                "target_bits": action_design.target_bits(goal[None, :], design.TASK),
                "partner_id": np.array([design.TARGET_WORKER], dtype=np.int8),
                "message_uniforms": np.zeros((1, design.MESSAGE_SLOTS), dtype=np.float64),
                "action_uniforms": np.zeros((1, design.HORIZON), dtype=np.float64),
                "capacity": np.full((1, design.SUBTASKS, 2), design.CAPACITY, dtype=np.int8),
            }
            tr = rollout(p, ep, representation, "live", sample=False, partner_filter=design.TARGET_WORKER)
            scores.append(float(tr["team_return"][0]))
        by_goal.append(float(np.mean(scores)))
    sequences = {str((int(goal[0]), int(goal[1]))): sequence(p, goal).tolist() for goal in patterns}
    return {
        "semantic_success_by_goal": by_goal,
        "semantic_success_mean": float(np.mean(by_goal)),
        "sender_sequences": sequences,
    }

def metrics(p,ep,representation,mode,*,partner_filter=design.TARGET_WORKER,override=None):
    channel="silent" if mode=="silent" else "live"; message_mode="natural" if mode in ("natural","silent") else ("permuted" if mode=="permuted" else "natural")
    tr=rollout(p,ep,representation,channel,sample=False,message_mode=message_mode,partner_filter=partner_filter,message_override=override); mask=tr["active"]; r=tr["team_return"][mask]
    return {"episodes":int(mask.sum()),"team_return_mean":float(r.mean()) if len(r) else None,"team_return_sd":float(r.std()) if len(r) else None,"positive_episode_rate":float((r>0).mean()) if len(r) else None}

def evaluate(p,seed,representation):
    held=design.heldout_goal(seed); out={}
    for kind in ("all","seen","heldout"):
        ep=design.balanced_eval_stream(seed+design.EVAL_SEED_OFFSET,4096,kind,held)
        out[kind]={"natural":metrics(p,ep,representation,"natural"),"silent":metrics(p,ep,representation,"silent"),"permuted":metrics(p,ep,representation,"permuted"),"recombined":metrics(p,ep,representation,"recombined",override=recombination_override(p,ep))}
    h=out["heldout"]; n=float(h["natural"]["team_return_mean"]); re=float(h["recombined"]["team_return_mean"])
    out["heldout_composable"]=(n>=0.60 and abs(re-n)<=0.02)
    out["heldout_recombined_gap"]=re-n
    out["codebook"] = semantic_probe(p, seed, representation)
    return out

def train_one(seed,condition,execution,parent_checkpoint,updates=None):
    representation,channel=design.parse_condition(condition); updates=design.UPDATES if updates is None else int(updates)
    run=Path(execution)/f"seed_{seed}_{condition}"; run.mkdir(parents=True,exist_ok=False)
    parent=policy.load(parent_checkpoint); p=policy.clone(parent); events=[]; rows=[]; parent_hash=policy.combined_hash(parent); start_all=time.perf_counter()
    parent_sender_sequences={str(tuple(goal)):sequence(parent,goal).tolist() for goal in ((0,0),(0,1),(1,0),(1,1))}
    for generation in design.GENERATIONS:
        role=design.EVENT_ROLES[generation]; p=policy.replace(p,seed,generation,role); start_hash=policy.combined_hash(p)
        checkpoints=sorted(set([u for u in design.CHECKPOINTS if u<=updates]+[updates])); cp_receipts={}
        if 0 in checkpoints: cp_receipts[0]=policy.save(run/f"checkpoint_g{generation}_{0:04d}.npz",p,0)
        event_start=time.perf_counter()
        for update in range(1,updates+1):
            ep=design.episode_stream(seed,design.BATCH_SIZE,generation=generation,update=update,support=design.SUPPORT,heldout_goal_index=design.heldout_goal(seed))
            tr=rollout(p,ep,representation,channel,sample=True); beta=design.entropy_coefficient(update); g=gradient(p,ep,tr,role,representation,beta)
            norm=float(np.sqrt(sum(float((v*v).sum()) for v in g.values()))); scale=min(1.0,5.0/max(norm,1e-12))
            if role=="worker": p["worker_logits"][design.TARGET_WORKER]-=design.LEARNING_RATE*scale*g["worker_logits"][design.TARGET_WORKER]
            else: p["sender_logits_hidden"]-=design.LEARNING_RATE*scale*g["sender_logits_hidden"]
            row={"generation":generation,"event_role":role,"update":update,"seed":seed,"condition":condition,"representation":representation,"channel":channel,"support":design.SUPPORT,"world_sha256":design.array_sha(ep["site_type"]),"goal_sha256":design.array_sha(ep["goal"]),"partner_sha256":design.array_sha(ep["partner_id"]),"message_uniform_sha256":design.array_sha(ep["message_uniforms"]),"action_uniform_sha256":design.array_sha(ep["action_uniforms"]),"return_mean":float(tr["team_return"][tr["active"]].mean()) if tr["active"].any() else 0.0,"active_count":int(tr["active"].sum()),"gradient_norm":norm,"gradient_clip_scale":scale,"parameter_sha256":policy.combined_hash(p),"elapsed_seconds":time.perf_counter()-event_start}
            if update in checkpoints:
                cp_receipts[update]=policy.save(run/f"checkpoint_g{generation}_{update:04d}.npz",p,update); row["checkpoint_sha256"]=cp_receipts[update]
            rows.append(row)
        evaluation=evaluate(p,seed,representation); events.append({"generation":generation,"event_role":role,"start_parameter_sha256":start_hash,"final_parameter_sha256":policy.combined_hash(p),"checkpoints":checkpoints,"checkpoint_sha256":{str(k):v for k,v in cp_receipts.items()},"evaluation":evaluation,"composable":bool(evaluation["heldout_composable"]),"elapsed_seconds":time.perf_counter()-event_start})
    log=run/"training.jsonl"; log.write_bytes(b"".join(json_bytes(row) for row in rows))
    result={"seed":seed,"condition":condition,"representation":representation,"channel":channel,"support":design.SUPPORT,"heldout_goal":design.heldout_goal(seed),"parent_checkpoint":str(parent_checkpoint),"parent_checkpoint_sha256":policy.file_hash(parent_checkpoint),"parent_parameter_sha256":parent_hash,"parent_sender_sequences":parent_sender_sequences,"updates_per_event":updates,"generations":list(design.GENERATIONS),"events":events,"training_log_sha256":policy.file_hash(log),"final_parameter_sha256":policy.combined_hash(p),"elapsed_seconds":time.perf_counter()-start_all}
    (run/"result.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n"); return result

def run_grid(prepared,out,updates=None,seeds=None,conditions=None):
    _,cfg=verify(prepared); out=Path(out); execution=out/"execution"; execution.mkdir(parents=True)
    seeds=tuple(design.SEEDS if seeds is None else seeds); conditions=tuple(design.CONDITIONS if conditions is None else conditions); design.require(set(seeds)<=set(design.SEEDS) and set(conditions)<=set(design.CONDITIONS),"invalid subset")
    allr=[]
    for seed in seeds:
        for condition in conditions:
            allr.append(train_one(seed,condition,execution,parent_path(cfg["parent_root"],seed),updates))
            (execution/"progress.json").write_text(json.dumps({"completed":len(allr),"total":len(seeds)*len(conditions),"seeds":list(seeds),"conditions":list(conditions)},indent=2))
    (execution/"results.json").write_text(json.dumps({"results":allr},ensure_ascii=False,indent=2)+"\n"); return allr

if __name__=="__main__":
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True); a=sub.add_parser("prepare"); a.add_argument("--out",required=True); a.add_argument("--parent-root",required=True); a.add_argument("--parent-analysis",required=True); a=sub.add_parser("execute"); a.add_argument("--out",required=True); a.add_argument("--prepared",required=True); a.add_argument("--updates",type=int,default=None); a.add_argument("--seeds"); a.add_argument("--conditions"); args=ap.parse_args()
    if args.cmd=="prepare": print(json.dumps(prepare(args.out,args.parent_root,args.parent_analysis),ensure_ascii=False))
    else:
        seeds=None if args.seeds is None else tuple(int(x) for x in args.seeds.split(",") if x); conditions=None if args.conditions is None else tuple(x for x in args.conditions.split(",") if x); print(json.dumps({"status":"completed","runs":len(run_grid(args.prepared,args.out,args.updates,seeds,conditions))},ensure_ascii=False))
