"""Streaming replay and paired-channel audit for multi-generation chains."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from . import design,runner,policy

def finite(x):
    if isinstance(x,dict): return all(finite(v) for v in x.values())
    if isinstance(x,list): return all(finite(v) for v in x)
    if isinstance(x,(float,int,np.number)): return math.isfinite(float(x))
    return True

def load_params(path):
    with np.load(path,allow_pickle=False) as z: return {k:np.asarray(z[k],dtype=np.float64).copy() for k in z.files if k!="update"}

def same_float(a,b):
    if a is None or b is None: return a is None and b is None
    return abs(float(a)-float(b))

def audit(prepared,execution,out):
    _,cfg=runner.verify(prepared); execution=Path(execution); progress=json.loads((execution/"progress.json").read_text()); files=sorted(execution.glob("seed_*/result.json"));
    design.require(progress["completed"]==progress["total"]==len(files),"run accounting mismatch")
    expected={(s,c) for s in design.SEEDS for c in design.CONDITIONS}; seen=set(); logs=checkpoints=0; max_error=0.0; rows=[]; paths={}
    for path in files:
        r=json.loads(path.read_text()); design.require(finite(r),"nonfinite result"); key=(int(r["seed"]),r["condition"]); design.require(key in expected and key not in seen,f"invalid or duplicate run {key}"); seen.add(key)
        seed=int(r["seed"]); rep,ch=design.parse_condition(r["condition"]); parent_path=runner.parent_path(cfg["parent_root"],seed); design.require(r["parent_checkpoint_sha256"]==runner.sha(parent_path),"parent hash mismatch")
        p=policy.load(parent_path); design.require(r["parent_parameter_sha256"]==policy.combined_hash(p),"parent parameter mismatch")
        expected_parent_sequences={str(tuple(goal)):runner.sequence(p,goal).tolist() for goal in ((0,0),(0,1),(1,0),(1,1))}
        design.require(r.get("parent_sender_sequences")==expected_parent_sequences,"parent codebook mismatch")
        run_dir=path.parent; log_path=run_dir/"training.jsonl"; paths[key]=log_path; row_count=0; event_rows={}
        for line in log_path.open():
            if not line.strip(): continue
            row=json.loads(line); row_count+=1; gen=int(row["generation"]); update=int(row["update"]); role=row["event_role"]; design.require(gen in design.GENERATIONS and role==design.EVENT_ROLES[gen],"bad event row")
            if gen not in event_rows: event_rows[gen]=0
            event_rows[gen]+=1
        design.require(row_count==len(design.GENERATIONS)*int(r["updates_per_event"]),"training row count mismatch")
        # Re-run the chain, this time checking every logged update and checkpoint.
        p=policy.load(parent_path)
        for gen in design.GENERATIONS:
            role=design.EVENT_ROLES[gen]; p=policy.replace(p,seed,gen,role); ev=next(x for x in r["events"] if int(x["generation"])==gen); design.require(ev["start_parameter_sha256"]==policy.combined_hash(p),"event start mismatch")
            cp0=run_dir/f"checkpoint_g{gen}_{0:04d}.npz"; design.require(cp0.is_file() and ev["checkpoint_sha256"].get("0")==runner.sha(cp0),"initial checkpoint receipt mismatch")
            z=load_params(cp0)
            for k in p: max_error=max(max_error,float(np.max(np.abs(z[k]-p[k]))))
            with log_path.open() as fh:
                # Skip rows from earlier generations; chain logs are ordered by event.
                for line in fh:
                    row=json.loads(line)
                    if int(row["generation"])!=gen: continue
                    update=int(row["update"]); ep=design.episode_stream(seed,design.BATCH_SIZE,generation=gen,update=update,support=design.SUPPORT,heldout_goal_index=design.heldout_goal(seed));
                    design.require(row["world_sha256"]==design.array_sha(ep["site_type"]) and row["goal_sha256"]==design.array_sha(ep["goal"]) and row["partner_sha256"]==design.array_sha(ep["partner_id"]) and row["message_uniform_sha256"]==design.array_sha(ep["message_uniforms"]) and row["action_uniform_sha256"]==design.array_sha(ep["action_uniforms"]),"stream mismatch")
                    tr=runner.rollout(p,ep,rep,ch,sample=True); g=runner.gradient(p,ep,tr,role,rep,design.entropy_coefficient(update)); norm=float(np.sqrt(sum(float((v*v).sum()) for v in g.values()))); scale=min(1.0,5.0/max(norm,1e-12));
                    max_error=max(max_error,abs(float(tr["team_return"][tr["active"]].mean())-float(row["return_mean"])),abs(norm-float(row["gradient_norm"])),abs(scale-float(row["gradient_clip_scale"])))
                    if role=="worker": p["worker_logits"][design.TARGET_WORKER]-=design.LEARNING_RATE*scale*g["worker_logits"][design.TARGET_WORKER]
                    else: p["sender_logits_hidden"]-=design.LEARNING_RATE*scale*g["sender_logits_hidden"]
                    design.require(row["parameter_sha256"]==policy.combined_hash(p),"parameter replay mismatch")
                    if update in ev["checkpoints"]:
                        cp=run_dir/f"checkpoint_g{gen}_{update:04d}.npz"; design.require(cp.is_file() and row.get("checkpoint_sha256")==runner.sha(cp),"checkpoint receipt mismatch"); got=load_params(cp)
                        for k in got: max_error=max(max_error,float(np.max(np.abs(got[k]-p[k]))))
                        checkpoints+=1
                    logs+=1
            design.require(ev["final_parameter_sha256"]==policy.combined_hash(p),"event final parameter mismatch")
            fresh=runner.evaluate(p,seed,rep)
            for kind in ("all","seen","heldout"):
                for mode in ("natural","silent","permuted","recombined"):
                    max_error=max(max_error,same_float(fresh[kind][mode]["team_return_mean"],ev["evaluation"][kind][mode]["team_return_mean"]))
            max_error=max(max_error,abs(float(fresh["heldout_recombined_gap"])-float(ev["evaluation"]["heldout_recombined_gap"])))
            for key in ("semantic_success_by_goal","semantic_success_mean"):
                if key == "semantic_success_by_goal":
                    got=fresh["codebook"][key]; logged=ev["evaluation"]["codebook"][key]
                    design.require(len(got)==len(logged),"semantic probe length mismatch")
                    max_error=max(max_error,max((abs(float(a)-float(b)) for a,b in zip(got,logged)),default=0.0))
                else:
                    max_error=max(max_error,abs(float(fresh["codebook"][key])-float(ev["evaluation"]["codebook"][key])))
        design.require(r["final_parameter_sha256"]==policy.combined_hash(p),"chain final parameter mismatch")
        rows.extend({"seed":seed,"condition":r["condition"],"representation":rep,"channel":ch,"generation":int(ev["generation"]),"event_role":ev["event_role"],"heldout_live_natural":ev["evaluation"]["heldout"]["natural"]["team_return_mean"],"composable":bool(ev["composable"]),"parent_composable":bool(cfg["parent_composable"][str(seed)])} for ev in r["events"])
    # Stream-pair the live and silent logs without retaining the million-row logs in RAM.
    pair_rows=0
    for seed in sorted({s for s,_ in seen}):
        for rep in design.REPRESENTATIONS:
            a=paths[(seed,f"{rep}_live")].open(); b=paths[(seed,f"{rep}_silent")].open()
            while True:
                la=a.readline(); lb=b.readline()
                if not la and not lb: break
                design.require(bool(la) and bool(lb),"paired log length mismatch")
                x=json.loads(la); y=json.loads(lb)
                for k in ("generation","event_role","update","world_sha256","goal_sha256","partner_sha256","message_uniform_sha256","action_uniform_sha256"): design.require(x[k]==y[k],f"paired stream mismatch {seed}/{rep}/{k}")
                pair_rows+=1
            a.close(); b.close()
    design.require(len(seen)==progress["total"] and max_error<1e-12,"audit failure")
    report={"schema":"multi_generation_chain_audit_v1","status":"passed","chains":len(files),"runs":len(files)*len(design.GENERATIONS),"training_log_rows":logs,"final_checkpoints":checkpoints,"max_abs_replay_error":max_error,"paired_channel_trajectory_rows":pair_rows,"rows":rows}
    Path(out).write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n"); print(json.dumps({k:report[k] for k in ("schema","status","chains","runs","training_log_rows","final_checkpoints","max_abs_replay_error","paired_channel_trajectory_rows")},ensure_ascii=False))

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--prepared",required=True); ap.add_argument("--execution",required=True); ap.add_argument("--out",required=True); a=ap.parse_args(); audit(a.prepared,a.execution,a.out)
