"""Independent replay audit for packet identity controls."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from research_program.triadic_message_study import runner as core
from research_program.triadic_action_dependency_study import dataset as task_dataset, environment as env
from research_program.triadic_content_response_study import dataset as content
from . import dataset, metrics

HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[1]; SOURCE=ROOT/"research_program/triadic_rule_formation_study/results/formation_001"; CONTENT=ROOT/"research_program/triadic_content_response_study/results/content_001"; TARGET="new_needs_and_layouts"; SEEDS=metrics.SEEDS; CONDITIONS=metrics.CONDITIONS; MODES=metrics.MODES; STEPS=metrics.STEPS; TOL=2e-12
def require(ok,message):
    if not ok: raise ValueError(message)
def sha(path): return dataset.sha(path)
def read(path): return dataset.read(path)
def route(t):
    t=np.asarray(t); require(t.shape[1:]==(3,4) and t.dtype.kind in "iu" and np.all((t>=0)&(t<8)),"Audit route"); vis=np.ones((3,3)); one=np.eye(8)[t]; return np.concatenate(((one[:,None]*vis[None,:,:,None,None]).reshape(len(t),3,96),np.broadcast_to(vis[None],(len(t),3,3))),axis=-1)
def replay(networks,x,natural,senders,donor):
    n=len(x); r1=route(natural[:,0]); enc=np.eye(8)[donor].reshape(n,32)
    for v in range(3):
        rows=np.flatnonzero(senders!=v); slots=32*senders[rows,None]+np.arange(32); r1[rows[:,None],v,slots]=enc[rows]
    z=np.concatenate((x,r1),-1); l2=np.stack([core.base.actor_forward(networks[3*a+1],z[:,a])[0].reshape(n,4,8) for a in range(3)],1); p2,_=core.base.policy_distribution(l2); second=np.argmax(p2,-1).astype(np.int8); r2=route(second); ai=np.concatenate((x,r1,r2),-1); logits=np.stack([core.base.actor_forward(networks[3*a+2],ai[:,a])[0] for a in range(3)],1); p,_=core.base.policy_distribution(logits); return np.stack((natural[:,0],second),1),p,np.argmax(p,-1).astype(np.int16),r1
def features(states):
    packed=np.asarray(states); worlds=[env.State(tuple(map(int,row[:3])),tuple(map(int,row[3:7])),tuple(map(int,row[7:10]))) for row in packed]; return task_dataset.encode_observations([{a:env.observe(w,a,information="PL") for a in env.AGENTS} for w in worlds])
def calc(p,row):
    p=np.asarray(p,dtype=np.float64); n=len(p); ix=np.arange(n); l=row["listeners"]; c=row["candidate_receiver_actions"]; cp=p[ix[:,None],l[:,None],c]; target=cp[ix,row["donor_endpoint"]]; cp[ix,row["donor_endpoint"]]=-np.inf; m=target-cp.max(1); G=int(row["group_index"].max())+1; B=int(row["background_index"].max())+1; tensor=m.reshape(G,B,4,4); off=~np.eye(4,dtype=bool); cell=tensor[:,:,off].reshape(G,B,12).mean(-1); gm=cell.mean(-1); return dict(M=float(gm.mean()),margin_mean=float(m.mean()),off_diagonal_margin_mean=float(m[row["host_endpoint"]!=row["donor_endpoint"]].mean()),group_means=gm.tolist(),off_diagonal_cell_means=cell.tolist())
def source_path(seed,condition,step,kind="trajectory"):
    f=SOURCE/"execution"/f"seed_{seed}_{condition}"; return f/(f"trajectory_{step:04d}_{TARGET}.npz" if kind=="trajectory" else f"checkpoint_{step:04d}.npz")
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def cmp(a,b,label,errors):
    a=np.asarray(a);b=np.asarray(b); require(a.shape==b.shape,"Shape "+label); e=float(np.max(np.abs(a-b))) if a.size else 0.; errors[label]=max(errors.get(label,0.),e); require(e<=TOL,label+" tolerance")
def audit(run,expected_plan):
    run=Path(run).resolve(); plan,prepared,control_maps=dataset.verify(run); require(sha(run/"plan.json")==expected_plan,"Plan hash"); ex=run/"execution"; result=dataset.read(ex/"results.json"); status=dataset.read(ex/"status.json"); require(status["status"]==result["status"]=="completed" and status["results_sha256"]==sha(ex/"results.json"),"Completed control");
    source_prepared=dataset.read(SOURCE/"prepared.json"); states=content.pack_states(source_prepared["partitions"][TARGET]); x=features(states); _,_,ca=dataset.source_arrays(); row=content.flatten_rows(ca); errors={}; counts=dict(records=0,live_files=0,silent_aliases=0,live_rows=0,silent_rows=0,independent_module_samples=0)
    same=__import__("research_program.triadic_packet_identity_control_study.runner",fromlist=["same_endpoint_values"]).same_endpoint_values();
    keys={(s,c,m,t) for s in SEEDS for c in CONDITIONS for m in MODES for t in STEPS}; actual={(r["seed"],r["condition"],r["mode"],r["update"]):r for r in result["records"]}; require(actual.keys()==keys,"Complete control grid")
    for s,c,m,t in [(s,c,m,t) for s in SEEDS for c in CONDITIONS for m in MODES for t in STEPS]:
        rec=actual[s,c,m,t]; src=source_path(s,c,t); require(rec["source_natural_path"]==str(src) and rec["source_natural_sha256"]==sha(src),"Natural identity"); pool=load(src); host=row["host_indices"]; donor=control_maps[f"{m}_donor_indices"]; senders=row["senders"]; natural=pool["messages"][host]; packets=pool["messages"][donor,0,senders,:]
        if rec["live"]:
            nets=core.load_networks(source_path(s,c,t,"checkpoint")); ps=[];ms=[];acs=[]
            for start in range(0,len(host),1024):
                sl=slice(start,min(start+1024,len(host))); mm,pp,aa,_=replay(nets,x[host[sl]],natural[sl],senders[sl],packets[sl]);ps.append(pp);ms.append(mm);acs.append(aa)
            p=np.concatenate(ps); mm=np.concatenate(ms); aa=np.concatenate(acs); saved=load(rec["path"]); require(rec["data_sha256"]==sha(rec["path"]),"Probe hash"); cmp(saved["action_probabilities"],p,"probability",errors); require(np.array_equal(saved["action_indices"],aa) and np.array_equal(saved["messages"],mm),"Saved actions/messages")
            for field,expected in (("host_indices",host),("donor_indices",donor),("source_group_index",control_maps[f"{m}_source_group_index"]),("source_packet_endpoint",control_maps[f"{m}_source_packet_endpoint"]),("group_index",row["group_index"]),("background_index",row["background_index"]),("host_endpoint",row["host_endpoint"]),("donor_endpoint",row["donor_endpoint"]),("senders",senders),("listeners",row["listeners"]),("candidate_receiver_actions",row["candidate_receiver_actions"]),("target_receiver_actions",row["target_receiver_actions"]),("donor_packets",packets)): require(np.array_equal(saved[field],expected),"Metadata "+field)
            z=calc(p,row)
            for field in ("M","margin_mean","off_diagonal_margin_mean","group_means","off_diagonal_cell_means"): cmp(rec["metrics"][field],z[field],"metric_"+field,errors)
            counts["live_files"]+=1;counts["live_rows"]+=len(host);counts["independent_module_samples"]+=6*len(host)
        else:
            z=calc(pool["action_probabilities"][host],row)
            for field in ("M","margin_mean","off_diagonal_margin_mean","group_means","off_diagonal_cell_means"): cmp(rec["metrics"][field],z[field],"silent_metric_"+field,errors)
            require(rec["path"] is None and rec["alias_of_source"] and rec["neural_forward_samples"]==0,"Silent alias")
            counts["silent_aliases"]+=1;counts["silent_rows"]+=len(host)
        counts["records"]+=1
    require(result["primary"] == metrics.primary(result["records"], same), "Primary")
    require(counts==dict(records=768,live_files=384,silent_aliases=384,live_rows=384*27648,silent_rows=384*27648,independent_module_samples=63700992),"Audit scope")
    return dict(status="passed",plan_sha256=expected_plan,scope=counts,max_errors=errors,primary=result["primary"],audit_source_sha256=sha(__file__))
def freeze(run,expected_plan):
    path=HERE/"audit_freeze_001.json";require(not path.exists(),"Never overwrite freeze");require(not (Path(run)/"execution").exists(),"Freeze precedes execution");require(sha(Path(run)/"plan.json")==expected_plan,"Plan hash"); val=dict(status="frozen_before_control_execution",at=time.time(),plan_sha256=expected_plan,source_sha256={str(HERE/"audit.py"):sha(HERE/"audit.py")},policy_output_files_read=0,neural_forward_samples=0,independent_replay_module_samples=63700992); dataset.write(path,val); return val
if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--run",required=True);ap.add_argument("--plan-sha",required=True);ap.add_argument("--out");ap.add_argument("--freeze",action="store_true");a=ap.parse_args()
    if a.freeze: print(json.dumps(freeze(a.run,a.plan_sha),ensure_ascii=False))
    else:
        out=Path(a.out).resolve();require(not out.exists(),"Never overwrite audit");out.mkdir(parents=True); start=time.perf_counter()
        try:
            value=audit(a.run,a.plan_sha);value["elapsed_seconds"]=time.perf_counter()-start;dataset.write(out/"verification.json",value);print(json.dumps(value,ensure_ascii=False))
        except BaseException as e: dataset.write(out/"failure.json",dict(status="failed",error=repr(e)));raise
