"""Bounded endpoint and categorical-trace audit for v0.37 origin runs."""
from __future__ import annotations
import argparse, hashlib, itertools, json, shutil, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent; SEEDS=(34101,34102,34103,34104); PARTITIONS=(1,2,3); CONDITIONS=("fixed_A","rotating_AB","random_ABC"); SCHEDULES=("A","B","C"); CHECKPOINTS=(0,100,600,1200)
TEAMS={"A":((0,1,2),(1,2,3),(2,3,0),(3,0,1)),"B":((0,2,3),(1,3,0),(2,0,1),(3,1,2)),"C":((0,3,1),(1,0,2),(2,1,3),(3,2,0))}
def read(path:Path): return json.loads(path.read_text())
def write(path:Path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
def sha(path:Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def npz(path:Path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key] for key in z.files}
class Checks:
    def __init__(self): self.count=0; self.coverage=Counter()
    def require(self,v,l): self.count+=1; assert bool(v),l
    def exact(self,a,b,l): self.require(np.array_equal(a,b),l)
    def add(self,k,v=1): self.coverage[k]+=v
def schedule_for(condition,seed,part,step):
    if condition=="fixed_A":return "A"
    if condition=="rotating_AB":return "A" if step%2==0 else "B"
    rng=np.random.default_rng(np.random.SeedSequence([34037,int(seed),int(part),int(step),77])); return str(rng.choice(np.asarray(SCHEDULES)))
def check_raw(raw,worlds,checks,label):
    for key in ("map_id","photo_ids","positions","shown"):checks.exact(raw[key],worlds[key],label+"/"+key)
    n=len(worlds["map_id"]); checks.require(raw["tokens"].shape==(n,2) and raw["tokens"].dtype.kind in "iu",label+" tokens"); checks.require(((raw["tokens"]>=0)&(raw["tokens"]<7)).all(),label+" token domain")
    for key in ("sender_log_probs_food","sender_log_probs_water"):
        checks.require(raw[key].shape==(n,7) and np.isfinite(raw[key]).all(),label+" component"); checks.require(np.allclose(np.exp(raw[key]).sum(-1),1,atol=1e-6),label+" component normalization")
    checks.require(raw["sender_log_probs"].shape==(n,49) and np.isfinite(raw["sender_log_probs"]).all(),label+" joint"); checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1),1,atol=1e-6),label+" joint normalization"); checks.require(np.allclose(raw["sender_log_probs"],(raw["sender_log_probs_food"][:,:,None]+raw["sender_log_probs_water"][:,None,:]).reshape(n,49),atol=1e-6,rtol=1e-6),label+" factorization"); checks.require(raw["receiver_logits"].shape==(49,2,6) and np.isfinite(raw["receiver_logits"]).all(),label+" receiver"); checks.add("protocol_worlds",n)
def draw(probs,uniforms):return np.minimum((np.cumsum(probs,axis=-1)<uniforms[...,None]).sum(-1),probs.shape[-1]-1)
def check_trace(path,worlds,condition,seed,part,checks):
    raw=npz(path); checks.require(raw["schedule_id"].shape==(1,),"schedule shape"); schedule=str(raw["schedule_id"][0]); step=int(raw["global_step"][0]); checks.require(schedule in SCHEDULES,"schedule domain"); checks.exact(np.asarray(schedule),np.asarray(schedule_for(condition,seed,part,step)),"schedule replay")
    slots={}
    for key,value in raw.items():
        if key.startswith("world__"):
            _,slot,field=key.split("__"); slots.setdefault(slot,{})[field]=value
    checks.exact(sorted(slots),[f"slot{i}" for i in range(4)],"trace slots"); prefixes=sorted({"__".join(key.split("__")[:3]) for key in raw if key.startswith("trace__")}); checks.exact(len(prefixes),4,"trace prefix count")
    for prefix in prefixes:
        _,slot,role=prefix.split("__"); i=int(slot.removeprefix("slot")); expected=f"r{TEAMS[schedule][i][0]}_sf{TEAMS[schedule][i][1]}_sw{TEAMS[schedule][i][2]}"; checks.exact(role,expected,"trace role"); fields={key[len(prefix)+2:]:value for key,value in raw.items() if key.startswith(prefix+"__")}; indices=slots[slot]["indices"]; uniforms=slots[slot]["uniforms"]
        checks.require(indices.shape==(240,) and np.issubdtype(indices.dtype,np.integer) and ((indices>=0)&(indices<len(worlds["map_id"]))).all(),"trace indices"); checks.require(uniforms.shape==(240,4) and np.isfinite(uniforms).all() and ((uniforms>=0)&(uniforms<=1)).all(),"trace uniforms"); positions=worlds["positions"][indices]; checks.exact(fields["uniforms"],uniforms,"trace uniforms link"); checks.exact(fields["positions"],positions,"trace positions"); checks.exact(fields["messages"],np.stack((fields["token_food"],fields["token_water"]),1),"trace messages")
        for key,col,tok in (("token_probabilities_food",0,"token_food"),("token_probabilities_water",1,"token_water")):
            probs=fields[key]; checks.require(probs.shape==(240,7) and np.isfinite(probs).all() and np.allclose(probs.sum(-1),1,atol=1e-6),"token probabilities"); checks.exact(draw(probs,uniforms[:,col]),fields[tok],"token replay")
        probs=fields["action_probabilities"]; checks.require(probs.shape==(240,2,6) and np.isfinite(probs).all() and np.allclose(probs.sum(-1),1,atol=1e-6),"action probabilities"); actions=fields["actions"]; checks.exact(draw(probs,uniforms[:,2:4]),actions,"action replay"); success=(actions==positions).astype(np.float32); checks.exact(fields["success"],success,"success"); reward=(.25*success.sum(1)+.5*success.prod(1)).astype(np.float32); checks.require(np.allclose(fields["reward"],reward,atol=1e-6,rtol=1e-6),"reward"); checks.require(np.allclose(fields["advantage"],reward-.5,atol=1e-6,rtol=1e-6),"advantage"); checks.add("trace_rows",240)
    checks.add("trace_files")
def run(out:Path):
    checks=Checks(); invocation=read(out/"invocation.json"); complete=read(out/"training_complete.json"); runs=read(out/"runs.json"); checks.exact(invocation["formal"],True,"formal invocation"); checks.exact(invocation["seeds"],list(SEEDS),"seeds"); checks.exact(invocation["partitions"],list(PARTITIONS),"partitions"); checks.exact(invocation["conditions"],list(CONDITIONS),"conditions"); checks.exact(invocation["updates"],1200,"updates"); checks.exact(invocation["checkpoints"],list(CHECKPOINTS),"checkpoints"); checks.exact(complete["status"],"complete","completion"); checks.exact(complete["formal"],True,"formal completion"); checks.exact(complete["probe"],"teacher_free_origin","probe"); checks.exact(complete["runs"],36,"runs"); checks.exact(runs["count"],36,"run manifest")
    for key in ("source_hashes","input_hashes"):
        checks.exact(invocation[key],complete[key],key+" identity")
        for path,digest in invocation[key].items(): checks.require(Path(path).is_file(),key+" bound path"); checks.exact(sha(Path(path)),digest,key+" bound hash")
    worlds=npz(out/"test_worlds.npz"); train_worlds=npz(out/"train_worlds.npz"); checks.exact(len(worlds["map_id"]),180,"test worlds"); checks.exact(len(train_worlds["map_id"]),720,"train worlds")
    for seed,part,condition in itertools.product(SEEDS,PARTITIONS,CONDITIONS):
        folder=out/"social"/f"s{seed}_p{part}_{condition}"; cfg=read(folder/"config.json"); checks.exact(cfg["condition"],condition,"config condition"); curve=read(folder/"curve.json"); checks.exact([x["update"] for x in curve],list(CHECKPOINTS),"curve updates")
        for item in curve:
            update=int(item["update"])
            for schedule in SCHEDULES:
                for slot in range(4):
                    path=folder/f"protocol_{schedule}_{update:04d}_team{slot}.npz"; rel=str(path.relative_to(out)); checks.require(rel in complete["files"],"protocol hash"); check_raw(npz(path),worlds,checks,f"protocol {condition}/{seed}/{part}/{schedule}/{update}/{slot}"); checks.add("protocol_tables")
        for name,step in (("train_0001.npz",0),("train_1200.npz",1199)):
            path=folder/name; rel=str(path.relative_to(out)); checks.require(rel in complete["files"],"trace hash"); check_trace(path,train_worlds,condition,seed,part,checks); checks.exact(npz(path)["global_step"],np.asarray([step],dtype=np.int64),"trace step")
        checks.require((folder/"final.pt").is_file(),"final state")
    result={"passed":True,"status":"passed_teacher_free_origin_trace_audit","formal":True,"probe":"teacher_free_origin","checks":checks.count,"coverage":dict(checks.coverage),"source_hashes":invocation["source_hashes"],"input_hashes":invocation["input_hashes"],"training_complete_sha256":sha(out/"training_complete.json"),"audit_source_sha256":sha(ROOT/"origin_audit.py"),"exclusions":["No optimizer-state replay; the audit checks endpoint tables, categorical traces, schedule replay, reward arithmetic and completion bindings."]}; write(out/"origin_audit.json",result); shutil.copy2(ROOT/"origin_audit.py",out/"origin_audit_source.py"); return result
def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); args=parser.parse_args(); started=time.monotonic()
    try:
        result=run(args.out.resolve()); result["seconds"]=time.monotonic()-started; write(args.out.resolve()/"origin_audit.json",result); print(json.dumps({k:result[k] for k in ("passed","status","checks","coverage","seconds")},ensure_ascii=False))
    except Exception as error:
        stamp=time.time_ns(); write(args.out.resolve()/f"origin_audit_failure_{stamp}.json",{"passed":False,"error":repr(error),"traceback":traceback.format_exc(),"source_sha256":sha(ROOT/"origin_audit.py")}); raise
if __name__=="__main__":main()
