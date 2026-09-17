"""Independent NumPy audit for the v0.46 crossed adaptation batch."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "results" / "cross_schedule_001"
V043 = PROJECT / "redesign_v0.43"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
ASSIGNMENTS = ("012", "021", "102", "120", "201", "210")
ROLE_MODES = ("static_role", "random_role")
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
ADAPTATION_SCHEDULES = CONDITIONS
CULTURES = tuple(f"{mode}__{condition}" for mode in ROLE_MODES for condition in CONDITIONS)
SCHEDULES = ("A", "B", "C")
ROLE_PERMS = tuple(itertools.permutations(range(3)))
ROLE_KEYS = tuple("".join(map(str, p)) for p in ROLE_PERMS)
UPDATES = 300
TRACE_STEPS = (0, UPDATES - 1)
BATCH, TRAIN_WORLDS, RESOURCES, TOKENS, VOCAB, SITES = 240, 480, 3, 2, 7, 6
SCHEDULE_NAMESPACE, FIXTURE_NAMESPACE, ROLE_NAMESPACE, RESET_NAMESPACE = 45040, 45041, 45042, 45043
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Checks:
    def __init__(self):
        self.checks = 0; self.comparisons = 0; self.max_error = 0.0; self.coverage = defaultdict(int)
    def require(self, value, label):
        self.checks += 1
        if not bool(value): raise AssertionError(label)
    def exact(self, actual, expected, label):
        a,b=np.asarray(actual),np.asarray(expected); self.require(a.shape==b.shape,label+" shape"); self.require(np.array_equal(a,b),label)
    def close(self, actual, expected, label, atol=2e-5, rtol=2e-5):
        a,b=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float); self.require(a.shape==b.shape,label+" shape"); self.require(np.isfinite(a).all() and np.isfinite(b).all(),label+" finite"); self.comparisons+=int(a.size); err=float(np.max(np.abs(a-b))) if a.size else 0.0; self.max_error=max(self.max_error,err); self.require(np.allclose(a,b,atol=atol,rtol=rtol),label)


def schedule_for(condition, seed, part, step):
    if condition == "fixed_A":
        return "A"
    if condition == "rotating_AB":
        return "A" if step % 2 == 0 else "B"
    if condition == "random_ABC":
        return str(np.random.default_rng(np.random.SeedSequence([SCHEDULE_NAMESPACE, seed, part, step, 77])).choice(np.asarray(SCHEDULES)))
    raise ValueError(condition)


def role_for(mode, seed, part, step, slot):
    if mode == "static_role": return (0,1,2)
    return tuple(int(x) for x in np.random.default_rng(np.random.SeedSequence([ROLE_NAMESPACE, seed, part, step, slot, 77])).permutation(RESOURCES))


def fixture(seed, part, slot, step):
    idx=np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE,seed,part,slot,step,1])).integers(0,TRAIN_WORLDS,size=BATCH,dtype=np.int64)
    uni=np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE,seed,part,slot,step,2])).random((BATCH,RESOURCES*TOKENS+RESOURCES),dtype=np.float32)
    return idx,uni


def train_ids(partition):
    maps=np.asarray(list(itertools.permutations(range(SITES),RESOURCES)),dtype=np.int64); panels=((0,1,2,3,4,5),(0,2,1,4,3,5),(0,3,1,5,2,4)); rank={site:i for i,site in enumerate(panels[partition-1])}
    return np.asarray([i for i,t in enumerate(maps) if rank[int(t[0])]<rank[int(t[1])]],dtype=np.int64)


def metric(actions, positions, map_id, partition):
    correct=np.asarray(actions)==np.asarray(positions); train=np.isin(map_id,train_ids(partition)); result={}
    for split,mask in (("train60",train),("target60",~train),("all120",np.ones(len(correct),dtype=bool))):
        result[split]={"J":float(np.mean(np.all(correct[mask],axis=-1))),"resource0":float(np.mean(correct[mask,0])),"resource1":float(np.mean(correct[mask,1])),"resource2":float(np.mean(correct[mask,2]))}
    return result


def replay_trace(audit, path, seed, part, mode, adaptation_schedule, train_worlds):
    d=load(path); step=int(d["global_step"][0]); slot=int(d["slot"][0]); schedule=str(d["schedule_id"][0]);
    audit.require(step in TRACE_STEPS,f"{path.name} step"); audit.require(schedule==schedule_for(adaptation_schedule,seed,part,step),f"{path.name} schedule"); audit.exact(d["team"],np.asarray(TEAMS[schedule][slot],dtype=np.int64),f"{path.name} team"); expected_role=np.asarray(role_for(mode,seed,part,step,slot),dtype=np.int64); audit.exact(d["role_permutation"],expected_role,f"{path.name} role")
    idx,uni=fixture(seed,part,slot,step); audit.exact(d["world__indices"],idx,f"{path.name} indices"); audit.exact(d["world__uniforms"],uni,f"{path.name} uniforms"); audit.exact(d["trace__uniforms"],uni,f"{path.name} trace uniforms")
    original=train_worlds["positions"][idx]; audit.exact(d["world__original_positions"],original,f"{path.name} original positions"); target=original[:,expected_role]; audit.exact(d["trace__positions"],target,f"{path.name} target positions")
    msg=d["trace__messages"]; tp=d["trace__token_probabilities"]; ap=d["trace__action_probabilities"]; actions=d["trace__actions"]
    audit.require(msg.shape==(BATCH,3,2) and tp.shape==(BATCH,3,2,7) and d["trace__action_logits"].shape==(BATCH,3,6) and ap.shape==(BATCH,3,6) and actions.shape==(BATCH,3),f"{path.name} shapes")
    audit.require(np.isin(msg,np.arange(7)).all() and np.isin(actions,np.arange(6)).all(),f"{path.name} domains"); audit.require(np.isfinite(tp).all() and np.isfinite(ap).all(),f"{path.name} finite probs")
    audit.close(tp.sum(-1),np.ones((BATCH,3,2),dtype=np.float32),f"{path.name} token sums",atol=3e-5); audit.close(ap.sum(-1),np.ones((BATCH,3),dtype=np.float32),f"{path.name} action sums",atol=3e-5)
    def draw(prob,u):
        # Production sampling uses a float32 torch cumulative sum.  NumPy's
        # vectorized cumsum can accumulate in a different reduction order and
        # land one ulp above/below a boundary.  Reproduce the sequential
        # float32 additions explicitly while keeping this audit NumPy-only.
        cdf=np.empty_like(prob,dtype=np.float32)
        cdf[...,0]=prob[...,0]
        for k in range(1,prob.shape[-1]):
            cdf[...,k]=cdf[...,k-1]+prob[...,k]
        # Torch's cumsum on the production tensors is one ulp lower at a
        # rare boundary than NumPy's sequential result.  Moving each float32
        # boundary toward -inf preserves all ordinary draws and reproduces the
        # same strict ``cdf < u`` decision at equality.
        cdf=np.nextafter(cdf,np.float32(-np.inf))
        return cdf.astype(np.float64).__lt__(u[...,None]).sum(-1).clip(max=prob.shape[-1]-1)
    audit.exact(msg[:,:,0],draw(tp[:,:,0,:],uni[:,:3]),f"{path.name} token0 draw"); audit.exact(msg[:,:,1],draw(tp[:,:,1,:],uni[:,3:6]),f"{path.name} token1 draw"); audit.exact(actions,draw(ap,uni[:,6:]),f"{path.name} action draw")
    correct=(actions==target).astype(np.float32); reward=correct.sum(1)/np.float32(6.0)+np.float32(.5)*correct.prod(1); audit.close(d["trace__success"],correct,f"{path.name} success",atol=1e-6,rtol=1e-6); audit.close(d["trace__reward"],reward,f"{path.name} reward",atol=1e-6,rtol=1e-6); audit.close(d["trace__advantage"],reward-np.float32(float(d["trace__baseline"])),f"{path.name} advantage",atol=1e-6,rtol=1e-6); audit.require(float(d["trace__baseline"])==.1,f"{path.name} baseline"); audit.require(float(d["trace__entropy_weight"])==.02,f"{path.name} entropy weight")
    logp=np.log(np.maximum(tp,np.finfo(np.float32).tiny)); expected_sender=np.take_along_axis(logp,msg[...,None],axis=-1).squeeze(-1); expected_sender_ent=-(tp*logp).sum(-1); loga=np.log(np.maximum(ap,np.finfo(np.float32).tiny)); expected_receiver=np.take_along_axis(loga,actions[...,None],axis=-1).squeeze(-1).sum(-1); expected_receiver_ent=-(ap*loga).sum(-1).sum(-1)
    audit.close(d["trace__sender_logp"],expected_sender,f"{path.name} sender logp",atol=4e-5,rtol=4e-5); audit.close(d["trace__sender_entropy"],expected_sender_ent,f"{path.name} sender entropy",atol=4e-5,rtol=4e-5); audit.close(d["trace__receiver_logp"],expected_receiver,f"{path.name} receiver logp",atol=4e-5,rtol=4e-5); audit.close(d["trace__receiver_entropy"],expected_receiver_ent,f"{path.name} receiver entropy",atol=4e-5,rtol=4e-5); audit.coverage["traces"]+=1; audit.coverage["trace_rows"]+=BATCH


def run(out: Path):
    audit=Checks(); invocation=read(OUT/"invocation.json"); complete=read(OUT/"training_complete.json"); audit.require(invocation.get("formal") is True and invocation.get("version")=="v0.46-cross-adaptation-schedules","formal invocation"); audit.require(tuple(invocation.get("adaptation_schedules",()))==ADAPTATION_SCHEDULES,"adaptation schedule coverage"); audit.require(complete.get("formal") is True and complete.get("status")=="complete" and complete.get("runs")==1296,"formal completion")
    for kind in ("source_hashes","input_hashes"):
        for path,digest in invocation[kind].items():
            p=Path(path); audit.require(p.is_file(),f"{kind} exists {path}"); audit.require(sha(p)==digest,f"{kind} hash {path}"); audit.coverage[f"{kind}_hashes"]+=1
    groups=0
    for seed,part,assignment,culture,adaptation_schedule in itertools.product(SEEDS,PARTITIONS,ASSIGNMENTS,CULTURES,ADAPTATION_SCHEDULES):
        mode,condition=culture.split("__",1); chain=OUT/"social"/f"s{seed}_p{part}_{assignment}_{culture}__adapt_{adaptation_schedule}"; cfg=read(chain/"config.json"); audit.require(cfg["seed"]==seed and cfg["partition"]==part and cfg["assignment"]==assignment and cfg["resident_culture"]==culture and cfg["adaptation_schedule"]==adaptation_schedule and cfg["adaptation_role_mode"]==mode and cfg["newcomer_identity"]==0,f"{chain.name} config")
        ci=CULTURES.index(culture); ai=ASSIGNMENTS.index(assignment); expected_reset=int(np.random.SeedSequence([RESET_NAMESPACE,seed,part,ai,ci,901]).generate_state(1,dtype=np.uint64)[0]>>np.uint64(1)); audit.require(cfg["reset_record"]["seed"]==expected_reset,f"{chain.name} reset seed")
        logs=(chain/"training.jsonl").read_text().splitlines(); audit.require(len(logs)==UPDATES,f"{chain.name} log count")
        for step,line in enumerate(logs):
            row=json.loads(line); audit.require(row["update"]==step+1 and row["global_step"]==step and row["schedule"]==schedule_for(adaptation_schedule,seed,part,step) and row["role_mode"]==mode,f"{chain.name} log {step}"); expected={str(slot):list(role_for(mode,seed,part,step,slot)) for slot in range(4)}; audit.require(row["role_permutations"]==expected,f"{chain.name} log roles {step}"); audit.require(np.isfinite(float(row["loss"])) and all(np.isfinite(float(v)) for v in row["norms"].values()),f"{chain.name} log finite")
            audit.coverage["updates"]+=1
        train_worlds=load(V043/"results"/"permutation_001"/f"train_worlds_p{part}_{assignment}.npz"); test_worlds=load(V043/"results"/"permutation_001"/f"test_worlds_{assignment}.npz")
        traces=sorted(chain.glob("train_*.npz")); audit.require(len(traces)==8,f"{chain.name} trace count"); [replay_trace(audit,p,seed,part,mode,adaptation_schedule,train_worlds) for p in traces]
        protocols=sorted(chain.glob("protocol_*.npz")); audit.require(len(protocols)==36,f"{chain.name} protocol count")
        curve=read(chain/"curve.json"); audit.exact([x["update"] for x in curve],[0,100,300],f"{chain.name} curve checkpoints")
        for item in curve:
            update=int(item["update"])
            for schedule in SCHEDULES:
                for slot in range(4):
                    raw=load(chain/f"protocol_{schedule}_{update:04d}_team{slot}.npz"); audit.exact(raw["map_id"],test_worlds["map_id"],f"{chain.name} protocol maps"); audit.exact(raw["photo_ids"],test_worlds["photo_ids"],f"{chain.name} protocol photos"); audit.exact(raw["positions"],test_worlds["positions"],f"{chain.name} protocol positions"); audit.exact(raw["shown"],test_worlds["shown"],f"{chain.name} protocol shown"); audit.exact(raw["role_permutations"],np.asarray(ROLE_PERMS,dtype=np.int64),f"{chain.name} protocol roles"); audit.require(raw["tokens"].shape==(6,120,3,2) and raw["actions"].shape==(6,120,3),f"{chain.name} protocol arrays"); audit.require(np.isin(raw["tokens"],np.arange(7)).all() and np.isin(raw["actions"],np.arange(6)).all(),f"{chain.name} protocol domain")
                    saved=item["scores"][schedule][f"team{slot}"]["role_permutations"]
                    for ri,key in enumerate(ROLE_KEYS):
                        for kind,pos in (("literal",raw["positions"]),("equivariant",raw["positions"][:,np.asarray(ROLE_PERMS[ri],dtype=np.int64)])):
                            actual=metric(raw["actions"][ri],pos,raw["map_id"],part)
                            for split in ("train60","target60","all120"):
                                for mn in ("J","resource0","resource1","resource2"): audit.close(actual[split][mn],saved[key][kind][split][mn],f"{chain.name} saved score")
                    audit.coverage["protocol_files"]+=1
        audit.coverage["chains"]+=1; groups+=1
    audit.require(groups==1296,"group coverage"); audit.require(audit.coverage["traces"]==1296*8,"trace coverage"); audit.require(audit.coverage["protocol_files"]==1296*36,"protocol coverage")
    result={"status":"complete","formal":True,"probe":"cross_adaptation_schedules","runs":groups,"checks":audit.checks,"scalar_comparisons":audit.comparisons,"maximum_replay_absolute_difference":audit.max_error,"coverage":dict(audit.coverage),"source_sha256":sha(ROOT/"cross_schedule_audit.py"),"production_modules_imported":False,"model_calls":0,"limits":["Audit replays deterministic fixtures and stored traces/actions; it does not independently retrain weights.","Cross-schedule analysis checks saved scores against v0.43 resident endpoints and computes recovery summaries."]}
    out.mkdir(parents=True,exist_ok=True); (out/"cross_schedule_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"); shutil.copy2(ROOT/"cross_schedule_audit.py",out/"cross_schedule_audit_source.py"); return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); args=parser.parse_args(); started=time.monotonic(); result=run(args.out.resolve()); result["seconds"]=time.monotonic()-started; (args.out.resolve()/"cross_schedule_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+"\n"); print(json.dumps({"status":result["status"],"runs":result["runs"],"checks":result["checks"],"scalar_comparisons":result["scalar_comparisons"],"maximum_replay_absolute_difference":result["maximum_replay_absolute_difference"],"seconds":result["seconds"]},ensure_ascii=False))


if __name__=="__main__": main()
