"""Independent NumPy reanalysis of teacher-free origin runs."""
from __future__ import annotations
import argparse, hashlib, itertools, json, shutil, time, traceback
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = (34101, 34102, 34103, 34104); PARTITIONS = (1, 2, 3); CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC"); SCHEDULES = ("A", "B", "C"); CHECKPOINTS = (0, 100, 600, 1200)
MAPS = tuple((food, water) for food in range(6) for water in range(6) if food != water); PANELS = ((0,1,2,3,4,5),(0,2,1,4,3,5),(0,3,1,5,2,4)); TARGET_PAIRS=((0,1),(0,3),(1,2),(1,4),(2,0),(2,5),(3,2),(3,4),(4,0),(4,5),(5,1),(5,3)); TRAIN_PAIRS=((0,2),(0,5),(1,0),(1,3),(2,1),(2,4),(3,1),(3,5),(4,2),(4,3),(5,0),(5,4))
TEAMS = {"A": ((0,1,2),(1,2,3),(2,3,0),(3,0,1)), "B": ((0,2,3),(1,3,0),(2,0,1),(3,1,2)), "C": ((0,3,1),(1,0,2),(2,1,3),(3,2,0))}

def read(path: Path): return json.loads(path.read_text())
def write(path: Path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def npz(path: Path):
    with np.load(path, allow_pickle=False) as z: return {key: z[key] for key in z.files}
def ids(partition, pairs):
    order = PANELS[partition - 1]; index = {pair: i for i, pair in enumerate(MAPS)}; return np.asarray(sorted(index[(order[i], order[j])] for i, j in pairs), dtype=np.int64)
def score(raw, partition):
    decoder = np.argmax(raw["receiver_logits"], axis=-1); action = decoder[7*raw["tokens"][:,0] + raw["tokens"][:,1]]; correct = action == raw["positions"]; train=np.isin(raw["map_id"],ids(partition,TRAIN_PAIRS)); target=np.isin(raw["map_id"],ids(partition,TARGET_PAIRS)); held=np.isin(raw["map_id"],np.setdiff1d(np.arange(30,dtype=np.int64),ids(partition,TRAIN_PAIRS)))
    return {split:{"J":float(np.mean(np.all(correct[mask],axis=-1))),"food":float(np.mean(correct[mask,0])),"water":float(np.mean(correct[mask,1]))} for split,mask in (("train12",train),("target12",target),("held18",held))}
def agreement(raw_by_slot, schedule):
    food, water = {}, {}
    for slot, team in enumerate(TEAMS[schedule]):
        _, sender_food, sender_water = team; food[sender_food] = raw_by_slot[slot]["tokens"][:,0]; water[sender_water] = raw_by_slot[slot]["tokens"][:,1]
    pairs = ((0,2),(1,3)); f = [float(np.mean(food[a] == food[b])) for a,b in pairs]; w = [float(np.mean(water[a] == water[b])) for a,b in pairs]
    return {"food":float(np.mean(f)),"water":float(np.mean(w)),"joint":float(np.mean([np.mean((food[a]==food[b]) & (water[a]==water[b])) for a,b in pairs]))}
def mean_sd(values):
    a=np.asarray(values,dtype=float); return {"mean":float(a.mean()),"sd":float(a.std(ddof=1)) if len(a)>1 else 0.0,"values":a.tolist()}
def auc(values): return float(np.trapezoid(np.asarray(values,dtype=float),np.asarray(CHECKPOINTS,dtype=float))/1200.0)

class Checks:
    def __init__(self): self.count=0; self.comparisons=0; self.max_error=0.0; self.coverage=Counter()
    def require(self,v,label): self.count+=1; assert bool(v),label
    def exact(self,a,b,label): self.require(np.array_equal(a,b),label)
    def close(self,a,b,label):
        x,y=np.asarray(a),np.asarray(b); self.require(x.shape==y.shape,label+" shape"); self.require(np.isfinite(x).all() and np.isfinite(y).all(),label+" finite"); self.comparisons+=int(x.size); e=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.0; self.max_error=max(self.max_error,e); self.require(np.allclose(x,y,atol=1e-6,rtol=1e-6),label+" numerical")
    def add(self,k,v=1): self.coverage[k]+=v

def check_raw(raw, worlds, checks, label):
    for key in ("map_id","photo_ids","positions","shown"): checks.exact(raw[key],worlds[key],label+"/"+key)
    n=len(worlds["map_id"]); checks.require(raw["tokens"].shape==(n,2) and raw["tokens"].dtype.kind in "iu",label+" tokens"); checks.require(((raw["tokens"]>=0)&(raw["tokens"]<7)).all(),label+" token domain")
    for key in ("sender_log_probs_food","sender_log_probs_water"):
        checks.require(raw[key].shape==(n,7) and np.isfinite(raw[key]).all(),label+" component"); checks.require(np.allclose(np.exp(raw[key]).sum(-1),1,atol=1e-6),label+" component normalization")
    checks.require(raw["sender_log_probs"].shape==(n,49) and np.isfinite(raw["sender_log_probs"]).all(),label+" joint"); checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1),1,atol=1e-6),label+" joint normalization"); checks.require(np.allclose(raw["sender_log_probs"],(raw["sender_log_probs_food"][:,:,None]+raw["sender_log_probs_water"][:,None,:]).reshape(n,49),atol=1e-6,rtol=1e-6),label+" factorization"); checks.require(raw["receiver_logits"].shape==(49,2,6) and np.isfinite(raw["receiver_logits"]).all(),label+" receiver"); checks.add("protocol_worlds",n)

def summarize(rows):
    return {"n":len(rows),"train_J":mean_sd([x["train_J"] for x in rows]),"target_J":mean_sd([x["target_J"] for x in rows]),"held_J":mean_sd([x["held_J"] for x in rows]),"agreement_joint":mean_sd([x["agreement_joint"] for x in rows]),"agreement_food":mean_sd([x["agreement_food"] for x in rows]),"agreement_water":mean_sd([x["agreement_water"] for x in rows])}

def run(out: Path):
    checks=Checks(); invocation=read(out/"invocation.json"); complete=read(out/"training_complete.json"); runs=read(out/"runs.json"); checks.exact(invocation["formal"],True,"formal invocation"); checks.exact(invocation["seeds"],list(SEEDS),"seeds"); checks.exact(invocation["partitions"],list(PARTITIONS),"partitions"); checks.exact(invocation["conditions"],list(CONDITIONS),"conditions"); checks.exact(invocation["updates"],1200,"updates"); checks.exact(invocation["checkpoints"],list(CHECKPOINTS),"checkpoints"); checks.exact(complete["status"],"complete","completion"); checks.exact(complete["formal"],True,"formal completion"); checks.exact(complete["probe"],"teacher_free_origin","probe"); checks.exact(complete["runs"],36,"runs"); checks.exact(runs["count"],36,"run manifest")
    for key in ("source_hashes","input_hashes"):
        checks.exact(invocation[key],complete[key],key+" identity")
        for path,digest in invocation[key].items(): checks.require(Path(path).is_file(),key+" bound path"); checks.exact(sha(Path(path)),digest,key+" bound hash")
    worlds=npz(out/"test_worlds.npz"); checks.exact(len(worlds["map_id"]),180,"test worlds"); rows=[]
    for seed,part,condition in itertools.product(SEEDS,PARTITIONS,CONDITIONS):
        folder=out/"social"/f"s{seed}_p{part}_{condition}"; cfg=read(folder/"config.json"); checks.exact(cfg["condition"],condition,"config condition"); curve=read(folder/"curve.json"); checks.exact([x["update"] for x in curve],list(CHECKPOINTS),"curve updates")
        for item in curve:
            update=int(item["update"])
            for schedule in SCHEDULES:
                raws=[]; metrics=[]
                for slot in range(4):
                    path=folder/f"protocol_{schedule}_{update:04d}_team{slot}.npz"; rel=str(path.relative_to(out)); checks.require(rel in complete["files"],"protocol completion hash"); raw=npz(path); check_raw(raw,worlds,checks,f"{condition}/{seed}/{part}/{schedule}/{update}/{slot}"); metric=score(raw,part); saved=item["scores"][schedule][f"team{slot}"]
                    for split in ("train12","target12","held18"):
                        for key in ("J","food","water"): checks.close(metric[split][key],saved[split]["pooled"][key],"saved metric")
                    raws.append(raw); metrics.append(metric); checks.add("protocol_tables")
                pooled={split:{key:float(np.mean([m[split][key] for m in metrics])) for key in ("J","food","water")} for split in ("train12","target12","held18")}; agree=agreement(raws,schedule); rows.append({"seed":seed,"partition":part,"condition":condition,"schedule":schedule,"update":update,"train_J":pooled["train12"]["J"],"target_J":pooled["target12"]["J"],"held_J":pooled["held18"]["J"],"agreement_food":agree["food"],"agreement_water":agree["water"],"agreement_joint":agree["joint"]})
    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for schedule in SCHEDULES:
            summary[condition][schedule] = {}
            for update in CHECKPOINTS:
                selected = [row for row in rows if row["condition"] == condition and row["schedule"] == schedule and row["update"] == update]
                summary[condition][schedule][str(update)] = summarize(selected)
    formation_auc = {}
    for condition in CONDITIONS:
        formation_auc[condition] = {}
        for schedule in SCHEDULES:
            values = []
            for seed, part in itertools.product(SEEDS, PARTITIONS):
                curve = [next(row for row in rows if row["seed"] == seed and row["partition"] == part and row["condition"] == condition and row["schedule"] == schedule and row["update"] == update)["target_J"] for update in CHECKPOINTS]
                values.append(auc(curve))
            formation_auc[condition][schedule] = mean_sd(values)
    record={"status":"complete","formal":True,"probe":"teacher_free_origin","seeds":list(SEEDS),"partitions":list(PARTITIONS),"conditions":list(CONDITIONS),"schedules":list(SCHEDULES),"checkpoints":list(CHECKPOINTS),"rows":rows,"summary":summary,"formation_auc":formation_auc,"checks":checks.count,"scalar_comparisons":checks.comparisons,"maximum_metric_absolute_difference":checks.max_error,"source_sha256":sha(ROOT/"origin_analysis.py"),"probe_training_complete_sha256":sha(out/"training_complete.json"),"coverage":dict(checks.coverage),"limits":["Private visual encoders are frozen; only communication modules start random and update.","The task is a two-resource complementary protocol and does not test grammar or world knowledge.","Joint policy-gradient formation can establish a protocol here, but it is not a model of human language history."]}; write(out/"origin_analysis.json",record); qa={"passed":True,"status":"passed_teacher_free_origin_recheck","formal":True,"checks":checks.count,"scalar_comparisons":checks.comparisons,"maximum_metric_absolute_difference":checks.max_error,"coverage":dict(checks.coverage),"analysis_source_sha256":sha(ROOT/"origin_analysis.py"),"analysis_sha256":sha(out/"origin_analysis.json"),"production_modules_imported":False,"model_calls":0}; write(out/"origin_raw_validation.json",qa); shutil.copy2(ROOT/"origin_analysis.py",out/"origin_analysis_source.py"); return record,qa

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); args=parser.parse_args(); started=time.monotonic()
    try:
        record,qa=run(args.out.resolve()); qa["seconds"]=time.monotonic()-started; write(args.out.resolve()/"origin_raw_validation.json",qa); print(json.dumps({k:record[k] for k in ("status","checks","scalar_comparisons","maximum_metric_absolute_difference")},ensure_ascii=False))
    except Exception as error:
        stamp=time.time_ns(); write(args.out.resolve()/f"origin_analysis_failure_{stamp}.json",{"status":"failed","error":repr(error),"traceback":traceback.format_exc(),"source_sha256":sha(ROOT/"origin_analysis.py")}); raise

if __name__ == "__main__": main()
