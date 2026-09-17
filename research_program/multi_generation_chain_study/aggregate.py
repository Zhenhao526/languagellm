"""Aggregate multi-generation chain outcomes."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
from . import design

def ci(values):
    x=np.asarray(values,dtype=float); n=len(x)
    if n==0:return [None,None]
    if n==1:return [float(x[0]),float(x[0])]
    critical=2.04 if n>=30 else 2.365 if n==8 else 1.96
    h=critical*float(x.std(ddof=1))/math.sqrt(n)
    return [float(x.mean()-h),float(x.mean()+h)]

def load(path):
    out={}
    for r in json.loads(Path(path).read_text())["results"]:
        k=(int(r["seed"]),r["condition"])
        if k in out: raise ValueError(k)
        out[k]=r
    return out

def heldout_value(event,mode="natural"):
    return float(event["evaluation"]["heldout"][mode]["team_return_mean"])

def sequence_fidelity(a,b):
    """Fraction of goal/slot entries that are identical between codebooks."""
    keys=sorted(set(a).intersection(b))
    values=[]
    for key in keys:
        xa=np.asarray(a[key],dtype=np.int64).ravel(); xb=np.asarray(b[key],dtype=np.int64).ravel()
        if len(xa)!=len(xb): continue
        values.extend((xa==xb).tolist())
    return float(np.mean(values)) if values else None

def aggregate(path):
    by=load(path); seeds=sorted({s for s,_ in by}); rows=[]
    for (s,c),r in sorted(by.items()):
        rep,ch=design.parse_condition(c)
        previous=None
        for ev in r["events"]:
            codebook=ev["evaluation"]["codebook"]
            rows.append({"seed":s,"condition":c,"representation":rep,"channel":ch,"generation":int(ev["generation"]),"event_role":ev["event_role"],"heldout_live_natural":heldout_value(ev,"natural"),"heldout_silent":heldout_value(ev,"silent"),"heldout_permuted":heldout_value(ev,"permuted"),"heldout_recombined":heldout_value(ev,"recombined"),"recombined_gap":float(ev["evaluation"]["heldout_recombined_gap"]),"composable":bool(ev["composable"]),"parent_composable":bool(r.get("parent_composable",False))})
            row=rows[-1]
            row["semantic_success_by_goal"]=list(codebook["semantic_success_by_goal"])
            row["semantic_success_mean"]=float(codebook["semantic_success_mean"])
            row["sender_sequences"]=codebook["sender_sequences"]
            row["sender_fidelity_to_parent"]=sequence_fidelity(codebook["sender_sequences"],r["parent_sender_sequences"])
            row["sender_fidelity_to_previous"]=sequence_fidelity(codebook["sender_sequences"],previous) if previous is not None else None
            previous=codebook["sender_sequences"]
    # parent status is injected from prepared by main; aggregate() receives only results path,
    # so rows are amended in main() from the frozen config.
    effects={}
    for rep in design.REPRESENTATIONS:
        for gen in design.GENERATIONS:
            live={(s,rep):next(x for x in rows if x["seed"]==s and x["representation"]==rep and x["channel"]=="live" and x["generation"]==gen) for s in seeds if any(x["seed"]==s and x["representation"]==rep and x["channel"]=="live" and x["generation"]==gen for x in rows)}
            silent={(s,rep):next(x for x in rows if x["seed"]==s and x["representation"]==rep and x["channel"]=="silent" and x["generation"]==gen) for s in seeds if any(x["seed"]==s and x["representation"]==rep and x["channel"]=="silent" and x["generation"]==gen for x in rows)}
            vals=[live[(s,rep)]["heldout_live_natural"]-silent[(s,rep)]["heldout_live_natural"] for s in seeds if (s,rep) in live and (s,rep) in silent]
            effects[f"live_minus_silent|{rep}|g{gen}"]={"values":vals,"mean":float(np.mean(vals)) if vals else None,"ci95_t":ci(vals)}
            vals=[live[(s,rep)]["heldout_live_natural"] for s in seeds if (s,rep) in live]
            effects[f"composable|{rep}|g{gen}"]={"values":vals,"mean":float(np.mean(vals)) if vals else None,"ci95_t":ci(vals),"passes":int(sum(x>=0.60 for x in vals)),"n":len(vals)}
            vals=[live[(s,rep)]["heldout_recombined"]-live[(s,rep)]["heldout_live_natural"] for s in seeds if (s,rep) in live]
            effects[f"recombined_minus_natural|{rep}|g{gen}"]={"values":vals,"mean":float(np.mean(vals)) if vals else None,"ci95_t":ci(vals)}
            vals=[live[(s,rep)]["semantic_success_mean"] for s in seeds if (s,rep) in live]
            effects[f"semantic|{rep}|g{gen}|live"]={"values":vals,"mean":float(np.mean(vals)) if vals else None,"ci95_t":ci(vals)}
            vals=[silent[(s,rep)]["semantic_success_mean"] for s in seeds if (s,rep) in silent]
            effects[f"semantic|{rep}|g{gen}|silent"]={"values":vals,"mean":float(np.mean(vals)) if vals else None,"ci95_t":ci(vals)}
            vals=[live[(s,rep)]["sender_fidelity_to_parent"] for s in seeds if (s,rep) in live and live[(s,rep)]["sender_fidelity_to_parent"] is not None]
            effects[f"sender_fidelity_to_parent|{rep}|g{gen}"]={"values":vals,"mean":float(np.mean(vals)) if vals else None,"ci95_t":ci(vals)}
            vals=[live[(s,rep)]["sender_fidelity_to_previous"] for s in seeds if (s,rep) in live and live[(s,rep)]["sender_fidelity_to_previous"] is not None]
            effects[f"sender_fidelity_to_previous|{rep}|g{gen}"]={"values":vals,"mean":float(np.mean(vals)) if vals else None,"ci95_t":ci(vals)}
    # Parent-stratum endpoint effects are kept separate from the overall means.
    strata={}
    for rep in design.REPRESENTATIONS:
        for gen in design.GENERATIONS:
            for status in (False,True):
                vals=[x["heldout_live_natural"] for x in rows if x["representation"]==rep and x["channel"]=="live" and x["generation"]==gen and x["parent_composable"]==status]
                gaps=[x["recombined_gap"] for x in rows if x["representation"]==rep and x["channel"]=="live" and x["generation"]==gen and x["parent_composable"]==status]
                strata[f"{rep}|g{gen}|parent_{'composable' if status else 'noncomposable'}"]={"n":len(vals),"natural_mean":float(np.mean(vals)) if vals else None,"natural_ci95_t":ci(vals),"recombined_gap_mean":float(np.mean(gaps)) if gaps else None,"composable_passes":int(sum(x>=0.60 and abs(g)<=0.02 for x,g in zip(vals,gaps)))}
    # live branch transition counts: composability is evaluated at each event.
    transitions={}
    for rep in design.REPRESENTATIONS:
        for a,b in zip(design.GENERATIONS[:-1],design.GENERATIONS[1:]):
            vals=[]
            for s in seeds:
                r=by.get((s,f"{rep}_live"));
                if r is None: continue
                ev={int(x["generation"]):x for x in r["events"]}; vals.append((bool(ev[a]["composable"]),bool(ev[b]["composable"])))
            transitions[f"{rep}|g{a}->g{b}"]={"n":len(vals),"counts":{f"{x}->{y}":sum(1 for a0,b0 in vals if a0==x and b0==y) for x in (False,True) for y in (False,True)}}
    return {"schema":"multi_generation_chain_aggregate_v1","runs":len(by),"chains":len(by),"rows":rows,"effects":effects,"strata":strata,"transitions":transitions}

def write_md(path,data):
    lines=["# Alternating multi-generation chain", "",f"- chains: {data['chains']}","- training support: leave-one-out","","| generation | role | representation | live held-out | silent held-out | live−silent | semantic live | sender fidelity to parent | composable passes |", "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for gen in design.GENERATIONS:
        role=design.EVENT_ROLES[gen]
        for rep in design.REPRESENTATIONS:
            x=data["effects"][f"composable|{rep}|g{gen}"]; e=data["effects"][f"live_minus_silent|{rep}|g{gen}"]
            silent=np.mean([r["heldout_live_natural"] for r in data["rows"] if r["generation"]==gen and r["representation"]==rep and r["channel"]=="silent"])
            sem=data["effects"][f"semantic|{rep}|g{gen}|live"]["mean"]
            fid=data["effects"][f"sender_fidelity_to_parent|{rep}|g{gen}"]["mean"]
            lines.append(f"| {gen} | `{role}` | `{rep}` | {x['mean']:.3f} | {silent:.3f} | {e['mean']:+.3f} [{e['ci95_t'][0]:+.3f},{e['ci95_t'][1]:+.3f}] | {sem:.3f} | {fid:.3f} | {x['passes']}/{x['n']} |")
    lines += ["","| transition | counts (False→False, False→True, True→False, True→True) |","|---|---|"]
    for key,v in data["transitions"].items():
        c=v["counts"]; lines.append(f"| `{key}` | {c['False->False']}, {c['False->True']}, {c['True->False']}, {c['True->True']} |")
    lines += ["","The chain is sequential: the final parameters of one replacement event initialize the next event. A pass is held-out live natural return ≥ 0.60 with absolute recombined−natural gap ≤ 0.02."]
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results",required=True); ap.add_argument("--prepared",required=True); ap.add_argument("--out",required=True); ap.add_argument("--markdown",required=True); a=ap.parse_args()
    d=aggregate(a.results); cfg=json.loads((Path(a.prepared)/"prepared.json").read_text());
    for row in d["rows"]: row["parent_composable"]=bool(cfg["parent_composable"][str(row["seed"])])
    Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n"); write_md(a.markdown,d); print(json.dumps({"status":"written","chains":d["chains"]},ensure_ascii=False))
if __name__=="__main__": main()
