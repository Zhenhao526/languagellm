"""Aggregate compositional signaling runs into compact causal summaries."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

T95 = {7: 2.365}

def ci(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) == 0: return [None, None]
    if len(x) == 1: return [float(x[0]), float(x[0])]
    t = T95.get(len(x)-1, 1.96); h = t * float(x.std(ddof=1)) / np.sqrt(len(x))
    return [float(x.mean()-h), float(x.mean()+h)]

def load(path):
    payload=json.loads(Path(path).read_text()); out={}
    for r in payload["results"]:
        key=(int(r["seed"]),r["condition"])
        if key in out: raise ValueError(f"duplicate {key}")
        out[key]=r
    return out

def mean_return(r, mode="natural", worker=None, split="heldout"):
    ids=[worker] if worker is not None else [int(x) for x in r["final"][split]["workers"].keys()]
    vals=[]
    for w in ids:
        x=r["final"][split]["workers"][str(w)][mode]["team_return_mean"]
        if x is not None: vals.append(float(x))
    return float(np.mean(vals)) if vals else np.nan

def effects(by, split="heldout"):
    out={}
    conds=sorted({c for _,c in by})
    for cond in conds:
        r0=by[next(k for k in by if k[1]==cond)]
        if r0["channel"] != "live": continue
        for label,a,b in (("natural-minus-closed","natural","closed"),("natural-minus-permuted","natural","permuted")):
            vals=[]
            for seed in sorted(s for s,c in by if c==cond): vals.append(mean_return(by[(seed,cond)],a,split=split)-mean_return(by[(seed,cond)],b,split=split))
            out[f"{cond}|{label}"]={"condition":cond,"label":label,"values":vals,"mean":float(np.mean(vals)),"ci95_t":ci(vals)}
        silent=cond.replace("_live_","_silent_")
        if (next(iter([s for s,c in by if c==cond]),None),silent) in by or silent in conds:
            vals=[]
            for seed in sorted({s for s,c in by if c==cond}&{s for s,c in by if c==silent}): vals.append(mean_return(by[(seed,cond)],"natural",split=split)-mean_return(by[(seed,silent)],"natural",split=split))
            out[f"{cond}|natural-minus-silent"]={"condition":cond,"label":"natural-minus-silent","values":vals,"mean":float(np.mean(vals)),"ci95_t":ci(vals)}
        if r0["form"]=="dual2":
            vals=[]
            for seed in sorted(s for s,c in by if c==cond):
                x=mean_return(by[(seed,cond)],"recombined",split=split); y=mean_return(by[(seed,cond)],"natural",split=split)
                if np.isfinite(x): vals.append(x-y)
            if vals: out[f"{cond}|recombined-minus-natural"]={"condition":cond,"label":"recombined-minus-natural","values":vals,"mean":float(np.mean(vals)),"ci95_t":ci(vals)}
    return out

def codebook(by, split="heldout"):
    rows=[]
    for (seed,cond),r in sorted(by.items()):
        c=r["final"][split]["codebook"]
        rows.append({"seed":seed,"condition":cond,"sender_sequence_agreement":c["sender_sequence_agreement"],"semantic_success_mean":c["semantic_success_mean"],"semantic_success_min":c["semantic_success_min"]})
    return rows

def write_md(path,data):
    lines=["# Compositional signaling compact aggregation","",f"- runs: {data['runs']}","- split: heldout","","| condition | natural | silent | Δ natural−silent | Δ natural−permuted | recombined−natural | sequence agreement |","|---|---:|---:|---:|---:|---:|---:|"]
    for cond in sorted({x["condition"] for x in data["codebook"]}):
        vals=[x for x in data["rows"] if x["condition"]==cond and x["mode"]=="natural"]; rmean=float(np.mean([x["return"] for x in vals])); silent=next((x for x in data["rows"] if x["condition"]==cond.replace("_live_","_silent_") and x["mode"]=="natural"),None)
        smean=float(np.mean([x["return"] for x in data["rows"] if x["condition"]==cond.replace("_live_","_silent_") and x["mode"]=="natural"])) if silent else np.nan
        f=lambda x:"NA" if not np.isfinite(x) else f"{x:.3f}"
        lines.append(f"| `{cond}` | {f(rmean)} | {f(smean)} | {f(data['effects'].get(cond+'|natural-minus-silent',{}).get('mean',np.nan))} | {f(data['effects'].get(cond+'|natural-minus-permuted',{}).get('mean',np.nan))} | {f(data['effects'].get(cond+'|recombined-minus-natural',{}).get('mean',np.nan))} | {f(np.mean([x['sender_sequence_agreement'] for x in data['codebook'] if x['condition']==cond]))} |")
    lines += ["","Natural-minus-silent is the primary channel-necessity contrast; natural-minus-permuted tests sender/receiver pairing. `recombined−natural` is a diagnostic for dual2 factor-slot recombination and is not by itself evidence for compositional language."]
    Path(path).write_text("\n".join(lines)+"\n",encoding="utf8")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--results",required=True); ap.add_argument("--out",required=True); ap.add_argument("--markdown",required=True)
    args=ap.parse_args(); by=load(args.results); rows=[]
    for (seed,cond),r in sorted(by.items()):
        for mode in ("natural","closed","permuted"):
            rows.append({"seed":seed,"condition":cond,"mode":mode,"return":mean_return(r,mode)})
    data={"schema":"compositional_signaling_aggregate_v1","runs":len(by),"rows":rows,"effects":effects(by),"codebook":codebook(by)}
    Path(args.out).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n"); write_md(args.markdown,data); print(json.dumps({"status":"written","runs":len(by)},ensure_ascii=False))
if __name__=="__main__": main()
