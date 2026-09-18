"""Aggregate structured factor-sharing results with paired seed contrasts."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

def ci(values):
    x=np.asarray(values,dtype=float); mean=float(x.mean()) if len(x) else float("nan"); sd=float(x.std(ddof=1)) if len(x)>1 else 0.0; half=1.96*sd/np.sqrt(len(x)) if len(x)>1 else 0.0
    return {"n":int(len(x)),"mean":mean,"sd":sd,"ci95":[mean-half,mean+half]}
def metric(row,kind,mode="natural"): return row["final"][kind][mode]["team_return_mean"]
def group(rows,keys):
    out=[]
    for key in sorted({tuple(r[k] for k in keys) for r in rows}):
        subset=[r for r in rows if tuple(r[k] for k in keys)==key]
        rec={k:v for k,v in zip(keys,key)}; rec["n"]=len(subset)
        kinds = tuple(k for k in ("all","heldout_combo","heldout_value") if k in subset[0]["final"])
        for kind in kinds:
            modes = ("natural","permuted","silent","recombined") if "recombined" in subset[0]["final"][kind] else ("natural","permuted","silent")
            for mode in modes:
                rec[f"{kind}_{mode}"]=ci([metric(r,kind,mode) for r in subset])
            rec[f"{kind}_message_gap"]=ci([metric(r,kind,"natural")-metric(r,kind,"permuted") for r in subset])
            rec[f"{kind}_live_silent"]=ci([metric(r,kind,"natural")-metric(r,kind,"silent") for r in subset])
        rec["fresh_sender_consistency"]=ci([r["final"]["fresh_sender_consistency"] for r in subset]); rec["community_codebook_hamming"]=ci([r["final"]["community_codebook_hamming"] for r in subset]); fac=[r["final"]["factor_slot_unique_fraction"] for r in subset if r["final"]["factor_slot_unique_fraction"] is not None]; rec["factor_slot_unique_fraction"]=ci(fac) if fac else None
        rec["functional_combo_count"]=sum(metric(r,"heldout_combo")>=0.60 for r in subset) if "heldout_combo" in subset[0]["final"] else None
        rec["functional_value_count"]=sum(metric(r,"heldout_value")>=0.60 for r in subset) if "heldout_value" in subset[0]["final"] else None
        out.append(rec)
    return out

def contrasts(children):
    by={(r["seed"],r["architecture"],r["population"],r["visibility"],r["support"]):r for r in children}; out=[]; seeds=sorted(set(r["seed"] for r in children))
    for population in ["aligned","conflict"]:
      for visibility in ["hidden","visible"]:
       for support in ["full","heldout_combo","heldout_value"]:
        vals={}
        for kind in ("all","heldout_combo","heldout_value"):
          vals[kind]=ci([metric(by[(s,"factorized",population,visibility,support)],kind)-metric(by[(s,"holistic",population,visibility,support)],kind) for s in seeds])
        out.append({"contrast":"factorized_minus_holistic","population":population,"visibility":visibility,"support":support,**vals})
    for architecture in ["holistic","factorized"]:
      for visibility in ["hidden","visible"]:
       for support in ["full","heldout_combo","heldout_value"]:
        vals={}
        for kind in ("all","heldout_combo","heldout_value"):
          vals[kind]=ci([metric(by[(s,architecture,"conflict",visibility,support)],kind)-metric(by[(s,architecture,"aligned",visibility,support)],kind) for s in seeds])
        out.append({"contrast":"conflict_minus_aligned","architecture":architecture,"visibility":visibility,"support":support,**vals})
    return out

def aggregate(payload): return {"schema":"structured_factorization_aggregate_v1","parent_summary":group(payload["parents"],["architecture","population"]),"child_summary":group(payload["children"],["architecture","population","visibility","support"]),"contrasts":contrasts(payload["children"]),"run_counts":{"parents":len(payload["parents"]),"children":len(payload["children"])}}
def main():
    p=argparse.ArgumentParser(); p.add_argument("--results",required=True); p.add_argument("--out",required=True); p.add_argument("--markdown",required=True); a=p.parse_args(); d=aggregate(json.loads(Path(a.results).read_text())); Path(a.out).write_text(json.dumps(d,ensure_ascii=False,indent=2)+"\n"); lines=["# Structured factor-sharing aggregate","",f"Parents: {d['run_counts']['parents']}; children: {d['run_counts']['children']}.","","## Child cells",""]
    for r in d["child_summary"]: lines.append(f"- {r['architecture']}/{r['population']}/{r['visibility']}/{r['support']}: all {r['all_natural']['mean']:.3f}, combo {r['heldout_combo_natural']['mean']:.3f}, value {r['heldout_value_natural']['mean']:.3f}, combo functional {r['functional_combo_count']}/{r['n']}, value functional {r['functional_value_count']}/{r['n']}")
    lines += ["","## Paired contrasts",""]
    for r in d["contrasts"]: lines.append(f"- {r['contrast']} {r.get('architecture',r.get('population',''))}/{r['visibility']}/{r['support']}: all {r['all']['mean']:.3f} [{r['all']['ci95'][0]:.3f}, {r['all']['ci95'][1]:.3f}], combo {r['heldout_combo']['mean']:.3f}, value {r['heldout_value']['mean']:.3f}")
    Path(a.markdown).write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
