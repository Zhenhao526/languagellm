"""Plot the matched full-combination receiver control."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(summary_path: str, output: str):
    summary_path = Path(summary_path).resolve(); output = Path(output).resolve()
    if output.exists(): raise ValueError(f"Refuse to overwrite figures: {output}")
    d = json.loads(summary_path.read_text(encoding="utf8")); output.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    colors = {"seen_joint_only": "#8aa4c7", "all_joint": "#2f6db0"}; labels=[]; vals=[]; lo=[]; hi=[]; bars=[]
    for schedule in ("static", "rematched"):
        for arm, label in (("seen_joint_only", "seen A"), ("all_joint", "all A")):
            x = d["cells"][f"{arm}/{schedule}/live"]["A"]["aligned_minus_placebo"]["plan_transfer"]
            labels.append(f"{schedule}\n{label}"); vals.append(100*x["mean"]); lo.append(100*(x["mean"]-x["ci95_t7"][0])); hi.append(100*(x["ci95_t7"][1]-x["mean"])); bars.append(colors[arm])
    pos = np.arange(len(vals)); axes[0].bar(pos, vals, color=bars, width=.72, edgecolor="white"); axes[0].errorbar(pos, vals, yerr=np.array([lo,hi]), fmt="none", ecolor="#222", capsize=3, linewidth=1); axes[0].axhline(0,color="#222",linewidth=.8); axes[0].set_xticks(pos,labels); axes[0].set_ylabel("aligned − placebo plan transfer (pp)"); axes[0].set_title("Matched receiver support control"); axes[0].grid(axis="y",alpha=.25)
    fields = (("plan_transfer","plan"),("physical","physical"),("q","Q"),("conditional_q","Q|physical"),("action_change","action change")); v=[]; l=[]; h=[]
    for field,_ in fields:
        x=d["interaction"][field]; v.append(100*x["mean"]); l.append(100*(x["mean"]-x["ci95_t7"][0])); h.append(100*(x["ci95_t7"][1]-x["mean"]))
    pos=np.arange(len(fields)); axes[1].bar(pos,v,color="#6d9e72",width=.62,edgecolor="white"); axes[1].errorbar(pos,v,yerr=np.array([l,h]),fmt="none",ecolor="#222",capsize=3,linewidth=1); axes[1].axhline(0,color="#222",linewidth=.8); axes[1].set_xticks(pos,[label for _,label in fields],rotation=25,ha="right"); axes[1].set_ylabel("all−seen × schedule interaction (pp)"); axes[1].set_title("Matched-control interaction"); axes[1].grid(axis="y",alpha=.25)
    fig.suptitle("Compositional holdout matched full-combination control",fontsize=13); fig.savefig(output/"matched_control.png",dpi=220); fig.savefig(output/"matched_control.pdf"); plt.close(fig)
    rec={"status":"rendered","visual_review":"pending","summary_sha256":hashlib.sha256(summary_path.read_bytes()).hexdigest(),"files":["matched_control.png","matched_control.pdf"]}; (output/"receipt.json").write_text(json.dumps(rec,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8"); return rec


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--summary",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(); print(json.dumps(plot(args.summary,args.output),ensure_ascii=False))
