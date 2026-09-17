"""Generate plots for local packet controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import dataset, metrics

ROOT=Path(__file__).resolve().parents[2]; CONTENT=ROOT/"research_program/triadic_content_response_study/results/content_001"


def read(path): return dataset.read(path)
def baseline_records():
    return {(r["seed"],r["condition"],r["update"]):r for r in read(CONTENT/"summary_001"/"summary.json")["records"]}


def plot_trajectory(summary,out):
    baseline=baseline_records(); fig,axes=plt.subplots(1,2,figsize=(12,4.4),sharey=True); palette=plt.get_cmap("tab10").colors; colors={m:palette[i] for i,m in enumerate(metrics.MODES)}; labels={"single_slot_cycle":"single slot","rank_canonical":"rank canonical","equality_pattern_relabel":"equality pattern"}; x=np.asarray(metrics.STEPS)
    for ax,rule in zip(axes,metrics.RULES):
        c=rule+"_PL_live"; vals=[[baseline[seed,c,step]["M"] for seed in metrics.SEEDS] for step in metrics.STEPS]; bm=np.mean(vals,axis=1); be=np.std(vals,axis=1,ddof=1)/4; ax.plot(x,bm*100,color="#222222",lw=2.2,label="natural packet"); ax.fill_between(x,(bm-be)*100,(bm+be)*100,color="#222222",alpha=.10)
        for mode in metrics.MODES:
            means=np.asarray([summary["trajectory"][mode][c][str(step)]["M"]["mean"] for step in metrics.STEPS]); errs=np.asarray([summary["trajectory"][mode][c][str(step)]["M"]["sample_sd"]/4 for step in metrics.STEPS]); ax.plot(x,means*100,color=colors[mode],lw=1.6,label=labels[mode]); ax.fill_between(x,(means-errs)*100,(means+errs)*100,color=colors[mode],alpha=.08)
        e=summary["ensemble_trajectory"][c]; means=np.asarray([e[str(step)]["mean"] for step in metrics.STEPS]); errs=np.asarray([e[str(step)]["sample_sd"]/4 for step in metrics.STEPS]); ax.plot(x,means*100,color="#000000",lw=2.8,ls="--",label="three-control mean"); ax.fill_between(x,(means-errs)*100,(means+errs)*100,color="#777777",alpha=.10); ax.axhline(0,color="#888888",lw=.8); ax.set_title(rule.capitalize()+" execution rule"); ax.set_xlabel("Formation update"); ax.grid(axis="y",alpha=.25)
    axes[0].set_ylabel("Four-choice margin M (percentage points)"); axes[-1].legend(frameon=False,fontsize=8,loc="lower left"); fig.suptitle("Local packet controls over formation",y=1.02); fig.tight_layout(); fig.savefig(out/"01_local_trajectory.png",dpi=180,bbox_inches="tight"); fig.savefig(out/"01_local_trajectory.pdf",bbox_inches="tight"); plt.close(fig)


def plot_endpoint(summary,out):
    baseline=baseline_records(); names=("natural",)+metrics.MODES+("ensemble",); labels={"natural":"natural","ensemble":"three-control mean","single_slot_cycle":"single slot","rank_canonical":"rank canonical","equality_pattern_relabel":"equality pattern"}; palette=plt.get_cmap("tab10").colors; colors={"natural":"#222222","ensemble":"#000000"}; colors.update({m:palette[i] for i,m in enumerate(metrics.MODES)}); fig,axes=plt.subplots(2,2,figsize=(12.3,7.1)); specs=(("M","Margin M (pp)"),("target_probability","Target probability (%)"),("hit4","Four-choice hit (%)"),("hit17","Full-action hit (%)")); x=np.arange(2); offsets=np.linspace(-.34,.34,len(names)); width=.13
    for ax,(key,ylabel) in zip(axes.flat,specs):
        for name,off in zip(names,offsets):
            means=[]; errs=[]
            for rule in metrics.RULES:
                c=rule+"_PL_live"
                if name=="natural": values=[baseline[seed,c,6000][key] for seed in metrics.SEEDS]; means.append(np.mean(values)); errs.append(np.std(values,ddof=1)/4)
                elif name=="ensemble": st=summary["ensemble_endpoint"][c][key]; means.append(st["mean"]); errs.append(st["sample_sd"]/4)
                else: st=summary["endpoint"][name][c][key]; means.append(st["mean"]); errs.append(st["sample_sd"]/4)
            ax.bar(x+off,np.asarray(means)*100,width,yerr=np.asarray(errs)*100,capsize=2,color=colors[name],alpha=.88,label=labels[name])
        ax.axhline(0,color="#888888",lw=.8); ax.set_xticks(x,["strict","reciprocal"]); ax.set_ylabel(ylabel); ax.set_title(ylabel.split(" (")[0]); ax.grid(axis="y",alpha=.25)
    axes[0,0].legend(frameon=False,fontsize=7,ncol=2,loc="lower left"); fig.suptitle("Endpoint behavior at update 6000",y=1.01); fig.tight_layout(); fig.savefig(out/"02_local_endpoint.png",dpi=180,bbox_inches="tight"); fig.savefig(out/"02_local_endpoint.pdf",bbox_inches="tight"); plt.close(fig)


def plot_selectivity(summary,out):
    names=list(metrics.MODES)+["ensemble"]; labels=["single slot","rank canonical","equality pattern","three-control mean"]; fig,ax=plt.subplots(figsize=(10.4,4.7)); stats=summary["paired_selectivity"]; x=np.arange(len(names)); width=.24
    for j,(rule,color) in enumerate((("strict","#c44e52"),("reciprocal","#4c72b0"),("both rules","#dd8452"))):
        vals=[]; lo=[]; hi=[]
        for name in names:
            st=stats[name]["rule_averaged"]["statistics"] if rule=="both rules" else stats[name][rule]; vals.append(100*st["mean"]); lo.append(100*(st["mean"]-st["ci95_lower"])); hi.append(100*(st["ci95_upper"]-st["mean"]))
        ax.bar(x+(j-1)*width,vals,width,yerr=np.asarray([lo,hi]),capsize=3,color=color,label=rule)
    ax.axhline(0,color="#888888",lw=.8); ax.set_xticks(x,labels,rotation=16,ha="right"); ax.set_ylabel("natural M − local-control M (percentage points)"); ax.set_title("Local-control selectivity"); ax.legend(frameon=False,fontsize=8); ax.grid(axis="y",alpha=.25); fig.tight_layout(); fig.savefig(out/"03_local_selectivity.png",dpi=180,bbox_inches="tight"); fig.savefig(out/"03_local_selectivity.pdf",bbox_inches="tight"); plt.close(fig)


def generate(run):
    run=Path(run).resolve(); summary=read(run/"summary_001"/"summary.json"); out=run/"figures_001"; out.mkdir(); plot_trajectory(summary,out); plot_endpoint(summary,out); plot_selectivity(summary,out); receipt={"status":"passed","summary_sha256":dataset.sha(run/"summary_001"/"summary.json"),"figures":{p.name:dataset.sha(p) for p in sorted(out.iterdir()) if p.is_file()},"labels":"English labels selected for portable font rendering"}; dataset.write(out/"receipt.json",receipt); return receipt


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--run",required=True); args=parser.parse_args(); print(json.dumps(generate(args.run),ensure_ascii=False))
