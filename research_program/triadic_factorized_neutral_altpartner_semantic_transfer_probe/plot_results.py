"""Plot aligned/placebo transfer and the rematching interaction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pct(x):
    return 100.0 * float(x)


def plot(summary_path, output):
    summary_path = Path(summary_path).resolve(); output = Path(output).resolve(); require(not output.exists(), "Refuse to overwrite figures")
    summary = json.loads(summary_path.read_text(encoding="utf8")); output.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    colors = {"static": "#2f6db0", "rematched": "#d66b2d"}
    labels = []; aligned = []; placebo = []; diff = []; errors = []; x = 0; positions = []
    for schedule in ("static", "rematched"):
        cell = summary["cells"][f"{schedule}/live"]
        for prefix, values, color in (("aligned", aligned, colors[schedule]), ("placebo", placebo, "#b9c3cf")):
            entry = cell[prefix]["plan_transfer"]; values.append(pct(entry["mean"])); errors.append((pct(entry["mean"]-entry["ci95_t15"][0]), pct(entry["ci95_t15"][1]-entry["mean"])))
        labels.extend([f"{schedule}\naligned", f"{schedule}\nplacebo"]); positions.extend([x, x+1]); x += 3
    vals = np.array([aligned[0], placebo[0], aligned[1], placebo[1]])
    err = np.array(errors).T
    axes[0].bar(positions, vals, color=[colors["static"], "#b9c3cf", colors["rematched"], "#b9c3cf"], width=0.72, edgecolor="white")
    axes[0].errorbar(positions, vals, yerr=err, fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    axes[0].axhline(0, color="#222222", linewidth=0.8); axes[0].set_xticks(positions, labels); axes[0].set_ylabel("plan transfer (percentage points)"); axes[0].set_title("Aligned versus placebo"); axes[0].grid(axis="y", alpha=0.25)
    fields = [("plan_transfer", "plan"), ("physical", "physical"), ("q", "Q"), ("conditional_q", "Q|physical"), ("action_change", "action change")]
    values=[]; lo=[]; hi=[]
    for field,_ in fields:
        entry=summary["interactions"][field]; values.append(pct(entry["mean"])); lo.append(pct(entry["mean"]-entry["ci95_t15"][0])); hi.append(pct(entry["ci95_t15"][1]-entry["mean"]))
    pos=np.arange(len(fields)); axes[1].bar(pos,values,color="#6d9e72",width=0.62,edgecolor="white"); axes[1].errorbar(pos,values,yerr=np.array([lo,hi]),fmt="none",ecolor="#222222",capsize=3,linewidth=1); axes[1].axhline(0,color="#222222",linewidth=0.8); axes[1].set_xticks(pos,[label for _,label in fields],rotation=25,ha="right"); axes[1].set_ylabel("rematched × communication (pp)"); axes[1].set_title("Interaction on aligned−placebo"); axes[1].grid(axis="y",alpha=0.25)
    fig.suptitle("Factorized neutral aligned/placebo message transfer",fontsize=13)
    fig.savefig(output/"semantic_transfer.png",dpi=220); fig.savefig(output/"semantic_transfer.pdf"); plt.close(fig)
    receipt=dict(status="rendered",visual_review="pending",summary_sha256=__import__("hashlib").sha256(summary_path.read_bytes()).hexdigest(),files=["semantic_transfer.png","semantic_transfer.pdf"])
    (output/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return receipt


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--summary",required=True); parser.add_argument("--output",required=True); args=parser.parse_args(); print(json.dumps(plot(args.summary,args.output),ensure_ascii=False))
