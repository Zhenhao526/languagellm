"""Make two fixed descriptive figures from summary JSON only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import dataset, metrics


def read(path):
    return dataset.read_json(path)


def plot(run):
    run = Path(run).resolve(); summary_dir = run / "summary_001"; summary = read(summary_dir / "summary.json")
    out = run / "figures_001"
    if out.exists():
        raise ValueError("Never overwrite figure output")
    out.mkdir()
    steps = np.asarray(metrics.STEPS, dtype=float)
    colors = {"strict_PL_silent": "#7f8c8d", "strict_PL_live": "#2878b5",
              "reciprocal_PL_silent": "#c18f00", "reciprocal_PL_live": "#c23b22"}
    labels = {"strict_PL_silent": "strict / silent", "strict_PL_live": "strict / live",
              "reciprocal_PL_silent": "reciprocal / silent", "reciprocal_PL_live": "reciprocal / live"}
    fig, ax = plt.subplots(figsize=(8.8, 5.2), constrained_layout=True)
    for condition in metrics.CONDITIONS:
        means = []; ses = []
        for step in metrics.STEPS:
            item = summary["trajectory"][condition][str(step)]["M"]
            means.append(100 * item["mean"]); ses.append(100 * item["sample_sd"] / np.sqrt(16))
        means = np.asarray(means); ses = np.asarray(ses)
        ax.plot(steps, means, marker="o", linewidth=2, color=colors[condition], label=labels[condition])
        ax.fill_between(steps, means - ses, means + ses, color=colors[condition], alpha=.12, linewidth=0)
    ax.axhline(0, color="#333333", linewidth=.8)
    ax.set_xscale("symlog", linthresh=100)
    ax.set_xticks(metrics.STEPS); ax.set_xticklabels([str(x) for x in metrics.STEPS])
    ax.set_xlabel("training updates (symlog x-axis)")
    ax.set_ylabel("four-choice content margin M (percentage points)")
    ax.set_title("Four-choice content margin over training\nshading: mean +/- SE across 16 paired initializations")
    ax.grid(axis="y", alpha=.25); ax.legend(frameon=False, ncol=2, loc="lower left")
    fig.savefig(out / "01_content_margin_trajectory.png", dpi=180); fig.savefig(out / "01_content_margin_trajectory.pdf")
    plt.close(fig)

    order = list(metrics.CONDITIONS)
    x = np.arange(len(order)); labels_short = [labels[c] for c in order]
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.7), constrained_layout=True)
    for ax, key, title, ylabel, ref in zip(axes,
        ("M", "hit4", "hit17"),
        ("content margin M", "target is top among 4 candidates", "target is top among all 17 actions"),
        ("percentage points", "rate (%)", "rate (%)"), (0, 25, None)):
        vals = np.asarray([summary["endpoint"][c][key]["mean"] for c in order])
        err = np.asarray([summary["endpoint"][c][key]["sample_sd"] / np.sqrt(16) for c in order])
        scale = 100 if key == "M" else 100
        ax.bar(x, scale * vals, yerr=scale * err, color=[colors[c] for c in order], alpha=.9,
               capsize=3, error_kw={"linewidth": 1}, width=.72)
        if ref is not None: ax.axhline(ref, color="#444444", linestyle="--", linewidth=.9)
        ax.axhline(0, color="#333333", linewidth=.7)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel); ax.set_xticks(x); ax.set_xticklabels(labels_short, rotation=35, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=.25)
    axes[1].text(.02, .96, "dashed = 25% four-choice chance baseline", transform=axes[1].transAxes, va="top", fontsize=8, color="#444444")
    fig.suptitle("Endpoint content diagnostics at update 6000 (mean +/- SE)", fontsize=13)
    fig.savefig(out / "02_endpoint_content_diagnostics.png", dpi=180); fig.savefig(out / "02_endpoint_content_diagnostics.pdf")
    plt.close(fig)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True)
    args = parser.parse_args(); print(plot(args.run))
