"""Generate manuscript-oriented plots for the packet identity controls."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import dataset, metrics

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "research_program/triadic_content_response_study/results/content_001"


def read(path):
    return dataset.read(path)


def pct(stat):
    return 100.0 * float(stat["mean"])


def pct_err(stat):
    return 100.0 * float(stat["sample_sd"]) / np.sqrt(16.0)


def baseline_records():
    summary = read(CONTENT / "summary_001" / "summary.json")
    return {(r["seed"], r["condition"], r["update"]): r for r in summary["records"]}, summary


def plot_trajectory(summary, out):
    baseline, _ = baseline_records()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    colors = {"baseline": "#222222", "endpoint_cycle": "#c44e52", "cross_group_cycle": "#4c72b0"}
    labels = {"baseline": "same-group natural packet", "endpoint_cycle": "within-group endpoint cycle", "cross_group_cycle": "cross-group cycle"}
    x = np.asarray(metrics.STEPS)
    for ax, rule in zip(axes, metrics.RULES):
        condition = rule + "_PL_live"
        base_means = []; base_err = []
        for step in metrics.STEPS:
            vals = [baseline[seed, condition, step]["M"] for seed in metrics.SEEDS]
            base_means.append(np.mean(vals)); base_err.append(np.std(vals, ddof=1) / np.sqrt(16))
        ax.plot(x, np.asarray(base_means) * 100, color=colors["baseline"], lw=2, label=labels["baseline"])
        ax.fill_between(x, (np.asarray(base_means) - np.asarray(base_err)) * 100,
                        (np.asarray(base_means) + np.asarray(base_err)) * 100, color=colors["baseline"], alpha=.10)
        for mode in metrics.MODES:
            means = [summary["trajectory"][mode][condition][str(step)]["M"]["mean"] for step in metrics.STEPS]
            errs = [summary["trajectory"][mode][condition][str(step)]["M"]["sample_sd"] / np.sqrt(16) for step in metrics.STEPS]
            means = np.asarray(means); errs = np.asarray(errs)
            ax.plot(x, means * 100, color=colors[mode], lw=2, label=labels[mode])
            ax.fill_between(x, (means - errs) * 100, (means + errs) * 100, color=colors[mode], alpha=.14)
        ax.axhline(0, color="#888888", lw=.8)
        ax.set_title(rule.capitalize() + " execution rule")
        ax.set_xlabel("Formation update")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Four-choice margin M (percentage points)")
    axes[-1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Packet identity controls over formation", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "01_identity_control_trajectory.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "01_identity_control_trajectory.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_endpoint_diagnostics(summary, out):
    baseline, _ = baseline_records()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2))
    specs = [("M", "Margin M (pp)"), ("target_probability", "Target probability (%)"),
             ("hit4", "Four-choice hit (%)"), ("hit17", "Full-action hit (%)")]
    colors = {"same": "#222222", "endpoint_cycle": "#c44e52", "cross_group_cycle": "#4c72b0"}
    labels = {"same": "natural", "endpoint_cycle": "endpoint cycle", "cross_group_cycle": "cross-group cycle"}
    x = np.arange(2); width = .24
    for ax, (key, ylabel) in zip(axes.flat, specs):
        for j, (name, offset) in enumerate((("same", -.24), ("endpoint_cycle", 0), ("cross_group_cycle", .24))):
            means = []; errs = []
            for rule in metrics.RULES:
                condition = rule + "_PL_live"
                if name == "same":
                    vals = [baseline[seed, condition, 6000][key] for seed in metrics.SEEDS]
                    means.append(np.mean(vals)); errs.append(np.std(vals, ddof=1) / np.sqrt(16))
                else:
                    stat = summary["endpoint"][name][condition][key]
                    means.append(stat["mean"]); errs.append(stat["sample_sd"] / np.sqrt(16))
            ax.bar(x + offset, np.asarray(means) * (100 if key != "M" else 100), width,
                   yerr=np.asarray(errs) * 100, capsize=3, color=colors[name], label=labels[name])
        ax.axhline(0, color="#888888", lw=.8)
        ax.set_xticks(x, ["strict", "reciprocal"])
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=.25)
        ax.set_title(ylabel.split(" (")[0])
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Endpoint behavior at update 6000", y=1.01)
    fig.tight_layout()
    fig.savefig(out / "02_identity_control_endpoint.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "02_identity_control_endpoint.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_selectivity(summary, out):
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    entries = [("strict", "endpoint_cycle"), ("strict", "cross_group_cycle"),
               ("reciprocal", "endpoint_cycle"), ("reciprocal", "cross_group_cycle"),
               ("both rules", "endpoint_cycle"), ("both rules", "cross_group_cycle")]
    labels = ["strict\nendpoint", "strict\ncross-group", "reciprocal\nendpoint", "reciprocal\ncross-group",
              "both\nendpoint", "both\ncross-group"]
    vals = []; lo = []; hi = []
    for rule, mode in entries:
        if rule == "both rules":
            stat = summary["paired_selectivity"]["rule_averaged"][mode]
        else:
            stat = summary["paired_selectivity"][mode][rule]
        vals.append(100 * stat["mean"]); lo.append(100 * (stat["mean"] - stat["ci95_lower"])); hi.append(100 * (stat["ci95_upper"] - stat["mean"]))
    x = np.arange(len(vals))
    colors = ["#c44e52", "#4c72b0", "#c44e52", "#4c72b0", "#dd8452", "#6b8e23"]
    ax.bar(x, vals, yerr=np.asarray([lo, hi]), capsize=4, color=colors)
    ax.axhline(0, color="#888888", lw=.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("same-group M − control M (percentage points)")
    ax.set_title("Packet endpoint selectivity")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out / "03_identity_selectivity.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / "03_identity_selectivity.pdf", bbox_inches="tight")
    plt.close(fig)


def generate(run):
    run = Path(run).resolve(); summary = read(run / "summary_001" / "summary.json")
    out = run / "figures_001"; out.mkdir()
    plot_trajectory(summary, out); plot_endpoint_diagnostics(summary, out); plot_selectivity(summary, out)
    receipt = {"status": "passed", "summary_sha256": dataset.sha(run / "summary_001" / "summary.json"),
               "figures": {p.name: dataset.sha(p) for p in sorted(out.iterdir()) if p.is_file()},
               "plot_runtime": "matplotlib", "labels": "English labels selected for portable font rendering"}
    dataset.write(out / "receipt.json", receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True)
    args = parser.parse_args(); print(json.dumps(generate(args.run), ensure_ascii=False))
