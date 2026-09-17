"""Generate publication-facing plots for the random permutation ensemble."""
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


def baseline_records():
    summary = read(CONTENT / "summary_001" / "summary.json")
    return {(r["seed"], r["condition"], r["update"]): r for r in summary["records"]}


def plot_trajectory(summary, out):
    baseline = baseline_records()
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5), sharey=True)
    palette = plt.get_cmap("tab10").colors
    colors = {mode: palette[i] for i, mode in enumerate(metrics.MODES)}
    labels = {mode: mode.replace("perm_", "perm ") for mode in metrics.MODES}
    x = np.asarray(metrics.STEPS)
    for ax, rule in zip(axes, metrics.RULES):
        condition = rule + "_PL_live"
        vals = [[baseline[seed, condition, step]["M"] for seed in metrics.SEEDS] for step in metrics.STEPS]
        bm = np.mean(vals, axis=1); be = np.std(vals, axis=1, ddof=1) / np.sqrt(16)
        ax.plot(x, bm * 100, color="#222222", lw=2.3, label="natural packet")
        ax.fill_between(x, (bm - be) * 100, (bm + be) * 100, color="#222222", alpha=.10)
        for mode in metrics.MODES:
            means = np.asarray([summary["trajectory"][mode][condition][str(step)]["M"]["mean"] for step in metrics.STEPS])
            errs = np.asarray([summary["trajectory"][mode][condition][str(step)]["M"]["sample_sd"] / np.sqrt(16) for step in metrics.STEPS])
            ax.plot(x, means * 100, color=colors[mode], lw=1.0, alpha=.55, label=labels[mode])
            ax.fill_between(x, (means - errs) * 100, (means + errs) * 100, color=colors[mode], alpha=.035)
        ens = summary["ensemble_trajectory"][condition]
        means = np.asarray([ens[str(step)]["mean"] for step in metrics.STEPS])
        errs = np.asarray([ens[str(step)]["sample_sd"] / np.sqrt(16) for step in metrics.STEPS])
        ax.plot(x, means * 100, color="#000000", lw=2.8, ls="--", label="six-permutation mean")
        ax.fill_between(x, (means - errs) * 100, (means + errs) * 100, color="#777777", alpha=.12)
        ax.axhline(0, color="#888888", lw=.8); ax.set_title(rule.capitalize() + " execution rule")
        ax.set_xlabel("Formation update"); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Four-choice margin M (percentage points)")
    axes[-1].legend(frameon=False, fontsize=7.5, loc="lower left", ncol=2)
    fig.suptitle("Random symbol permutations over formation", y=1.02)
    fig.tight_layout(); fig.savefig(out / "01_ensemble_trajectory.png", dpi=180, bbox_inches="tight"); fig.savefig(out / "01_ensemble_trajectory.pdf", bbox_inches="tight"); plt.close(fig)


def plot_endpoint(summary, out):
    baseline = baseline_records()
    names = ("natural",) + metrics.MODES + ("ensemble",)
    labels = {"natural": "natural", "ensemble": "six-permutation mean"}
    labels.update({mode: mode.replace("perm_", "perm ") for mode in metrics.MODES})
    colors = {"natural": "#222222", "ensemble": "#000000"}
    colors.update({mode: plt.get_cmap("tab10").colors[i] for i, mode in enumerate(metrics.MODES)})
    fig, axes = plt.subplots(2, 2, figsize=(12.7, 7.2))
    specs = (("M", "Margin M (pp)"), ("target_probability", "Target probability (%)"),
             ("hit4", "Four-choice hit (%)"), ("hit17", "Full-action hit (%)"))
    x = np.arange(2); offsets = np.linspace(-.38, .38, len(names)); width = .105
    for ax, (key, ylabel) in zip(axes.flat, specs):
        for name, offset in zip(names, offsets):
            means = []; errs = []
            for rule in metrics.RULES:
                condition = rule + "_PL_live"
                if name == "natural":
                    values = [baseline[seed, condition, 6000][key] for seed in metrics.SEEDS]
                    means.append(np.mean(values)); errs.append(np.std(values, ddof=1) / np.sqrt(16))
                elif name == "ensemble":
                    stat = summary["ensemble_endpoint"][condition][key]
                    means.append(stat["mean"]); errs.append(stat["sample_sd"] / np.sqrt(16))
                else:
                    stat = summary["endpoint"][name][condition][key]
                    means.append(stat["mean"]); errs.append(stat["sample_sd"] / np.sqrt(16))
            ax.bar(x + offset, np.asarray(means) * 100, width, yerr=np.asarray(errs) * 100,
                   capsize=2, color=colors[name], alpha=.88, label=labels[name])
        ax.axhline(0, color="#888888", lw=.8); ax.set_xticks(x, ["strict", "reciprocal"])
        ax.set_ylabel(ylabel); ax.set_title(ylabel.split(" (")[0]); ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(frameon=False, fontsize=7, ncol=2, loc="lower left")
    fig.suptitle("Endpoint behavior at update 6000", y=1.01)
    fig.tight_layout(); fig.savefig(out / "02_ensemble_endpoint.png", dpi=180, bbox_inches="tight"); fig.savefig(out / "02_ensemble_endpoint.pdf", bbox_inches="tight"); plt.close(fig)


def plot_selectivity(summary, out):
    names = list(metrics.MODES) + ["ensemble"]
    labels = [name.replace("perm_", "perm ") if name != "ensemble" else "six-permutation mean" for name in names]
    fig, ax = plt.subplots(figsize=(11.2, 4.8)); stats = summary["paired_selectivity"]
    x = np.arange(len(names)); width = .24
    rule_specs = (("strict", "#c44e52"), ("reciprocal", "#4c72b0"), ("both rules", "#dd8452"))
    for j, (rule, color) in enumerate(rule_specs):
        vals = []; lo = []; hi = []
        for name in names:
            stat = stats[name]["rule_averaged"]["statistics"] if rule == "both rules" else stats[name][rule]
            vals.append(100 * stat["mean"]); lo.append(100 * (stat["mean"] - stat["ci95_lower"])); hi.append(100 * (stat["ci95_upper"] - stat["mean"]))
        ax.bar(x + (j - 1) * width, vals, width, yerr=np.asarray([lo, hi]), capsize=3, color=color, label=rule)
    ax.axhline(0, color="#888888", lw=.8); ax.set_xticks(x, labels, rotation=20, ha="right")
    ax.set_ylabel("natural M − permuted M (percentage points)"); ax.set_title("Permutation selectivity")
    ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(out / "03_ensemble_selectivity.png", dpi=180, bbox_inches="tight"); fig.savefig(out / "03_ensemble_selectivity.pdf", bbox_inches="tight"); plt.close(fig)


def generate(run):
    run = Path(run).resolve(); summary = read(run / "summary_001" / "summary.json")
    out = run / "figures_001"; out.mkdir()
    plot_trajectory(summary, out); plot_endpoint(summary, out); plot_selectivity(summary, out)
    receipt = {"status": "passed", "summary_sha256": dataset.sha(run / "summary_001" / "summary.json"),
               "figures": {p.name: dataset.sha(p) for p in sorted(out.iterdir()) if p.is_file()},
               "labels": "English labels selected for portable font rendering"}
    dataset.write(out / "receipt.json", receipt); return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True); args = parser.parse_args()
    print(json.dumps(generate(args.run), ensure_ascii=False))
