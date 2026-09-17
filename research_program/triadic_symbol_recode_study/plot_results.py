"""Generate plots for symbol-identity and packet-position recodings."""
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
    baseline = baseline_records(); fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    colors = {"baseline": "#222222", "global_symbol_permutation": "#c44e52", "position_rotation": "#4c72b0", "position_reverse": "#dd8452"}
    labels = {"baseline": "natural packet", "global_symbol_permutation": "global symbol permutation", "position_rotation": "position rotation", "position_reverse": "position reverse"}
    x = np.asarray(metrics.STEPS)
    for ax, rule in zip(axes, metrics.RULES):
        condition = rule + "_PL_live"; vals = []
        for step in metrics.STEPS:
            vals.append([baseline[seed, condition, step]["M"] for seed in metrics.SEEDS])
        bm = np.mean(vals, axis=1); be = np.std(vals, axis=1, ddof=1) / np.sqrt(16)
        ax.plot(x, bm * 100, color=colors["baseline"], lw=2, label=labels["baseline"]); ax.fill_between(x, (bm - be) * 100, (bm + be) * 100, color=colors["baseline"], alpha=.10)
        for mode in metrics.MODES:
            means = np.asarray([summary["trajectory"][mode][condition][str(step)]["M"]["mean"] for step in metrics.STEPS]); errs = np.asarray([summary["trajectory"][mode][condition][str(step)]["M"]["sample_sd"] / np.sqrt(16) for step in metrics.STEPS])
            ax.plot(x, means * 100, color=colors[mode], lw=2, label=labels[mode]); ax.fill_between(x, (means - errs) * 100, (means + errs) * 100, color=colors[mode], alpha=.12)
        ax.axhline(0, color="#888888", lw=.8); ax.set_title(rule.capitalize() + " execution rule"); ax.set_xlabel("Formation update"); ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Four-choice margin M (percentage points)"); axes[-1].legend(frameon=False, fontsize=8, loc="lower left"); fig.suptitle("Recoding controls over formation", y=1.02); fig.tight_layout()
    fig.savefig(out / "01_recoding_trajectory.png", dpi=180, bbox_inches="tight"); fig.savefig(out / "01_recoding_trajectory.pdf", bbox_inches="tight"); plt.close(fig)


def plot_endpoint(summary, out):
    baseline = read(CONTENT / "summary_001" / "summary.json"); b = {(r["seed"], r["condition"], r["update"]): r for r in baseline["records"]}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2)); specs = [("M", "Margin M (pp)"), ("target_probability", "Target probability (%)"), ("hit4", "Four-choice hit (%)"), ("hit17", "Full-action hit (%)")]
    colors = {"natural": "#222222", "global_symbol_permutation": "#c44e52", "position_rotation": "#4c72b0", "position_reverse": "#dd8452"}; labels = {"natural": "natural", "global_symbol_permutation": "global symbol", "position_rotation": "position rotation", "position_reverse": "position reverse"}; x = np.arange(2); offsets = (-.27, -.09, .09, .27)
    for ax, (key, ylabel) in zip(axes.flat, specs):
        for name, offset in zip(("natural",) + metrics.MODES, offsets):
            means = []; errs = []
            for rule in metrics.RULES:
                condition = rule + "_PL_live"
                if name == "natural":
                    values = [b[seed, condition, 6000][key] for seed in metrics.SEEDS]; means.append(np.mean(values)); errs.append(np.std(values, ddof=1) / np.sqrt(16))
                else:
                    stat = summary["endpoint"][name][condition][key]; means.append(stat["mean"]); errs.append(stat["sample_sd"] / np.sqrt(16))
            ax.bar(x + offset, np.asarray(means) * 100, .18, yerr=np.asarray(errs) * 100, capsize=3, color=colors[name], label=labels[name])
        ax.axhline(0, color="#888888", lw=.8); ax.set_xticks(x, ["strict", "reciprocal"]); ax.set_ylabel(ylabel); ax.set_title(ylabel.split(" (")[0]); ax.grid(axis="y", alpha=.25)
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left"); fig.suptitle("Endpoint behavior at update 6000", y=1.01); fig.tight_layout(); fig.savefig(out / "02_recoding_endpoint.png", dpi=180, bbox_inches="tight"); fig.savefig(out / "02_recoding_endpoint.pdf", bbox_inches="tight"); plt.close(fig)


def plot_selectivity(summary, out):
    fig, ax = plt.subplots(figsize=(9.5, 4.6)); stats = summary["paired_selectivity"]
    x = np.arange(len(metrics.MODES)); width = .24
    rule_specs = (("strict", "#c44e52"), ("reciprocal", "#4c72b0"), ("both rules", "#dd8452"))
    for j, (rule, color) in enumerate(rule_specs):
        vals = []; lo = []; hi = []
        for mode in metrics.MODES:
            st = stats[mode]["rule_averaged"]["statistics"] if rule == "both rules" else stats[mode][rule]
            vals.append(100 * st["mean"]); lo.append(100 * (st["mean"] - st["ci95_lower"])); hi.append(100 * (st["ci95_upper"] - st["mean"]))
        ax.bar(x + (j - 1) * width, vals, width, yerr=np.asarray([lo, hi]), capsize=3, color=color, label=rule)
    ax.axhline(0, color="#888888", lw=.8); ax.set_xticks(x, ["global symbol", "position rotation", "position reverse"]); ax.set_ylabel("same-group M − recoded M (percentage points)"); ax.set_title("Recoding selectivity"); ax.legend(frameon=False, fontsize=8); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(out / "03_recoding_selectivity.png", dpi=180, bbox_inches="tight"); fig.savefig(out / "03_recoding_selectivity.pdf", bbox_inches="tight"); plt.close(fig)


def generate(run):
    run = Path(run).resolve(); summary = read(run / "summary_001" / "summary.json"); out = run / "figures_001"; out.mkdir(); plot_trajectory(summary, out); plot_endpoint(summary, out); plot_selectivity(summary, out)
    receipt = {"status": "passed", "summary_sha256": dataset.sha(run / "summary_001" / "summary.json"), "figures": {p.name: dataset.sha(p) for p in sorted(out.iterdir()) if p.is_file()}, "labels": "English labels selected for portable font rendering"}; dataset.write(out / "receipt.json", receipt); return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True); args = parser.parse_args(); print(json.dumps(generate(args.run), ensure_ascii=False))
