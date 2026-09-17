"""Plots for the two-generation protocol chain."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import design, summarize


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    data = json.loads(Path(path).read_text())
    require(len(data.get("runs", [])) == 64, "Expected 64 chain runs")
    return data


def mean_ci(values):
    x = np.asarray(values, dtype=np.float64)
    mean = x.mean(axis=0)
    half = summarize.T7_975 * x.std(axis=0, ddof=1) / math.sqrt(x.shape[0])
    return mean, mean - half, mean + half


def plot(results, output):
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    runs = results["runs"]
    by = {(r["seed"], r["generation"], r["schedule"], bool(r["live"])): r for r in runs}
    seeds = list(design.SEEDS)
    updates = np.asarray(design.CHECKPOINTS, dtype=np.float64)
    colors = {"static": "#386cb0", "rematched": "#d95f02"}

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, generation in zip(axes, design.GENERATIONS):
        ax.axhline(0, color="0.35", lw=0.8)
        for schedule in design.SCHEDULES:
            gains = []
            for seed in seeds:
                live = np.asarray([row["target_trajectory"]["q_rate"] for row in sorted(by[seed, generation, schedule, True]["trajectory"], key=lambda x: x["update"])])
                silent = np.asarray([row["target_trajectory"]["q_rate"] for row in sorted(by[seed, generation, schedule, False]["trajectory"], key=lambda x: x["update"])])
                gains.append((live - silent) * 100)
            mean, lo, hi = mean_ci(gains)
            ax.plot(updates, mean, marker="o", color=colors[schedule], label=schedule)
            ax.fill_between(updates, lo, hi, color=colors[schedule], alpha=0.14)
        ax.set_title(f"generation {generation}")
        ax.set_xlabel("adaptation updates")
    axes[0].set_ylabel("live − silent monitor Q (pp)")
    axes[1].legend(frameon=False)
    fig.suptitle("Routed packet advantage across the learner-replacement chain")
    fig.tight_layout()
    fig.savefig(output / "q_gain_trajectories.png", dpi=180)
    fig.savefig(output / "q_gain_trajectories.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.3, 4.6))
    x = np.arange(len(design.GENERATIONS))
    width = 0.32
    for j, schedule in enumerate(design.SCHEDULES):
        means, los, his = [], [], []
        for generation in design.GENERATIONS:
            vals = [
                (by[seed, generation, schedule, True]["final"]["new_layouts"]["q_rate"] -
                 by[seed, generation, schedule, False]["final"]["new_layouts"]["q_rate"]) * 100
                for seed in seeds
            ]
            _, lo, hi = mean_ci(np.asarray(vals))
            means.append(float(np.mean(vals))); los.append(float(np.mean(vals) - lo)); his.append(float(hi - np.mean(vals)))
        ax.errorbar(x + (j - 0.5) * width, means, yerr=np.vstack((los, his)), fmt="o", capsize=3, label=schedule)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, ["generation 2\nreplace A", "generation 3\nreplace B"])
    ax.set_ylabel("final heldout live − silent Q (pp)")
    ax.set_title("Routing advantage by generation and schedule")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "q_gain_by_generation.png", dpi=180)
    fig.savefig(output / "q_gain_by_generation.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels, rows = [], []
    for generation in design.GENERATIONS:
        for schedule in design.SCHEDULES:
            vals = [
                (by[seed, generation, schedule, True]["final"]["new_layouts"]["q_rate"] -
                 by[seed, generation, schedule, True]["inherited_parent"]["new_layouts"]["q_rate"]) * 100
                for seed in seeds
            ]
            rows.append(vals)
            labels.append(f"gen {generation}\n{schedule}")
    positions = np.arange(len(rows))
    ax.boxplot(rows, positions=positions, widths=0.62, showfliers=False)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(positions, labels)
    ax.set_ylabel("live child − live parent final Q (pp)")
    ax.set_title("Parent-to-child live retention")
    fig.tight_layout()
    fig.savefig(output / "live_retention.png", dpi=180)
    fig.savefig(output / "live_retention.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    labels, rows = [], []
    for generation in design.GENERATIONS:
        for schedule in design.SCHEDULES:
            for live in (True, False):
                vals = [by[seed, generation, schedule, live]["final"]["new_layouts"]["q_rate"] * 100 for seed in seeds]
                rows.append(vals)
                labels.append(f"gen {generation}\n{schedule}\n{'live' if live else 'silent'}")
    positions = np.arange(len(rows))
    ax.boxplot(rows, positions=positions, widths=0.62, showfliers=False)
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("final heldout Q (%)")
    ax.set_title("Absolute adaptation quality across generations")
    fig.tight_layout()
    fig.savefig(output / "absolute_q_by_generation.png", dpi=180)
    fig.savefig(output / "absolute_q_by_generation.pdf")
    plt.close(fig)

    receipt = dict(status="generated", visual_review="pending", source=str(results.get("source", "execution/results.json")),
                   figures=[p.name for p in sorted(output.glob("*.png"))])
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(load(args.results), args.output)
