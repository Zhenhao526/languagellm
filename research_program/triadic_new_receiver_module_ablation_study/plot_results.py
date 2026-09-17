"""Plots for the module-selective transmission ablation."""
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
    require(len(data.get("runs", [])) == 64, "Expected 64 selective runs")
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
    by = {(r["seed"], r["schedule"], r["arm"], bool(r["live"])): r for r in runs}
    full_data = json.loads(design.FULL_RESULT.read_text())
    for r in full_data["runs"]:
        by[(r["seed"], r["schedule"], "full", bool(r["live"]))] = r
    seeds = list(design.SEEDS)
    arms = ("full",) + design.ARMS
    colors = {"full": "#222222", "action_only": "#386cb0", "sender_only": "#d95f02"}
    updates = np.asarray(design.CHECKPOINTS, dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), sharey=True)
    for ax, schedule in zip(axes, design.SCHEDULES):
        ax.axhline(0, color="0.35", lw=0.8)
        for arm in arms:
            gains = []
            for seed in seeds:
                live = np.asarray([row["target_trajectory"]["q_rate"] for row in sorted(by[seed, schedule, arm, True]["trajectory"], key=lambda x: x["update"])])
                silent = np.asarray([row["target_trajectory"]["q_rate"] for row in sorted(by[seed, schedule, arm, False]["trajectory"], key=lambda x: x["update"])])
                gains.append((live - silent) * 100)
            mean, lo, hi = mean_ci(gains)
            ax.plot(updates, mean, marker="o", color=colors[arm], label=arm)
            ax.fill_between(updates, lo, hi, color=colors[arm], alpha=0.12)
        ax.set_title(schedule)
        ax.set_xlabel("adaptation updates")
    axes[0].set_ylabel("live − silent monitor Q (pp)")
    axes[1].legend(frameon=False)
    fig.suptitle("Module ablation: routed packet advantage")
    fig.tight_layout()
    fig.savefig(output / "q_gain_trajectories.png", dpi=180)
    fig.savefig(output / "q_gain_trajectories.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(len(arms))
    width = 0.26
    for j, schedule in enumerate(design.SCHEDULES):
        gains = []
        for arm in arms:
            vals = [
                (by[seed, schedule, arm, True]["final"]["new_layouts"]["q_rate"] -
                 by[seed, schedule, arm, False]["final"]["new_layouts"]["q_rate"]) * 100
                for seed in seeds
            ]
            gains.append(vals)
        mean, lo, hi = mean_ci(np.asarray(gains).T)
        ax.errorbar(x + (j - 0.5) * width, mean, yerr=np.vstack((mean - lo, hi - mean)),
                    fmt="o", capsize=3, label=schedule)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, ["full", "action-only", "sender-only"])
    ax.set_ylabel("final heldout live − silent Q (pp)")
    ax.set_title("Final routed packet advantage by trainable module")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "q_gain_by_arm.png", dpi=180)
    fig.savefig(output / "q_gain_by_arm.pdf")
    plt.close(fig)

    # Absolute final Q compares adaptation quality, not only the channel gain.
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    labels = []
    rows = []
    for schedule in design.SCHEDULES:
        for arm in arms:
            live = [by[seed, schedule, arm, True]["final"]["new_layouts"]["q_rate"] * 100 for seed in seeds]
            silent = [by[seed, schedule, arm, False]["final"]["new_layouts"]["q_rate"] * 100 for seed in seeds]
            rows.extend([live, silent])
            labels.append(f"{schedule}\n{arm}\nlive")
            labels.append(f"{schedule}\n{arm}\nsilent")
    positions = np.arange(len(rows))
    ax.boxplot(rows, positions=positions, widths=0.62, showfliers=False)
    ax.set_xticks(positions, labels, rotation=45, ha="right")
    ax.set_ylabel("final heldout Q (%)")
    ax.set_title("Absolute adaptation quality")
    fig.tight_layout()
    fig.savefig(output / "absolute_q_by_arm.png", dpi=180)
    fig.savefig(output / "absolute_q_by_arm.pdf")
    plt.close(fig)

    receipt = dict(status="generated", visual_review="pending",
                   source=str(results.get("source", "execution/results.json")),
                   figures=[p.name for p in sorted(output.glob("*.png"))])
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(load(args.results), args.output)
