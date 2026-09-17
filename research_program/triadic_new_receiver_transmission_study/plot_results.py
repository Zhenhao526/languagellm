"""Plots for the new-receiver transmission experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import summarize


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    data = json.loads(Path(path).read_text())
    require("runs" in data and len(data["runs"]) == 32, "Expected completed 32-run results")
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
    by = {(r["seed"], r["schedule"], bool(r["live"])): r for r in runs}
    seeds = sorted({r["seed"] for r in runs})
    updates = summarize.UPDATES
    colors = {"static": "#386cb0", "rematched": "#d95f02"}

    # Primary trajectory: paired live-minus-silent monitor Q-rate.
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.axhline(0, color="0.35", lw=0.8)
    for schedule in ("static", "rematched"):
        gains = []
        for seed in seeds:
            live = np.asarray([row["target_trajectory"]["q_rate"] for row in sorted(by[seed, schedule, True]["trajectory"], key=lambda x: x["update"])])
            silent = np.asarray([row["target_trajectory"]["q_rate"] for row in sorted(by[seed, schedule, False]["trajectory"], key=lambda x: x["update"])])
            gains.append((live - silent) * 100)
        mean, lo, hi = mean_ci(gains)
        ax.plot(updates, mean, marker="o", color=colors[schedule], label=schedule)
        ax.fill_between(updates, lo, hi, color=colors[schedule], alpha=0.16)
    ax.set(xlabel="adaptation updates", ylabel="live − silent Q-rate (pp)", title="New receiver: routed packet advantage")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "q_live_minus_silent_trajectories.png", dpi=180)
    fig.savefig(output / "q_live_minus_silent_trajectories.pdf")
    plt.close(fig)

    # Final heldout-layout gain by seed and schedule.
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    x = np.arange(len(seeds))
    width = 0.38
    for offset, schedule in ((-width / 2, "static"), (width / 2, "rematched")):
        gains = np.asarray([
            (by[seed, schedule, True]["final"]["new_layouts"]["q_rate"] -
             by[seed, schedule, False]["final"]["new_layouts"]["q_rate"]) * 100
            for seed in seeds
        ])
        ax.bar(x + offset, gains, width, color=colors[schedule], label=schedule)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, [str(seed) for seed in seeds], rotation=45, ha="right")
    ax.set(xlabel="source seed", ylabel="final heldout Q gain (pp)", title="New receiver endpoint: live − silent")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "q_final_gain_by_seed.png", dpi=180)
    fig.savefig(output / "q_final_gain_by_seed.pdf")
    plt.close(fig)

    # Final heldout metrics, schedule means with paired seed intervals.
    metrics = ("q_rate", "conditional_q_rate", "physical_execution_rate", "target_pair_legal_rate", "engagement_rate")
    labels = ("Q", "Q | physical", "physical", "target pair", "engagement")
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(len(metrics))
    for offset, schedule in ((-0.18, "static"), (0.18, "rematched")):
        diffs = np.asarray([
            [by[seed, schedule, True]["final"]["new_layouts"][key] -
             by[seed, schedule, False]["final"]["new_layouts"][key] for key in metrics]
            for seed in seeds
        ]) * 100
        mean, lo, hi = mean_ci(diffs)
        ax.errorbar(x + offset, mean, yerr=np.vstack((mean - lo, hi - mean)), fmt="o", capsize=3,
                    color=colors[schedule], label=schedule)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("live − silent difference (pp)")
    ax.set_title("Heldout endpoint components")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "endpoint_component_gains.png", dpi=180)
    fig.savefig(output / "endpoint_component_gains.pdf")
    plt.close(fig)

    receipt = dict(status="generated", visual_review="pending", source=str(results.get("source", "execution/results.json")),
                   figures=[path.name for path in sorted(output.glob("*.png"))])
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(load(args.results), args.output)
