"""Figures for the joint-combination holdout experiment."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import design, summarize


ARMS = ("seen_joint_only", "all_needs_control")
ARM_LABELS = {
    "seen_joint_only": "seen-joint-only adaptation",
    "all_needs_control": "all-needs control",
}
ARM_COLORS = {"seen_joint_only": "#386cb0", "all_needs_control": "#d95f02"}
SCHEDULE_COLORS = {"static": "#386cb0", "rematched": "#d95f02"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    data = json.loads(Path(path).read_text())
    require(len(data.get("runs", [])) == 32 and len(data.get("all_needs_control_runs", [])) == 32,
            "Expected 32 runs in each arm")
    return data


def mean_ci(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1 or x.ndim == 2, "Invalid values")
    if x.ndim == 1:
        x = x[:, None]
    mean = x.mean(axis=0)
    half = summarize.T7_975 * x.std(axis=0, ddof=1) / math.sqrt(x.shape[0])
    return mean, mean - half, mean + half


def trajectory(run, partition="heldout", key="q_rate"):
    rows = sorted(run["trajectory"], key=lambda row: int(row["update"]))
    return np.asarray([row["target_trajectory"][partition][key] for row in rows], dtype=np.float64)


def final(run, partition="heldout", key="q_rate"):
    return float(run["final"][partition][key])


def run_maps(data):
    return {
        "seen_joint_only": {(r["seed"], r["schedule"], bool(r["live"])): r for r in data["runs"]},
        "all_needs_control": {(r["seed"], r["schedule"], bool(r["live"])): r for r in data["all_needs_control_runs"]},
    }


def plot(results, output):
    output = Path(output)
    output.mkdir(parents=False, exist_ok=False)
    by = run_maps(results)
    seeds = list(design.SEEDS)
    updates = np.asarray(design.CHECKPOINTS, dtype=np.float64)

    # Routing gain trajectories on the genuinely heldout joint combinations.
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=True)
    for ax, schedule in zip(axes, design.SCHEDULES):
        ax.axhline(0, color="0.35", lw=0.8)
        for arm in ARMS:
            gains = [trajectory(by[arm][seed, schedule, True]) - trajectory(by[arm][seed, schedule, False]) for seed in seeds]
            mean, lo, hi = mean_ci(np.asarray(gains) * 100.0)
            ax.plot(updates, mean, marker="o", color=ARM_COLORS[arm], label=ARM_LABELS[arm])
            ax.fill_between(updates, lo, hi, color=ARM_COLORS[arm], alpha=0.14)
        ax.set_title(schedule)
        ax.set_xlabel("adaptation updates")
        ax.set_xticks(updates)
    axes[0].set_ylabel("heldout live − silent Q (percentage points)")
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("Routed-packet advantage on unseen joint combinations")
    fig.tight_layout()
    fig.savefig(output / "heldout_q_gain_trajectories.png", dpi=180)
    fig.savefig(output / "heldout_q_gain_trajectories.pdf")
    plt.close(fig)

    # Endpoint routing gain by adaptation arm and schedule.
    fig, ax = plt.subplots(figsize=(8.7, 4.7))
    x = np.arange(len(design.SCHEDULES), dtype=float)
    width = 0.34
    for j, arm in enumerate(ARMS):
        means, low, high = [], [], []
        for schedule in design.SCHEDULES:
            values = [final(by[arm][seed, schedule, True]) - final(by[arm][seed, schedule, False]) for seed in seeds]
            m, l, h = mean_ci(np.asarray(values) * 100.0)
            means.append(float(m[0])); low.append(float(m[0] - l[0])); high.append(float(h[0] - m[0]))
        ax.errorbar(x + (j - 0.5) * width, means, yerr=np.vstack((low, high)), fmt="o", capsize=3,
                    color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, ["static", "rematched"])
    ax.set_ylabel("final heldout live − silent Q (percentage points)")
    ax.set_title("Endpoint routing gain by adaptation regime")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "heldout_q_gain_endpoint.png", dpi=180)
    fig.savefig(output / "heldout_q_gain_endpoint.pdf")
    plt.close(fig)

    # Centered trajectory AUC, which discounts the shared initial difference.
    fig, ax = plt.subplots(figsize=(8.7, 4.7))
    for j, arm in enumerate(ARMS):
        means, low, high = [], [], []
        for schedule in design.SCHEDULES:
            values = [summarize.centered_auc(trajectory(by[arm][seed, schedule, True]) - trajectory(by[arm][seed, schedule, False])) for seed in seeds]
            m, l, h = mean_ci(np.asarray(values) * 100.0)
            means.append(float(m[0])); low.append(float(m[0] - l[0])); high.append(float(h[0] - m[0]))
        ax.errorbar(x + (j - 0.5) * width, means, yerr=np.vstack((low, high)), fmt="o", capsize=3,
                    color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, ["static", "rematched"])
    ax.set_ylabel("centered heldout live − silent Q AUC (percentage points)")
    ax.set_title("Cumulative routing gain after the shared start")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "heldout_q_gain_auc.png", dpi=180)
    fig.savefig(output / "heldout_q_gain_auc.pdf")
    plt.close(fig)

    # Compositional transfer gap on the same heldout layouts.
    fig, ax = plt.subplots(figsize=(8.7, 4.7))
    for j, arm in enumerate(ARMS):
        means, low, high = [], [], []
        for schedule in design.SCHEDULES:
            values = [final(by[arm][seed, schedule, True], "heldout") - final(by[arm][seed, schedule, True], "seen_new") for seed in seeds]
            m, l, h = mean_ci(np.asarray(values) * 100.0)
            means.append(float(m[0])); low.append(float(m[0] - l[0])); high.append(float(h[0] - m[0]))
        ax.errorbar(x + (j - 0.5) * width, means, yerr=np.vstack((low, high)), fmt="o", capsize=3,
                    color=ARM_COLORS[arm], label=ARM_LABELS[arm])
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, ["static", "rematched"])
    ax.set_ylabel("heldout Q − seen-new-layout Q (percentage points)")
    ax.set_title("Joint-combination transfer gap for live agents")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "compositional_transfer_gap.png", dpi=180)
    fig.savefig(output / "compositional_transfer_gap.pdf")
    plt.close(fig)

    # Absolute heldout quality, preserving the live/silent comparison.
    fig, ax = plt.subplots(figsize=(10.0, 4.8))
    labels, centers = [], []
    positions = []
    pos = 0.0
    for arm in ARMS:
        for schedule in design.SCHEDULES:
            for live in (True, False):
                values = [final(by[arm][seed, schedule, live]) * 100.0 for seed in seeds]
                m, l, h = mean_ci(np.asarray(values))
                color = ARM_COLORS[arm] if live else "#bdbdbd"
                ax.errorbar(pos, float(m[0]), yerr=np.asarray([[float(m[0] - l[0])], [float(h[0] - m[0])]]),
                            fmt="o", capsize=3, color=color)
                labels.append(f"{schedule}\n{'live' if live else 'silent'}")
                positions.append(pos)
                pos += 1.0
            pos += 0.35
    ax.set_xticks(positions, labels, rotation=35, ha="right")
    ax.set_ylabel("final heldout Q (%)")
    ax.set_title("Absolute heldout task quality")
    ax.text(0.01, 0.98, "blue/orange = live arm; gray = silent", transform=ax.transAxes, va="top", fontsize=9, color="0.3")
    fig.tight_layout()
    fig.savefig(output / "absolute_heldout_q.png", dpi=180)
    fig.savefig(output / "absolute_heldout_q.pdf")
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
