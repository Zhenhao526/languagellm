"""Plot the read-only summaries for the factorial-payoff confirmation.

The script consumes only summary JSON and the post-hoc cross-split probe JSON;
it does not open model checkpoints or regenerate evaluations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


UPDATES = np.array([0, 100, 500, 1500, 3000, 6000], dtype=float)
UPDATE_POSITIONS = np.arange(len(UPDATES), dtype=float)


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _pp(x: float) -> float:
    return 100.0 * float(x)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial Unicode MS", "DejaVu Sans"],
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 220,
        }
    )


def plot_trajectory(summary: dict, out: Path) -> None:
    rows = [
        r
        for r in summary["trajectory_summaries"]
        if r["partition"] == "heldout_resource"
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    colors = {"partial": "#4472C4", "all_or_nothing": "#C55A11"}
    labels = {"partial": "partial", "all_or_nothing": "all-or-nothing"}
    for ax, rule in zip(axes, ("strict", "reciprocal")):
        for payoff in ("partial", "all_or_nothing"):
            for live, ls, marker in ((False, "--", "o"), (True, "-", "s")):
                subset = [
                    r
                    for r in rows
                    if r["rule"] == rule and r["payoff"] == payoff and r["live"] == live
                ]
                subset.sort(key=lambda r: r["update"])
                y = [_pp(r["correct_executed_pair_rate"]) for r in subset]
                ax.plot(
                    UPDATE_POSITIONS,
                    y,
                    color=colors[payoff],
                    linestyle=ls,
                    marker=marker,
                    linewidth=1.7,
                    markersize=3.5,
                    label=f"{labels[payoff]} / {'live' if live else 'silent'}",
                )
        ax.set_title(rule)
        ax.set_xlabel("training updates")
        ax.set_xticks(UPDATE_POSITIONS, [str(int(u)) for u in UPDATES])
        ax.set_xlim(-0.2, len(UPDATES) - 0.8)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("correct executed pair (%)")
    axes[0].legend(loc="upper left", frameon=False, ncol=2)
    fig.suptitle("Heldout-resource partner coordination during training", y=1.02)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_interactions(summary: dict, probe: dict, out: Path) -> None:
    # Within-heldout Q is the pre-specified confirmation target; partner is a
    # mechanism secondary. The cross-split probe was added after seeing the
    # within-heldout null and is labelled exploratory in the figure.
    within_q = summary["primary"]["statistics"]
    within_partner = summary["primary"]["partner_secondary"]["statistics"]
    cross_q = probe["primary"]["statistics"]
    cross_partner = probe["partner_secondary"]["statistics"]
    values = np.array(
        [
            _pp(within_q["mean"]),
            _pp(within_partner["mean"]),
            _pp(cross_q["mean"]),
            _pp(cross_partner["mean"]),
        ]
    )
    lows = np.array(
        [
            _pp(within_q["ci95_lower"]),
            _pp(within_partner["ci95_lower"]),
            _pp(cross_q["ci95_lower"]),
            _pp(cross_partner["ci95_lower"]),
        ]
    )
    highs = np.array(
        [
            _pp(within_q["ci95_upper"]),
            _pp(within_partner["ci95_upper"]),
            _pp(cross_q["ci95_upper"]),
            _pp(cross_partner["ci95_upper"]),
        ]
    )
    errors = np.vstack((values - lows, highs - values))
    fig, axes = plt.subplots(1, 2, figsize=(9.3, 4.0), sharey=True)
    panels = [
        (axes[0], (0, 1), "Pre-specified within-heldout target", "Q / partner"),
        (axes[1], (2, 3), "Exploratory cross-split probe", "Q / partner"),
    ]
    for ax, idxs, title, _ in panels:
        idx = np.array(idxs)
        ax.bar(
            np.arange(2),
            values[idx],
            yerr=errors[:, idx],
            capsize=4,
            color=["#4472C4", "#70AD47"],
            edgecolor="#333333",
            linewidth=0.6,
        )
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks([0, 1], ["Q", "partner"])
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        for j, val in enumerate(values[idx]):
            ax.text(j, val + (1.0 if val >= 0 else -2.2), f"{val:.1f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=8)
    axes[0].set_ylabel("interaction (percentage points)\nmean ± approximate t15 95% CI")
    axes[1].text(
        0.5,
        -0.22,
        "cross-split = seen training need → heldout object×attribute need",
        transform=axes[1].transAxes,
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.suptitle("Communication interaction by evaluation boundary", y=1.02)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_seed_forest(probe: dict, out: Path) -> None:
    rows = probe["primary"]["by_seed"]
    rows = sorted(rows, key=lambda r: r["seed"])
    y = np.arange(len(rows))
    x = np.array([_pp(r["rule_mean_interaction"]) for r in rows])
    strict = np.array([_pp(r["by_rule"]["strict"]["interaction"]) for r in rows])
    reciprocal = np.array([_pp(r["by_rule"]["reciprocal"]["interaction"]) for r in rows])
    mean = _pp(probe["primary"]["statistics"]["mean"])
    lo = _pp(probe["primary"]["statistics"]["ci95_lower"])
    hi = _pp(probe["primary"]["statistics"]["ci95_upper"])
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.axvspan(lo, hi, color="#D9EAD3", alpha=0.75, label="mean CI")
    ax.axvline(mean, color="#38761D", linewidth=1.4, label=f"mean {mean:.1f} pp")
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.scatter(strict, y, color="#4472C4", marker="o", s=28, label="strict")
    ax.scatter(reciprocal, y, color="#C55A11", marker="s", s=28, label="reciprocal")
    ax.scatter(x, y, color="#222222", marker="|", s=120, linewidths=1.5, label="seed mean")
    ax.set_yticks(y, [str(r["seed"]) for r in rows])
    ax.set_xlabel("cross-split Q interaction (percentage points)")
    ax.set_ylabel("independent seed")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title("Cross-split Q interaction is positive in all 16 seed blocks")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    _style()
    summary = _load(args.source / "summary_001" / "summary.json")
    probe = _load(args.probe / "probe.json")
    plot_trajectory(summary, args.output / "trajectory_partner_rate")
    plot_interactions(summary, probe, args.output / "interaction_comparison")
    plot_seed_forest(probe, args.output / "cross_split_seed_forest")
    receipt = {
        "status": "generated",
        "source": str(args.source),
        "probe": str(args.probe),
        "outputs": [
            "trajectory_partner_rate.png",
            "trajectory_partner_rate.pdf",
            "interaction_comparison.png",
            "interaction_comparison.pdf",
            "cross_split_seed_forest.png",
            "cross_split_seed_forest.pdf",
        ],
        "model_calls": 0,
        "optimizer_updates": 0,
    }
    with (args.output / "receipt.json").open("w", encoding="utf-8") as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
