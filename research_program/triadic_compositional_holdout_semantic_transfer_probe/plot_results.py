"""Plot the heldout-combination semantic-transfer probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pct(x):
    return 100.0 * float(x)


def plot(summary_path: str, output: str):
    summary_path = Path(summary_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refuse to overwrite figures: {output}")
    summary = json.loads(summary_path.read_text(encoding="utf8"))
    output.mkdir(parents=True)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8), constrained_layout=True)
    colors = {"static": "#2f6db0", "rematched": "#d66b2d"}
    labels, values, lower, upper, bar_colors = [], [], [], [], []
    for schedule in ("static", "rematched"):
        cell = summary["cells"][f"{schedule}/live"]
        for actor, label, color in (("0", "A", colors[schedule]), ("parent_BC", "B/C", "#b9c3cf")):
            entry = cell[actor]["aligned_minus_placebo"]["plan_transfer"]
            labels.append(f"{schedule}\n{label}")
            values.append(pct(entry["mean"]))
            lower.append(pct(entry["mean"] - entry["ci95_t7"][0]))
            upper.append(pct(entry["ci95_t7"][1] - entry["mean"]))
            bar_colors.append(color)
    x = np.arange(len(values))
    axes[0].bar(x, values, color=bar_colors, width=0.72, edgecolor="white")
    axes[0].errorbar(x, values, yerr=np.array([lower, upper]), fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    axes[0].axhline(0, color="#222222", linewidth=0.8)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("aligned − placebo plan transfer (pp)")
    axes[0].set_title("Heldout joint-combination content transfer")
    axes[0].grid(axis="y", alpha=0.25)

    fields = (("plan_transfer", "plan"), ("physical", "physical"), ("q", "Q"),
              ("conditional_q", "Q|physical"), ("action_change", "action change"))
    inter_values, inter_low, inter_high = [], [], []
    for field, _ in fields:
        entry = summary["interaction"][field]
        inter_values.append(pct(entry["mean"]))
        inter_low.append(pct(entry["mean"] - entry["ci95_t7"][0]))
        inter_high.append(pct(entry["ci95_t7"][1] - entry["mean"]))
    x2 = np.arange(len(fields))
    axes[1].bar(x2, inter_values, color="#6d9e72", width=0.62, edgecolor="white")
    axes[1].errorbar(x2, inter_values, yerr=np.array([inter_low, inter_high]), fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    axes[1].axhline(0, color="#222222", linewidth=0.8)
    axes[1].set_xticks(x2, [label for _, label in fields], rotation=25, ha="right")
    axes[1].set_ylabel("rematched × schedule interaction (pp)")
    axes[1].set_title("A−B/C interaction")
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Compositional holdout semantic-transfer probe", fontsize=13)
    fig.savefig(output / "semantic_transfer.png", dpi=220)
    fig.savefig(output / "semantic_transfer.pdf")
    plt.close(fig)
    receipt = {
        "status": "rendered",
        "visual_review": "pending",
        "summary_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
        "files": ["semantic_transfer.png", "semantic_transfer.pdf"],
    }
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.summary, args.output), ensure_ascii=False))
