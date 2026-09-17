"""Plot the slot/recode role cells and C-minus-A contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pct(x):
    return 100.0 * float(x)


def main(summary_path: str, out_dir: str) -> None:
    summary = json.loads(Path(summary_path).read_text(encoding="utf8"))
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    variants = summary["contract"]["variants"]
    x = np.arange(len(variants))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)

    ax = axes[0]
    for role, color in (("A", "#2563eb"), ("C", "#dc2626")):
        for schedule, marker in (("static", "o"), ("rematched", "s")):
            values = []
            lows = []
            highs = []
            for variant in variants:
                entry = summary["variant_cells"][variant][role][f"{schedule}/live"]["aligned_minus_placebo"]["plan_transfer"]
                values.append(pct(entry["mean"]))
                lows.append(pct(entry["mean"] - entry["ci95_t7"][0]))
                highs.append(pct(entry["ci95_t7"][1] - entry["mean"]))
            offset = -0.12 if schedule == "static" else 0.12
            ax.errorbar(x + offset, values, yerr=[lows, highs], fmt=marker + "-", color=color, capsize=3, label=f"{role} {schedule}")
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x, variants, rotation=35, ha="right")
    ax.set_ylabel("aligned−placebo plan transfer (pp)")
    ax.set_title("Role cells")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    ax = axes[1]
    values = []; lows = []; highs = []
    for variant in variants:
        entry = summary["role_contrasts"][variant]["plan_transfer"]
        values.append(pct(entry["mean"]))
        lows.append(pct(entry["mean"] - entry["ci95_t7"][0]))
        highs.append(pct(entry["ci95_t7"][1] - entry["mean"]))
    ax.errorbar(x, values, yerr=[lows, highs], fmt="o-", color="#111827", capsize=3)
    ax.axhline(0, color="#444", linewidth=0.8)
    ax.set_xticks(x, variants, rotation=35, ha="right")
    ax.set_ylabel("C−A plan transfer (pp)")
    ax.set_title("Role contrast")
    fig.savefig(out / "slot_recode_transfer.png", dpi=180)
    fig.savefig(out / "slot_recode_transfer.pdf")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.summary, args.out)
