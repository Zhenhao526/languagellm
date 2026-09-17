"""Plot exploratory FI locality effects from the JSON-only probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


MODES = ("cross_closed", "symbol_permutation", "position_rotation", "position_reverse",
         "mask_sender_A", "mask_sender_B", "mask_sender_C")
LABELS = ("closed", "symbol\nperm.", "position\nrotate", "position\nreverse", "mask A", "mask B", "mask C")
METRICS = (("q_rate", "Δ team Q (pp)"), ("physical_execution_rate", "Δ physical execution (pp)"),
           ("proposal_legal_rate", "Δ legal proposal (pp)"))


def plot(source, output):
    source = Path(source); output = Path(output); output.mkdir(parents=False, exist_ok=False)
    data = json.loads((source / "execution/results.json").read_text(encoding="utf8"))
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.0), sharex="col")
    x = np.arange(len(MODES)); width = 0.36
    for row, (metric, ylabel) in enumerate(METRICS):
        for col, schedule in enumerate(("static", "rematched")):
            ax = axes[row, col]
            for offset, channel, color in ((-width / 2, "live", "#2f6f9f"), (width / 2, "own", "#c97b29")):
                means = []; lows = []; highs = []
                for mode in MODES:
                    s = data["summaries"][f"{schedule}_{channel}_{mode}"][metric]
                    # Report loss as natural minus intervention; positive means a drop.
                    means.append(-100 * s["mean"]); lows.append(-100 * s["ci95_upper"]); highs.append(-100 * s["ci95_lower"])
                means = np.asarray(means); yerr = np.vstack((means - lows, highs - means))
                ax.bar(x + offset, means, width, yerr=yerr, capsize=3, label="FI-live" if channel == "live" else "FI-silent", color=color, alpha=.9)
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{schedule} schedule")
            ax.grid(axis="y", alpha=.25)
            if row == 0 and col == 0:
                ax.legend(frameon=False, loc="upper right")
            if row == len(METRICS) - 1:
                ax.set_xticks(x, LABELS)
    fig.suptitle("FI policy sensitivity to local message interventions\n(post-hoc exploratory; bars are natural − intervention, percentage points)", y=.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .965))
    png = output / "locality_effects.png"; pdf = output / "locality_effects.pdf"
    fig.savefig(png, dpi=180); fig.savefig(pdf); plt.close(fig)
    receipt = {"status": "generated", "source": str(source), "png": str(png), "pdf": str(pdf), "visual_review": "pending"}
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); plot(args.source, args.output)

