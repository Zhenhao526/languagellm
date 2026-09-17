"""Plot the v0.42 endpoint intervention results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "sequence_intervention_001"
ASSIGNMENTS = ("canonical", "cyclic")
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
BLOCKS = ("012", "021", "102", "120", "201", "210")


def read(path):
    return json.loads(path.read_text())


def pct(x):
    return 100.0 * float(x)


def rows_for(rows, assignment=None, condition=None):
    return [
        row
        for row in rows
        if (assignment is None or row["assignment"] == assignment)
        and (condition is None or row["condition"] == condition)
    ]


def direct(rows, key, split="target60", metric="J"):
    return np.asarray([row[key][split][metric] for row in rows], dtype=float)


def nested(rows, key, value, split="target60", metric="J"):
    return np.asarray([row[key][str(value)][split][metric] for row in rows], dtype=float)


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    data = read(out / "sequence_intervention_analysis.json")
    rows = data["rows"]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)
    colors = {"fixed_A": "#4C78A8", "rotating_AB": "#F58518", "random_ABC": "#54A24B"}
    labels = {"fixed_A": "fixed A", "rotating_AB": "rotating A/B", "random_ABC": "random A/B/C"}

    # A: endpoint score and basic sequence interventions.
    ax = axes[0, 0]
    x = np.arange(len(CONDITIONS))
    width = 0.18
    variants = [("baseline", "baseline", "#333333"), ("swap", "swap t0/t1", "#E45756"), ("shuffle_t1", "shuffle t1", "#B279A2")]
    for i, (key, label, color) in enumerate(variants):
        vals = []
        errs = []
        for condition in CONDITIONS:
            m, s = mean_sd(direct(rows_for(rows, condition=condition), key))
            vals.append(pct(m)); errs.append(pct(s))
        ax.bar(x + (i - 1) * width, vals, width, yerr=errs, capsize=2, label=label, color=color, alpha=0.9)
    ax.set_xticks(x, [labels[c] for c in CONDITIONS], rotation=18, ha="right")
    ax.set_ylabel("target-60 joint J (%)")
    ax.set_title("Endpoint function and token interventions")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # B: all mask values, preserving the no-reserved-symbol caveat.
    ax = axes[0, 1]
    mask_x = np.arange(7)
    for condition in CONDITIONS:
        subset = rows_for(rows, condition=condition)
        t0 = [pct(mean_sd(nested(subset, "mask_t0", value))[0]) for value in mask_x]
        t1 = [pct(mean_sd(nested(subset, "mask_t1", value))[0]) for value in mask_x]
        ax.plot(mask_x, t0, marker="o", color=colors[condition], linestyle="-", label=labels[condition] + " / mask t0")
        ax.plot(mask_x, t1, marker="s", color=colors[condition], linestyle="--", alpha=0.75, label=labels[condition] + " / mask t1")
    ax.set_xticks(mask_x)
    ax.set_xlabel("replacement vocabulary value")
    ax.set_ylabel("target-60 joint J (%)")
    ax.set_title("Mask sensitivity across all seven symbols")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # C: target score for every resource-block permutation.
    ax = axes[0, 2]
    x = np.arange(len(BLOCKS))
    for condition in CONDITIONS:
        subset = rows_for(rows, condition=condition)
        literal = []
        equiv = []
        for block in BLOCKS:
            literal.append(pct(mean_sd([row["block_permutations"][block]["literal"]["target60"]["J"] for row in subset])[0]))
            equiv.append(pct(mean_sd([row["block_permutations"][block]["equivariant"]["target60"]["J"] for row in subset])[0]))
        ax.plot(x, literal, marker="o", color=colors[condition], linestyle="-", label=labels[condition] + " / literal")
        ax.plot(x, equiv, marker="s", color=colors[condition], linestyle="--", alpha=0.75, label=labels[condition] + " / permuted target")
    ax.set_xticks(x, BLOCKS)
    ax.set_xlabel("resource block permutation")
    ax.set_ylabel("target-60 joint J (%)")
    ax.set_title("Resource-block permutation: literal vs equivariant")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=7, ncol=2)

    # D: token0/token1 mask averages and swap drop.
    ax = axes[1, 0]
    x = np.arange(len(CONDITIONS))
    bars = []
    for i, (key, label, color) in enumerate((("baseline", "baseline", "#333333"), ("mask_t0", "mask t0 (mean)", "#E45756"), ("mask_t1", "mask t1 (mean)", "#72B7B2"), ("swap", "swap", "#B279A2"))):
        vals = []
        for condition in CONDITIONS:
            subset = rows_for(rows, condition=condition)
            if key.startswith("mask"):
                vals.append(pct(np.mean([nested(subset, key, v).mean() for v in range(7)])))
            else:
                vals.append(pct(mean_sd(direct(subset, key))[0]))
        bars.append((i, vals, label, color))
    for i, vals, label, color in bars:
        ax.bar(x + (i - 1.5) * width, vals, width, label=label, color=color, alpha=0.9)
    ax.set_xticks(x, [labels[c] for c in CONDITIONS], rotation=18, ha="right")
    ax.set_ylabel("target-60 joint J (%)")
    ax.set_title("Which token position carries function?")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # E: cyclic site relocation. Shift 0 is identity.
    ax = axes[1, 1]
    x = np.arange(6)
    for condition in CONDITIONS:
        subset = rows_for(rows, condition=condition)
        vals = [pct(mean_sd([row["site_relocations"][str(shift)]["target60"]["J"] for row in subset])[0]) for shift in range(6)]
        ax.plot(x, vals, marker="o", color=colors[condition], label=labels[condition])
    ax.set_xticks(x, ["identity", "+1", "+2", "+3", "+4", "+5"])
    ax.set_xlabel("site-ID cyclic relocation")
    ax.set_ylabel("target-60 joint J (%)")
    ax.set_title("Counterfactual site relocation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # F: canonical/cyclic assignment comparison for the unmodified endpoint.
    ax = axes[1, 2]
    x = np.arange(len(CONDITIONS))
    for i, assignment in enumerate(ASSIGNMENTS):
        vals = [pct(mean_sd(direct(rows_for(rows, assignment=assignment, condition=condition), "baseline"))[0]) for condition in CONDITIONS]
        ax.bar(x + (i - 0.5) * width * 1.25, vals, width * 1.25, label=assignment, color=("#79706E" if assignment == "canonical" else "#D5A657"))
    ax.set_xticks(x, [labels[c] for c in CONDITIONS], rotation=18, ha="right")
    ax.set_ylabel("target-60 joint J (%)")
    ax.set_title("Resource assignment at the endpoint")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("v0.42 endpoint counterfactual assay", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figure = out / "figures" / "01_sequence_intervention.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "complete",
        "figure": str(figure.resolve()),
        "analysis": str((out / "sequence_intervention_analysis.json").resolve()),
        "panels": [
            "endpoint function and token interventions",
            "seven-value token mask sensitivity",
            "literal versus equivariant resource-block permutation",
            "token-position function comparison",
            "cyclic site relocation",
            "canonical versus cyclic resource assignment",
        ],
        "matplotlib": plt.matplotlib.__version__,
    }
    (out / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
