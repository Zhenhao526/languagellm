"""Plot the v0.41 two-token results from the independent analysis JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
ASSIGNMENTS = ("canonical", "cyclic")
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
ASSIGNMENT_LABELS = {"canonical": "canonical resource order", "cyclic": "cyclic resource order"}
CONDITION_LABELS = {"fixed_A": "fixed A", "rotating_AB": "rotating A/B", "random_ABC": "random A/B/C"}
COLORS = {"fixed_A": "#2563eb", "rotating_AB": "#d97706", "random_ABC": "#059669"}


def read(path: Path):
    return json.loads(path.read_text())


def run(out: Path):
    analysis = read(out / "two_token_analysis.json")
    effects = analysis["condition_effects"]
    summary = analysis["summary"]
    generations = np.asarray([0, 100, 600, 1200])
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    fig.suptitle("Formation of a sequential two-token protocol", fontsize=17, fontweight="bold")

    for ax, assignment in zip(axes[0, :2], ASSIGNMENTS):
        for condition in CONDITIONS:
            item = effects[assignment][condition]
            mean = np.asarray([item[str(update)]["mean"] for update in generations]) * 100
            sd = np.asarray([item[str(update)]["sd"] for update in generations]) * 100
            ax.plot(generations, mean, marker="o", lw=2.2, color=COLORS[condition], label=CONDITION_LABELS[condition])
            ax.fill_between(generations, mean - sd, mean + sd, color=COLORS[condition], alpha=0.12, linewidth=0)
        ax.set_title(ASSIGNMENT_LABELS[assignment])
        ax.set_xlabel("Training update")
        ax.set_ylabel("Target-60 joint success (%)")
        ax.set_ylim(0, 40)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="best")

    # Difference induced by moving resource identities across architecture slots.
    ax = axes[0, 2]
    x = np.arange(len(CONDITIONS))
    width = 0.23
    for index, schedule in enumerate(SCHEDULES):
        values = [analysis["assignment_effects"][condition][schedule]["cyclic_minus_canonical_target_J"] * 100 for condition in CONDITIONS]
        ax.bar(x + (index - 1) * width, values, width, label=f"evaluation {schedule}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title("Cyclic minus canonical at update 1200")
    ax.set_xticks(x, [CONDITION_LABELS[c] for c in CONDITIONS], rotation=20, ha="right")
    ax.set_ylabel("Target-60 joint success difference (pp)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # Pair agreement versus the stronger requirement that all three resource
    # pairs agree simultaneously.
    ax = axes[1, 0]
    labels = []
    pair = []
    joint = []
    for assignment in ASSIGNMENTS:
        for condition in CONDITIONS:
            labels.append(f"{assignment}\n{condition.replace('_', '/')}")
            pair.append(np.mean([summary[assignment][condition]["1200"][s]["pair_agreement"]["mean"] for s in SCHEDULES]) * 100)
            joint.append(np.mean([summary[assignment][condition]["1200"][s]["sequence_joint_agreement"]["mean"] for s in SCHEDULES]) * 100)
    x = np.arange(len(labels))
    ax.bar(x - 0.18, pair, 0.36, label="per-resource pair agreement")
    ax.bar(x + 0.18, joint, 0.36, label="three-resource sequence agreement")
    ax.set_title("Generation-4 agreement")
    ax.set_xticks(x, labels, rotation=34, ha="right", fontsize=7)
    ax.set_ylabel("Agreement (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    # Information carried by each token position about the sender's target
    # location. The pair value is a reference for the full sequence.
    ax = axes[1, 1]
    token0, token1, pair_info, prefix_tv = [], [], [], []
    labels = []
    for assignment in ASSIGNMENTS:
        for condition in CONDITIONS:
            labels.append(f"{assignment}\n{condition.replace('_', '/')}")
            token0.append(np.mean([summary[assignment][condition]["1200"][s]["nmi_token0_position"]["mean"] for s in SCHEDULES]) * 100)
            token1.append(np.mean([summary[assignment][condition]["1200"][s]["nmi_token1_position"]["mean"] for s in SCHEDULES]) * 100)
            pair_info.append(np.mean([summary[assignment][condition]["1200"][s]["nmi_pair_position"]["mean"] for s in SCHEDULES]) * 100)
            prefix_tv.append(np.mean([summary[assignment][condition]["1200"][s]["prefix_sensitivity_tv"]["mean"] for s in SCHEDULES]) * 100)
    x = np.arange(len(labels))
    for index, values in enumerate((token0, token1, pair_info)):
        ax.bar(x + (index - 1) * 0.23, values, 0.23, label=("token 0", "token 1", "token pair")[index])
    ax.set_title("Position information in the sequence")
    ax.set_xticks(x, labels, rotation=34, ha="right", fontsize=7)
    ax.set_ylabel("Normalized mutual information (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax2 = ax.twinx()
    prefix_line, = ax2.plot(x, prefix_tv, color="#7c3aed", marker="D", lw=1.8, label="prefix sensitivity")
    ax2.set_ylim(0, 20)
    ax2.set_ylabel("Prefix sensitivity (TV, %)", color="#7c3aed")
    ax2.tick_params(axis="y", colors="#7c3aed")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [prefix_line], labels + ["prefix sensitivity"], frameon=False, fontsize=8, loc="lower right")

    # Resource-wise target accuracy for the final protocol.
    ax = axes[1, 2]
    resource_values = {resource: [] for resource in ("resource0", "resource1", "resource2")}
    labels = []
    for assignment in ASSIGNMENTS:
        for condition in CONDITIONS:
            labels.append(f"{assignment}\n{condition.replace('_', '/')}")
            for resource in resource_values:
                resource_values[resource].append(np.mean([summary[assignment][condition]["1200"][s][resource]["mean"] for s in SCHEDULES]) * 100)
    x = np.arange(len(labels))
    for index, resource in enumerate(resource_values):
        ax.bar(x + (index - 1) * 0.24, resource_values[resource], 0.24, label=f"resource {index}")
    ax.set_title("Generation-4 accuracy by resource")
    ax.set_xticks(x, labels, rotation=34, ha="right", fontsize=7)
    ax.set_ylabel("Target-60 accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    figure = out / "figures" / "01_two_token_outcomes.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=220)
    plt.close(fig)
    metadata = {
        "status": "complete",
        "figure": str(figure.resolve()),
        "analysis": str((out / "two_token_analysis.json").resolve()),
        "panels": [
            "target success curves by resource assignment",
            "cyclic minus canonical assignment effect",
            "per-resource versus sequence agreement",
            "information carried by token positions",
            "resource-wise target accuracy",
        ],
        "matplotlib": matplotlib.__version__,
    }
    (out / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())


if __name__ == "__main__":
    main()
