"""Plot the v0.40 triad transfer outcomes from the independent analysis JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
FORMATIONS = ("origin_fixed_A", "origin_rotating_AB", "origin_random_ABC")
TRANSMISSIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
FORMATION_LABELS = {
    "origin_fixed_A": "Formation: fixed A",
    "origin_rotating_AB": "Formation: rotating A/B",
    "origin_random_ABC": "Formation: random A/B/C",
}
TRANSMISSION_LABELS = {"fixed_A": "fixed A", "rotating_AB": "rotating A/B", "random_ABC": "random A/B/C"}
COLORS = {"fixed_A": "#2563eb", "rotating_AB": "#d97706", "random_ABC": "#059669"}


def read(path: Path):
    return json.loads(path.read_text())


def run(out: Path):
    analysis = read(out / "triad_transfer_analysis.json")
    summary = analysis["summary"]
    effects = analysis["condition_effects"]
    generations = np.arange(5)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.8), constrained_layout=True)
    fig.suptitle("Generational transfer of a three-resource, three-token protocol", fontsize=17, fontweight="bold")

    # Top row: pooled target-60 joint success through four replacements.
    for ax, formation in zip(axes[0], FORMATIONS):
        for transmission in TRANSMISSIONS:
            item = effects[formation][transmission]
            mean = np.asarray([item[str(g)]["mean"] for g in generations]) * 100
            sd = np.asarray([item[str(g)]["sd"] for g in generations]) * 100
            ax.plot(generations, mean, marker="o", lw=2.2, color=COLORS[transmission], label=TRANSMISSION_LABELS[transmission])
            ax.fill_between(generations, mean - sd, mean + sd, color=COLORS[transmission], alpha=0.12, linewidth=0)
        ax.set_title(FORMATION_LABELS[formation])
        ax.set_xticks(generations)
        ax.set_xlabel("Replacement generation")
        ax.set_ylabel("Target-60 joint success (%)")
        ax.set_ylim(0, 35)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(frameon=False, fontsize=8, loc="best")

    # Bottom-left: final target-J factorial surface.
    ax = axes[1, 0]
    matrix = np.asarray([[effects[f][t]["4"]["mean"] * 100 for t in TRANSMISSIONS] for f in FORMATIONS])
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=max(35, float(matrix.max())))
    ax.set_title("Generation-4 target-60 joint success")
    ax.set_xticks(range(3), [TRANSMISSION_LABELS[t] for t in TRANSMISSIONS], rotation=22, ha="right")
    ax.set_yticks(range(3), [FORMATION_LABELS[f].replace("Formation: ", "") for f in FORMATIONS])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color="white" if matrix[i, j] < 20 else "black", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

    # Bottom-middle: final agreement surface, averaged over evaluation teams.
    ax = axes[1, 1]
    agreement = np.asarray(
        [
            [
                np.mean([summary[f][t]["4"][s]["agreement_joint"]["mean"] for s in SCHEDULES]) * 100
                for t in TRANSMISSIONS
            ]
            for f in FORMATIONS
        ]
    )
    im = ax.imshow(agreement, cmap="magma", vmin=0, vmax=100)
    ax.set_title("Generation-4 joint three-token agreement")
    ax.set_xticks(range(3), [TRANSMISSION_LABELS[t] for t in TRANSMISSIONS], rotation=22, ha="right")
    ax.set_yticks(range(3), [FORMATION_LABELS[f].replace("Formation: ", "") for f in FORMATIONS])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{agreement[i, j]:.1f}", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="%")

    # Bottom-right: resource-wise target accuracy for every factorial cell.
    ax = axes[1, 2]
    labels = []
    resource_values = {resource: [] for resource in ("resource0", "resource1", "resource2")}
    for formation in FORMATIONS:
        for transmission in TRANSMISSIONS:
            labels.append(f"{formation.removeprefix('origin_')}\n{transmission.replace('_', '/')}")
            for resource in resource_values:
                resource_values[resource].append(
                    np.mean([summary[formation][transmission]["4"][s][resource]["mean"] for s in SCHEDULES]) * 100
                )
    x = np.arange(len(labels))
    width = 0.24
    bars = []
    for k, resource in enumerate(resource_values):
        bar = ax.bar(x + (k - 1) * width, resource_values[resource], width, label=f"Resource {k}")
        bars.append(bar)
    ax.set_title("Generation-4 target-60 accuracy by resource")
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8)

    out_fig = out / "figures" / "01_triad_transfer_outcomes.png"
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=220)
    plt.close(fig)
    metadata = {
        "status": "complete",
        "figure": str(out_fig.resolve()),
        "analysis": str((out / "triad_transfer_analysis.json").resolve()),
        "panels": [
            "pooled target-60 joint success curves by formation condition and transmission condition",
            "generation-4 target-60 factorial surface",
            "generation-4 joint three-token agreement surface",
            "generation-4 resource-wise target-60 accuracy",
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
