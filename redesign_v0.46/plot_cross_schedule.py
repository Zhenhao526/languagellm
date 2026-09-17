"""Plot the v0.46 fully crossed newcomer adaptation assay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CULTURES = (
    "static_role__fixed_A",
    "static_role__rotating_AB",
    "static_role__random_ABC",
    "random_role__fixed_A",
    "random_role__rotating_AB",
    "random_role__random_ABC",
)
LABELS = {
    "static_role__fixed_A": "static / A",
    "static_role__rotating_AB": "static / A-B",
    "static_role__random_ABC": "static / A-B-C",
    "random_role__fixed_A": "random / A",
    "random_role__rotating_AB": "random / A-B",
    "random_role__random_ABC": "random / A-B-C",
}
SCHEDULES = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULE_LABELS = {"fixed_A": "fixed A", "rotating_AB": "rotating A-B", "random_ABC": "random A-B-C"}
COLORS = {"fixed_A": "#4C78A8", "rotating_AB": "#F58518", "random_ABC": "#54A24B"}
UPDATES = (0, 100, 300)


def read(path: Path):
    return json.loads(path.read_text())


def pct(x):
    return 100.0 * float(x)


def stats(summary, culture, adaptation, update, metric):
    item = summary[culture][adaptation][str(update)][metric]
    return pct(item["mean"]), pct(item["sd"])


def endpoint_matrix(summary, metric):
    return np.asarray([[stats(summary, c, a, 300, metric)[0] for a in SCHEDULES] for c in CULTURES])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    data = read(out / "cross_schedule_analysis.json")
    summary = data["summary"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)

    def heatmap(ax, matrix, title, label, cmap, vmax):
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(3), [SCHEDULE_LABELS[a] for a in SCHEDULES], rotation=25, ha="right")
        ax.set_yticks(np.arange(6), [LABELS[c] for c in CULTURES])
        ax.set_xlabel("newcomer adaptation schedule")
        ax.set_title(title)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white" if value > 0.55 * vmax else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    heatmap(
        axes[0, 0],
        endpoint_matrix(summary, "identity_equivariant_target60_J"),
        "Endpoint functional recovery",
        "target-60 J (%)",
        "YlGnBu",
        100,
    )
    heatmap(
        axes[0, 1],
        endpoint_matrix(summary, "role_spread_target60_J"),
        "Endpoint role-permutation spread",
        "spread (percentage points)",
        "OrRd",
        100,
    )
    heatmap(
        axes[0, 2],
        endpoint_matrix(summary, "newcomer_resident_pair_agreement"),
        "Endpoint surface convention alignment",
        "pair agreement (%)",
        "Purples",
        100,
    )

    # Learning curves pooled by resident role regime, with adaptation schedule
    # as colour. Pooling is across the three resident formation topologies.
    ax = axes[1, 0]
    for mode, linestyle in (("static_role", "-"), ("random_role", "--")):
        for adaptation in SCHEDULES:
            cultures = [c for c in CULTURES if c.startswith(mode + "__")]
            values, sds = [], []
            for update in UPDATES:
                per_culture = [stats(summary, c, adaptation, update, "identity_equivariant_target60_J")[0] for c in cultures]
                values.append(float(np.mean(per_culture)))
                sds.append(float(np.std(per_culture, ddof=1)) if len(per_culture) > 1 else 0.0)
            x = np.asarray(UPDATES)
            y = np.asarray(values)
            sd = np.asarray(sds)
            ax.plot(x, y, marker="o", lw=2, ls=linestyle, color=COLORS[adaptation], label=("static" if mode == "static_role" else "random") + " / " + SCHEDULE_LABELS[adaptation])
            ax.fill_between(x, y - sd, y + sd, color=COLORS[adaptation], alpha=0.06)
    ax.set_xticks(UPDATES)
    ax.set_xlabel("adaptation update")
    ax.set_ylabel("identity-equivariant target-60 J (%)")
    ax.set_title("Recovery curves by adaptation schedule")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    # Paired endpoint gain relative to update 0 for each culture.
    ax = axes[1, 1]
    x = np.arange(len(CULTURES))
    width = 0.25
    for i, adaptation in enumerate(SCHEDULES):
        gains = []
        for culture in CULTURES:
            j0 = stats(summary, culture, adaptation, 0, "identity_equivariant_target60_J")[0]
            j3 = stats(summary, culture, adaptation, 300, "identity_equivariant_target60_J")[0]
            gains.append(j3 - j0)
        ax.bar(x + (i - 1) * width, gains, width, color=COLORS[adaptation], label=SCHEDULE_LABELS[adaptation])
    ax.set_xticks(x, [LABELS[c] for c in CULTURES], rotation=30, ha="right")
    ax.set_ylabel("J gain from update 0 to 300 (pp)")
    ax.set_title("Social-learning gain by culture and schedule")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    # Endpoint performance versus surface agreement, all 18 cells.
    ax = axes[1, 2]
    for culture in CULTURES:
        mode = culture.split("__", 1)[0]
        for adaptation in SCHEDULES:
            j, _ = stats(summary, culture, adaptation, 300, "identity_equivariant_target60_J")
            pair, _ = stats(summary, culture, adaptation, 300, "newcomer_resident_pair_agreement")
            nmi, _ = stats(summary, culture, adaptation, 300, "newcomer_pair_position_nmi")
            ax.scatter(pair, j, s=30 + 0.5 * nmi, color=COLORS[adaptation], marker="o" if mode == "static_role" else "s", alpha=0.85)
    ax.set_xlabel("newcomer–resident pair agreement (%)")
    ax.set_ylabel("identity-equivariant target-60 J (%)")
    ax.set_title("Endpoint function versus surface form")
    ax.set_xlim(0, 65)
    ax.set_ylim(0, 65)
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.97, "circle: static resident roles\nsquare: random resident roles\ncolour: adaptation schedule", transform=ax.transAxes, va="top", fontsize=7)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("v0.46 crossed resident culture × newcomer adaptation schedule", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure = out / "figures" / "01_cross_schedule.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "complete",
        "figure": str(figure),
        "analysis": str((out / "cross_schedule_analysis.json").resolve()),
        "panels": ["endpoint J heatmap", "endpoint role spread heatmap", "endpoint pair agreement heatmap", "pooled recovery curves", "paired recovery gain", "endpoint function versus surface form"],
        "matplotlib": plt.matplotlib.__version__,
    }
    (out / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
