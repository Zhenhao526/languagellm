"""Plot the v0.48 bounded resident co-adaptation assay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CULTURES = (
    "static_role__fixed_A", "static_role__rotating_AB", "static_role__random_ABC",
    "random_role__fixed_A", "random_role__rotating_AB", "random_role__random_ABC",
)
LABELS = {
    "static_role__fixed_A": "static / A", "static_role__rotating_AB": "static / A-B", "static_role__random_ABC": "static / A-B-C",
    "random_role__fixed_A": "random / A", "random_role__rotating_AB": "random / A-B", "random_role__random_ABC": "random / A-B-C",
}
RESIDENT_MODES = ("resident_frozen", "resident_sender_sparse", "resident_receiver_sparse", "resident_both_sparse")
MODE_LABELS = {
    "resident_frozen": "frozen",
    "resident_sender_sparse": "sender sparse",
    "resident_receiver_sparse": "receiver sparse",
    "resident_both_sparse": "both sparse",
}
COLORS = {
    "resident_frozen": "#7A7A7A",
    "resident_sender_sparse": "#4C78A8",
    "resident_receiver_sparse": "#F58518",
    "resident_both_sparse": "#54A24B",
}
UPDATES = (0, 100, 300, 600)


def read(path: Path):
    return json.loads(path.read_text())


def pct(x):
    return 100.0 * float(x)


def mean(summary, culture, mode, update, metric):
    return float(summary[culture][mode][str(update)][metric]["mean"])


def matrix(summary, metric, update, scale=1.0):
    return np.asarray([[scale * mean(summary, culture, mode, update, metric) for mode in RESIDENT_MODES] for culture in CULTURES])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    analysis = read(out / "coadaptation_analysis.json")
    statistics = read(out / "coadaptation_statistics.json")
    summary = analysis["summary"]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)

    def heatmap(ax, values, title, label, cmap, vmax=None, fmt=".1f"):
        if vmax is None:
            vmax = float(np.max(values)) if np.max(values) > 0 else 1.0
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(RESIDENT_MODES)), [MODE_LABELS[m] for m in RESIDENT_MODES], rotation=28, ha="right")
        ax.set_yticks(np.arange(len(CULTURES)), [LABELS[c] for c in CULTURES])
        ax.set_xlabel("resident adaptation mode")
        ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                ax.text(j, i, format(value, fmt), ha="center", va="center", fontsize=8, color="white" if value > 0.58 * vmax else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    heatmap(
        axes[0, 0],
        matrix(summary, "identity_equivariant_target60_J", 600, scale=100.0),
        "Endpoint functional recovery",
        "target-60 J (%)",
        "YlGnBu",
        vmax=100.0,
    )
    drift_values = matrix(summary, "resident_communication_drift_mean", 600, scale=100.0)
    heatmap(
        axes[0, 1],
        drift_values,
        "Endpoint resident communication drift",
        "relative L2 drift (%)",
        "magma",
        vmax=max(2.0, float(np.max(drift_values)) * 1.05),
        fmt=".2f",
    )
    spread_values = matrix(summary, "role_spread_target60_J", 600, scale=100.0)
    heatmap(
        axes[0, 2],
        spread_values,
        "Endpoint role-permutation spread",
        "spread (percentage points)",
        "OrRd",
        vmax=max(1.0, float(np.max(spread_values)) * 1.05),
        fmt=".1f",
    )

    ax = axes[1, 0]
    for mode in RESIDENT_MODES:
        for role_mode, linestyle in (("static_role", "-"), ("random_role", "--")):
            cultures = [c for c in CULTURES if c.startswith(role_mode + "__")]
            x = np.asarray(UPDATES)
            y = np.asarray([np.mean([pct(mean(summary, c, mode, update, "identity_equivariant_target60_J")) for c in cultures]) for update in UPDATES])
            spread = np.asarray([np.std([pct(mean(summary, c, mode, update, "identity_equivariant_target60_J")) for c in cultures], ddof=1) for update in UPDATES])
            ax.plot(x, y, marker="o", lw=2, ls=linestyle, color=COLORS[mode], label=("static" if role_mode == "static_role" else "random") + " / " + MODE_LABELS[mode])
            ax.fill_between(x, y - spread, y + spread, color=COLORS[mode], alpha=0.06)
    ax.set_xticks(UPDATES)
    ax.set_xlabel("adaptation update")
    ax.set_ylabel("identity-equivariant target-60 J (%)")
    ax.set_title("Recovery over the 600-update horizon")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    x = np.arange(len(CULTURES))
    width = 0.20
    gains = statistics["recovery_gains"]
    for i, mode in enumerate(RESIDENT_MODES):
        values = [pct(gains[c][mode]["mean"]) for c in CULTURES]
        ax.bar(x + (i - 1.5) * width, values, width, color=COLORS[mode], label=MODE_LABELS[mode])
    ax.set_xticks(x, [LABELS[c] for c in CULTURES], rotation=30, ha="right")
    ax.set_ylabel("J gain from update 0 to 600 (pp)")
    ax.set_title("Bounded co-adaptation gain")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 2]
    for culture in CULTURES:
        marker = "o" if culture.startswith("static_role") else "s"
        for mode in RESIDENT_MODES:
            pair = pct(mean(summary, culture, mode, 600, "newcomer_resident_pair_agreement"))
            j = pct(mean(summary, culture, mode, 600, "identity_equivariant_target60_J"))
            nmi = pct(mean(summary, culture, mode, 600, "newcomer_pair_position_nmi"))
            ax.scatter(pair, j, s=30 + 0.5 * nmi, color=COLORS[mode], marker=marker, alpha=0.85)
    ax.set_xlabel("newcomer–resident pair agreement (%)")
    ax.set_ylabel("identity-equivariant target-60 J (%)")
    ax.set_title("Function versus surface convention")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.97, "circle: static resident roles\nsquare: random resident roles\ncolour: resident mode", transform=ax.transAxes, va="top", fontsize=7)

    handles, labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("v0.48 bounded resident co-adaptation × newcomer recovery", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))

    figure = out / "figures" / "01_coadaptation.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "complete",
        "figure": str(figure),
        "analysis": str((out / "coadaptation_analysis.json").resolve()),
        "statistics": str((out / "coadaptation_statistics.json").resolve()),
        "panels": [
            "endpoint target-60 J heatmap",
            "endpoint resident communication drift heatmap",
            "endpoint role-permutation spread heatmap",
            "pooled recovery curves at updates 0, 100, 300, 600",
            "bounded co-adaptation gain bars",
            "endpoint function versus surface convention",
        ],
        "matplotlib": plt.matplotlib.__version__,
    }
    (out / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
