"""Plot the v0.45 newcomer social-learning assay."""
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
COLORS = {"fixed_A": "#4C78A8", "rotating_AB": "#F58518", "random_ABC": "#54A24B"}
UPDATES = (0, 100, 300)


def read(path: Path):
    return json.loads(path.read_text())


def pct(x):
    return 100.0 * float(x)


def stats(summary, culture, update, metric):
    item = summary[culture][str(update)][metric]
    return pct(item["mean"]), pct(item["sd"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    data = read(out / "newcomer_adaptation_analysis.json")
    summary = data["summary"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)

    def line_panel(ax, metric, ylabel, title, ylim=None, sd=True):
        x = np.asarray(UPDATES)
        for culture in CULTURES:
            mode, condition = culture.split("__", 1)
            vals = np.asarray([stats(summary, culture, u, metric)[0] for u in UPDATES])
            sds = np.asarray([stats(summary, culture, u, metric)[1] for u in UPDATES])
            style = "-" if mode == "static_role" else "--"
            color = COLORS[condition]
            ax.plot(x, vals, marker="o", lw=2, ls=style, color=color, label=LABELS[culture])
            if sd:
                ax.fill_between(x, vals - sds, vals + sds, color=color, alpha=0.07)
        ax.set_xticks(UPDATES)
        ax.set_xlabel("adaptation update")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)

    line_panel(
        axes[0, 0],
        "identity_equivariant_target60_J",
        "identity-equivariant target-60 J (%)",
        "Newcomer functional recovery",
        (0, 100),
    )
    line_panel(
        axes[0, 1],
        "role_spread_target60_J",
        "role spread (percentage points)",
        "Across-role convention spread",
        (0, 100),
    )
    line_panel(
        axes[0, 2],
        "newcomer_resident_pair_agreement",
        "newcomer–resident pair agreement (%)",
        "Surface convention alignment",
        (0, 100),
    )
    line_panel(
        axes[1, 0],
        "newcomer_pair_position_nmi",
        "pair-position NMI (%)",
        "Functional position encoding",
        (0, 105),
    )

    # Endpoint comparison between fixed-role and random-role resident cultures.
    ax = axes[1, 1]
    conditions = ("fixed_A", "rotating_AB", "random_ABC")
    x = np.arange(len(conditions))
    width = 0.34
    static = np.asarray([stats(summary, "static_role__" + c, 300, "identity_equivariant_target60_J") for c in conditions])
    random = np.asarray([stats(summary, "random_role__" + c, 300, "identity_equivariant_target60_J") for c in conditions])
    ax.bar(x - width / 2, static[:, 0], width, yerr=static[:, 1], capsize=3, color="#9ecae1", label="static resident roles")
    ax.bar(x + width / 2, random[:, 0], width, yerr=random[:, 1], capsize=3, color="#2171b5", label="random resident roles")
    ax.set_xticks(x, ["A", "A-B", "A-B-C"])
    ax.set_xlabel("resident partner topology")
    ax.set_ylabel("identity-equivariant target-60 J (%)")
    ax.set_title("Endpoint recovery by resident role regime")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    # Endpoint relationship between task performance and surface agreement.
    ax = axes[1, 2]
    for culture in CULTURES:
        mode, condition = culture.split("__", 1)
        j, _ = stats(summary, culture, 300, "identity_equivariant_target60_J")
        pair, _ = stats(summary, culture, 300, "newcomer_resident_pair_agreement")
        nmi, _ = stats(summary, culture, 300, "newcomer_pair_position_nmi")
        ax.scatter(pair, j, s=35 + 0.7 * nmi, color=COLORS[condition], marker="o" if mode == "static_role" else "s", alpha=0.9)
        ax.annotate(LABELS[culture], (pair, j), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xlabel("newcomer–resident pair agreement (%)")
    ax.set_ylabel("identity-equivariant target-60 J (%)")
    ax.set_title("Endpoint performance versus surface alignment")
    ax.set_xlim(0, 55)
    ax.set_ylim(0, 55)
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.96, "circle: static roles\nsquare: random roles\nsize: pair-position NMI", transform=ax.transAxes, va="top", fontsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("v0.45 newcomer adaptation in frozen resident cultures", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure = out / "figures" / "01_newcomer_adaptation.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "complete",
        "figure": str(figure),
        "analysis": str((out / "newcomer_adaptation_analysis.json").resolve()),
        "panels": [
            "functional recovery",
            "role spread",
            "surface convention alignment",
            "functional position encoding",
            "endpoint role-regime comparison",
            "endpoint performance versus surface alignment",
        ],
        "matplotlib": plt.matplotlib.__version__,
    }
    (out / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
