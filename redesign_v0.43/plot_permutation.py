"""Plot the v0.43 resource-permutation/role-randomization analysis."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
ROLE_MODES = ("static_role", "random_role")
ROLE_KEYS = ("012", "021", "102", "120", "201", "210")
ASSIGNMENTS = ROLE_KEYS
UPDATES = (0, 100, 300, 600)


def read(path):
    return json.loads(path.read_text())


def pct(x):
    return 100.0 * float(x)


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def endpoint(data, mode, condition, role="012", kind="equivariant"):
    return data["summary"][mode][condition]["600"][role][kind]


def endpoint_values(data, mode, condition, role="012", kind="equivariant"):
    return data["summary"][mode][condition]["600"][role][kind]["values"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    data = read(out / "permutation_analysis.json")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)
    colors = {"fixed_A": "#4C78A8", "rotating_AB": "#F58518", "random_ABC": "#54A24B"}
    labels = {"fixed_A": "fixed A", "rotating_AB": "rotating A/B", "random_ABC": "random A/B/C"}
    mode_colors = {"static_role": "#4C78A8", "random_role": "#E45756"}
    mode_labels = {"static_role": "static role", "random_role": "random role"}

    # A: main endpoint result, using the identity role ordering.
    ax = axes[0, 0]
    x = np.arange(len(CONDITIONS))
    width = 0.36
    for i, mode in enumerate(ROLE_MODES):
        vals, errs = [], []
        for condition in CONDITIONS:
            m, s = mean_sd(endpoint_values(data, mode, condition, "012", "equivariant"))
            vals.append(pct(m)); errs.append(pct(s))
        ax.bar(x + (i - 0.5) * width, vals, width, yerr=errs, capsize=2, color=mode_colors[mode], alpha=0.9, label=mode_labels[mode])
    ax.set_xticks(x, [labels[c] for c in CONDITIONS], rotation=18, ha="right")
    ax.set_ylabel("target-60 joint J (%)")
    ax.set_title("Identity-role endpoint performance")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # B: all receiver-side role permutations; dashed lines are role-randomized formation.
    ax = axes[0, 1]
    x = np.arange(len(ROLE_KEYS))
    for condition in CONDITIONS:
        for mode in ROLE_MODES:
            vals = [pct(endpoint(data, mode, condition, role, "equivariant")["mean"]) for role in ROLE_KEYS]
            ax.plot(x, vals, marker="o", color=colors[condition], linestyle="-" if mode == "static_role" else "--", alpha=0.85, label=labels[condition] + " / " + mode_labels[mode])
    ax.set_xticks(x, ROLE_KEYS)
    ax.set_xlabel("resource-role permutation")
    ax.set_ylabel("target-60 equivariant J (%)")
    ax.set_title("Role-order sensitivity at the endpoint")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    # C: literal/equivariant gap averaged over five non-identity role orders.
    ax = axes[0, 2]
    x = np.arange(len(CONDITIONS))
    group_width = 0.18
    variants = (("static_role", "literal", "static literal", "#9ecae1"), ("static_role", "equivariant", "static equivariant", "#3182bd"), ("random_role", "literal", "random literal", "#fcbba1"), ("random_role", "equivariant", "random equivariant", "#de2d26"))
    for i, (mode, kind, label, color) in enumerate(variants):
        vals, errs = [], []
        for condition in CONDITIONS:
            samples = []
            for role in ROLE_KEYS[1:]:
                samples.extend(endpoint_values(data, mode, condition, role, kind))
            m, s = mean_sd(samples)
            vals.append(pct(m)); errs.append(pct(s))
        ax.bar(x + (i - 1.5) * group_width, vals, group_width, yerr=errs, capsize=2, color=color, label=label)
    ax.set_xticks(x, [labels[c] for c in CONDITIONS], rotation=18, ha="right")
    ax.set_ylabel("non-identity target-60 J (%)")
    ax.set_title("Literal versus target-axis-synchronized scoring")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    # D: all six resource assignments under randomized role formation.
    ax = axes[1, 0]
    matrix = np.asarray([[pct(data["assignment_summary"][assignment]["random_role"][condition]["012"]["equivariant"]["mean"]) for condition in CONDITIONS] for assignment in ASSIGNMENTS])
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(70.0, float(matrix.max())))
    ax.set_xticks(np.arange(len(CONDITIONS)), [labels[c] for c in CONDITIONS], rotation=20, ha="right")
    ax.set_yticks(np.arange(len(ASSIGNMENTS)), ASSIGNMENTS)
    ax.set_xlabel("formation topology")
    ax.set_ylabel("resource-column assignment")
    ax.set_title("All six resource assignments (random role)")
    for i in range(len(ASSIGNMENTS)):
        for j in range(len(CONDITIONS)):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8, color="black" if matrix[i, j] < 45 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="target-60 J (%)")

    # E: sequence agreement and position information at endpoint.
    ax = axes[1, 1]
    x = np.arange(len(CONDITIONS))
    width = 0.12
    seq_keys = (("token0_agreement", "token0 agreement", "#4C78A8"), ("token1_agreement", "token1 agreement", "#F58518"), ("pair_agreement", "pair agreement", "#54A24B"))
    for i, mode in enumerate(ROLE_MODES):
        for j, (key, label, color) in enumerate(seq_keys):
            vals = [pct(data["sequence_summary"][mode][condition][key]["mean"]) for condition in CONDITIONS]
            offset = (i * len(seq_keys) + j - 2.5) * width
            ax.bar(x + offset, vals, width, color=color, alpha=0.55 if mode == "static_role" else 0.95, label=mode_labels[mode] + " / " + label if i == 0 else (mode_labels[mode] + " / " + label))
    # Keep the legend readable by deduplicating labels.
    handles, labels_ = ax.get_legend_handles_labels()
    dedup = dict(zip(labels_, handles))
    ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=6.5, ncol=2)
    ax.set_xticks(x, [labels[c] for c in CONDITIONS], rotation=18, ha="right")
    ax.set_ylabel("agreement (%)")
    ax.set_title("Emergent sequence structure")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.25)

    # F: learning curves for the identity role, with a compact uncertainty band.
    ax = axes[1, 2]
    for condition in CONDITIONS:
        for mode in ROLE_MODES:
            means, lows, highs = [], [], []
            for update in UPDATES:
                values = data["summary"][mode][condition][str(update)]["012"]["equivariant"]["values"]
                m, s = mean_sd(values)
                means.append(pct(m)); lows.append(max(0.0, pct(m - s))); highs.append(min(100.0, pct(m + s)))
            style = "-" if mode == "static_role" else "--"
            ax.plot(UPDATES, means, marker="o", color=colors[condition], linestyle=style, label=labels[condition] + " / " + mode_labels[mode])
            ax.fill_between(UPDATES, lows, highs, color=colors[condition], alpha=0.06)
    ax.set_xlabel("formation update")
    ax.set_ylabel("target-60 equivariant J (%)")
    ax.set_title("Learning curves: identity role")
    ax.set_xticks(UPDATES)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5, ncol=2)

    fig.suptitle("v0.43 resource permutation and role randomization", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figure = out / "figures" / "01_permutation_formation.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    metadata = {
        "status": "complete", "figure": str(figure.resolve()), "analysis": str((out / "permutation_analysis.json").resolve()),
        "panels": ["identity-role endpoint performance", "all role-permutation equivariant curves", "literal/equivariant comparison", "six resource assignments", "sequence agreement", "identity-role learning curves"],
        "matplotlib": plt.matplotlib.__version__,
    }
    (out / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
