"""Plot the v0.44 endpoint partner-transfer assay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
CULTURES = ("static_role__fixed_A", "static_role__rotating_AB", "static_role__random_ABC", "random_role__fixed_A", "random_role__rotating_AB", "random_role__random_ABC")
CULTURE_LABELS = {"static_role__fixed_A": "static / A", "static_role__rotating_AB": "static / A-B", "static_role__random_ABC": "static / A-B-C", "random_role__fixed_A": "random / A", "random_role__rotating_AB": "random / A-B", "random_role__random_ABC": "random / A-B-C"}
CATEGORIES = ("same_culture", "same_role_mode_different_topology", "different_role_mode_same_topology", "different_role_mode_and_topology")
CATEGORY_LABELS = {"same_culture": "same culture", "same_role_mode_different_topology": "same role mode", "different_role_mode_same_topology": "different mode / same topology", "different_role_mode_and_topology": "different mode + topology"}
REPLACEMENTS = ("none", "slot0", "slot1", "slot2", "all")
REPLACEMENT_LABELS = {"none": "none", "slot0": "sender slot 0", "slot1": "sender slot 1", "slot2": "sender slot 2", "all": "all senders"}


def read(path):
    return json.loads(path.read_text())


def pct(x):
    return 100.0 * float(x)


def average_category(data, category, replacement, metric="equivariant_target_J"):
    values = [data["category_summary"][category][culture][replacement][metric]["mean"] for culture in CULTURES]
    return float(np.mean(values))


def direct(data, recipient, donor, replacement, schedule="A", role="012", metric="equivariant_target60_J"):
    return data["summary"][recipient][donor][replacement][schedule][role][metric]["mean"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    data = read(out / "partner_transfer_analysis.json")
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)
    colors = {"same_culture": "#4C78A8", "same_role_mode_different_topology": "#F58518", "different_role_mode_same_topology": "#54A24B", "different_role_mode_and_topology": "#E45756"}

    # A: endpoint target J by donor category and replacement size.
    ax = axes[0, 0]
    x = np.arange(len(REPLACEMENTS))
    for category in CATEGORIES:
        vals = [pct(average_category(data, category, replacement)) for replacement in REPLACEMENTS]
        ax.plot(x, vals, marker="o", color=colors[category], label=CATEGORY_LABELS[category])
    ax.set_xticks(x, [REPLACEMENT_LABELS[r] for r in REPLACEMENTS], rotation=22, ha="right")
    ax.set_ylabel("equivariant target-60 J (%)")
    ax.set_title("Compatibility by cultural distance")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5)

    # B: single-sender slot versus all-sender replacement, excluding same culture.
    ax = axes[0, 1]
    x = np.arange(4)
    for category in CATEGORIES[1:]:
        vals = [pct(average_category(data, category, replacement)) for replacement in REPLACEMENTS[1:]]
        ax.plot(x, vals, marker="o", color=colors[category], label=CATEGORY_LABELS[category])
    ax.set_xticks(x, ["slot 0", "slot 1", "slot 2", "all"], rotation=18, ha="right")
    ax.set_ylabel("equivariant target-60 J (%)")
    ax.set_title("One partner versus complete replacement")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6.5)

    # C: cultural compatibility matrix for one representative evaluation view.
    ax = axes[0, 2]
    matrix = np.asarray([[pct(direct(data, recipient, donor, "all")) for donor in CULTURES] for recipient in CULTURES])
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(35.0, float(matrix.max())))
    ax.set_xticks(np.arange(6), [CULTURE_LABELS[c] for c in CULTURES], rotation=45, ha="right")
    ax.set_yticks(np.arange(6), [CULTURE_LABELS[c] for c in CULTURES])
    ax.set_xlabel("donor culture")
    ax.set_ylabel("recipient culture")
    ax.set_title("All-sender compatibility (schedule A, role 012)")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7, color="black" if matrix[i, j] < 18 else "white")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="target-60 J (%)")

    # D: token pair agreement matrix; diagonal is 100% by construction.
    ax = axes[1, 0]
    matrix = np.asarray([[pct(data["token_summary"][recipient + "__" + donor]["pair"]["mean"]) for donor in CULTURES] for recipient in CULTURES])
    im = ax.imshow(matrix, cmap="magma", vmin=0, vmax=100)
    ax.set_xticks(np.arange(6), [CULTURE_LABELS[c] for c in CULTURES], rotation=45, ha="right")
    ax.set_yticks(np.arange(6), [CULTURE_LABELS[c] for c in CULTURES])
    ax.set_xlabel("donor culture")
    ax.set_ylabel("recipient culture")
    ax.set_title("Cross-culture message pair agreement")
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if matrix[i, j] < 55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="pair agreement (%)")

    # E: native and cross-culture all-sender replacement by recipient culture.
    ax = axes[1, 1]
    x = np.arange(6)
    native = [pct(average_category(data, "same_culture", "none"))] * 6
    # Use category summaries for each recipient to retain recipient-specific
    # endpoint variation.
    native = [pct(data["category_summary"]["same_culture"][c]["none"]["equivariant_target_J"]["mean"]) for c in CULTURES]
    cross = [pct(np.mean([data["category_summary"][cat][c]["all"]["equivariant_target_J"]["mean"] for cat in CATEGORIES[1:]])) for c in CULTURES]
    ax.bar(x - 0.18, native, 0.36, color="#4C78A8", label="native")
    ax.bar(x + 0.18, cross, 0.36, color="#E45756", label="cross-culture all")
    ax.set_xticks(x, [CULTURE_LABELS[c] for c in CULTURES], rotation=28, ha="right")
    ax.set_ylabel("equivariant target-60 J (%)")
    ax.set_title("Native versus replaced sender performance")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    # F: literal/equivariant scores for the three donor-distance categories.
    ax = axes[1, 2]
    x = np.arange(3)
    width = 0.18
    for i, replacement in enumerate(("slot0", "slot1", "all")):
        eq = [pct(average_category(data, cat, replacement, "equivariant_target_J")) for cat in CATEGORIES[1:]]
        lit = [pct(np.mean([data["category_summary"][cat][culture][replacement]["literal_target_J"]["mean"] for culture in CULTURES])) for cat in CATEGORIES[1:]]
        ax.bar(x + (i - 1) * width, eq, width, color=("#3182bd", "#6baed6", "#9ecae1")[i], label=replacement + " / equivariant")
        ax.bar(x + (i - 1) * width, lit, width, color=("#de2d26", "#fc9272", "#fcbba1")[i], alpha=0.55, label=replacement + " / literal")
    ax.set_xticks(x, ["same mode, other topology", "other mode, same topology", "other mode + topology"], rotation=20, ha="right")
    ax.set_ylabel("target-60 J (%)")
    ax.set_title("Literal/equivariant transfer gap")
    ax.legend(frameon=False, fontsize=6.5, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("v0.44 endpoint partner replacement and cultural compatibility", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    figure = out / "figures" / "01_partner_transfer.png"
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, bbox_inches="tight")
    plt.close(fig)
    metadata = {"status": "complete", "figure": str(figure.resolve()), "analysis": str((out / "partner_transfer_analysis.json").resolve()), "panels": ["compatibility by cultural distance", "single versus complete replacement", "cultural compatibility matrix", "message agreement matrix", "native versus replaced performance", "literal/equivariant transfer gap"], "matplotlib": plt.matplotlib.__version__}
    (out / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
