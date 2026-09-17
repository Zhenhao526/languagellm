"""Plot the v0.47 communication-reset granularity assay."""
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
RESET_MODES = ("sender_only", "receiver_only", "both")
RESET_LABELS = {"sender_only": "sender reset", "receiver_only": "receiver reset", "both": "both reset"}
COLORS = {"sender_only": "#4C78A8", "receiver_only": "#F58518", "both": "#54A24B"}
UPDATES = (0, 100, 300)


def read(path: Path):
    return json.loads(path.read_text())


def pct(x):
    return 100.0 * float(x)


def stats(summary, culture, reset_mode, update, metric):
    item = summary[culture][reset_mode][str(update)][metric]
    return pct(item["mean"]), pct(item["sd"])


def matrix(summary, metric, update=300):
    return np.asarray([[stats(summary, c, mode, update, metric)[0] for mode in RESET_MODES] for c in CULTURES])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); out = args.out.resolve()
    data = read(out / "reset_granularity_analysis.json"); summary = data["summary"]
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=220)

    def heatmap(ax, values, title, label, cmap, vmax):
        im = ax.imshow(values, cmap=cmap, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(3), [RESET_LABELS[m] for m in RESET_MODES], rotation=25, ha="right")
        ax.set_yticks(np.arange(6), [LABELS[c] for c in CULTURES]); ax.set_xlabel("reset granularity"); ax.set_title(title)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]; ax.text(j, i, f"{value:.1f}", ha="center", va="center", fontsize=8, color="white" if value > 0.55 * vmax else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=label)

    heatmap(axes[0, 0], matrix(summary, "identity_equivariant_target60_J"), "Endpoint functional recovery", "target-60 J (%)", "YlGnBu", 100)
    heatmap(axes[0, 1], matrix(summary, "role_spread_target60_J"), "Endpoint role-permutation spread", "spread (percentage points)", "OrRd", 100)
    heatmap(axes[0, 2], matrix(summary, "newcomer_resident_pair_agreement"), "Endpoint surface convention alignment", "pair agreement (%)", "Purples", 100)

    ax = axes[1, 0]
    for mode in RESET_MODES:
        for role_mode, linestyle in (("static_role", "-"), ("random_role", "--")):
            cultures = [c for c in CULTURES if c.startswith(role_mode + "__")]
            y = [float(np.mean([stats(summary, c, mode, update, "identity_equivariant_target60_J")[0] for c in cultures])) for update in UPDATES]
            sd = [float(np.std([stats(summary, c, mode, update, "identity_equivariant_target60_J")[0] for c in cultures], ddof=1)) for update in UPDATES]
            x = np.asarray(UPDATES); y = np.asarray(y); sd = np.asarray(sd)
            ax.plot(x, y, marker="o", lw=2, ls=linestyle, color=COLORS[mode], label=("static" if role_mode == "static_role" else "random") + " / " + RESET_LABELS[mode])
            ax.fill_between(x, y - sd, y + sd, color=COLORS[mode], alpha=0.06)
    ax.set_xticks(UPDATES); ax.set_xlabel("adaptation update"); ax.set_ylabel("identity-equivariant target-60 J (%)"); ax.set_title("Recovery by reset granularity"); ax.set_ylim(0, 100); ax.grid(alpha=0.25)

    ax = axes[1, 1]; x = np.arange(len(CULTURES)); width = 0.25
    for i, mode in enumerate(RESET_MODES):
        gains = [stats(summary, c, mode, 300, "identity_equivariant_target60_J")[0] - stats(summary, c, mode, 0, "identity_equivariant_target60_J")[0] for c in CULTURES]
        ax.bar(x + (i - 1) * width, gains, width, color=COLORS[mode], label=RESET_LABELS[mode])
    ax.set_xticks(x, [LABELS[c] for c in CULTURES], rotation=30, ha="right"); ax.set_ylabel("J gain from update 0 to 300 (pp)"); ax.set_title("Social-learning gain"); ax.grid(axis="y", alpha=0.25); ax.legend(frameon=False, fontsize=7)

    ax = axes[1, 2]
    for culture in CULTURES:
        marker = "o" if culture.startswith("static_role") else "s"
        for mode in RESET_MODES:
            pair = stats(summary, culture, mode, 300, "newcomer_resident_pair_agreement")[0]; j = stats(summary, culture, mode, 300, "identity_equivariant_target60_J")[0]; nmi = stats(summary, culture, mode, 300, "newcomer_pair_position_nmi")[0]
            ax.scatter(pair, j, s=30 + 0.5 * nmi, color=COLORS[mode], marker=marker, alpha=0.85)
    ax.set_xlabel("newcomer–resident pair agreement (%)"); ax.set_ylabel("identity-equivariant target-60 J (%)"); ax.set_title("Function versus surface form"); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(alpha=0.25); ax.text(0.02, 0.97, "circle: static resident roles\nsquare: random resident roles\ncolour: reset mode", transform=ax.transAxes, va="top", fontsize=7)
    handles, labels = axes[1, 0].get_legend_handles_labels(); fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.005)); fig.suptitle("v0.47 communication-reset granularity × resident culture", fontsize=16, y=0.995); fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    figure = out / "figures" / "01_reset_granularity.png"; figure.parent.mkdir(parents=True, exist_ok=True); fig.savefig(figure, bbox_inches="tight"); plt.close(fig)
    metadata = {"status": "complete", "figure": str(figure), "analysis": str((out / "reset_granularity_analysis.json").resolve()), "panels": ["endpoint J heatmap", "endpoint role spread heatmap", "endpoint pair agreement heatmap", "pooled recovery curves", "reset-mode recovery gain", "endpoint function versus surface form"], "matplotlib": plt.matplotlib.__version__}
    (out / "plot_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"); print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
