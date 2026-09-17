"""Plot audited FI single-slot sensitivity results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ("slot_payload_zero", "slot_symbol_shift", "slot_symbol_xor4")
FAMILY_LABELS = ("payload zero", "symbol +1", "symbol XOR4")
METRICS = (
    ("q_rate", "team Q loss (pp)"),
    ("physical_execution_rate", "physical execution loss (pp)"),
    ("proposal_legal_rate", "proposal legality loss (pp)"),
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def value(data, schedule, family, metric):
    v = data["summaries"][f"{schedule}_live_{family}"][metric]
    return 100 * float(v["mean"]), 100 * float(v["ci95_lower"]), 100 * float(v["ci95_upper"])


def plot(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite figure directory")
    data = json.loads((source / "execution/results.json").read_text(encoding="utf8"))
    require(data.get("status") == "completed_json_only_single_slot_probe", "Unexpected result status")
    output.mkdir(parents=True)

    # Family-level summary.
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(11.5, 9.0), sharex="col")
    x = np.arange(len(FAMILIES))
    for row_index, (metric, ylabel) in enumerate(METRICS):
        for col, schedule in enumerate(("static", "rematched")):
            ax = axes[row_index, col]
            triples = [value(data, schedule, family, metric) for family in FAMILIES]
            means = np.asarray([t[0] for t in triples])
            low = np.asarray([t[1] for t in triples])
            high = np.asarray([t[2] for t in triples])
            yerr = np.vstack((means - low, high - means))
            ax.bar(x, means, yerr=yerr, capsize=4, color=("#2f6f9f", "#c97b29", "#8a6bbd"), edgecolor="white", linewidth=.5)
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{schedule} schedule — FI-live")
            ax.grid(axis="y", alpha=.25)
            ax.set_xticks(x, FAMILY_LABELS if row_index == len(METRICS) - 1 else ["", "", ""])
    fig.suptitle("FI single-slot sensitivity\n(natural − intervention, percentage points; 16 paired seeds)", y=.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .965))
    family_png = output / "single_slot_family_effects.png"
    family_pdf = output / "single_slot_family_effects.pdf"
    fig.savefig(family_png, dpi=180)
    fig.savefig(family_pdf)
    plt.close(fig)

    # Per-slot Q heatmap: rows are transformations, columns are slots.
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True, constrained_layout=True)
    for ax, schedule in zip(axes, ("static", "rematched")):
        matrix = np.zeros((len(FAMILIES), 4), dtype=float)
        for i, family in enumerate(FAMILIES):
            for slot in range(4):
                k = f"{schedule}_live_{family}_{slot}"
                matrix[i, slot] = 100 * float(data["summaries"][k]["q_rate"]["mean"])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(0.8, float(matrix.max()) * 1.05), aspect="auto")
        ax.set_title(f"{schedule} schedule")
        ax.set_xticks(range(4), ["slot 0", "slot 1", "slot 2", "slot 3"])
        ax.set_yticks(range(len(FAMILIES)), FAMILY_LABELS)
        for i in range(len(FAMILIES)):
            for j in range(4):
                ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black")
    axes[0].set_ylabel("single-slot transformation")
    # A horizontal bar keeps the two panel titles and the right panel clear.
    fig.colorbar(im, ax=axes.ravel().tolist(), orientation="horizontal", shrink=.72, pad=.10, label="team Q loss (pp)")
    fig.suptitle("Locality of FI message-code sensitivity by slot", fontsize=12)
    heat_png = output / "single_slot_q_heatmap.png"
    heat_pdf = output / "single_slot_q_heatmap.pdf"
    fig.savefig(heat_png, dpi=180)
    fig.savefig(heat_pdf)
    plt.close(fig)

    receipt = {
        "status": "generated",
        "source": str(source),
        "source_results_sha256": sha(source / "execution/results.json"),
        "png": [str(family_png), str(heat_png)],
        "pdf": [str(family_pdf), str(heat_pdf)],
        "visual_review": "pending",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
