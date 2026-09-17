"""Plots for the audited single-slot FI probe."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FAMILIES = ("slot_payload_zero", "slot_symbol_shift", "slot_symbol_xor4")
FAMILY_LABELS = ("payload zero", "symbol +1", "symbol XOR 4")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def draw(summary, output):
    output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), sharey=False)
    x = np.arange(3); width = .34
    for ax, metric, title in zip(axes, ("q_rate", "physical_execution_rate"), ("Q rate", "physical execution")):
        for j, schedule in enumerate(("static", "rematched")):
            means = []; low = []; high = []
            for family in FAMILIES:
                vals = []
                for slot in range(4):
                    st = summary["summaries"][f"{schedule}_live_{family}_{slot}"][metric]
                    vals.append(st["mean"])
                st_mean = float(np.mean(vals))
                # Use the family-level precomputed interval for exact paired
                # seed averaging rather than averaging slot-level half-widths.
                fam_st = summary["summaries"][f"{schedule}_live_{family}"][metric]
                means.append(100 * fam_st["mean"]); low.append(100 * (fam_st["mean"] - fam_st["ci95_lower"])); high.append(100 * (fam_st["ci95_upper"] - fam_st["mean"]))
            ax.bar(x + (j - .5) * width, means, width, yerr=np.asarray([low, high]), capsize=3, color=("#4c72b0" if j == 0 else "#dd8452"), label=schedule)
        ax.axhline(0, color="#777", lw=.8); ax.set_xticks(x, FAMILY_LABELS); ax.set_ylabel("natural − single-slot intervention (pp)"); ax.set_title(title); ax.grid(axis="y", alpha=.22)
    axes[1].legend(frameon=False); fig.suptitle("Single-slot FI sensitivity (three selected senders, four slots)", y=1.02); fig.tight_layout()
    for ext in ("png", "pdf"): fig.savefig(output / f"slot_family_effects.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
    for ax, schedule in zip(axes, ("static", "rematched")):
        matrix = np.asarray([[100 * summary["summaries"][f"{schedule}_live_{family}_{slot}"]["q_rate"]["mean"] for slot in range(4)] for family in FAMILIES])
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(1.0, float(matrix.max())), aspect="auto")
        ax.set_xticks(range(4), ["slot 0", "slot 1", "slot 2", "slot 3"]); ax.set_yticks(range(3), FAMILY_LABELS); ax.set_title(schedule)
        for i in range(3):
            for j in range(4): ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black")
    fig.subplots_adjust(left=.16, right=.86, bottom=.16, top=.82, wspace=.16)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=.035, pad=.04); cbar.set_label("Q effect (pp)")
    fig.suptitle("Slot-wise Q effects", y=.96)
    for ext in ("png", "pdf"): fig.savefig(output / f"slot_q_heatmap.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    receipt = dict(status="generated", summary_sha256=sha(Path(summary["summary_path"])), png=[str(output / "slot_family_effects.png"), str(output / "slot_q_heatmap.png")], pdf=[str(output / "slot_family_effects.pdf"), str(output / "slot_q_heatmap.pdf")], visual_review="pending")
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n"); print(json.dumps(receipt, ensure_ascii=False))


def main(path, output):
    path = Path(path).resolve(); data = json.loads(path.read_text()); data["summary_path"] = str(path); draw(data, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--summary", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); main(args.summary, args.out)
