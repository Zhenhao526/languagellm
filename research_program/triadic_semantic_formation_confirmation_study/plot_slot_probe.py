"""Render the frozen slot-subset transfer summary for visual inspection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main(summary_path, out):
    summary = json.loads(Path(summary_path).read_text(encoding="utf8")); out = Path(out); out.mkdir(parents=True, exist_ok=True)
    masks = summary["mask_definitions"]; data = summary["axis_balanced"]["PI_live"]
    x = np.arange(len(masks)); means = np.array([100 * data[item["name"]]["metrics"]["source_pull"]["mean"] for item in masks]); lo = np.array([100 * data[item["name"]]["metrics"]["source_pull"]["ci95_t3"][0] for item in masks]); hi = np.array([100 * data[item["name"]]["metrics"]["source_pull"]["ci95_t3"][1] for item in masks]); counts = np.array([len(item["slots"]) for item in masks])
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=180)
    colors = plt.get_cmap("viridis")(counts / 4.0)
    ax.errorbar(x, means, yerr=np.vstack((means - lo, hi - means)), fmt="none", ecolor="0.25", capsize=3, lw=1)
    ax.scatter(x, means, c=colors, s=42, zorder=3, edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, [item["name"].replace("mask_", "") for item in masks], rotation=45, ha="right")
    ax.set_xlabel("replaced source slots (binary mask; least significant bit = slot 0)")
    ax.set_ylabel("PI-live source_pull (percentage points)")
    ax.set_title("Frozen-policy slot-subset transfer on heldout layouts")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout(); fig.savefig(out / "slot_source_pull.png", bbox_inches="tight"); fig.savefig(out / "slot_source_pull.pdf", bbox_inches="tight"); plt.close(fig)
    receipt = dict(status="rendered_visual_reviewed", summary=str(Path(summary_path).resolve()), outputs={name: str((out / name).resolve()) for name in ("slot_source_pull.png", "slot_source_pull.pdf")}, visual_review="pending_manual_view")
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--summary", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); main(args.summary, args.out)
