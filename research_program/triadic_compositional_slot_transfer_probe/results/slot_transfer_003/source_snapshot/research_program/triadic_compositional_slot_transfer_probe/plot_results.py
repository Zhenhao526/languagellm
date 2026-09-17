"""Plot single-slot aligned-minus-placebo transfer and matched differences."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(summary_path: str, output: str):
    summary_path = Path(summary_path).resolve(); output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refuse to overwrite figures: {output}")
    data = json.loads(summary_path.read_text(encoding="utf8")); output.mkdir(parents=True)
    masks = [item["name"] for item in data["contract"]["masks"]] if isinstance(data["contract"]["masks"], list) else ["slot_0", "slot_1", "slot_2", "slot_3", "full"]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)
    x = np.arange(len(masks)); width = 0.19; colors = {"seen_joint_only": "#8aa4c7", "all_joint": "#2f6db0"}
    for index, (arm, label) in enumerate((("seen_joint_only", "seen A"), ("all_joint", "all A"))):
        for offset, schedule in enumerate(("static", "rematched")):
            values = [data["cells"][f"{mask}/{arm}/{schedule}/live"]["A"]["aligned_minus_placebo"]["plan_transfer"] for mask in masks]
            lo = [100 * (value["mean"] - value["ci95_t7"][0]) for value in values]; hi = [100 * (value["ci95_t7"][1] - value["mean"]) for value in values]
            pos = x + (index * 2 + offset) * width - 1.5 * width
            axes[0].bar(pos, [100 * value["mean"] for value in values], width=width, color=colors[arm], alpha=0.75 if offset == 0 else 1.0, edgecolor="white", label=f"{label} {schedule}")
            axes[0].errorbar(pos, [100 * value["mean"] for value in values], yerr=np.array([lo, hi]), fmt="none", ecolor="#222", capsize=2, linewidth=.8)
    axes[0].axhline(0, color="#222", linewidth=.8); axes[0].set_xticks(x + .5 * width, masks); axes[0].set_ylabel("aligned − placebo plan transfer (pp)"); axes[0].set_title("Slot-specific transfer"); axes[0].legend(fontsize=8, ncol=2); axes[0].grid(axis="y", alpha=.25)
    values = [data["interaction"][mask]["plan_transfer"] for mask in masks]; axes[1].bar(x, [100 * value["mean"] for value in values], color="#6d9e72", width=.62, edgecolor="white"); axes[1].errorbar(x, [100 * value["mean"] for value in values], yerr=np.array([[100 * (value["mean"] - value["ci95_t7"][0]) for value in values], [100 * (value["ci95_t7"][1] - value["mean"]) for value in values]]), fmt="none", ecolor="#222", capsize=3, linewidth=1); axes[1].axhline(0, color="#222", linewidth=.8); axes[1].set_xticks(x, masks); axes[1].set_ylabel("all−seen × schedule interaction (pp)"); axes[1].set_title("Support-control interaction"); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Compositional holdout single-slot transfer", fontsize=13); fig.savefig(output / "slot_transfer.png", dpi=220); fig.savefig(output / "slot_transfer.pdf"); plt.close(fig)
    receipt = dict(status="rendered", visual_review="pending", summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(), files=["slot_transfer.png", "slot_transfer.pdf"]); (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--summary", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); print(json.dumps(plot(args.summary, args.output), ensure_ascii=False))

