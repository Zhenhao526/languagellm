"""Plots for the role-wise receiver crossover summary."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(summary_path, output):
    summary_path = Path(summary_path).resolve(); output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"Refuse to overwrite figures: {output}")
    data = json.loads(summary_path.read_text(encoding="utf8")); summary = data["summary"]; output.mkdir(parents=True)
    roles = list(summary["roles"]); schedules = ("static", "rematched"); x = np.arange(len(roles)); width = .34
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), constrained_layout=True)
    colors = {"static": "#386cb0", "rematched": "#d95f02"}
    for offset, schedule in ((-width / 2, "static"), (width / 2, "rematched")):
        cells = [summary["roles"][role][schedule]["q_rate"]["centered_AUC"] for role in roles]
        values = [100 * cell["mean"] for cell in cells]; lo = [100 * (cell["mean"] - cell["ci95_t7"][0]) for cell in cells]; hi = [100 * (cell["ci95_t7"][1] - cell["mean"]) for cell in cells]
        axes[0].bar(x + offset, values, width, color=colors[schedule], label=schedule, edgecolor="white"); axes[0].errorbar(x + offset, values, yerr=np.array([lo, hi]), fmt="none", ecolor="#222", capsize=3, linewidth=1)
    axes[0].axhline(0, color="#222", linewidth=.8); axes[0].set_xticks(x, roles); axes[0].set_ylabel("live − silent Q AUC (pp)"); axes[0].set_title("Role-specific adaptation"); axes[0].legend(frameon=False); axes[0].grid(axis="y", alpha=.25)
    metrics = (("q_rate", "Q"), ("physical_execution_rate", "physical"), ("conditional_q_rate", "Q | physical"), ("target_pair_legal_rate", "target pair"))
    for offset, role in enumerate(roles):
        cells = [summary["roles"][role]["static"][metric]["endpoint"] for metric, _ in metrics]; values = [100 * cell["mean"] for cell in cells]; lo = [100 * (cell["mean"] - cell["ci95_t7"][0]) for cell in cells]; hi = [100 * (cell["ci95_t7"][1] - cell["mean"]) for cell in cells]
        axes[1].errorbar(np.arange(len(metrics)) + (offset - 1) * .22, values, yerr=np.array([lo, hi]), fmt="o", capsize=3, label=f"{role} static", color=("#1b9e77", "#7570b3", "#d95f02")[offset])
    axes[1].axhline(0, color="#222", linewidth=.8); axes[1].set_xticks(np.arange(len(metrics)), [label for _, label in metrics]); axes[1].set_ylabel("static endpoint live − silent (pp)"); axes[1].set_title("Endpoint components"); axes[1].legend(frameon=False, fontsize=8); axes[1].grid(axis="y", alpha=.25)
    fig.suptitle("Receiver-role crossover on the same frozen protocol", fontsize=13); fig.savefig(output / "role_crossover.png", dpi=220); fig.savefig(output / "role_crossover.pdf"); plt.close(fig)
    receipt = dict(status="rendered", visual_review="pending", summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(), files=["role_crossover.png", "role_crossover.pdf"]); (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--summary", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); plot(args.summary, args.output)

