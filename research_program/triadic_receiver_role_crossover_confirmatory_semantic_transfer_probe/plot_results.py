"""Plot role-stratified aligned/placebo transfer and schedule interactions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def pct(x):
    return 100.0 * float(x)


def plot(summary_path, output):
    summary_path = Path(summary_path).resolve(); output = Path(output).resolve()
    if output.exists():
        raise ValueError("Refuse to overwrite figures")
    summary = json.loads(summary_path.read_text(encoding="utf8")); output.mkdir(parents=True)
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), constrained_layout=True)
    roles = list(summary["role_cells"].keys()); colors = {"static": "#2f6db0", "rematched": "#d66b2d"}
    positions = []; values = []; lo = []; hi = []; labels = []; x = 0
    for role in roles:
        for schedule in ("static", "rematched"):
            entry = summary["role_cells"][role][f"{schedule}/live"]["aligned_minus_placebo"]["plan_transfer"]
            positions.append(x); values.append(pct(entry["mean"])); lo.append(pct(entry["mean"] - entry["ci95_t7"][0])); hi.append(pct(entry["ci95_t7"][1] - entry["mean"]))
            labels.append(f"{role}\n{schedule}"); x += 1
        x += 0.5
    axes[0].bar(positions, values, color=[colors["static"], colors["rematched"]] * len(roles), width=0.72, edgecolor="white")
    axes[0].errorbar(positions, values, yerr=np.array([lo, hi]), fmt="none", ecolor="#222222", capsize=3, linewidth=1)
    axes[0].axhline(0, color="#222222", linewidth=0.8); axes[0].set_xticks(positions, labels); axes[0].set_ylabel("aligned−placebo plan transfer (pp)"); axes[0].set_title("Role-specific content-transfer probe"); axes[0].grid(axis="y", alpha=0.25)
    fields = [("plan_transfer", "plan"), ("physical", "physical"), ("q", "Q"), ("conditional_q", "Q|physical"), ("partner_transfer", "partner"), ("action_change", "action")]
    x = np.arange(len(fields)); width = 0.24
    for index, role in enumerate(roles):
        entries = [summary["interactions"][role][field] for field, _ in fields]
        vals = [pct(entry["mean"]) for entry in entries]; lows = [pct(entry["mean"] - entry["ci95_t7"][0]) for entry in entries]; highs = [pct(entry["ci95_t7"][1] - entry["mean"]) for entry in entries]
        axes[1].bar(x + (index - 1) * width, vals, width=width, label=role, edgecolor="white")
        axes[1].errorbar(x + (index - 1) * width, vals, yerr=np.array([lows, highs]), fmt="none", ecolor="#222222", capsize=2, linewidth=0.8)
    axes[1].axhline(0, color="#222222", linewidth=0.8); axes[1].set_xticks(x, [label for _, label in fields], rotation=25, ha="right"); axes[1].set_ylabel("rematched × communication (pp)"); axes[1].set_title("Role interaction diagnostics"); axes[1].legend(title="replaced role"); axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Receiver-role crossover: aligned/placebo message transfer", fontsize=13)
    fig.savefig(output / "role_semantic_transfer.png", dpi=220); fig.savefig(output / "role_semantic_transfer.pdf"); plt.close(fig)
    receipt = dict(status="rendered", visual_review="pending", summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest(), files=["role_semantic_transfer.png", "role_semantic_transfer.pdf"])
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--summary", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); print(json.dumps(plot(args.summary, args.output), ensure_ascii=False))
