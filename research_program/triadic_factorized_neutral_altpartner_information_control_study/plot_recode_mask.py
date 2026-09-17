"""Plot the audited FI recode/mask probe.

The probe is post-hoc and read-only.  All plotted effects are natural minus
intervention in percentage points; positive values therefore mean that the
intervention lowered the metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


T15 = 2.1314495455597715
TRANSFORMS = ("symbol_ensemble", "position", "field")
LABELS = ("6 symbol\nbijections", "2 slot\npermutations", "payload\nzero", "visibility\nzero")
METRICS = (
    ("q_rate", "team Q loss (pp)"),
    ("physical_execution_rate", "physical execution loss (pp)"),
    ("target_pair_legal_rate", "target-pair rate change (pp)"),
)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stat(data: dict, schedule: str, transform: str, metric: str) -> tuple[float, float, float]:
    key = f"{schedule}_live_{transform}"
    value = data["summaries"][key][metric]
    # The JSON stores natural minus intervention already.
    return 100 * value["mean"], 100 * value["ci95_lower"], 100 * value["ci95_upper"]


def direct_stat(data: dict, schedule: str, transform: str, metric: str) -> tuple[float, float, float]:
    if transform == "payload_zero":
        return stat(data, schedule, "field", metric)
    if transform == "visibility_zero":
        # The field summary averages payload and visibility; use the direct
        # transform summary for the visibility control.
        value = data["summaries"][f"{schedule}_live_cross_visibility_zero"][metric]
        return 100 * value["mean"], 100 * value["ci95_lower"], 100 * value["ci95_upper"]
    return stat(data, schedule, transform, metric)


def symbol_sender_stats(data: dict, schedule: str):
    rows = [row for policy in data["policies"]
            if policy["schedule"] == schedule and policy["trained_channel"] == "live"
            for row in policy["rows"]]
    result = []
    for sender in range(3):
        per_seed = []
        for seed in range(66701, 66717):
            values = [
                row["natural_minus_intervened"]["q_rate"]
                for row in rows
                if row["seed"] == seed and row["sender"] == sender and row["transform"].startswith("perm_")
            ]
            require(len(values) == 6, "Expected six symbol permutations per sender and seed")
            per_seed.append(float(np.mean(values)))
        x = np.asarray(per_seed, dtype=np.float64)
        mean = float(x.mean())
        half = T15 * float(x.std(ddof=1)) / np.sqrt(len(x))
        result.append((100 * mean, 100 * (mean - half), 100 * (mean + half)))
    return result


def plot(source: str, output: str) -> dict:
    source_path = Path(source).resolve()
    output_path = Path(output).resolve()
    require(not output_path.exists(), "Refuse to overwrite figure directory")
    result_path = source_path / "execution" / "results.json"
    data = json.loads(result_path.read_text(encoding="utf8"))
    require(data["status"] == "completed_json_only_recode_mask_probe", "Unexpected probe result status")
    output_path.mkdir(parents=True)

    # Summary by intervention family.  Symbol and position families are
    # aggregated across their fixed transformations; field is split into its
    # payload and visibility controls.
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.0), sharex="col")
    x = np.arange(len(LABELS))
    for row_index, (metric, ylabel) in enumerate(METRICS):
        for col, schedule in enumerate(("static", "rematched")):
            ax = axes[row_index, col]
            means, lows, highs = [], [], []
            for transform in TRANSFORMS[:2]:
                mean, low, high = stat(data, schedule, transform, metric)
                means.append(mean); lows.append(low); highs.append(high)
            for transform in ("payload_zero", "visibility_zero"):
                mean, low, high = direct_stat(data, schedule, transform, metric)
                means.append(mean); lows.append(low); highs.append(high)
            means = np.asarray(means); lows = np.asarray(lows); highs = np.asarray(highs)
            yerr = np.vstack((means - lows, highs - means))
            ax.bar(x, means, color=("#2f6f9f", "#2f6f9f", "#c97b29", "#8a6bbd"),
                   yerr=yerr, capsize=3, edgecolor="white", linewidth=.5)
            ax.axhline(0, color="black", linewidth=.8)
            ax.set_ylabel(ylabel)
            ax.set_title(f"{schedule} schedule — FI-live")
            ax.grid(axis="y", alpha=.25)
            if row_index == len(METRICS) - 1:
                ax.set_xticks(x, LABELS)
            else:
                ax.set_xticks(x, [""] * len(x))
    fig.suptitle("FI post-hoc recode/mask sensitivity\n(natural − intervention, percentage points; 16 paired seeds)",
                 y=.995, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .965))
    summary_png = output_path / "recode_mask_effects.png"
    summary_pdf = output_path / "recode_mask_effects.pdf"
    fig.savefig(summary_png, dpi=180)
    fig.savefig(summary_pdf)
    plt.close(fig)

    # Sender-specific symbol sensitivity.  This is descriptive: each point is
    # the six-permutation mean within a seed, followed by a t15 interval.
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)
    sender_labels = ("A", "B", "C")
    for ax, schedule in zip(axes, ("static", "rematched")):
        values = symbol_sender_stats(data, schedule)
        means = np.asarray([v[0] for v in values]); lows = np.asarray([v[1] for v in values]); highs = np.asarray([v[2] for v in values])
        yerr = np.vstack((means - lows, highs - means))
        ax.bar(np.arange(3), means, yerr=yerr, capsize=4, color="#2f6f9f")
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_title(f"{schedule} schedule")
        ax.set_xticks(np.arange(3), sender_labels)
        ax.set_xlabel("selected sender")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("team Q loss (pp)\n(symbol-bijection mean)")
    fig.suptitle("Sender-specific sensitivity to six symbol bijections\n(natural − intervention, percentage points)", y=.99, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    sender_png = output_path / "sender_symbol_effects.png"
    sender_pdf = output_path / "sender_symbol_effects.pdf"
    fig.savefig(sender_png, dpi=180)
    fig.savefig(sender_pdf)
    plt.close(fig)

    receipt = {
        "status": "generated",
        "source": str(source_path),
        "source_results_sha256": sha(result_path),
        "png": [str(summary_png), str(sender_png)],
        "pdf": [str(summary_pdf), str(sender_pdf)],
        "visual_review": "pending",
    }
    (output_path / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt, ensure_ascii=False))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.source, args.output)
