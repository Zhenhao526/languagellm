"""Render the v0.35 newcomer-adaptation curves and paired contrasts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
import sys
sys.path.insert(0, str(PROJECT / "redesign_v0.9/.analysis_deps"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


COLORS = {"A": "#527A9E", "B": "#C47A3A", "C": "#8C6BB1"}
COND_COLORS = {"fixed_A": "#527A9E", "rotating_AB": "#2A8C82"}
COND_LABELS = {"fixed_A": "固定 A", "rotating_AB": "轮换 A/B"}
SCHEDULE_LABELS = {"A": "A（固定训练）", "B": "B（轮换训练）", "C": "C（未见）"}


def sha(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def setup():
    try:
        font = font_manager.findfont("PingFang SC", fallback_to_default=False)
    except Exception:
        font = font_manager.findfont("DejaVu Sans")
    plt.rcParams.update({
        "font.family": ["PingFang SC", "DejaVu Sans"], "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 9,
        "ytick.labelsize": 9, "axes.unicode_minus": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "savefig.facecolor": "white", "axes.spines.top": False,
        "axes.spines.right": False,
    })
    return font


def pct(x):
    return 100.0 * float(x)


def save(fig, directory: Path, stem: str):
    result = {}
    for ext in ("png", "pdf"):
        path = directory / f"{stem}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.14)
        result[path.name] = sha(path)
    plt.close(fig)
    return result


def format_axis(ax, title, ylabel="目标 J（%）"):
    ax.set_title(title, loc="left", pad=8)
    ax.set_ylim(0, 55)
    ax.set_yticks(np.arange(0, 56, 10))
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="#C9CED3", alpha=.58, lw=.7)
    ax.set_axisbelow(True)


def curve_panel(ax, analysis, condition, evidence):
    times = analysis["checkpoints"]
    for schedule in analysis["schedules"]:
        summary = analysis["summary"][condition][schedule]
        mean = np.asarray([summary["curve"][str(t)]["mean"] for t in times]) * 100
        sd = np.asarray([summary["curve"][str(t)]["sd"] for t in times]) * 100
        ax.plot(times, mean, marker="o", ms=4, lw=2.2, color=COLORS[schedule], label=SCHEDULE_LABELS[schedule])
        ax.fill_between(times, np.maximum(0, mean - sd), np.minimum(55, mean + sd), color=COLORS[schedule], alpha=.12, linewidth=0)
        evidence["curves"].setdefault(condition, {})[schedule] = {"mean_percent": mean.tolist(), "sd_percent": sd.tolist()}
    format_axis(ax, "固定居民：" + COND_LABELS[condition])
    ax.set_xlabel("新主体通信模块更新次数")
    ax.set_xlim(0, max(times))
    ax.set_xticks(times)


def endpoint_panel(ax, analysis, identity, evidence):
    schedules = analysis["schedules"]
    x = np.arange(len(schedules))
    width = .34
    for j, condition in enumerate(analysis["conditions"]):
        vals = [pct(analysis["summary"][condition][s]["final_target_J"]["mean"]) for s in schedules]
        err = [pct(analysis["summary"][condition][s]["final_target_J"]["sd"]) for s in schedules]
        ax.bar(x + (j - .5) * width, vals, width, yerr=err, capsize=3, color=COND_COLORS[condition], alpha=.86, label=COND_LABELS[condition])
        evidence["final_endpoint"][condition] = {"mean_percent": vals, "sd_percent": err}
    ax.axhline(100 / 36, color="#6D747A", lw=1, ls="--", label="均匀随机 J≈2.78%")
    format_axis(ax, "600 更新后的身份留出", "目标 J（%）")
    ax.set_xticks(x, [f"{s}：{SCHEDULE_LABELS[s].split('（')[0]}" for s in schedules])
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def auc_panel(ax, analysis, evidence):
    schedules = analysis["schedules"]
    x = np.arange(len(schedules))
    width = .34
    for j, condition in enumerate(analysis["conditions"]):
        vals = [pct(analysis["summary"][condition][s]["auc_target_J"]["mean"]) for s in schedules]
        err = [pct(analysis["summary"][condition][s]["auc_target_J"]["sd"]) for s in schedules]
        ax.bar(x + (j - .5) * width, vals, width, yerr=err, capsize=3, color=COND_COLORS[condition], alpha=.86, label=COND_LABELS[condition])
        evidence["auc"][condition] = {"mean_percent": vals, "sd_percent": err}
    format_axis(ax, "适应速度：目标 J 的 AUC", "AUC（%·更新归一化）")
    ax.set_xticks(x, schedules)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def paired_panel(ax, analysis, evidence):
    times = analysis["checkpoints"]
    for schedule in analysis["schedules"]:
        entry = analysis["paired_rotating_minus_fixed"][schedule]
        mean = np.asarray([entry[str(t)]["mean"] for t in times]) * 100
        sd = np.asarray([entry[str(t)]["sd"] for t in times]) * 100
        ax.plot(times, mean, marker="o", ms=4, lw=2.2, color=COLORS[schedule], label=schedule)
        ax.fill_between(times, mean - sd, mean + sd, color=COLORS[schedule], alpha=.12, linewidth=0)
        evidence["paired_rotating_minus_fixed"][schedule] = {"mean_percent": mean.tolist(), "sd_percent": sd.tolist()}
    ax.axhline(0, color="#4D5964", lw=.9)
    ax.set_title("轮换减固定：同 seed×panel×身份配对", loc="left", pad=8)
    ax.set_ylabel("目标 J 差（百分点）")
    ax.set_xlabel("更新次数")
    ax.set_xlim(0, max(times)); ax.set_xticks(times)
    ax.set_ylim(-25, 40); ax.set_yticks(np.arange(-20, 41, 10))
    ax.grid(axis="y", color="#C9CED3", alpha=.58, lw=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def type_panel(ax, analysis, evidence):
    types = ["0", "1"]
    schedules = analysis["schedules"]
    x = np.arange(len(schedules)); width = .34
    for j, private_type in enumerate(types):
        vals = [pct(analysis["private_type_summary"]["rotating_AB"][private_type][s]["final_target_J"]["mean"]) for s in schedules]
        ax.bar(x + (j - .5) * width, vals, width, color=("#6B8EAA", "#D59659")[j], alpha=.88, label=f"私有类型 {private_type}")
        evidence["rotating_private_type"][private_type] = vals
    format_axis(ax, "轮换条件：私有视觉类型的终点", "目标 J（%）")
    ax.set_xticks(x, schedules)
    ax.legend(frameon=False, fontsize=8, loc="upper left")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); out = args.out.resolve(); font = setup()
    analysis = read(out / "adaptation_analysis.json")
    validation = read(out / "adaptation_raw_validation.json")
    audit = read(out / "adaptation_audit.json")
    if not (analysis["formal"] and analysis["status"] == "complete" and validation["passed"] and audit["passed"]):
        raise ValueError("formal adaptation analysis, validation and audit are required")
    directory = out / "figures"; directory.mkdir(exist_ok=True)
    evidence = {"curves": {}, "final_endpoint": {}, "auc": {}, "paired_rotating_minus_fixed": {}, "rotating_private_type": {}}
    fig, axes = plt.subplots(2, 3, figsize=(15.3, 8.5))
    fig.subplots_adjust(left=.06, right=.985, top=.82, bottom=.16, hspace=.50, wspace=.28)
    fig.suptitle("v0.35：新主体能在线学会共同符号，但拓扑轮换改变了迁移—性能权衡", fontsize=17, y=.98)
    fig.text(.5, .943, "均值 ± 1 SD；固定居民来自 v0.34 终点，只有 newcomer 的通信模块更新。", ha="center", fontsize=10.5)
    curve_panel(axes[0, 0], analysis, "fixed_A", evidence)
    curve_panel(axes[0, 1], analysis, "rotating_AB", evidence)
    endpoint_panel(axes[0, 2], analysis, None, evidence)
    auc_panel(axes[1, 0], analysis, evidence)
    paired_panel(axes[1, 1], analysis, evidence)
    type_panel(axes[1, 2], analysis, evidence)
    fig.text(.5, .075, "A/B 是训练中出现的伙伴拓扑；C 从未出现。阴影为跨 24 个运行的标准差。", ha="center", fontsize=10.2)
    fig.text(.5, .04, "零样本身份留出几乎停留在随机水平；在线适应只更新通信模块，不更新私有视觉编码器。", ha="center", fontsize=10.2, color="#43505C")
    figures = save(fig, directory, "01_adaptation_outcomes")
    record = {
        "status": "rendered_pending_visual_QA", "formal": True,
        "plot_source_sha256": sha(Path(__file__)), "analysis_sha256": sha(out / "adaptation_analysis.json"),
        "raw_validation_sha256": sha(out / "adaptation_raw_validation.json"), "audit_sha256": sha(out / "adaptation_audit.json"),
        "font_path": font, "figure_sha256": figures, "plotted_values": evidence,
        "axes_fixed_before_results": True, "source_and_condition_selection": False,
    }
    write(directory / "figure_source.json", record)
    print(json.dumps({"status": record["status"], "directory": str(directory), "files": list(figures)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
