"""Render v0.36 iterated replacement outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "redesign_v0.9/.analysis_deps"))
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
from matplotlib import pyplot as plt


COLORS = {"fixed_A": "#527A9E", "rotating_AB": "#2A8C82", "random_ABC": "#8C6BB1"}
LABELS = {"fixed_A": "固定 A", "rotating_AB": "轮换 A/B", "random_ABC": "随机 A/B/C"}
SCHEDULE_COLORS = {"A": "#527A9E", "B": "#C47A3A", "C": "#8C6BB1"}


def sha(path: Path | str): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path: Path): return json.loads(path.read_text())
def write(path: Path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
def pct(x): return 100.0 * float(x)


def setup():
    try: font = font_manager.findfont("PingFang SC", fallback_to_default=False)
    except Exception: font = font_manager.findfont("DejaVu Sans")
    plt.rcParams.update({"font.family": ["PingFang SC", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})
    return font


def save(fig, directory: Path, stem: str):
    values = {}
    for ext in ("png", "pdf"):
        path = directory / f"{stem}.{ext}"; fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.14); values[path.name] = sha(path)
    plt.close(fig); return values


def style(ax, title, ylabel="目标 J（%）"):
    ax.set_title(title, loc="left", pad=8); ax.set_ylabel(ylabel); ax.set_ylim(0, 65); ax.set_yticks(np.arange(0, 66, 10)); ax.grid(axis="y", color="#C9CED3", alpha=.58, lw=.7); ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); out = args.out.resolve(); font = setup()
    analysis = read(out / "iterated_analysis.json"); validation = read(out / "iterated_raw_validation.json"); audit = read(out / "iterated_audit.json")
    if not (analysis["formal"] and analysis["status"] == "complete" and validation["passed"] and audit["passed"]): raise ValueError("formal iterated outputs required")
    directory = out / "figures"; directory.mkdir(exist_ok=True); evidence = {"target_curves": {}, "drift_curves": {}, "final": {}, "final_held": {}}
    generations = np.asarray(analysis["generations"], dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(15.3, 8.6)); fig.subplots_adjust(left=.06, right=.985, top=.82, bottom=.16, hspace=.50, wspace=.28)
    fig.suptitle("v0.36：有限社会学习瓶颈下的代际传递与协议漂移", fontsize=17, y=.98); fig.text(.5, .943, "每代重置一个身份的通信模块并训练 300 次；初始四主体群体在所有条件中相同。", ha="center", fontsize=10.5)
    for col, schedule in enumerate(("A", "B", "C")):
        ax = axes[0, col]
        for condition in analysis["conditions"]:
            means = np.asarray([analysis["summary"][condition][str(g)][schedule]["target_J"]["mean"] for g in generations]) * 100
            sds = np.asarray([analysis["summary"][condition][str(g)][schedule]["target_J"]["sd"] for g in generations]) * 100
            ax.plot(generations, means, marker="o", ms=3.8, lw=2.1, color=COLORS[condition], label=LABELS[condition]); ax.fill_between(generations, np.maximum(0, means-sds), np.minimum(65, means+sds), color=COLORS[condition], alpha=.10, linewidth=0)
            evidence["target_curves"].setdefault(schedule, {})[condition] = {"mean_percent": means.tolist(), "sd_percent": sds.tolist()}
        style(ax, f"{chr(65+col)}  评估拓扑 {schedule}"); ax.set_xlabel("替换代次（0=共同初始群体）"); ax.set_xlim(0, 8); ax.set_xticks(generations)
        if col == 0: ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 0]
    for condition in analysis["conditions"]:
        means = np.asarray([analysis["summary"][condition][str(g)]["A"]["token_change_prev"]["mean"] for g in range(1, 9)]) * 100
        sds = np.asarray([analysis["summary"][condition][str(g)]["A"]["token_change_prev"]["sd"] for g in range(1, 9)]) * 100
        ax.plot(np.arange(1, 9), means, marker="o", ms=3.8, lw=2.1, color=COLORS[condition], label=LABELS[condition]); ax.fill_between(np.arange(1, 9), np.maximum(0, means-sds), np.minimum(65, means+sds), color=COLORS[condition], alpha=.10, linewidth=0); evidence["drift_curves"][condition] = {"mean_percent": means.tolist(), "sd_percent": sds.tolist()}
    style(ax, "D  A 协议的相邻代 token 变化", "变化率（%）"); ax.set_xlabel("新一代完成后的替换次数"); ax.set_xlim(1, 8); ax.set_xticks(np.arange(1, 9)); ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax = axes[1, 1]; x = np.arange(3); width = .25
    for j, condition in enumerate(analysis["conditions"]):
        vals = [pct(analysis["summary"][condition]["8"][s]["target_J"]["mean"]) for s in ("A", "B", "C")]; err = [pct(analysis["summary"][condition]["8"][s]["target_J"]["sd"]) for s in ("A", "B", "C")]; ax.bar(x + (j - 1) * width, vals, width, yerr=err, capsize=3, color=COLORS[condition], alpha=.86, label=LABELS[condition]); evidence["final"][condition] = {"mean_percent": vals, "sd_percent": err}
    style(ax, "E  第 8 代的三拓扑终点"); ax.set_xticks(x, ["A", "B", "C"]); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 2]; x = np.arange(3)
    for j, schedule in enumerate(("A", "B", "C")):
        vals = [pct(analysis["summary"][condition]["8"][schedule]["held_J"]["mean"]) for condition in analysis["conditions"]]; err = [pct(analysis["summary"][condition]["8"][schedule]["held_J"]["sd"]) for condition in analysis["conditions"]]; ax.bar(x + (j - 1) * width, vals, width, yerr=err, capsize=3, color=SCHEDULE_COLORS[schedule], alpha=.86, label=f"拓扑 {schedule}"); evidence["final_held"][schedule] = {"mean_percent": vals, "sd_percent": err}
    style(ax, "F  第 8 代 held18 泛化"); ax.set_xticks(x, [LABELS[c] for c in analysis["conditions"]], rotation=12, ha="right"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(.5, .075, "固定 A 的代际传递保持并增强 A；轮换和随机日程逐渐扩大 B/C 的可读范围，但伴随更高的早期漂移。", ha="center", fontsize=10.2); fig.text(.5, .04, "阴影和误差线为跨 12 条 seed×panel 链的标准差；token 变化率是表面输出诊断，不是语义距离。", ha="center", fontsize=10.2, color="#43505C")
    figures = save(fig, directory, "01_iterated_outcomes")
    source = {"status": "rendered_pending_visual_QA", "formal": True, "plot_source_sha256": sha(Path(__file__)), "analysis_sha256": sha(out / "iterated_analysis.json"), "raw_validation_sha256": sha(out / "iterated_raw_validation.json"), "audit_sha256": sha(out / "iterated_audit.json"), "font_path": font, "figure_sha256": figures, "plotted_values": evidence, "axes_fixed_before_results": True, "source_and_condition_selection": False}
    write(directory / "figure_source.json", source); print(json.dumps({"status": source["status"], "directory": str(directory), "files": list(figures)}, ensure_ascii=False))


if __name__ == "__main__": main()
