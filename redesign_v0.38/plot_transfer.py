"""Render v0.38 formation-by-transmission transfer outcomes."""
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


COLORS = {"origin_fixed_A": "#527A9E", "origin_rotating_AB": "#2A8C82", "origin_random_ABC": "#8C6BB1"}
LABELS = {"origin_fixed_A": "形成：固定 A", "origin_rotating_AB": "形成：轮换 A/B", "origin_random_ABC": "形成：随机 A/B/C"}
TRANSMISSION_LABELS = {"fixed_A": "传递：固定 A", "rotating_AB": "传递：轮换 A/B", "random_ABC": "传递：随机 A/B/C"}
SCHEDULE_COLORS = {"A": "#527A9E", "B": "#C47A3A", "C": "#8C6BB1"}


def sha(path: Path | str): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path: Path): return json.loads(path.read_text())
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
    ax.set_title(title, loc="left", pad=8); ax.set_ylabel(ylabel); ax.set_ylim(0, 70); ax.set_yticks(np.arange(0, 71, 10)); ax.grid(axis="y", color="#C9CED3", alpha=.58, lw=.7); ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); out = args.out.resolve(); font = setup()
    analysis = read(out / "transfer_analysis.json"); validation = read(out / "transfer_raw_validation.json"); audit = read(out / "transfer_audit.json")
    if not (analysis["formal"] and analysis["status"] == "complete" and validation["passed"] and audit["passed"]): raise ValueError("formal transfer outputs required")
    directory = out / "figures"; directory.mkdir(exist_ok=True); evidence = {"curves": {}, "final_eval_A": {}, "retention": {}}
    generations = np.asarray(analysis["generations"], dtype=int); formations = analysis["formation_conditions"]; transmissions = analysis["transmission_conditions"]; schedules = analysis["schedules"]
    fig, axes = plt.subplots(2, 3, figsize=(15.3, 8.6)); fig.subplots_adjust(left=.06, right=.985, top=.82, bottom=.16, hspace=.50, wspace=.28)
    fig.suptitle("v0.38：形成拓扑 × 传递拓扑的代际可传递性", fontsize=17, y=.98); fig.text(.5, .943, "初始群体来自 v0.37 的无教师终点；每代重置一个身份并训练 300 次。", ha="center", fontsize=10.5)
    for col, transmission in enumerate(transmissions):
        ax = axes[0, col]
        for formation in formations:
            means = np.asarray([analysis["summary"][formation][transmission][str(g)]["A"]["target_J"]["mean"] for g in generations]) * 100
            sds = np.asarray([analysis["summary"][formation][transmission][str(g)]["A"]["target_J"]["sd"] for g in generations]) * 100
            ax.plot(generations, means, marker="o", ms=3.8, lw=2.1, color=COLORS[formation], label=LABELS[formation]); ax.fill_between(generations, np.maximum(0, means-sds), np.minimum(70, means+sds), color=COLORS[formation], alpha=.10, linewidth=0)
            evidence["curves"].setdefault(transmission, {})[formation] = {"mean_percent": means.tolist(), "sd_percent": sds.tolist()}
        style(ax, f"{chr(65+col)}  {TRANSMISSION_LABELS[transmission]}"); ax.set_xlabel("替换代次（0=形成终点）"); ax.set_xlim(0, int(generations[-1])); ax.set_xticks(generations)
        if col == 0: ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 0]; x = np.arange(len(formations)); width = .24
    for j, transmission in enumerate(transmissions):
        vals = [pct(analysis["summary"][formation][transmission][str(generations[-1])]["A"]["target_J"]["mean"]) for formation in formations]; err = [pct(analysis["summary"][formation][transmission][str(generations[-1])]["A"]["target_J"]["sd"]) for formation in formations]
        ax.bar(x + (j - 1) * width, vals, width, yerr=err, capsize=3, color=SCHEDULE_COLORS[schedules[j]], alpha=.86, label=TRANSMISSION_LABELS[transmission]); evidence["final_eval_A"][transmission] = {"mean_percent": vals, "sd_percent": err}
    style(ax, "D  第 4 代 A 拓扑终点"); ax.set_xlabel("协议形成日程"); ax.set_xticks(x, ["固定 A", "轮换 A/B", "随机 A/B/C"], rotation=12, ha="right"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 1]; x = np.arange(len(formations));
    for j, transmission in enumerate(transmissions):
        vals = [pct(analysis["retention"][formation][transmission]["A"]["mean"]) for formation in formations]; err = [pct(analysis["retention"][formation][transmission]["A"]["sd"]) for formation in formations]
        ax.bar(x + (j - 1) * width, vals, width, yerr=err, capsize=3, color=SCHEDULE_COLORS[schedules[j]], alpha=.86, label=TRANSMISSION_LABELS[transmission]); evidence["retention"].setdefault(transmission, {"mean_percent": vals, "sd_percent": err})
    ax.axhline(0, color="#4b5560", lw=.8); ax.set_title("E  A 拓扑的代际保留变化", loc="left", pad=8); ax.set_ylabel("第 4 代 − 形成终点（百分点）"); ax.grid(axis="y", color="#C9CED3", alpha=.58, lw=.7); ax.set_axisbelow(True); ax.set_xticks(x, ["固定 A", "轮换 A/B", "随机 A/B/C"], rotation=12, ha="right"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 2]; x = np.arange(len(schedules)); width = .24
    for j, transmission in enumerate(transmissions):
        vals = [pct(analysis["summary"][formation][transmission][str(generations[-1])][schedule]["target_J"]["mean"]) for schedule in schedules]; err = [pct(analysis["summary"][formation][transmission][str(generations[-1])][schedule]["target_J"]["sd"]) for schedule in schedules]
        ax.bar(x + (j - 1) * width, vals, width, yerr=err, capsize=3, color=COLORS[formations[j]], alpha=.86, label=TRANSMISSION_LABELS[transmission])
    style(ax, "F  第 4 代三拓扑读出", "目标 J（%）"); ax.set_xticks(x, ["A", "B", "C"]); ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.text(.5, .075, "形成日程决定初始协议的可传递性；传递日程决定这种协议在有限社会学习瓶颈下保留还是重编码。", ha="center", fontsize=10.2); fig.text(.5, .04, "阴影和误差线为跨 12 条 seed×panel 链的标准差；token 变化率是表面输出诊断。", ha="center", fontsize=10.2, color="#43505C")
    figures = save(fig, directory, "01_transfer_outcomes"); source = {"status": "rendered_pending_visual_QA", "formal": True, "plot_source_sha256": sha(Path(__file__)), "analysis_sha256": sha(out / "transfer_analysis.json"), "raw_validation_sha256": sha(out / "transfer_raw_validation.json"), "audit_sha256": sha(out / "transfer_audit.json"), "font_path": font, "figure_sha256": figures, "plotted_values": evidence, "axes_fixed_before_results": True, "source_and_condition_selection": False}; (directory / "figure_source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n"); print(json.dumps({"status": source["status"], "directory": str(directory), "files": list(figures)}, ensure_ascii=False))


if __name__ == "__main__": main()
