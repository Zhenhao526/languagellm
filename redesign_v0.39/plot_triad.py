"""Render v0.39 three-resource composition outcomes."""
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
def pct(x): return 100 * float(x)


def setup():
    try: font = font_manager.findfont("PingFang SC", fallback_to_default=False)
    except Exception: font = font_manager.findfont("DejaVu Sans")
    plt.rcParams.update({"font.family": ["PingFang SC", "DejaVu Sans"], "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.unicode_minus": False, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white", "axes.spines.top": False, "axes.spines.right": False})
    return font


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); out = args.out.resolve(); font = setup(); d = read(out / "triad_analysis.json"); v = read(out / "triad_raw_validation.json"); a = read(out / "triad_audit.json")
    if not (d["formal"] and d["status"] == "complete" and v["passed"] and a["passed"]): raise ValueError("formal triad outputs required")
    directory = out / "figures"; directory.mkdir(exist_ok=True); conditions = d["conditions"]; schedules = d["schedules"]; checkpoints = d["checkpoints"]; evidence = {"curves": {}, "final": {}, "agreement": {}}
    fig, axes = plt.subplots(2, 3, figsize=(15.3, 8.6)); fig.subplots_adjust(left=.06, right=.985, top=.82, bottom=.16, hspace=.50, wspace=.28); fig.suptitle("v0.39：三资源三 token 协议的形成与拓扑覆盖", fontsize=17, y=.98); fig.text(.5, .943, "每个资源由一个私有 sender 观察并发一个 token；receiver 只读三 token 组合。", ha="center", fontsize=10.5)
    for col, schedule in enumerate(schedules):
        ax = axes[0, col]
        for condition in conditions:
            x = d["summary"][condition][schedule]; means = np.asarray([x[str(u)]["target_J"]["mean"] for u in checkpoints]) * 100; sds = np.asarray([x[str(u)]["target_J"]["sd"] for u in checkpoints]) * 100; ax.plot(checkpoints, means, marker="o", ms=3.8, lw=2.1, color=COLORS[condition], label=LABELS[condition]); ax.fill_between(checkpoints, np.maximum(0, means-sds), np.minimum(100, means+sds), color=COLORS[condition], alpha=.10, linewidth=0); evidence["curves"].setdefault(schedule, {})[condition] = {"mean_percent": means.tolist(), "sd_percent": sds.tolist()}
        ax.set_title(f"{chr(65+col)}  评估拓扑 {schedule}", loc="left", pad=8); ax.set_ylabel("联合 J（%）"); ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20)); ax.grid(axis="y", color="#C9CED3", alpha=.58, lw=.7); ax.set_axisbelow(True); ax.set_xlabel("更新步"); ax.set_xlim(0, 1200); ax.set_xticks(checkpoints)
        if col == 0: ax.legend(frameon=False, fontsize=8, loc="upper left")
    x = np.arange(len(conditions)); width = .25; ax = axes[1, 0]
    for j, schedule in enumerate(schedules):
        vals = [pct(d["summary"][condition][schedule]["1200"]["target_J"]["mean"]) for condition in conditions]; err = [pct(d["summary"][condition][schedule]["1200"]["target_J"]["sd"]) for condition in conditions]; ax.bar(x + (j - 1) * width, vals, width, yerr=err, capsize=3, color=SCHEDULE_COLORS[schedule], alpha=.86, label=f"评估拓扑 {schedule}"); evidence["final"].setdefault(schedule, {"mean_percent": vals, "sd_percent": err})
    ax.set_title("D  第 1200 步联合成功率", loc="left", pad=8); ax.set_ylabel("联合 J（%）"); ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20)); ax.set_xticks(x, [LABELS[c] for c in conditions], rotation=12, ha="right"); ax.grid(axis="y", color="#C9CED3", alpha=.58, lw=.7); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 1]; resource_names = ["资源 0", "资源 1", "资源 2", "三 token"]; x_ag = np.arange(len(resource_names))
    for j, condition in enumerate(conditions):
        vals = [pct(d["summary"][condition]["A"]["1200"][f"agreement_resource{k}"]["mean"]) for k in range(3)] + [pct(d["summary"][condition]["A"]["1200"]["agreement_joint"]["mean"])]; err = [pct(d["summary"][condition]["A"]["1200"][f"agreement_resource{k}"]["sd"]) for k in range(3)] + [pct(d["summary"][condition]["A"]["1200"]["agreement_joint"]["sd"])]; ax.bar(x_ag + (j - 1) * width, vals, width, yerr=err, capsize=3, color=COLORS[condition], alpha=.86, label=LABELS[condition]); evidence["agreement"][condition] = {"mean_percent": vals, "sd_percent": err}
    ax.set_title("E  A 拓扑下的同型 token 一致率", loc="left", pad=8); ax.set_ylabel("一致率（%）"); ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20)); ax.set_xticks(x_ag, resource_names, rotation=12, ha="right"); ax.grid(axis="y", color="#C9CED3", alpha=.58, lw=.7); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1, 2];
    for j, condition in enumerate(conditions):
        vals = [pct(d["summary"][condition]["A"]["1200"][f"resource{k}"]["mean"]) for k in range(3)]; err = [pct(d["summary"][condition]["A"]["1200"][f"resource{k}"]["sd"]) for k in range(3)]; ax.bar(x + (j - 1) * width, vals, width, yerr=err, capsize=3, color=COLORS[condition], alpha=.86, label=LABELS[condition])
    ax.set_title("F  第 1200 步分资源正确率", loc="left", pad=8); ax.set_ylabel("正确率（%）"); ax.set_ylim(0, 100); ax.set_yticks(np.arange(0, 101, 20)); ax.set_xticks(x, ["资源 0", "资源 1", "资源 2"]); ax.legend(frameon=False, fontsize=8, loc="upper left"); ax.grid(axis="y", color="#C9CED3", alpha=.58, lw=.7); ax.set_axisbelow(True)
    fig.text(.5, .075, "三 token 组合把双资源实验中的互补角色扩展为三个资源槽；联合 J 与单资源一致率同时报告。", ha="center", fontsize=10.2); fig.text(.5, .04, "阴影和误差线为跨 12 条 seed×panel 运行的标准差；一致率是离散输出诊断。", ha="center", fontsize=10.2, color="#43505C")
    figures = {}
    for ext in ("png", "pdf"):
        path = directory / f"01_triad_outcomes.{ext}"; fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.14); figures[path.name] = sha(path)
    plt.close(fig); source = {"status": "rendered_pending_visual_QA", "formal": True, "plot_source_sha256": sha(Path(__file__)), "analysis_sha256": sha(out / "triad_analysis.json"), "raw_validation_sha256": sha(out / "triad_raw_validation.json"), "audit_sha256": sha(out / "triad_audit.json"), "font_path": font, "figure_sha256": figures, "plotted_values": evidence, "axes_fixed_before_results": True, "source_and_condition_selection": False}; (directory / "figure_source.json").write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n"); print(json.dumps({"status": source["status"], "directory": str(directory), "files": list(figures)}, ensure_ascii=False))


if __name__ == "__main__": main()
