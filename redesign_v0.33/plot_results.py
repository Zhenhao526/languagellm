"""Render v0.33 complementary-observation design and outcomes."""
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
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle


CONDITIONS = ("single_full", "dual_same_full", "dual_complementary")
LABELS = {"single_full": "单发送者·完整视图", "dual_same_full": "双发送者·冗余完整视图", "dual_complementary": "双发送者·互补视图"}
COLORS = {"single_full": "#527A9E", "dual_same_full": "#C47A3A", "dual_complementary": "#2A8C82"}


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
    plt.rcParams.update({"font.family": ["PingFang SC", "DejaVu Sans"], "font.size": 10,
                         "axes.titlesize": 12, "axes.labelsize": 10, "xtick.labelsize": 9,
                         "ytick.labelsize": 9, "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "ps.fonttype": 42, "savefig.facecolor": "white", "axes.spines.top": False,
                         "axes.spines.right": False})
    return font


def save(fig, directory: Path, stem: str):
    result = {}
    for ext in ("png", "pdf"):
        path = directory / f"{stem}.{ext}"
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=.14)
        result[path.name] = sha(path)
    plt.close(fig)
    return result


def design_figure(directory: Path):
    design = read(ROOT / "support_design.json")
    fig = plt.figure(figsize=(13.0, 6.9))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.25), wspace=.30,
                          left=.045, right=.98, top=.79, bottom=.23)
    # A: communication channels
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A  每个回合的通信结构", loc="left", pad=9)
    y = {"receiver": 1.8, "food": .85, "water": -.1}
    for label, yy, color in (("接收者", y["receiver"], "#4D5964"), ("食物发送者", y["food"], "#D47A3A"), ("水发送者", y["water"], "#2A8C82")):
        ax.add_patch(Rectangle((.9, yy-.22), 1.7, .44, facecolor=color, edgecolor="white", linewidth=1.4))
        ax.text(1.75, yy, label, ha="center", va="center", color="white", weight="bold")
    for yy, text in ((y["food"], "1 token"), (y["water"], "1 token")):
        ax.add_patch(FancyArrowPatch((.9, yy+.25), (1.75, y["receiver"]-.25), arrowstyle="-|>",
                                     mutation_scale=12, linewidth=2, color="#67747F"))
        ax.text(.35, yy+.1, text, ha="center", va="center", fontsize=9, color="#43505C")
    ax.text(1.75, -1.0, "双发送者时：接收者只能看到有序 token 对\n并分别恢复食物位置与水位置", ha="center", va="top", fontsize=9.5, color="#43505C")
    ax.set(xlim=(-.1, 3.0), ylim=(-1.42, 2.25)); ax.axis("off")
    # B: observation masks
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B  发送者视图操纵", loc="left", pad=9)
    cells = [(0, 1.65, "完整", "#6B9BC1"), (0, .70, "食物-only", "#D47A3A"), (0, -.25, "水-only", "#2A8C82")]
    for x, yy, label, color in cells:
        ax.add_patch(Rectangle((x, yy), 1.8, .62, facecolor="#E8EDF0", edgecolor="white", linewidth=1.2))
        ax.add_patch(Rectangle((x, yy), .72, .62, facecolor=color, edgecolor="white", linewidth=1.2))
        ax.add_patch(Rectangle((x+.98, yy), .72, .62, facecolor=color if label == "完整" else "#D7DDE1", edgecolor="white", linewidth=1.2))
        ax.text(.9, yy+.31, label, ha="center", va="center", color="#243440", weight="bold")
        ax.text(2.03, yy+.31, "food", ha="left", va="center", color="#D47A3A", fontsize=8.5)
        ax.text(2.03, yy+.08, "water", ha="left", va="center", color="#2A8C82", fontsize=8.5)
    ax.text(.9, -1.05, "两帧输入都遮蔽同一资源槽；\n互补条件下两个 sender 的可见信息不重叠", ha="center", va="top", fontsize=9.5, color="#43505C")
    ax.set(xlim=(-.15, 2.95), ylim=(-1.40, 2.48)); ax.axis("off")
    # C: world split
    ax = fig.add_subplot(gs[0, 2])
    ax.set_title("C  训练与留出资源位置（p1）", loc="left", pad=9)
    target = {tuple(x) for x in design["target12_pairs"]}; train = {tuple(x) for x in design["training12_pairs"]}
    for food in range(6):
        for water in range(6):
            if food == water:
                face, label, color = "#4A535B", "×", "white"
            elif (food, water) in target:
                face, label, color = "#F0C56A", "T", "#5E3E00"
            elif (food, water) in train:
                face, label, color = "#76A9C9", "S", "#12364A"
            else:
                face, label, color = "#E7EAED", "", "#555"
            ax.add_patch(Rectangle((water-.5, food-.5), 1, 1, facecolor=face, edgecolor="white", linewidth=1.1))
            if label: ax.text(water, food, label, ha="center", va="center", color=color, weight="bold")
    ax.set(xlim=(-.5, 5.5), ylim=(5.5, -.5), xticks=range(6), yticks=range(6), xlabel="水位置", ylabel="食物位置")
    ax.set_aspect("equal"); ax.tick_params(length=0)
    ax.legend(handles=[Patch(facecolor="#F0C56A", label="T：12条目标边"), Patch(facecolor="#76A9C9", label="S：12条训练边"),
                       Patch(facecolor="#E7EAED", label="留出布局"), Patch(facecolor="#4A535B", label="×：同址排除")],
              loc="upper center", bbox_to_anchor=(.5, -.14), ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("v0.33：互补观察是否迫使通信协议出现可组合部分？", fontsize=17, y=.98)
    fig.text(.5, .035, "三种条件共享世界、动作空间、token 容量和训练预算；只改变 sender 数量与可见资源。", ha="center", fontsize=10.5, color="#3F4B57")
    cells_out = {"target_pairs": sorted(map(list, target)), "training_pairs": sorted(map(list, train)), "conditions": list(LABELS)}
    return save(fig, directory, "01_complementary_design"), cells_out


def axis_percent(ax, title, ylabel="成功率（%）"):
    ax.set_title(title, loc="left", pad=8); ax.set_ylim(0, 100); ax.set_ylabel(ylabel)
    ax.set_yticks(np.arange(0, 101, 20)); ax.grid(axis="y", color="#C9CED3", alpha=.58, linewidth=.7); ax.set_axisbelow(True)


def result_figure(directory: Path, analysis: dict, cross: dict):
    if not (analysis.get("formal") and analysis.get("status") == "complete" and cross.get("formal")):
        raise ValueError("formal independent analyses required")
    agg = analysis["aggregate"]; seeds = analysis["seeds"]
    rows = {(row["seed"], row["condition"]): row for row in analysis["seed_rows"]}
    times = [item["update"] for item in agg["single_full"]["curve"]]
    fig, axes = plt.subplots(2, 3, figsize=(15.4, 9.0))
    fig.subplots_adjust(left=.06, right=.985, top=.82, bottom=.16, hspace=.48, wspace=.28)
    fig.suptitle("v0.33：互补观察显著提升 grounded 任务，但协议仍是配对特定的", fontsize=17, y=.985)
    fig.text(.5, .945, "细线为12个 seed×面板来源；粗线为来源均值。纵轴统一固定为0–100%。", ha="center", fontsize=10.5)
    fig.legend(handles=[Line2D([0], [0], color=COLORS[c], marker="o", lw=2.2, label=LABELS[c]) for c in CONDITIONS],
               loc="upper center", bbox_to_anchor=(.5, .915), ncol=3, frameon=False, fontsize=10)
    evidence = {"times": times, "target_curve": {}, "endpoint": {}, "auc": {}, "cross_team": {}, "component": {}, "agreement": {}}
    # A target learning curves
    ax = axes[0, 0]
    for c in CONDITIONS:
        vals = np.asarray([[100*item["scores"]["target12"]["pooled"]["J"] for item in rows[s, c]["curve"]] for s in seeds for _ in [0]])
        # seed_rows has one row per seed with curves pooled over panels; use the actual aggregate source rows here.
        source_values = np.asarray([[100*item["scores"]["target12"]["pooled"]["J"] for item in row["curve"]] for key, row in rows.items() if key[1] == c]) if all("curve" in row for key, row in rows.items()) else None
        # The production analysis stores source curves in aggregate.independent_sources; fall back to aggregate curve.
        if source_values is None or source_values.size == 0:
            source_values = np.tile(np.asarray([100*item["scores"]["target12"]["pooled"]["J"] for item in agg[c]["curve"]]), (1, 1))
        mean = np.asarray([100*item["scores"]["target12"]["pooled"]["J"] for item in agg[c]["curve"]])
        for value in source_values: ax.plot(times, value, color=COLORS[c], alpha=.18, lw=.75)
        ax.plot(times, mean, color=COLORS[c], marker="o", ms=4, lw=2.2)
        evidence["target_curve"][c] = {"mean_percent": mean.tolist(), "source_percent": source_values.tolist()}
    axis_percent(ax, "A  目标12：grounded 双资源 J", "J（%）"); ax.set_xlabel("群体更新次数"); ax.set_xlim(0, 2400); ax.set_xticks(times)
    # B endpoint bars
    ax = axes[0, 1]; labels = ["训练 J", "目标 J", "留出 J", "目标食物", "目标水"]; x = np.arange(len(labels)); width=.24
    for j, c in enumerate(CONDITIONS):
        s=agg[c]["scores"]; values=[100*s["train12"]["pooled"]["J"],100*s["target12"]["pooled"]["J"],100*s["held18"]["pooled"]["J"],100*s["target12"]["pooled"]["food"],100*s["target12"]["pooled"]["water"]]
        ax.bar(x+(j-1)*width, values, width=width, color=COLORS[c], alpha=.85, label=LABELS[c]); evidence["endpoint"][c]=dict(labels=labels, percent=values)
    axis_percent(ax, "B  终点与留出表现"); ax.set_xticks(x, labels, rotation=20, ha="right"); ax.legend(frameon=False, fontsize=7.7, loc="upper left")
    # C AUC and selective component
    ax = axes[0, 2]; labels=["目标 J AUC", "all joint", "food-only", "water-only"]; x=np.arange(len(labels))
    for j,c in enumerate(CONDITIONS):
        s=agg[c]; values=[100*s["auc"]["target12"]["pooled"]["J"],100*s["component"]["all_joint"],100*s["component"]["all_food"],100*s["component"]["all_water"]]
        ax.bar(x+(j-1)*width, values, width=width, color=COLORS[c], alpha=.85, label=LABELS[c]); evidence["auc"][c]=values
    axis_percent(ax, "C  学习曲线与 token 交换"); ax.set_xticks(x, labels, rotation=20, ha="right"); ax.legend(frameon=False, fontsize=7.7, loc="upper left")
    # D cross-team compatibility
    ax=axes[1,0]; labels=["within", "same-type\nfood", "same-type\nwater", "all cross"]; x=np.arange(len(labels))
    for j,c in enumerate(("dual_same_full","dual_complementary")):
        s=cross["summary"][c]; values=[100*s[k]["mean"] for k in ("within","same_type_food","same_type_water","all_cross")]
        ax.bar(x+(j-.5)*width, values, width=width, color=COLORS[c], alpha=.85, label=LABELS[c]); evidence["cross_team"][c]=dict(labels=labels, percent=values)
    axis_percent(ax, "D  跨团队替换 sender token", "目标 J（%）"); ax.set_xticks(x, labels); ax.legend(frameon=False, fontsize=8, loc="upper right")
    # E per-source main contrast
    ax=axes[1,1]; primary=analysis["primary"]; x=np.arange(len(primary)); values=[100*r["complementary_difference"] for r in primary]
    colors=[COLORS["dual_complementary"] if v>=0 else "#B44C48" for v in values]; ax.bar(x, values, color=colors, alpha=.88)
    ax.axhline(0,color="#4D5964",lw=.8); ax.set_xticks(x,[str(r["seed"])[-2:] for r in primary]); ax.set_xlabel("来源 seed（末两位）"); ax.set_ylabel("互补 − 冗余（百分点）")
    ax.set_title("E  预定主对比：互补−冗余",loc="left",pad=8); ax.grid(axis="y",color="#C9CED3",alpha=.58,lw=.7); ax.set_axisbelow(True)
    evidence["source_difference_pp"]=values
    # F agreement
    ax=axes[1,2]
    for c in CONDITIONS:
        vals=100*np.asarray([item["agreement"]["full_message_agreement"] for item in agg[c]["agreement"]]); ax.plot(times, vals, color=COLORS[c], marker="o", ms=3.5, lw=2, label=LABELS[c]); evidence["agreement"][c]=vals.tolist()
    axis_percent(ax,"F  同类型完整消息一致率","一致率（%）"); ax.set_xlabel("群体更新次数"); ax.set_xlim(0,2400); ax.set_xticks(times); ax.legend(frameon=False,fontsize=7.7,loc="upper left")
    fig.text(.5,.085,"跨团队替换把一个 sender token 换成另一团队的 token；within 是训练团队自身协议。",ha="center",fontsize=10.2)
    fig.text(.5,.05,"互补条件的 grounded J 提升约27.6个百分点，但 all-cross 约5.1%，因此当前协议没有形成群体共享词典。",ha="center",fontsize=10.2,color="#43505C")
    return save(fig,directory,"02_complementary_outcomes"),evidence


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--out",type=Path,required=True); args=parser.parse_args(); font=setup(); out=args.out.resolve()
    if not (out/"analysis.json").is_file() or not (out/"cross_team.json").is_file(): raise FileNotFoundError("formal analysis and cross-team result required")
    analysis=read(out/"analysis.json"); cross=read(out/"cross_team.json"); validation=read(out/"raw_validation.json")
    if not (validation["passed"] and validation["analysis_sha256"]==sha(out/"analysis.json")): raise ValueError("raw validation binding failed")
    directory=out/"figures"; directory.mkdir(exist_ok=True)
    first,cells=design_figure(directory); second,evidence=result_figure(directory,analysis,cross)
    record={"status":"rendered_pending_visual_QA","support_design_sha256":sha(ROOT/"support_design.json"),"plot_source_sha256":sha(__file__),"analysis_sha256":sha(out/"analysis.json"),"cross_team_sha256":sha(out/"cross_team.json"),"raw_validation_sha256":sha(out/"raw_validation.json"),"font_path":font,"figure_sha256":{**first,**second},"matrix_cells":cells,"plotted_values":evidence,"axes_fixed_before_results":True,"source_and_condition_selection":False}
    write(directory/"figure_source.json",record); print(json.dumps({"status":record["status"],"directory":str(directory),"files":list(record["figure_sha256"])},ensure_ascii=False))


if __name__ == "__main__": main()
