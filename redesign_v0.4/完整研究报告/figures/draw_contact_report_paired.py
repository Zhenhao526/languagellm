#!/usr/bin/env python3
"""Report-only figure of matched old-contact trajectories; no training or reanalysis writes.

Run from the project root with:
    .venv/bin/python redesign_v0.4/完整研究报告/figures/draw_contact_report_paired.py

All plotted outcomes come from 512-case checkpoint evaluations. The separate
4096-case terminal evaluations are validated and documented, but never plotted.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
V04 = HERE.parents[1]
SOURCE = V04 / "results" / "contact_001"
SEEDS = (101, 202, 303)
UPDATES = (0, 1, 5, 10, 20, 50, 100, 200, 400, 600)
CONDITIONS = {"FF": "fixed_to_fixed", "FR": "fixed_to_rotating"}
GROUPS = ("original", "cross")
COLORS = ("#3776AB", "#D48230", "#9572A9")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def group_value(record: dict, group: str, mode: str) -> float:
    pairs = [pair for pair in record["evaluation"]["pairs"] if pair["group"] == group]
    assert len(pairs) == (2 if group == "original" else 4)
    return float(np.mean([pair["tasks"]["full"][mode]["mean_reward_per_step"]
                          for pair in pairs]))


def load_paired_data() -> tuple[dict, dict]:
    data = {}
    audit = {
        "source": str(SOURCE),
        "contrast": "FR minus FF, same fixed-partner social-training checkpoint",
        "units": "percentage points",
        "group_weighting": "equal weight per pair within group, then equal weight per seed",
        "updates": UPDATES,
        "checkpoint_cases_per_pair_per_mode": 512,
        "separate_terminal_cases_per_pair_per_mode": 4096,
        "terminal_evaluation_used_in_plot": False,
        "curves": {},
    }
    for seed in SEEDS:
        curves = {}
        seed_audit = {"conditions": {}}
        for short, condition in CONDITIONS.items():
            directory = SOURCE / f"{condition}_s{seed}"
            curve_path = directory / "learning_curve.json"
            curves[short] = read_json(curve_path)
            assert tuple(record["update"] for record in curves[short]) == UPDATES
            terminal = read_json(directory / "result.json")["evaluation"]
            assert terminal["cases_per_pair_task_mode"] == 4096
            seed_audit["conditions"][short] = {
                "learning_curve_file": str(curve_path.relative_to(V04)),
                "learning_curve_sha256": sha256(curve_path),
                "checkpoint_0000_sha256": sha256(directory / "checkpoint_0000.pt"),
                "checkpoint_master_seed": curves[short][0]["evaluation"]["master_seed"],
                "terminal_master_seed": terminal["master_seed"],
            }
        assert (seed_audit["conditions"]["FF"]["checkpoint_0000_sha256"] ==
                seed_audit["conditions"]["FR"]["checkpoint_0000_sha256"])
        for ff, fr in zip(curves["FF"], curves["FR"]):
            assert ff["evaluation"]["master_seed"] == fr["evaluation"]["master_seed"]
            for record in (ff, fr):
                assert record["evaluation"]["cases_per_pair_task_mode"] == 512
                assert record["evaluation"]["evaluation"] == "checkpoint"
            ff_pairs = {tuple(pair["agents"]): pair for pair in ff["evaluation"]["pairs"]}
            fr_pairs = {tuple(pair["agents"]): pair for pair in fr["evaluation"]["pairs"]}
            assert ff_pairs.keys() == fr_pairs.keys()
            for pair in ff_pairs:
                hashes = []
                for condition_pairs in (ff_pairs, fr_pairs):
                    for mode in ("normal", "shuffle"):
                        outcome = condition_pairs[pair]["tasks"]["full"][mode]
                        assert outcome["cases"] == 512
                        hashes.append(outcome["external_cases_sha256"])
                assert len(set(hashes)) == 1
        data[seed] = {}
        for group in GROUPS:
            normal = []
            gap = []
            for ff, fr in zip(curves["FF"], curves["FR"]):
                ff_normal, fr_normal = [group_value(record, group, "normal") for record in (ff, fr)]
                ff_shuffle, fr_shuffle = [group_value(record, group, "shuffle") for record in (ff, fr)]
                normal.append(100 * (fr_normal - ff_normal))
                gap.append(100 * ((fr_normal - fr_shuffle) - (ff_normal - ff_shuffle)))
            assert normal[0] == 0 and gap[0] == 0
            data[seed][group] = {"normal": normal, "normal_minus_shuffle": gap}
        seed_audit["same_checkpoint_and_case_hashes_verified"] = True
        seed_audit["differences_in_percentage_points"] = data[seed]
        audit["curves"][str(seed)] = seed_audit
    return data, audit


def draw(data: dict, audit: dict) -> None:
    plt.rcParams.update({
        "font.family": ["PingFang SC", "DejaVu Sans"],
        "font.size": 12.5,
        "axes.titlesize": 13,
        "axes.labelsize": 12.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(8.35, 6.8), sharex=True, sharey="row")
    fig.subplots_adjust(left=.145, right=.978, bottom=.248, top=.912, wspace=.13, hspace=.22)
    labels = ("原搭档 AB、CD", "交叉配对 AC、AD、BC、BD")
    metrics = ("normal", "normal_minus_shuffle")
    panel_letters = (("a", "b"), ("c", "d"))
    mean_values = {}
    for col, group in enumerate(GROUPS):
        mean_values[group] = {}
        for row, metric in enumerate(metrics):
            ax = axes[row, col]
            arrays = []
            for seed, color in zip(SEEDS, COLORS):
                y = np.asarray(data[seed][group][metric])
                arrays.append(y)
                ax.plot(UPDATES, y, color=color, linewidth=1.4, marker="o",
                        markersize=3.6, markeredgewidth=.4, markeredgecolor="white",
                        alpha=.88, label=f"群体 {seed}", zorder=3)
            mean = np.mean(arrays, axis=0)
            mean_values[group][metric] = mean.tolist()
            ax.plot(UPDATES, mean, color="#17222D", linewidth=2.35, label="3 群体均值", zorder=4)
            ax.axhline(0, color="#77828B", linewidth=.9, linestyle=(0, (4, 3)), zorder=1)
            ax.grid(axis="y", color="#DDE2E6", linewidth=.6, zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines[["bottom", "left"]].set_color("#86929B")
            ax.tick_params(length=3, width=.7, color="#86929B")
            ax.set_xlim(-10, 615)
            ax.set_xticks([0, 200, 400, 600])
            ax.text(.025, .945, panel_letters[row][col], transform=ax.transAxes,
                    va="top", ha="left", fontsize=12, fontweight="bold")
            if row == 0:
                ax.set_title(labels[col], pad=10)
                ax.set_ylim(-12.5, 35)
                ax.set_yticks([-10, 0, 10, 20, 30])
            else:
                ax.set_ylim(-14, 30)
                ax.set_yticks([-10, 0, 10, 20, 30])
                ax.set_xlabel("继续训练的更新次数", labelpad=7)
    axes[0, 0].set_ylabel("成功率处理差\n（百分点）", labelpad=9)
    axes[1, 0].set_ylabel("消息打乱落差的处理差\n（百分点）", labelpad=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.53, .095),
               ncol=4, frameon=False, handlelength=1.8, columnspacing=1.25)
    fig.text(.145, .075, "处理差 = 同起点改为轮换（FR）− 继续固定（FF）", fontsize=12)
    fig.text(.145, .043, "消息打乱落差 = 正常通信成功率 − 打乱消息成功率", fontsize=12)
    fig.text(.145, .012, "过程：每配对每模式 512 案例；4096 案例终点评估另见正文表。", fontsize=11.5, color="#4D5862")
    for extension in ("png", "pdf"):
        fig.savefig(HERE / f"contact_report_paired.{extension}", dpi=300,
                    metadata={"Creator": "draw_contact_report_paired.py"} if extension == "pdf" else None)
    plt.close(fig)
    audit["three_seed_mean_differences_in_percentage_points"] = mean_values
    audit["caption"] = (
        "旧接触实验的同起点配对差：固定搭档训练形成的同一群体继续固定（FF）或改为轮换（FR）。"
        "细线与点保留 3 个群体的全部 10 个检查点，粗线为群体均值，未绘制置信区间。"
        "左列为 AB、CD 两对均值，右列为 AC、AD、BC、BD 四对均值；先在群体内对配对等权平均，"
        "再计算 FR−FF。上行为正常通信成功率之差，下行为正常减打乱消息落差的处理差。"
        "所有数据来自每配对每模式 512 个固定案例的过程评估；正文的最终表使用另行评估的 4096 案例，"
        "因此其 600 更新终点数值不必与本图完全一致。点间连线仅便于阅读，不表示未观测更新上的轨迹。"
    )
    (HERE / "contact_report_paired_data.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"outputs": [str(HERE / f"contact_report_paired.{extension}") for extension in ("png", "pdf")],
                      "paired_same_origin_validation": "passed", "plotted_evaluation_cases": 512,
                      "update_600_mean_differences_pp": {
                          group: {metric: values[-1] for metric, values in metrics.items()}
                          for group, metrics in mean_values.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    draw(*load_paired_data())
