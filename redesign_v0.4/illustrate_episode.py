"""Illustrate one real held-out pilot trial without assigning message meanings.

The figure is an explicitly selected successful example, not an aggregate
result. No experiment is launched, and smoke-test traces are rejected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
SELECTION_RULE = (
    "First successful trial in trace order with one homogeneous local resource "
    "set and one mixed set, and a nonblank message from the homogeneous-side "
    "agent. No message meaning or causal effect is inferred from this trial."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_example(trace: Path) -> dict:
    if "smoke" in str(trace).lower():
        raise ValueError("Smoke-test traces cannot be used for this pilot illustration")
    if not trace.parent.name.startswith("communicate_s"):
        raise ValueError("Expected a communicate_sSEED held-out trace")
    with trace.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            kinds = np.asarray(row["kinds"])
            actions = np.asarray(row["actions"])
            sent = np.asarray(row["sent"])
            delivered = np.asarray(row["delivered"])
            for episode in range(len(row["reward"])):
                local = kinds[episode]
                homogeneous = local[:, 0] == local[:, 1]
                if homogeneous.sum() != 1 or row["reward"][episode] != 1:
                    continue
                forced_agent = int(np.flatnonzero(homogeneous)[0])
                if sent[episode, forced_agent] == 0:
                    continue
                if not np.array_equal(sent[episode], delivered[episode]):
                    raise ValueError("Expected normal delivery, not shuffled messages")
                chosen = local[np.arange(2), actions[episode]]
                if set(chosen.tolist()) != {0, 1}:
                    raise ValueError("Recorded successful reward contradicts the selected resources")
                inventory = row["inventory"][episode]
                next_inventory = row["next_inventory"][episode]
                if inventory != [0, 0] or next_inventory != [0, 0]:
                    raise ValueError("This figure describes the unbuffered capacity-one condition")
                return {
                    "trace_line": line_number,
                    "episode_index_zero_based": episode,
                    "step_index_zero_based": row["step"],
                    "inventory": inventory,
                    "next_inventory": next_inventory,
                    "kinds": local.tolist(),
                    "image_indices": row["image_ids"][episode],
                    "sent": sent[episode].tolist(),
                    "delivered": delivered[episode].tolist(),
                    "actions": actions[episode].tolist(),
                    "selected_kinds": chosen.tolist(),
                    "reward": row["reward"][episode],
                }
    raise ValueError("No successful trial matches the declared illustration-selection rule")


def font_setup() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = [name for name in ("PingFang SC", "Heiti SC", "Arial Unicode MS") if name in available]
    if not preferred:
        raise RuntimeError("A Chinese-capable matplotlib font is required")
    plt.rcParams.update({"font.family": preferred + ["DejaVu Sans"], "axes.unicode_minus": False})


def render(example: dict, entries: list[dict], output: Path, run_name: str) -> None:
    font_setup()
    fig = plt.figure(figsize=(15, 8.5), facecolor="#ffffff")
    dark, muted, green = "#173042", "#586976", "#16836A"
    fig.text(.05, .947, "两个主体如何完成同一天的采集", fontsize=23, color=dark, weight="bold")
    fig.text(
        .05, .906,
        f"真实留出照片试次  ·  {run_name}  ·  第 {example['episode_index_zero_based'] + 1} 个回合 / 第 {example['step_index_zero_based'] + 1} 天",
        fontsize=11, color=muted,
    )
    fig.text(.05, .863, "公共状态：食物 0，水 0；当天需食物 1、水 1。两人先同时发信号，再各选一张图片采集。", fontsize=12, color=dark)

    photo_positions = ((.062, .276), (.532, .746))
    credits = []
    for who, name in enumerate(("A", "B")):
        left = .044 + who * .470
        fig.add_artist(FancyBboxPatch(
            (left, .300), .438, .517,
            boxstyle="round,pad=0.007,rounding_size=0.012",
            transform=fig.transFigure, facecolor="#F3F6F8", edgecolor="#D8E1E6", zorder=0,
        ))
        fig.text(left + .020, .775, f"主体 {name} 的私人观察", fontsize=17, color=dark, weight="bold")
        fig.text(left + .020, .737, "只看到自己的两张照片", fontsize=11, color=muted)
        for option in range(2):
            entry = entries[example["image_indices"][who][option]]
            path = ROOT / entry["path"]
            selected = example["actions"][who] == option
            axis = fig.add_axes([photo_positions[who][option], .448, .192, .255])
            with Image.open(path) as picture:
                axis.imshow(picture.convert("RGB"))
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)
            axis.add_patch(Rectangle((0, 0), 1, 1, transform=axis.transAxes,
                                     fill=False, linewidth=5 if selected else 1.2,
                                     edgecolor=green if selected else "#CAD6DD", clip_on=False))
            fig.text(photo_positions[who][option] + .096, .411,
                     f"选项 {option}  ·  {'实际选择' if selected else '未选择'}",
                     ha="center", fontsize=12, color=green if selected else muted,
                     weight="bold" if selected else "normal")
            credits.append({"agent": name, "option_index": option, "selected": selected,
                            "id": entry["id"], "title": entry["title"], "author": entry["author"],
                            "license": entry["license"], "source_url": entry["source_url"],
                            "license_url": entry["license_url"], "path": str(path)})
        other = "B" if who == 0 else "A"
        own_sent = example["sent"][who]
        received = example["delivered"][1 - who]
        fig.text(left + .020, .357,
                 f"发给 {other}：编号 {own_sent}      收到 {other}：编号 {received}",
                 fontsize=13, color=dark)
        fig.text(left + .020, .319, "编号仅表示离散信号，不为它指定词义。", fontsize=10.5, color=muted)

    selected_names = ["食物" if kind == 0 else "水" for kind in example["selected_kinds"]]
    fig.add_artist(FancyBboxPatch(
        (.049, .172), .905, .086,
        boxstyle="round,pad=0.007,rounding_size=0.010",
        transform=fig.transFigure, facecolor="#E8F5EF", edgecolor="#C0DDCF", zorder=0,
    ))
    fig.text(.068, .213,
             f"实际结果：A 采到{selected_names[0]}，B 采到{selected_names[1]}；两种需求均满足，奖励 = {example['reward']:g}。",
             fontsize=15, color=green, weight="bold", va="center")
    fig.text(.05, .129, "此图从成功试次中按预定规则挑选，用于解释任务，不能代替总体统计或消息干预证据。", fontsize=11, color=muted)
    fig.text(.05, .096, "文字与绿色边框均为事后标注；主体接收自己的图像特征、公共状态及伙伴的离散信号。", fontsize=10.5, color=muted)
    fig.text(.05, .063, "照片来源与各自许可详见 data/ATTRIBUTION.md；本图所用图片及试次位置记录于同名 JSON。", fontsize=10.5, color=muted)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    example["photo_credits"] = credits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=ROOT / "results/pilot_001/communicate_s101/heldout_trace.jsonl")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/pilot_001/task_example.png")
    args = parser.parse_args()
    if not args.trace.is_file():
        parser.error(f"Formal pilot trace is not available yet: {args.trace}")
    entries = json.loads(args.manifest.read_text())["images"]
    example = choose_example(args.trace)
    for who in range(2):
        for option in range(2):
            entry = entries[example["image_indices"][who][option]]
            expected = "food" if example["kinds"][who][option] == 0 else "water"
            if entry["category"] != expected or entry["split"] != "test":
                raise ValueError("Trace/manifest mismatch or non-held-out photograph")
            if sha256(ROOT / entry["path"]) != entry["sha256"]:
                raise ValueError("Photo bytes differ from the recorded manifest")
    example.update({"source_trace": str(args.trace.resolve()), "source_trace_sha256": sha256(args.trace),
                    "source_manifest": str(args.manifest.resolve()), "source_manifest_sha256": sha256(args.manifest),
                    "selection_rule": SELECTION_RULE,
                    "interpretation": "Selected successful illustration only; no symbol meanings or causal effects assigned.",
                    "attribution": str(ROOT / "data/ATTRIBUTION.md")})
    render(example, entries, args.output, f"{args.trace.parent.parent.name}/{args.trace.parent.name}")
    args.output.with_suffix(".json").write_text(json.dumps(example, ensure_ascii=False, indent=2))
    print(json.dumps({"figure": str(args.output.resolve()), "provenance": str(args.output.with_suffix('.json').resolve()),
                      "episode_index_zero_based": example["episode_index_zero_based"],
                      "step_index_zero_based": example["step_index_zero_based"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
