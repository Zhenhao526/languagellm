"""Pool the two independent role-stratified semantic-transfer seed blocks.

This is a descriptive secondary analysis.  It consumes only the two frozen
summary.json files and performs no model forward or optimizer update.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
T975_DF15 = 2.131449545559323


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean_ci(values: list[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1)) if n > 1 else 0.0
    half = T975_DF15 * sd / math.sqrt(n) if n > 1 else 0.0
    return {
        "mean": mean,
        "sd": sd,
        "n": n,
        "ci95_t15": [mean - half, mean + half],
        "seed_values": values,
    }


def role_values(summary: dict, role: str, schedule: str) -> list[float]:
    return summary["role_cells"][role][f"{schedule}/live"]["aligned_minus_placebo"]["plan_transfer"]["seed_values"]


def contrast_values(summary: dict, name: str) -> list[float]:
    return summary["role_contrasts"][name]["plan_transfer"]["seed_values"]


def run(first: Path, confirm: Path, out: Path) -> None:
    first = first.resolve()
    confirm = confirm.resolve()
    out = out.resolve()
    if out.exists():
        raise ValueError(f"Refuse to overwrite {out}")
    first_summary = json.loads(first.read_text())
    confirm_summary = json.loads(confirm.read_text())
    result = {
        "schema": "triadic_receiver_role_crossover_pooled_semantic_transfer_v1",
        "status": "completed_json_only_pool",
        "source_blocks": {
            "role_semantic_transfer_001": str(first),
            "role_semantic_transfer_confirm_001": str(confirm),
        },
        "no_model_calls": True,
        "optimizer_updates": 0,
        "role_cells": {},
        "role_contrasts": {},
        "interpretation_boundary": (
            "Pooled aligned-minus-placebo is a descriptive frozen task content-transfer diagnostic; "
            "it is not lexical meaning, compositionality, intergenerational transmission or human language-origin evidence."
        ),
    }
    for role in ("A", "B", "C"):
        result["role_cells"][role] = {}
        for schedule in ("static", "rematched"):
            initial = role_values(first_summary, role, schedule)
            independent = role_values(confirm_summary, role, schedule)
            result["role_cells"][role][schedule] = {
                "pooled_16_seed": mean_ci(initial + independent),
                "block_means": {
                    "role_semantic_transfer_001": sum(initial) / len(initial),
                    "role_semantic_transfer_confirm_001": sum(independent) / len(independent),
                },
                "block_delta_confirm_minus_first": sum(independent) / len(independent) - sum(initial) / len(initial),
            }
    for name in ("B_minus_A", "C_minus_A"):
        initial = contrast_values(first_summary, name)
        independent = contrast_values(confirm_summary, name)
        result["role_contrasts"][name] = {
            "pooled_16_seed": mean_ci(initial + independent),
            "block_means": {
                "role_semantic_transfer_001": sum(initial) / len(initial),
                "role_semantic_transfer_confirm_001": sum(independent) / len(independent),
            },
            "block_delta_confirm_minus_first": sum(independent) / len(independent) - sum(initial) / len(initial),
        }
    out.mkdir(parents=True)
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 两个独立源种子区块的角色分层内容转移合并",
        "",
        "这是只读取两个冻结 `summary.json` 的描述性二次分析；没有新增前向或优化更新。区间以 16 个独立源种子为统计单位，使用 t(15)。单位为百分点。",
        "",
        "## 角色×日程 pooled aligned−placebo",
        "",
        "| 角色 | static（16 seeds） | rematched（16 seeds） |",
        "|---|---:|---:|",
    ]
    for role in ("A", "B", "C"):
        vals = []
        for schedule in ("static", "rematched"):
            e = result["role_cells"][role][schedule]["pooled_16_seed"]
            vals.append(f"{e['mean']*100:+.4f} [{e['ci95_t15'][0]*100:+.4f}, {e['ci95_t15'][1]*100:+.4f}]")
        lines.append(f"| {role} | {vals[0]} | {vals[1]} |")
    lines += [
        "",
        "## 角色差值（每个 seed 先平均两个日程）",
        "",
        "| 对比 | pooled 16 seeds | 首批均值 | 确认批均值 | 确认−首批 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("B_minus_A", "C_minus_A"):
        e = result["role_contrasts"][name]["pooled_16_seed"]
        b = result["role_contrasts"][name]["block_means"]
        d = result["role_contrasts"][name]["block_delta_confirm_minus_first"]
        lines.append(
            f"| {name} | {e['mean']*100:+.4f} [{e['ci95_t15'][0]*100:+.4f}, {e['ci95_t15'][1]*100:+.4f}] | "
            f"{b['role_semantic_transfer_001']*100:+.4f} | {b['role_semantic_transfer_confirm_001']*100:+.4f} | {d*100:+.4f} pp |"
        )
    lines += [
        "",
        "C−A 在两个区块中方向一致，合并后仍低于零；B−A 的区间跨零。区块合并只作为同一架构和同一任务协议下的描述性精度提升，不把它当作新的预注册推断。",
        "",
        "[机器可读汇总](summary.json)",
    ]
    table_path = out / "汇总表.md"
    table_path.write_text("\n".join(lines) + "\n")
    receipt = {
        "status": "completed",
        "optimizer_updates": 0,
        "model_forward_samples": 0,
        "input_summary_sha256": {str(first): sha(first), str(confirm): sha(confirm)},
        "outputs": {"summary.json": sha(summary_path), "汇总表.md": sha(table_path)},
    }
    (out / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    run(Path(args.first), Path(args.confirm), Path(args.out))
