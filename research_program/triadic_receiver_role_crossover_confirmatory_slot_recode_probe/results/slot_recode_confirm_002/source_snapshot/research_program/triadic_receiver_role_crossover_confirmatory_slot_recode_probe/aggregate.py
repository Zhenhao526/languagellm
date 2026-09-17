"""Aggregate slot/recode variants by role and schedule."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import design

T8 = 2.3646242510102993
FIELDS = ("plan_transfer", "partner_transfer", "physical", "q", "conditional_q", "action_change")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(values):
    values = [float(value) for value in values]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    half = T8 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return dict(seed_values=values, mean=mean, sd=sd, ci95_t7=[mean - half, mean + half], n=len(values))


def group_value(group, prefix, field):
    return float(group[f"{prefix}_{field}"])


def policy_variant_balanced(row, variant):
    axes = {}
    for axis in design.AXES:
        groups = [row["groups"][f"{axis}/{sender}/{variant}"] for sender in range(3)]
        axes[axis] = {}
        for prefix in ("aligned", "placebo"):
            axes[axis][prefix] = {field: sum(group_value(group, prefix, field) for group in groups) / 3.0 for field in FIELDS}
        axes[axis]["aligned_minus_placebo"] = {field: axes[axis]["aligned"][field] - axes[axis]["placebo"][field] for field in FIELDS}
    return {
        prefix: {field: sum(axes[axis][prefix][field] for axis in design.AXES) / len(design.AXES) for field in FIELDS}
        for prefix in ("aligned", "placebo", "aligned_minus_placebo")
    }, axes


def analyze(probe, audit, out):
    probe = Path(probe).resolve(); audit = Path(audit).resolve(); out = Path(out).resolve()
    require(not out.exists(), "Refuse to overwrite summary")
    paths = [probe / "execution/results.json", probe / "execution/status.json", probe / "execution/receipt.json", probe / "prepared.json", probe / "inputs.json", probe / "plan.json", probe / "freeze.json", audit / "verification.json", audit / "receipt.json"]
    require(all(path.is_file() for path in paths), "Missing aggregate input")
    result = json.loads((probe / "execution/results.json").read_text(encoding="utf8"))
    status = json.loads((probe / "execution/status.json").read_text(encoding="utf8"))
    receipt = json.loads((probe / "execution/receipt.json").read_text(encoding="utf8"))
    verification = json.loads((audit / "verification.json").read_text(encoding="utf8"))
    static = json.loads((probe / "prepared.json").read_text(encoding="utf8"))
    require(result.get("status") == "completed_role_crossover_slot_recode_probe" and len(result.get("rows", [])) == 64, "Incomplete probe grid")
    require(result.get("variants") == list(design.VARIANT_NAMES), "Variant grid mismatch")
    require(status.get("status") == "completed" and status.get("results_sha256") == sha(paths[0]) and receipt.get("results_sha256") == sha(paths[0]), "Result receipt mismatch")
    require(verification.get("status") == "passed" and verification.get("max_abs_error") <= 1e-12 and verification.get("policy_blocks") == 64, "Audit gate failed")
    rows = result["rows"]
    variant_cells = {}
    seed_cells = {}
    for variant in design.VARIANT_NAMES:
        variant_cells[variant] = {}
        for role in design.ROLES:
            variant_cells[variant][role] = {}
            for schedule in design.SCHEDULES:
                for live in (True, False):
                    selected = [row for row in rows if row["role"] == role and row["schedule"] == schedule and bool(row["live"]) == live]
                    require(len(selected) == len(design.SEEDS), "Seed count in role cell")
                    balanced_rows = []
                    for row in selected:
                        balanced, axes = policy_variant_balanced(row, variant)
                        row.setdefault("variant_axis_balanced", {})[variant] = balanced
                        row.setdefault("variant_axis_cells", {})[variant] = axes
                        balanced_rows.append(balanced)
                        seed_cells[f"{row['seed']}/{row['role']}/{row['schedule']}/{ 'live' if row['live'] else 'silent'}/{variant}"] = balanced
                    variant_cells[variant][role][f"{schedule}/{'live' if live else 'silent'}"] = {
                        prefix: {field: stat([balanced[prefix][field] for balanced in balanced_rows]) for field in FIELDS}
                        for prefix in ("aligned", "placebo", "aligned_minus_placebo")
                    }

    role_contrasts = {}
    interactions = {}
    for variant in design.VARIANT_NAMES:
        interactions[variant] = {}
        for role in design.ROLES:
            interactions[variant][role] = {}
            for field in FIELDS:
                values = []
                for index, seed in enumerate(design.SEEDS):
                    live_static = variant_cells[variant][role]["static/live"]["aligned_minus_placebo"][field]["seed_values"][index]
                    silent_static = variant_cells[variant][role]["static/silent"]["aligned_minus_placebo"][field]["seed_values"][index]
                    live_rematched = variant_cells[variant][role]["rematched/live"]["aligned_minus_placebo"][field]["seed_values"][index]
                    silent_rematched = variant_cells[variant][role]["rematched/silent"]["aligned_minus_placebo"][field]["seed_values"][index]
                    values.append((live_rematched - silent_rematched) - (live_static - silent_static))
                interactions[variant][role][field] = stat(values)
        role_contrasts[variant] = {}
        for field in FIELDS:
            values = []
            for index, seed in enumerate(design.SEEDS):
                c = sum(variant_cells[variant]["C"][f"{schedule}/live"]["aligned_minus_placebo"][field]["seed_values"][index] - variant_cells[variant]["C"][f"{schedule}/silent"]["aligned_minus_placebo"][field]["seed_values"][index] for schedule in design.SCHEDULES) / len(design.SCHEDULES)
                a = sum(variant_cells[variant]["A"][f"{schedule}/live"]["aligned_minus_placebo"][field]["seed_values"][index] - variant_cells[variant]["A"][f"{schedule}/silent"]["aligned_minus_placebo"][field]["seed_values"][index] for schedule in design.SCHEDULES) / len(design.SCHEDULES)
                values.append(c - a)
            role_contrasts[variant][field] = stat(values)

    input_hashes = {str(path): sha(path) for path in paths}
    summary = dict(
        status="completed", created_at=datetime.now(timezone.utc).isoformat(), probe=str(probe),
        audit=dict(path=str(audit), verification_sha256=sha(audit / "verification.json"), status=verification["status"], max_abs_error=verification["max_abs_error"]),
        contract=dict(seeds=list(design.SEEDS), roles=list(design.ROLES), schedules=list(design.SCHEDULES), channels=["live", "silent"], axes=list(design.AXES), variants=list(design.VARIANT_NAMES), policy_blocks=64, cases_per_policy=76032, worlds=int(static["partition"]["world_count"]), model_forward_samples=result["model_forward_samples"], optimizer_updates=0),
        variant_cells=variant_cells, role_contrasts=role_contrasts, interactions=interactions, seed_cells=seed_cells,
        interpretation_boundary="Slot-selective and fixed-token recoding responses are frozen-policy task diagnostics; they are not lexical meaning, compositional syntax or language-origin evidence.",
        input_sha256=input_hashes,
    )
    out.mkdir(parents=True)
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    fmt = lambda entry: f"{100 * entry['mean']:+.4f} [{100 * entry['ci95_t7'][0]:+.4f}, {100 * entry['ci95_t7'][1]:+.4f}]"
    lines = [
        "# 接收者角色 C−A：单槽与符号重编码内容转移",
        "",
        "A/C 两个替换角色在三个需求轴和三个发送者内先等权，再以 8 个种子计算 t(7) 区间。单位为百分点；silent 是闭通道逐行对照。",
        "",
        "| 变体 | A static | A rematched | C static | C rematched |",
        "|---|---:|---:|---:|---:|",
    ]
    for variant in design.VARIANT_NAMES:
        vals = [
            variant_cells[variant]["A"]["static/live"]["aligned_minus_placebo"]["plan_transfer"],
            variant_cells[variant]["A"]["rematched/live"]["aligned_minus_placebo"]["plan_transfer"],
            variant_cells[variant]["C"]["static/live"]["aligned_minus_placebo"]["plan_transfer"],
            variant_cells[variant]["C"]["rematched/live"]["aligned_minus_placebo"]["plan_transfer"],
        ]
        lines.append(f"| {variant} | {' | '.join(fmt(value) for value in vals)} |")
    lines += [
        "",
        "## C−A 角色差值（每个 seed 先平均两个日程）",
        "",
        "| 变体 | 计划转移 C−A | 物理执行 C−A | Q C−A | Q|physical C−A | 动作改变 C−A |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for variant in design.VARIANT_NAMES:
        row = role_contrasts[variant]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(variant, *(fmt(row[field]) for field in ("plan_transfer", "physical", "q", "conditional_q", "action_change"))))
    lines += [
        "",
        "`full` 是整包 aligned/placebo 参照；`slot_i` 只替换首窗第 i 槽；`recode_add1` 和 `recode_xor4` 对供体包应用固定八符号双射。silent 下所有变体应逐行相等。",
        "",
        "[机器可读汇总](summary.json)",
        f"[独立审计](../../{audit.name}/verification.json)",
    ]
    table_path = out / "汇总表.md"
    table_path.write_text("\n".join(lines) + "\n", encoding="utf8")
    receipt_out = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), outputs={path.name: sha(path) for path in (summary_path, table_path)}, model_forward_samples=result["model_forward_samples"], optimizer_updates=0)
    (out / "receipt.json").write_text(json.dumps(receipt_out, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    return dict(status="completed", output=str(out), summary_sha256=sha(summary_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--audit", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); print(json.dumps(analyze(args.probe, args.audit, args.out), ensure_ascii=False))
