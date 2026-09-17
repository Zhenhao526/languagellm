"""Aggregate the frozen single-slot transfer probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

T7 = 2.3646242510102993
FIELDS = ("plan_transfer", "partner_transfer", "physical", "q", "conditional_q", "action_change")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(values):
    values = [float(value) for value in values]; mean = sum(values) / len(values); sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1)) if len(values) > 1 else 0.0; half = T7 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return dict(seed_values=values, mean=mean, sd=sd, ci95_t7=[mean - half, mean + half], min=min(values), max=max(values), n=len(values))


def actor_mask_summary(row, mask_name, sender=0):
    groups = [row["groups"][f"{axis}/{sender}/{mask_name}"] for axis in ("kind", "length")]
    output = {}
    for prefix in ("aligned", "placebo"):
        output[prefix] = {field: sum(float(group[f"{prefix}_{field}"]) for group in groups) / len(groups) for field in FIELDS}
    output["aligned_minus_placebo"] = {field: output["aligned"][field] - output["placebo"][field] for field in FIELDS}
    return output


def analyze(probe, audit, out):
    probe = Path(probe).resolve(); audit = Path(audit).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse to overwrite summary")
    paths = [probe / "execution/results.json", probe / "execution/status.json", probe / "execution/receipt.json", probe / "prepared.json", probe / "plan.json", probe / "freeze.json", audit / "verification.json", audit / "receipt.json"]
    require(all(path.is_file() for path in paths), "Missing aggregate input")
    result = json.loads((probe / "execution/results.json").read_text()); status = json.loads((probe / "execution/status.json").read_text()); receipt = json.loads((probe / "execution/receipt.json").read_text()); verification = json.loads((audit / "verification.json").read_text()); prepared = json.loads((probe / "prepared.json").read_text())
    require(result["status"] == "completed_compositional_slot_transfer_probe" and len(result["rows"]) == 64 and result["masks"] == 5, "Incomplete result")
    require(status["status"] == "completed" and status["results_sha256"] == sha(probe / "execution/results.json") == receipt["results_sha256"], "Result receipt mismatch")
    require(verification["status"] == "passed" and verification["max_abs_error"] <= 1e-12, "Audit failed")
    masks = [item["name"] for item in prepared["masks"]]; rows = result["rows"]; cells = {}
    for mask in masks:
        for arm in ("seen_joint_only", "all_joint"):
            for schedule in ("static", "rematched"):
                for live in (True, False):
                    key = f"{mask}/{arm}/{schedule}/{'live' if live else 'silent'}"; selected = [row for row in rows if row["arm"] == arm and row["schedule"] == schedule and bool(row["live"]) == live]; require(len(selected) == 8, "Cell size")
                    cells[key] = {"A": {prefix: {field: stat([actor_mask_summary(row, mask)[prefix][field] for row in selected]) for field in FIELDS} for prefix in ("aligned", "placebo", "aligned_minus_placebo")}}
    matched = {}
    for mask in masks:
        for schedule in ("static", "rematched"):
            for live in (True, False):
                suffix = "live" if live else "silent"; full = cells[f"{mask}/all_joint/{schedule}/{suffix}"]["A"]["aligned_minus_placebo"]; seen = cells[f"{mask}/seen_joint_only/{schedule}/{suffix}"]["A"]["aligned_minus_placebo"]
                matched[f"{mask}/{schedule}/{suffix}"] = {field: stat([full[field]["seed_values"][i] - seen[field]["seed_values"][i] for i in range(8)]) for field in FIELDS}
    interaction = {}
    for mask in masks:
        interaction[mask] = {field: stat([matched[f"{mask}/rematched/live"][field]["seed_values"][i] - matched[f"{mask}/static/live"][field]["seed_values"][i] for i in range(8)]) for field in FIELDS}
    out.mkdir(parents=True); inputs = {str(path): sha(path) for path in paths}; summary = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), probe=str(probe), audit=dict(path=str(audit), verification_sha256=sha(audit / "verification.json"), max_abs_error=verification["max_abs_error"]), contract=dict(seeds=list(range(66701, 66709)), arms=["seen_joint_only", "all_joint"], schedules=["static", "rematched"], channels=["live", "silent"], masks=masks, axes=["kind", "length"], sender="A", case_count=3456, policy_blocks=64, worlds_per_policy=11232, optimizer_updates=0, model_forward_samples=result["model_forward_samples"]), cells=cells, matched_all_minus_seen=matched, interaction=interaction, interpretation_boundary="Slot-selective aligned-minus-placebo transfer is a frozen-policy behavioral diagnostic, not lexical meaning, compositional syntax or language-origin evidence.", input_sha256=inputs)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    fmt = lambda value: f"{100 * value['mean']:+.4f} [{100 * value['ci95_t7'][0]:+.4f}, {100 * value['ci95_t7'][1]:+.4f}]"
    lines = ["# 单槽 aligned/placebo 联合组合留出转移", "", "单位为百分点，先在 kind、length 两条轴上等权，再在 8 个种子上计算 t(7) 区间；A 指每个策略的接收者角色。", "", "| 掩码 | seen static/live | all static/live | seen rematched/live | all rematched/live |", "|---|---:|---:|---:|---:|"]
    for mask in masks:
        values = [cells[f"{mask}/{arm}/{schedule}/live"]["A"]["aligned_minus_placebo"]["plan_transfer"] for arm, schedule in (("seen_joint_only", "static"), ("all_joint", "static"), ("seen_joint_only", "rematched"), ("all_joint", "rematched"))]
        lines.append(f"| {mask} | {' | '.join(fmt(value) for value in values)} |" if False else f"| {mask} | {fmt(values[0])} | {fmt(values[1])} | {fmt(values[2])} | {fmt(values[3])} |")
    lines += ["", "## all−seen live 差值", "", "| 掩码 | static | rematched | rematched−static |", "|---|---:|---:|---:|"]
    for mask in masks:
        static = matched[f"{mask}/static/live"]["plan_transfer"]; rematched = matched[f"{mask}/rematched/live"]["plan_transfer"]; inter = interaction[mask]["plan_transfer"]
        lines.append(f"| {mask} | {fmt(static)} | {fmt(rematched)} | {fmt(inter)} |")
    audit_name = Path(audit).name
    lines += ["", "silent 条件中每个掩码的 aligned 与 placebo 应逐行相等；`full` 是整包替换参照，四个 `slot_i` 是只替换一个首窗位置。", "", "[机器可读汇总](summary.json)", f"[独立审计](../../{audit_name}/verification.json)"]
    (out / "汇总表.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    receipt_out = dict(status="completed", script_sha256=sha(__file__), outputs={path.name: sha(path) for path in out.iterdir() if path.is_file()}, model_forward_samples=result["model_forward_samples"], optimizer_updates=0)
    (out / "receipt.json").write_text(json.dumps(receipt_out, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    return dict(status="completed", output=str(out), summary_sha256=sha(out / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--audit", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); print(json.dumps(analyze(args.probe, args.audit, args.out), ensure_ascii=False))
