"""Role-wise aggregation for aligned/placebo message transfer."""
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


def policy_axis_balanced(row):
    output = {}
    for axis in design.AXES:
        groups = [row["groups"][f"{axis}/{sender}"] for sender in range(3)]
        output[axis] = {}
        for prefix in ("aligned", "placebo"):
            output[axis][prefix] = {field: sum(group_value(group, prefix, field) for group in groups) / 3.0 for field in FIELDS}
        output[axis]["aligned_minus_placebo"] = {field: output[axis]["aligned"][field] - output[axis]["placebo"][field] for field in FIELDS}
    return {
        prefix: {field: sum(output[axis][prefix][field] for axis in design.AXES) / len(design.AXES) for field in FIELDS}
        for prefix in ("aligned", "placebo", "aligned_minus_placebo")
    }, output


def analyze(probe, audit, out):
    probe = Path(probe).resolve(); audit = Path(audit).resolve(); out = Path(out).resolve()
    require(not out.exists(), "Refuse to overwrite summary")
    result_path = probe / "execution/results.json"; status_path = probe / "execution/status.json"; receipt_path = probe / "execution/receipt.json"
    verification_path = audit / "verification.json"; audit_receipt_path = audit / "receipt.json"
    for path in (result_path, status_path, receipt_path, verification_path, audit_receipt_path, probe / "prepared.json", probe / "inputs.json", probe / "plan.json", probe / "freeze.json"):
        require(path.is_file(), "Missing input " + str(path))
    result = json.loads(result_path.read_text(encoding="utf8")); status = json.loads(status_path.read_text(encoding="utf8")); receipt = json.loads(receipt_path.read_text(encoding="utf8")); verification = json.loads(verification_path.read_text(encoding="utf8")); static = json.loads((probe / "prepared.json").read_text(encoding="utf8"))
    require(result.get("status") == "completed_factorized_semantic_transfer_probe" and len(result.get("rows", [])) == 96, "Incomplete probe grid")
    require(status.get("status") == "completed" and status.get("results_sha256") == sha(result_path) and receipt.get("results_sha256") == sha(result_path), "Result receipt mismatch")
    require(verification.get("status") == "passed" and verification.get("max_abs_error") <= 1e-12 and verification.get("policy_blocks") == 96, "Audit gate failed")
    rows = result["rows"]
    require(len({(int(row["seed"]), row["role"], row["schedule"], bool(row["live"])) for row in rows}) == 96, "Duplicate policy rows")
    role_cells = {}; seed_cells = {}
    for row in rows:
        balanced, axes = policy_axis_balanced(row)
        row["axis_balanced"] = balanced; row["axis_cells"] = axes
        seed_cells[f"{row['seed']}/{row['role']}/{row['schedule']}/{ 'live' if row['live'] else 'silent'}"] = balanced
    for role in design.ROLES:
        role_cells[role] = {}
        for schedule in design.SCHEDULES:
            for live in (True, False):
                selected = [row for row in rows if row["role"] == role and row["schedule"] == schedule and bool(row["live"]) == live]
                require(len(selected) == len(design.SEEDS), "Seed count in role cell")
                role_cells[role][f"{schedule}/{'live' if live else 'silent'}"] = {
                    prefix: {field: stat([row["axis_balanced"][prefix][field] for row in selected]) for field in FIELDS}
                    for prefix in ("aligned", "placebo", "aligned_minus_placebo")
                }

    interactions = {}
    for role in design.ROLES:
        interactions[role] = {}
        for field in FIELDS:
            values = []
            for seed in design.SEEDS:
                lookup = {(row["seed"], row["role"], row["schedule"], bool(row["live"])): row["axis_balanced"]["aligned_minus_placebo"][field] for row in rows}
                values.append((lookup[seed, role, "rematched", True] - lookup[seed, role, "rematched", False]) - (lookup[seed, role, "static", True] - lookup[seed, role, "static", False]))
            interactions[role][field] = stat(values)

    role_contrasts = {}
    for role in design.ROLES:
        if role == "A":
            continue
        role_contrasts[f"{role}_minus_A"] = {}
        for field in FIELDS:
            values = []
            for seed in design.SEEDS:
                lookup = {(row["seed"], row["role"], row["schedule"], bool(row["live"])): row["axis_balanced"]["aligned_minus_placebo"][field] for row in rows}
                values.append(sum(lookup[seed, role, schedule, True] - lookup[seed, role, schedule, False] - lookup[seed, "A", schedule, True] + lookup[seed, "A", schedule, False] for schedule in design.SCHEDULES) / len(design.SCHEDULES))
            role_contrasts[f"{role}_minus_A"][field] = stat(values)

    summary = dict(
        status="completed", created_at=datetime.now(timezone.utc).isoformat(), probe=str(probe),
        audit=dict(path=str(audit), verification_sha256=sha(verification_path), status=verification["status"], max_abs_error=verification["max_abs_error"]),
        contract=dict(seeds=list(design.SEEDS), roles=list(design.ROLES), schedules=list(design.SCHEDULES), channels=["live", "silent"], axes=list(design.AXES), policy_blocks=96, cases_per_policy=76032, worlds=int(static["partition"]["world_count"]), model_forward_samples=result["model_forward_samples"], optimizer_updates=0),
        role_cells=role_cells, interactions=interactions, role_contrasts=role_contrasts, seed_cells=seed_cells,
        interpretation_boundary="Aligned-minus-placebo is a role-stratified frozen task content-transfer diagnostic; absolute action changes are not lexical or language-origin evidence.",
        input_sha256={str(path): sha(path) for path in (result_path, status_path, receipt_path, verification_path, audit_receipt_path, probe / "prepared.json", probe / "inputs.json", probe / "plan.json", probe / "freeze.json")},
    )
    out.mkdir(parents=True)
    summary_path = out / "summary.json"; summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    fmt = lambda entry: f"{100*entry['mean']:+.4f} [{100*entry['ci95_t7'][0]:+.4f}, {100*entry['ci95_t7'][1]:+.4f}]"
    lines = ["# 接收者角色分层 aligned/placebo 消息转移", "", "先在 kind、length、destination 三轴和三个发送者内等权，再对 8 个种子汇总。单位为百分点；区间为种子层 t(7) 区间。", "", "| 角色 | static aligned−placebo | rematched aligned−placebo |", "|---|---:|---:|"]
    for role in design.ROLES:
        lines.append(f"| {role} | {fmt(role_cells[role]['static/live']['aligned_minus_placebo']['plan_transfer'])} | {fmt(role_cells[role]['rematched/live']['aligned_minus_placebo']['plan_transfer'])} |")
    lines += ["", "## 角色差值（每个 seed 先平均两个 schedule）", "", "| 对比 | aligned−placebo plan transfer |", "|---|---:|"]
    for key, value in role_contrasts.items():
        lines.append(f"| {key} | {fmt(value['plan_transfer'])} |")
    lines += ["", "## rematched×communication 交互", "", "| 角色 | aligned−placebo plan transfer | physical | Q | Q|physical | partner | action change |", "|---|---:|---:|---:|---:|---:|---:|"]
    for role in design.ROLES:
        row = interactions[role]
        lines.append("| {} | {} | {} | {} | {} | {} | {} |".format(role, *(fmt(row[f]) for f in FIELDS)))
    lines += ["", "aligned−placebo 接近零时，消息改变行动不等于恢复源端需求内容；silent 行应逐行相等。", "", "[JSON](summary.json)", "[独立审计](../audit_role_semantic_transfer_001/verification.json)"]
    (out / "汇总表.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    receipt_out = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), outputs={path.name: sha(path) for path in out.iterdir() if path.is_file()}, optimizer_updates=0, model_forward_samples=result["model_forward_samples"])
    (out / "receipt.json").write_text(json.dumps(receipt_out, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    return dict(status="completed", output=str(out), summary_sha256=sha(summary_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--audit", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); print(json.dumps(analyze(args.probe, args.audit, args.out), ensure_ascii=False))
