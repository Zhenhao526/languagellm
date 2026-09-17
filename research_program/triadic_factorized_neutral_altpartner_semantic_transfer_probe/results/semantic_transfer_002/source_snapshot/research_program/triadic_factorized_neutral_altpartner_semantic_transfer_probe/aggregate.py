"""Aggregate aligned/placebo factorized semantic-transfer rows."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from . import design

T15 = 2.1314495455597715
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
    half = T15 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return dict(seed_values=values, mean=mean, sd=sd, ci95_t15=[mean - half, mean + half], n=len(values))


def group_value(group, prefix, field):
    if field == "plan_transfer":
        return float(group[f"{prefix}_plan_transfer"])
    if field == "partner_transfer":
        return float(group[f"{prefix}_partner_transfer"])
    if field == "physical":
        return float(group[f"{prefix}_physical"])
    if field == "q":
        return float(group[f"{prefix}_q"])
    if field == "conditional_q":
        return float(group[f"{prefix}_conditional_q"])
    if field == "action_change":
        return float(group[f"{prefix}_action_change"])
    raise KeyError(field)


def policy_axis_balanced(row):
    output = {}
    for axis in design.AXES:
        groups = [row["groups"][f"{axis}/{sender}"] for sender in range(3)]
        output[axis] = {}
        for prefix in ("aligned", "placebo"):
            output[axis][prefix] = {field: sum(group_value(group, prefix, field) for group in groups) / 3.0 for field in FIELDS}
        output[axis]["aligned_minus_placebo"] = {field: output[axis]["aligned"][field] - output[axis]["placebo"][field] for field in FIELDS}
    return {prefix: {field: sum(output[axis][prefix][field] for axis in design.AXES) / len(design.AXES) for field in FIELDS}
            for prefix in ("aligned", "placebo", "aligned_minus_placebo")}, output


def analyze(probe, audit, out):
    probe = Path(probe).resolve(); audit = Path(audit).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse to overwrite summary")
    result_path = probe / "execution/results.json"; status_path = probe / "execution/status.json"; receipt_path = probe / "execution/receipt.json"; verification_path = audit / "verification.json"; audit_receipt_path = audit / "receipt.json"
    for path in (result_path, status_path, receipt_path, verification_path, audit_receipt_path, probe / "prepared.json", probe / "inputs.json", probe / "plan.json", probe / "freeze.json"):
        require(path.is_file(), "Missing input " + str(path))
    result = json.loads(result_path.read_text(encoding="utf8")); status = json.loads(status_path.read_text(encoding="utf8")); receipt = json.loads(receipt_path.read_text(encoding="utf8")); verification = json.loads(verification_path.read_text(encoding="utf8")); static = json.loads((probe / "prepared.json").read_text(encoding="utf8"))
    require(result.get("status") == "completed_factorized_semantic_transfer_probe" and len(result.get("rows", [])) == 64, "Incomplete probe grid")
    require(status.get("status") == "completed" and status.get("results_sha256") == sha(result_path) and receipt.get("results_sha256") == sha(result_path), "Result receipt mismatch")
    require(verification.get("status") == "passed" and verification.get("max_abs_error") <= 1e-12 and verification.get("policy_blocks") == 64, "Audit gate failed")
    rows = result["rows"]; require(len({(int(row["seed"]), row["schedule"], bool(row["live"])) for row in rows}) == 64, "Duplicate policy rows")
    out.mkdir(parents=True); cells={}; seed_cells={}
    for row in rows:
        balanced, axes = policy_axis_balanced(row)
        seed_cells[f"{row['seed']}/{row['schedule']}/{ 'live' if row['live'] else 'silent'}"] = balanced
        row["axis_balanced"] = balanced
        row["axis_cells"] = axes
    for schedule in design.SCHEDULES:
        for live in (True, False):
            selected = [row for row in rows if row["schedule"] == schedule and bool(row["live"]) == live]
            cells[f"{schedule}/{'live' if live else 'silent'}"] = {prefix: {field: stat([row["axis_balanced"][prefix][field] for row in selected]) for field in FIELDS} for prefix in ("aligned", "placebo", "aligned_minus_placebo")}
    interactions = {}
    for field in FIELDS:
        values=[]
        for seed in design.SEEDS:
            lookup={(row["seed"],row["schedule"],bool(row["live"])): row["axis_balanced"]["aligned_minus_placebo"][field] for row in rows}
            values.append((lookup[seed,"rematched",True]-lookup[seed,"rematched",False])-(lookup[seed,"static",True]-lookup[seed,"static",False]))
        interactions[field]=stat(values)
    summary=dict(status="completed",created_at=datetime.now(timezone.utc).isoformat(),probe=str(probe),audit=dict(path=str(audit),verification_sha256=sha(verification_path),status=verification["status"],max_abs_error=verification["max_abs_error"]),contract=dict(seeds=list(design.SEEDS),schedules=list(design.SCHEDULES),channels=["live","silent"],axes=list(design.AXES),policy_blocks=64,cases_per_policy=76032,worlds=56160,model_forward_samples=result["model_forward_samples"],optimizer_updates=0),axis_balanced_seed=seed_cells,cells=cells,interactions=interactions,interpretation_boundary="Aligned-minus-placebo is a frozen task content-transfer diagnostic; absolute action changes are not lexical or language-origin evidence.",input_sha256={str(path):sha(path) for path in (result_path,status_path,receipt_path,verification_path,audit_receipt_path,probe/"prepared.json",probe/"inputs.json",probe/"plan.json",probe/"freeze.json")})
    (out / "summary.json").write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    lines=["# 因子化中性行动 aligned/placebo 消息转移汇总","","先在 kind、length、destination 三轴内等权，再对16个种子汇总。aligned 是同背景源端首窗消息移植，placebo 是相同发送者×轴×背景层内循环错配。单位为百分点；区间为种子层描述性 t(15) 区间。","","| 训练安排 | 通道 | aligned plan transfer | placebo plan transfer | aligned−placebo |","|---|---|---:|---:|---:|"]
    for schedule in design.SCHEDULES:
        for live in (True,False):
            cell=cells[f"{schedule}/{'live' if live else 'silent'}"]; fmt=lambda prefix:f"{100*cell[prefix]['plan_transfer']['mean']:+.4f} [{100*cell[prefix]['plan_transfer']['ci95_t15'][0]:+.4f}, {100*cell[prefix]['plan_transfer']['ci95_t15'][1]:+.4f}]"; lines.append(f"| {schedule} | {'live' if live else 'silent'} | {fmt('aligned')} | {fmt('placebo')} | {fmt('aligned_minus_placebo')} |")
    lines += ["","## rematched×communication interaction","","| 指标 | 均值 | t(15) 区间 |","|---|---:|---:|"]
    for field in FIELDS:
        x=interactions[field]; lines.append(f"| {field} | {100*x['mean']:+.4f} | [{100*x['ci95_t15'][0]:+.4f}, {100*x['ci95_t15'][1]:+.4f}] |")
    lines += ["","aligned−placebo 接近零时，消息改变行动不等于恢复源端需求内容；silent 条件应逐行相等。","","[JSON](summary.json)；[独立审计](../../../triadic_factorized_neutral_altpartner_semantic_transfer_probe/audit_semantic_transfer_001/verification.json)。"]
    (out/"汇总表.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    receipt=dict(status="completed",created_at=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),outputs={path.name:sha(path) for path in out.iterdir() if path.is_file()},optimizer_updates=0,model_forward_samples=result["model_forward_samples"])
    (out/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,sort_keys=True,indent=2)+"\n",encoding="utf8")
    return dict(status="completed",output=str(out),summary_sha256=sha(out/"summary.json"))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--probe",required=True); parser.add_argument("--audit",required=True); parser.add_argument("--out",required=True); args=parser.parse_args(); print(json.dumps(analyze(args.probe,args.audit,args.out),ensure_ascii=False))
