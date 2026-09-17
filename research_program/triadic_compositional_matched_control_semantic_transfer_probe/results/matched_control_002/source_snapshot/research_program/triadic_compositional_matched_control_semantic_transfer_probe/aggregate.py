"""Aggregate matched full-combination and seen-combination transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

AXES = ("kind", "length")
FIELDS = ("plan_transfer", "partner_transfer", "physical", "q", "conditional_q", "action_change")
T7 = 2.3646242510102993


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stat(values):
    values = [float(x) for x in values]; mean = sum(values) / len(values); sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0; half = T7 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return dict(seed_values=values, mean=mean, sd=sd, ci95_t7=[mean - half, mean + half], n=len(values))


def gvalue(group, prefix, field):
    if field in ("plan_transfer", "partner_transfer"): return float(group[f"{prefix}_{field}"])
    return float(group[f"{prefix}_{field}"])


def actor_summary(row, sender=0):
    groups = [row["groups"][f"{axis}/{sender}"] for axis in AXES]
    out = {}
    for prefix in ("aligned", "placebo"):
        out[prefix] = {field: sum(gvalue(group, prefix, field) for group in groups) / len(groups) for field in FIELDS}
    out["aligned_minus_placebo"] = {field: out["aligned"][field] - out["placebo"][field] for field in FIELDS}
    return out


def analyze(probe, audit, out):
    probe = Path(probe).resolve(); audit = Path(audit).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse to overwrite summary")
    result_path = probe / "execution/results.json"; status_path = probe / "execution/status.json"; receipt_path = probe / "execution/receipt.json"; verification_path = audit / "verification.json"; audit_receipt_path = audit / "receipt.json"
    for path in (result_path, status_path, receipt_path, verification_path, audit_receipt_path, probe / "prepared.json", probe / "plan.json", probe / "freeze.json"): require(path.is_file(), "Missing input " + str(path))
    result = json.loads(result_path.read_text()); status = json.loads(status_path.read_text()); receipt = json.loads(receipt_path.read_text()); verification = json.loads(verification_path.read_text())
    require(result.get("status") == "completed_compositional_matched_control_semantic_transfer_probe" and len(result.get("rows", [])) == 64, "Incomplete result")
    require(status.get("status") == "completed" and status.get("results_sha256") == sha(result_path) == receipt.get("results_sha256"), "Receipt mismatch")
    require(verification.get("status") == "passed" and verification.get("max_abs_error", 1) <= 1e-12, "Audit failed")
    rows = result["rows"]
    cells = {}
    for arm in ("seen_joint_only", "all_joint"):
        for schedule in ("static", "rematched"):
            for live in (True, False):
                key = f"{arm}/{schedule}/{'live' if live else 'silent'}"; selected = [r for r in rows if r["arm"] == arm and r["schedule"] == schedule and bool(r["live"]) == live]; require(len(selected) == 8, "cell size")
                cells[key] = {"A": {prefix: {field: stat([actor_summary(row)[prefix][field] for row in selected]) for field in FIELDS} for prefix in ("aligned", "placebo", "aligned_minus_placebo")}}
    matched = {}
    for schedule in ("static", "rematched"):
        for live in (True, False):
            suffix = "live" if live else "silent"; full = cells[f"all_joint/{schedule}/{suffix}"]["A"]["aligned_minus_placebo"]; seen = cells[f"seen_joint_only/{schedule}/{suffix}"]["A"]["aligned_minus_placebo"]
            matched[f"{schedule}/{suffix}"] = {field: stat([full[field]["seed_values"][i] - seen[field]["seed_values"][i] for i in range(8)]) for field in FIELDS}
    interaction = {field: stat([matched["rematched/live"][field]["seed_values"][i] - matched["static/live"][field]["seed_values"][i] for i in range(8)]) for field in FIELDS}
    out.mkdir(parents=True); summary = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), probe=str(probe), audit=dict(path=str(audit), verification_sha256=sha(verification_path), max_abs_error=verification["max_abs_error"]), contract=dict(seeds=list(range(66701, 66709)), arms=["seen_joint_only", "all_joint"], schedules=["static", "rematched"], channels=["live", "silent"], axes=list(AXES), case_count=3456, policy_blocks=64, worlds_per_policy=11232, optimizer_updates=0, model_forward_samples=result["model_forward_samples"]), cells=cells, matched_all_minus_seen=matched, interaction=interaction, interpretation_boundary="This matched-control posthoc probe is a heldout behavioral diagnostic, not lexical or language-origin evidence.", input_sha256={str(p): sha(p) for p in (result_path, status_path, receipt_path, verification_path, audit_receipt_path, probe / "prepared.json", probe / "plan.json", probe / "freeze.json")})
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    lines = ["# 同架构 A 全组合控制：联合组合留出 aligned/placebo 转移", "", "seen_joint_only 与 all_joint 使用相同父代 B/C、相同新 A 初始化和相同随机流；区别只在 A 的训练组合支持。单位为百分点，区间为 t(7)。", "", "| 条件 | seen A | all-joint A | all−seen |", "|---|---:|---:|---:|"]
    fmt = lambda x: f"{100*x['mean']:+.4f} [{100*x['ci95_t7'][0]:+.4f}, {100*x['ci95_t7'][1]:+.4f}]"
    for key in ("static/live", "rematched/live", "static/silent", "rematched/silent"):
        seen = cells[f"seen_joint_only/{key}"]["A"]["aligned_minus_placebo"]["plan_transfer"]; full = cells[f"all_joint/{key}"]["A"]["aligned_minus_placebo"]["plan_transfer"]; gap = matched[key]["plan_transfer"]; lines.append(f"| {key} | {fmt(seen)} | {fmt(full)} | {fmt(gap)} |")
    lines += ["", "## all−seen live schedule interaction", "", "| 指标 | 均值 | t(7) 区间 |", "|---|---:|---:|"]
    for field in FIELDS:
        x = interaction[field]; lines.append(f"| {field} | {100*x['mean']:+.4f} | [{100*x['ci95_t7'][0]:+.4f}, {100*x['ci95_t7'][1]:+.4f}] |")
    lines += ["", "静默条件的 aligned 与 placebo 逐行相等；正的 all−seen 表示全组合训练比只见组合训练保留更多留出内容转移。", "", "[JSON](summary.json)；[独立审计](../../audit_matched_control_002/verification.json)。"]
    (out / "汇总表.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    rec = dict(status="completed", script_sha256=sha(__file__), outputs={p.name: sha(p) for p in out.iterdir() if p.is_file()}); (out / "receipt.json").write_text(json.dumps(rec, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8"); return dict(status="completed", output=str(out), summary_sha256=sha(out / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--audit", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); print(json.dumps(analyze(args.probe, args.audit, args.out), ensure_ascii=False))
