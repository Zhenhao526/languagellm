"""Aggregate the frozen slot-subset message-transfer probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (49301, 49302, 49303, 49304)
CONDITIONS = ("PI_live", "PI_silent")
AXES = ("kind_wood_fiber", "length_short_long", "destination_L_R")
MASKS = tuple((f"mask_{mask:02d}", tuple(slot for slot in range(4) if mask & (1 << slot))) for mask in range(16))
FIELDS = ("source_pull", "greedy_source_pull", "delta_target_set_mass", "delta_source_set_mass", "delta_action_change_rate")
TCRIT_DF3_95 = 3.182446305284263


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_new(path, value):
    path = Path(path); require(not path.exists(), "Refuse to overwrite " + str(path)); path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def stat(values):
    values = [float(x) for x in values]; mean = sum(values) / len(values); sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0; half = TCRIT_DF3_95 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return dict(seed_values=values, mean=mean, sd=sd, ci95_t3=[mean - half, mean + half], min=min(values), max=max(values), n=len(values))


def fields(row):
    return dict(source_pull=float(row["source_pull"]), greedy_source_pull=float(row["greedy_source_pull"]), delta_target_set_mass=float(row["delta"]["target_set_mass"]), delta_source_set_mass=float(row["delta"]["source_set_mass"]), delta_action_change_rate=float(row["delta"]["action_change_rate"]))


def summarize(policies, condition, mask_name, axis=None):
    seed_means = {}
    for seed in SEEDS:
        rows = [row for policy in policies if int(policy["seed"]) == seed and policy["condition"] == condition for row in policy["policy_rows"] if row["mask"] == mask_name and (axis is None or row["axis"] == axis)]
        require(rows, "Empty group")
        seed_means[str(seed)] = {field: sum(fields(row)[field] for row in rows) / len(rows) for field in FIELDS}
    return dict(condition=condition, mask=mask_name, axis=axis, cases_per_seed=len(rows), seed_means=seed_means, metrics={field: stat([seed_means[str(seed)][field] for seed in SEEDS]) for field in FIELDS})


def axis_balanced(policies, condition, mask_name):
    axes = {axis: summarize(policies, condition, mask_name, axis) for axis in AXES}
    seed_means = {str(seed): {field: sum(axes[axis]["seed_means"][str(seed)][field] for axis in AXES) / len(AXES) for field in FIELDS} for seed in SEEDS}
    return dict(condition=condition, mask=mask_name, weighting="equal semantic axis, then equal directed case and background", seed_means=seed_means, metrics={field: stat([seed_means[str(seed)][field] for seed in SEEDS]) for field in FIELDS})


def analyze(probe, audit, out):
    probe = Path(probe).resolve(); audit = Path(audit).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse to overwrite summary")
    result_path = probe / "execution/results.json"; status_path = probe / "execution/status.json"; receipt_path = probe / "execution/receipt.json"; audit_path = audit / "verification.json"; audit_receipt = audit / "receipt.json"
    for path in (result_path, status_path, receipt_path, audit_path, audit_receipt, probe / "prepared.json", probe / "manifest.json", probe / "plan.json", probe / "freeze.json"):
        require(path.is_file(), "Missing input " + str(path))
    result = read(result_path); status = read(status_path); receipt = read(receipt_path); verification = read(audit_path); prepared = read(probe / "prepared.json"); freeze = read(probe / "freeze.json")
    expected_rows = 8 * 240 * len(MASKS)
    require(result["status"] == "completed_json_only_semantic_transfer_slot_probe" and result["policy_blocks"] == 8 and result["masks"] == len(MASKS) and result["intervention_rows"] == expected_rows, "Incomplete slot result")
    require(result["live_rows"] == 960 * (len(MASKS) - 1) and result["alias_rows"] == 960 * (len(MASKS) + 1) and status["status"] == "completed" and status["results_sha256"] == sha(result_path) and receipt["results_sha256"] == sha(result_path), "Result/status mismatch")
    require(verification["status"] == "passed" and verification["max_abs_error"] == 0.0 and verification["probe"] == str(probe) and verification["rows_replayed"] == expected_rows and verification["worlds"] == expected_rows * 36, "Audit gate failed")
    require(sha(probe / "plan.json") == freeze["plan_sha256"] and sha(probe / "prepared.json") == freeze["prepared_sha256"] and prepared["manifest_sha256"] == sha(probe / "manifest.json"), "Probe freeze mismatch")
    policies = result["policies"]; require({(int(p["seed"]), p["condition"]) for p in policies} == {(s, c) for s in SEEDS for c in CONDITIONS}, "Policy grid mismatch")
    out.mkdir(parents=True); inputs = {str(path): sha(path) for path in (result_path, status_path, receipt_path, audit_path, audit_receipt, probe / "prepared.json", probe / "manifest.json", probe / "plan.json", probe / "freeze.json")}
    axis_balanced_data = {condition: {mask_name: axis_balanced(policies, condition, mask_name) for mask_name, _ in MASKS} for condition in CONDITIONS}
    summary = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), probe=str(probe), audit=dict(path=str(audit), status=verification["status"], verification_sha256=sha(audit_path), max_abs_error=verification["max_abs_error"]), contract=dict(seeds=list(SEEDS), conditions=list(CONDITIONS), axes=list(AXES), masks=len(MASKS), directed_cases_per_policy=240, backgrounds_per_case=36, intervention_rows=expected_rows, live_rows=result["live_rows"], alias_rows=result["alias_rows"], worlds=expected_rows * 36, model_forward_samples=result["model_forward_samples"], optimizer_updates=0, model_calls=0), axis_balanced=axis_balanced_data, mask_definitions=[dict(name=name, slots=list(slots)) for name, slots in MASKS], interpretation_boundary="Slot-subset transfer is a frozen-policy factorization diagnostic, not lexical meaning or compositional grammar evidence.", input_sha256=inputs)
    write_new(out / "summary.json", summary)
    lines = ["# 槽位子集消息移植汇总", "", "每个有向案例把源端改变者首窗消息的一个槽位子集移植到目标发送者位置，未替换槽位保留目标消息；`mask_00` 是自然别名，`mask_15` 是完整源消息。先在三条需求轴内等权，再对轴等权、四个种子等权。", "", "| 掩码 | 槽位 | PI-live source_pull | PI-live 贪心source_pull | 动作改变率 |", "|---|---|---:|---:|---:|"]
    live = axis_balanced_data["PI_live"]
    for mask_name, slots in MASKS:
        cell = live[mask_name]["metrics"]; fmt = lambda field: f"{100*cell[field]['mean']:+.4f} [{100*cell[field]['ci95_t3'][0]:+.4f}, {100*cell[field]['ci95_t3'][1]:+.4f}]"
        lines.append(f"| {mask_name} | {','.join(map(str, slots)) or '—'} | {fmt('source_pull')} | {fmt('greedy_source_pull')} | {fmt('delta_action_change_rate')} |")
    lines += ["", "只有完整掩码相对于自然别名的结果才对应整包移植；单槽和子集结果用于检查响应是否可由少数位置复用。所有 PI-silent 掩码都是自然路由别名。", "", "[JSON](summary.json)；[执行结果](../execution/results.json)；[独立审计](../../../../research_program/triadic_partner_ecology_study/audit_semantic_transfer_slot_001/verification.json)。"]
    (out / "汇总表.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    receipt_out = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), input_files=len(inputs), outputs={path.name: sha(path) for path in out.iterdir() if path.is_file()}, model_forward_samples=result["model_forward_samples"], optimizer_updates=0)
    write_new(out / "receipt.json", receipt_out)
    return dict(status="completed", output=str(out), summary_sha256=sha(out / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--audit", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); print(json.dumps(analyze(args.probe, args.audit, args.out), ensure_ascii=False))
