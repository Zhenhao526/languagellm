"""Aggregate the factorial heldout aligned/placebo transfer probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


SEEDS = tuple(range(62101, 62117))
REGIMES = ("factorial_holdout", "saturated")
RULES = ("strict", "reciprocal")
CHANNELS = ("live", "silent")
AXES = ("kind", "length", "destination")
FIELDS = ("source_pull", "source_pair_hit", "target_pair_hit", "physical_execution_rate", "pair_change_rate", "action_change_rate")
TCRIT_DF15_95 = 2.1314495455597715


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
    values = [float(x) for x in values]; mean = sum(values) / len(values); sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0; half = TCRIT_DF15_95 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return dict(seed_values=values, mean=mean, sd=sd, ci95_t15=[mean - half, mean + half], min=min(values), max=max(values), n=len(values))


def metric_fields(row, prefix):
    value = row[prefix]
    return dict(source_pull=float(value["source_pull"]), source_pair_hit=float(value["intervened"]["source_pair_hit"]), target_pair_hit=float(value["intervened"]["target_pair_hit"]), physical_execution_rate=float(value["intervened"]["physical_execution_rate"]), pair_change_rate=float(value["intervened"]["pair_change_rate"]), action_change_rate=float(value["action_change_rate"]))


def policy_axis_means(policies, seed, regime, rule, channel, axis=None):
    selected = [policy for policy in policies if int(policy["seed"]) == seed and policy["regime"] == regime and policy["rule"] == rule and policy["channel"] == channel]
    require(len(selected) == 1, "Policy identity")
    rows = [row for row in selected[0]["policy_rows"] if axis is None or row["axis"] == axis]
    require(rows, "Empty axis group")
    result = {}
    for prefix in ("aligned", "placebo"):
        result[prefix] = {field: sum(metric_fields(row, prefix)[field] for row in rows) / len(rows) for field in FIELDS}
    result["aligned_minus_placebo"] = {field: result["aligned"][field] - result["placebo"][field] for field in FIELDS}
    return result


def axis_balanced_seed(policies, seed, regime, rule, channel):
    axes = {axis: policy_axis_means(policies, seed, regime, rule, channel, axis) for axis in AXES}
    return {prefix: {field: sum(axes[axis][prefix][field] for axis in AXES) / len(AXES) for field in FIELDS} for prefix in ("aligned", "placebo", "aligned_minus_placebo")}


def analyze(probe, audit, out):
    probe = Path(probe).resolve(); audit = Path(audit).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse to overwrite summary")
    result_path = probe / "execution/results.json"; status_path = probe / "execution/status.json"; receipt_path = probe / "execution/receipt.json"; audit_path = audit / "verification.json"; audit_receipt = audit / "receipt.json"
    for path in (result_path, status_path, receipt_path, audit_path, audit_receipt, probe / "prepared.json", probe / "manifest.json", probe / "plan.json", probe / "freeze.json", probe / "cases.json"): require(path.is_file(), "Missing input " + str(path))
    result = read(result_path); status = read(status_path); receipt = read(receipt_path); verification = read(audit_path); prepared = read(probe / "prepared.json"); freeze = read(probe / "freeze.json")
    expected_rows = 128 * 2568; expected_worlds = expected_rows * 36; expected_live_rows = 64 * 2568
    require(result["status"] == "completed_json_only_factorial_heldout_semantic_transfer_probe" and result["policy_blocks"] == 128 and result["intervention_rows"] == expected_rows, "Incomplete result grid")
    require(result["live_rows"] == expected_live_rows and result["alias_rows"] == expected_live_rows and status["status"] == "completed" and status["results_sha256"] == sha(result_path) and receipt["results_sha256"] == sha(result_path), "Result/status mismatch")
    require(verification["status"] == "passed" and verification["max_abs_error"] == 0.0 and verification["probe"] == str(probe) and verification["rows_replayed"] == expected_rows and verification["worlds"] == expected_worlds, "Audit gate failed")
    require(sha(probe / "plan.json") == freeze["plan_sha256"] and sha(probe / "prepared.json") == freeze["prepared_sha256"] and prepared["manifest_sha256"] == sha(probe / "manifest.json"), "Freeze mismatch")
    policies = result["policies"]; require({(int(p["seed"]), p["regime"], p["rule"], p["channel"]) for p in policies} == {(s, g, r, c) for s in SEEDS for g in REGIMES for r in RULES for c in CHANNELS}, "Policy grid mismatch")
    out.mkdir(parents=True); inputs = {str(path): sha(path) for path in (result_path, status_path, receipt_path, audit_path, audit_receipt, probe / "prepared.json", probe / "manifest.json", probe / "plan.json", probe / "freeze.json", probe / "cases.json")}
    seed_cells = {(regime, rule, channel, seed): axis_balanced_seed(policies, seed, regime, rule, channel) for regime in REGIMES for rule in RULES for channel in CHANNELS for seed in SEEDS}
    cells = {}
    for regime in REGIMES:
        for rule in RULES:
            for channel in CHANNELS:
                cell = {prefix: {field: stat([seed_cells[regime, rule, channel, seed][prefix][field] for seed in SEEDS]) for field in FIELDS} for prefix in ("aligned", "placebo", "aligned_minus_placebo")}
                cells[regime, rule, channel] = cell
    interactions = {}
    for rule in RULES:
        per_seed = {}
        for seed in SEEDS:
            live = seed_cells["factorial_holdout", rule, "live", seed]["aligned_minus_placebo"]["source_pull"] - seed_cells["factorial_holdout", rule, "silent", seed]["aligned_minus_placebo"]["source_pull"]
            sat = seed_cells["saturated", rule, "live", seed]["aligned_minus_placebo"]["source_pull"] - seed_cells["saturated", rule, "silent", seed]["aligned_minus_placebo"]["source_pull"]
            per_seed[str(seed)] = dict(factorial_live_minus_silent=live, saturated_live_minus_silent=sat, factorial_minus_saturated=live - sat)
        interactions[rule] = dict(seed_values=per_seed, factorial_minus_saturated=stat([per_seed[str(seed)]["factorial_minus_saturated"] for seed in SEEDS]))
    seed_cells_json = {"%s/%s/%s/%s" % key: value for key, value in seed_cells.items()}
    summary = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), probe=str(probe), audit=dict(path=str(audit), status=verification["status"], verification_sha256=sha(audit_path), max_abs_error=verification["max_abs_error"]), contract=dict(seeds=list(SEEDS), regimes=list(REGIMES), rules=list(RULES), channels=list(CHANNELS), axes=list(AXES), cases_per_policy=2568, backgrounds_per_case=36, policy_blocks=128, intervention_rows=expected_rows, live_rows=expected_live_rows, alias_rows=expected_live_rows, worlds=expected_worlds, model_forward_samples=result["model_forward_samples"], optimizer_updates=0), axis_balanced_seed=seed_cells_json, cells={"%s/%s/%s" % key: value for key, value in cells.items()}, interactions=interactions, interpretation_boundary="Correct source messages are informative only if aligned-minus-placebo is positive; absolute transfer response is general task sensitivity and not a lexical or language-origin measure.", input_sha256=inputs)
    write_new(out / "summary.json", summary)
    lines = ["# 因素留出 aligned/placebo 消息转移汇总", "", "所有数值先在kind、length、destination三轴内等权，再对四个或十六个配对初始化等权。aligned是源端首窗消息移植，placebo在同一改变主体×轴×方向层内循环错配源消息。单位为百分点，区间为种子层描述性 t 区间。", "", "| 训练臂 | 规则 | 通道 | aligned source_pull | placebo source_pull | aligned−placebo |", "|---|---|---|---:|---:|---:|"]
    for regime in REGIMES:
        for rule in RULES:
            for channel in CHANNELS:
                cell = cells[regime, rule, channel]
                fmt = lambda prefix: f"{100*cell[prefix]['source_pull']['mean']:+.4f} [{100*cell[prefix]['source_pull']['ci95_t15'][0]:+.4f}, {100*cell[prefix]['source_pull']['ci95_t15'][1]:+.4f}]"
                lines.append(f"| {regime} | {rule} | {channel} | {fmt('aligned')} | {fmt('placebo')} | {fmt('aligned_minus_placebo')} |")
    lines += ["", "## 训练臂 live−silent 交互（aligned−placebo source_pull）", "", "| 规则 | factorial live−silent | saturated live−silent | factorial−saturated |", "|---|---:|---:|---:"]
    for rule in RULES:
        interaction = interactions[rule]; fvals = [interaction["seed_values"][str(seed)]["factorial_live_minus_silent"] for seed in SEEDS]; svals = [interaction["seed_values"][str(seed)]["saturated_live_minus_silent"] for seed in SEEDS]; d = interaction["factorial_minus_saturated"]
        lines.append(f"| {rule} | {100*sum(fvals)/len(fvals):+.4f} | {100*sum(svals)/len(svals):+.4f} | {100*d['mean']:+.4f} [{100*d['ci95_t15'][0]:+.4f}, {100*d['ci95_t15'][1]:+.4f}] |")
    lines += ["", "aligned−placebo 若接近零，说明消息改变行动但没有显示源端需求—消息对齐；若因素留出臂相对于 saturated 臂也没有正向交互，则不支持未见对象×长度组合产生额外的内容约定。", "", "[JSON](summary.json)；[执行结果](../execution/results.json)；[独立审计](../../../triadic_factorial_formation_study/audit_semantic_transfer_heldout_001/verification.json)。"]
    (out / "汇总表.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    write_new(out / "receipt.json", dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), input_files=len(inputs), outputs={path.name: sha(path) for path in out.iterdir() if path.is_file()}, model_forward_samples=result["model_forward_samples"], optimizer_updates=0))
    return dict(status="completed", output=str(out), summary_sha256=sha(out / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--audit", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); print(json.dumps(analyze(args.probe, args.audit, args.out), ensure_ascii=False))
