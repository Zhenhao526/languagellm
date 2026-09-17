"""Aggregate the completed semantic-transfer probe without loading models."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


SEEDS = (49101, 49102, 49103, 49104)
CONDITIONS = ("PI_live", "PI_silent")
AXES = ("kind_wood_fiber", "length_short_long", "destination_L_R")
DIRECTIONS = ("before_to_after", "after_to_before")
VALUE_FIELDS = (
    "source_pull",
    "greedy_source_pull",
    "delta_target_set_mass",
    "delta_source_set_mass",
    "delta_target_set_hit",
    "delta_source_set_hit",
    "delta_action_change_rate",
)
TCRIT_DF3_95 = 3.182446305284263


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write_new(path, value):
    path = Path(path)
    require(not path.exists(), "Refuse to overwrite " + str(path))
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf8")


def now():
    return datetime.now(timezone.utc).isoformat()


def values_from_row(row):
    return {
        "source_pull": float(row["source_pull"]),
        "greedy_source_pull": float(row["greedy_source_pull"]),
        "delta_target_set_mass": float(row["delta"]["target_set_mass"]),
        "delta_source_set_mass": float(row["delta"]["source_set_mass"]),
        "delta_target_set_hit": float(row["delta"]["target_set_hit"]),
        "delta_source_set_hit": float(row["delta"]["source_set_hit"]),
        "delta_action_change_rate": float(row["delta"]["action_change_rate"]),
    }


def stat(values):
    values = [float(x) for x in values]
    require(values and all(math.isfinite(x) for x in values), "Nonfinite aggregate values")
    mean = sum(values) / len(values)
    if len(values) > 1:
        sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
        half = TCRIT_DF3_95 * sd / math.sqrt(len(values))
    else:
        sd = 0.0
        half = 0.0
    return dict(seed_values=values, mean=mean, sd=sd, ci95_t3=[mean - half, mean + half], min=min(values), max=max(values), n=len(values))


def group_seed_values(policies, condition, axis=None, direction=None):
    result = {seed: [] for seed in SEEDS}
    for policy in policies:
        if policy["condition"] != condition:
            continue
        seed = int(policy["seed"])
        for row in policy["policy_rows"]:
            if axis is not None and row["axis"] != axis:
                continue
            if direction is not None and row["direction"] != direction:
                continue
            result[seed].append(values_from_row(row))
    return result


def summarize_group(policies, condition, axis=None, direction=None):
    grouped = group_seed_values(policies, condition, axis, direction)
    seed_means = {}
    for seed, rows in grouped.items():
        require(rows, f"Empty group {condition}/{axis}/{direction}/{seed}")
        seed_means[str(seed)] = {field: sum(row[field] for row in rows) / len(rows) for field in VALUE_FIELDS}
    metrics = {}
    for field in VALUE_FIELDS:
        metrics[field] = stat([seed_means[str(seed)][field] for seed in SEEDS])
    return dict(
        condition=condition,
        axis=axis,
        direction=direction,
        cases_per_seed=len(next(iter(grouped.values()))),
        seed_means=seed_means,
        metrics=metrics,
    )


def axis_balanced(policies, condition):
    by_axis = {axis: summarize_group(policies, condition, axis=axis) for axis in AXES}
    seed_means = {}
    for seed in SEEDS:
        seed_means[str(seed)] = {
            field: sum(by_axis[axis]["seed_means"][str(seed)][field] for axis in AXES) / len(AXES)
            for field in VALUE_FIELDS
        }
    return dict(
        condition=condition,
        weighting="equal semantic axis; within axis equal directed case; within case equal heldout layout×owner",
        seed_means=seed_means,
        metrics={field: stat([seed_means[str(seed)][field] for seed in SEEDS]) for field in VALUE_FIELDS},
    )


def analyze(probe, audit, out):
    probe = Path(probe).resolve()
    audit = Path(audit).resolve()
    out = Path(out).resolve()
    require(not out.exists(), "Refuse to overwrite summary output")
    execution = probe / "execution"
    result_path = execution / "results.json"
    status_path = execution / "status.json"
    receipt_path = execution / "receipt.json"
    audit_verification = audit / "verification.json"
    audit_receipt = audit / "receipt.json"
    for path in (result_path, status_path, receipt_path, audit_verification, audit_receipt, probe / "prepared.json", probe / "manifest.json", probe / "plan.json", probe / "freeze.json"):
        require(path.is_file(), "Missing input " + str(path))
    result = read(result_path)
    status = read(status_path)
    receipt = read(receipt_path)
    verification = read(audit_verification)
    probe_prepared = read(probe / "prepared.json")
    plan = read(probe / "plan.json")
    freeze = read(probe / "freeze.json")
    require(result["status"] == "completed_json_only_semantic_transfer_probe", "Probe incomplete")
    require(status["status"] == "completed" and status["results_sha256"] == sha(result_path), "Execution status/hash mismatch")
    require(receipt["results_sha256"] == sha(result_path), "Execution receipt/hash mismatch")
    require(verification["status"] == "passed" and verification["max_abs_error"] == 0.0, "Independent audit not passed")
    require(verification["probe"] == str(probe), "Audit probe binding mismatch")
    require(sha(probe / "plan.json") == freeze["plan_sha256"], "Plan freeze mismatch")
    require(sha(probe / "prepared.json") == freeze["prepared_sha256"], "Prepared freeze mismatch")
    require(probe_prepared["manifest_sha256"] == sha(probe / "manifest.json"), "Manifest hash chain mismatch")
    require(len(result["policies"]) == 8 and result["intervention_rows"] == 1920, "Unexpected result grid")
    require(result["live_rows"] == result["alias_rows"] == 960, "Unexpected live/alias counts")
    require(verification["rows_replayed"] == 1920 and verification["worlds"] == 69120, "Audit count mismatch")
    policies = result["policies"]
    require({(int(p["seed"]), p["condition"]) for p in policies} == {(s, c) for s in SEEDS for c in CONDITIONS}, "Policy grid mismatch")
    require(all(len(p["policy_rows"]) == 240 for p in policies), "Per-policy case count mismatch")
    axis_counts = {axis: sum(row["axis"] == axis for p in policies for row in p["policy_rows"]) for axis in AXES}
    require(axis_counts == {"kind_wood_fiber": 384, "length_short_long": 384, "destination_L_R": 1152}, "Axis count mismatch")
    out.mkdir(parents=True)
    inputs = {str(path): sha(path) for path in (result_path, status_path, receipt_path, audit_verification, audit_receipt, probe / "prepared.json", probe / "manifest.json", probe / "plan.json", probe / "freeze.json")}
    by_axis = {condition: {axis: summarize_group(policies, condition, axis=axis) for axis in AXES} for condition in CONDITIONS}
    by_direction = {condition: {direction: summarize_group(policies, condition, direction=direction) for direction in DIRECTIONS} for condition in CONDITIONS}
    live_minus_silent = {}
    for axis in AXES:
        live_minus_silent[axis] = {}
        for field in VALUE_FIELDS:
            live_values = by_axis["PI_live"][axis]["metrics"][field]["seed_values"]
            silent_values = by_axis["PI_silent"][axis]["metrics"][field]["seed_values"]
            live_minus_silent[axis][field] = stat([x - y for x, y in zip(live_values, silent_values)])
    summary = dict(
        status="completed",
        created_at=now(),
        probe=str(probe),
        contract=dict(
            seeds=list(SEEDS),
            conditions=list(CONDITIONS),
            axes=list(AXES),
            directions=list(DIRECTIONS),
            directed_cases_per_policy=240,
            case_counts_by_axis={"kind_wood_fiber": 48, "length_short_long": 48, "destination_L_R": 144},
            backgrounds_per_case=36,
            intervention_rows=1920,
            live_rows=960,
            exact_alias_rows=960,
            worlds=69120,
            model_forward_samples=207360,
            optimizer_updates=0,
            model_calls=0,
        ),
        audit=dict(path=str(audit), status=verification["status"], verification_sha256=sha(audit_verification), max_abs_error=verification["max_abs_error"], independent_replay=verification["independent_replay"]),
        axis_balanced={condition: axis_balanced(policies, condition) for condition in CONDITIONS},
        by_axis=by_axis,
        by_direction=by_direction,
        live_minus_silent=live_minus_silent,
        interpretation_boundary="Positive source_pull means the transplanted source message increases source-endpoint action-set mass relative to target-endpoint mass. This is a fixed-policy directional sensitivity result, not evidence of lexical meaning, compositional syntax or language origin.",
        input_sha256=inputs,
    )
    write_new(out / "summary.json", summary)
    lines = [
        "# 消息移植探针汇总",
        "",
        "本批次把固定角色、听者两端完整成功动作集合不相交的源端首窗消息，移植到目标需求的 live 跨视角输入；PI-silent 是相同策略的闭路别名。每个有向案例在 36 个 heldout layout×owner 背景上求均值，再对四个独立种子等权。source_pull=(源动作集合质量增量)−(目标动作集合质量增量)。",
        "",
        "## 轴均衡的种子均值与 t(3) 95% 区间",
        "",
        "| 条件 | source_pull | greedy_source_pull | Δ目标集合质量 | Δ源集合质量 | 动作改变率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        cell = summary["axis_balanced"][condition]["metrics"]
        def fmt(field, scale=100):
            s = cell[field]
            return f"{scale*s['mean']:+.4f} [{scale*s['ci95_t3'][0]:+.4f}, {scale*s['ci95_t3'][1]:+.4f}]"
        lines.append(f"| {condition} | {fmt('source_pull')} | {fmt('greedy_source_pull')} | {fmt('delta_target_set_mass')} | {fmt('delta_source_set_mass')} | {fmt('delta_action_change_rate')} |")
    lines += ["", "## 按语义轴（PI-live）", "", "| 轴 | source_pull | greedy_source_pull | Δ目标集合质量 | Δ源集合质量 | 动作改变率 |", "|---|---:|---:|---:|---:|---:|"]
    for axis in AXES:
        cell = summary["by_axis"]["PI_live"][axis]["metrics"]
        def fmt_axis(field):
            s = cell[field]
            return f"{100*s['mean']:+.4f} [{100*s['ci95_t3'][0]:+.4f}, {100*s['ci95_t3'][1]:+.4f}]"
        lines.append(f"| {axis} | {fmt_axis('source_pull')} | {fmt_axis('greedy_source_pull')} | {fmt_axis('delta_target_set_mass')} | {fmt_axis('delta_source_set_mass')} | {fmt_axis('delta_action_change_rate')} |")
    lines += ["", "正值表示向源端动作集合移动；区间只是四个独立初始化上的描述性 t(3) 区间，不能替代预注册假设检验。PI-silent 所有这些差值逐行为零，且已通过独立重放。", "", "[JSON](summary.json) 保存每种子、每轴、每方向和 live−silent 配对差；[执行结果](../execution/results.json)；[独立审计](../../audit_semantic_transfer_unique_001/verification.json)。"]
    (out / "汇总表.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    receipt_out = dict(status="completed", created_at=now(), script_sha256=sha(__file__), input_files=len(inputs), outputs={path.name: sha(path) for path in out.iterdir() if path.is_file()}, model_forward_samples=207360, optimizer_updates=0)
    write_new(out / "receipt.json", receipt_out)
    return dict(status="completed", output=str(out), summary_sha256=sha(out / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.probe, args.audit, args.out), ensure_ascii=False))
