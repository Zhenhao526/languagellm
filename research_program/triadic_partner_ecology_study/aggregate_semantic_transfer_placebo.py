"""Aggregate the aligned versus misaligned-message placebo."""
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
FIELDS = ("source_pull", "greedy_source_pull", "delta_target_set_mass", "delta_source_set_mass", "delta_action_change_rate")
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
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def stat(values):
    values = [float(x) for x in values]
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1)) if len(values) > 1 else 0.0
    half = TCRIT_DF3_95 * sd / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return dict(seed_values=values, mean=mean, sd=sd, ci95_t3=[mean - half, mean + half], min=min(values), max=max(values), n=len(values))


def row_fields(row, prefix):
    values = row[prefix]
    return dict(
        source_pull=float(values["source_pull"]),
        greedy_source_pull=float(values["greedy_source_pull"]),
        delta_target_set_mass=float(values["delta"]["target_set_mass"]),
        delta_source_set_mass=float(values["delta"]["source_set_mass"]),
        delta_action_change_rate=float(values["delta"]["action_change_rate"]),
    )


def group(policies, condition, axis=None):
    result = {seed: [] for seed in SEEDS}
    for policy in policies:
        if policy["condition"] != condition:
            continue
        for row in policy["policy_rows"]:
            if axis is None or row["axis"] == axis:
                result[int(policy["seed"])].append(row)
    return result


def summarize(policies, condition, axis=None):
    grouped = group(policies, condition, axis)
    seed_means = {}
    for seed, rows in grouped.items():
        require(rows, "Empty group")
        cells = {}
        for prefix in ("aligned", "placebo"):
            cells[prefix] = {field: sum(row_fields(row, prefix)[field] for row in rows) / len(rows) for field in FIELDS}
        cells["aligned_minus_placebo"] = {field: cells["aligned"][field] - cells["placebo"][field] for field in FIELDS}
        seed_means[str(seed)] = cells
    metrics = {}
    for prefix in ("aligned", "placebo", "aligned_minus_placebo"):
        metrics[prefix] = {field: stat([seed_means[str(seed)][prefix][field] for seed in SEEDS]) for field in FIELDS}
    return dict(condition=condition, axis=axis, cases_per_seed=len(next(iter(grouped.values()))), seed_means=seed_means, metrics=metrics)


def balanced(policies, condition):
    axes = {axis: summarize(policies, condition, axis) for axis in AXES}
    seed_means = {}
    for seed in SEEDS:
        seed_means[str(seed)] = {}
        for prefix in ("aligned", "placebo", "aligned_minus_placebo"):
            seed_means[str(seed)][prefix] = {field: sum(axes[axis]["seed_means"][str(seed)][prefix][field] for axis in AXES) / len(AXES) for field in FIELDS}
    return dict(condition=condition, weighting="equal semantic axis, then equal directed case and background", seed_means=seed_means, metrics={prefix: {field: stat([seed_means[str(seed)][prefix][field] for seed in SEEDS]) for field in FIELDS} for prefix in ("aligned", "placebo", "aligned_minus_placebo")})


def analyze(probe, audit, out):
    probe = Path(probe).resolve(); audit = Path(audit).resolve(); out = Path(out).resolve(); require(not out.exists(), "Refuse to overwrite summary")
    execution = probe / "execution"; result_path = execution / "results.json"; status_path = execution / "status.json"; receipt_path = execution / "receipt.json"; audit_path = audit / "verification.json"; audit_receipt = audit / "receipt.json"
    for path in (result_path, status_path, receipt_path, audit_path, audit_receipt, probe / "prepared.json", probe / "manifest.json", probe / "plan.json", probe / "freeze.json"):
        require(path.is_file(), "Missing input " + str(path))
    result = read(result_path); status = read(status_path); receipt = read(receipt_path); verification = read(audit_path); prepared = read(probe / "prepared.json"); freeze = read(probe / "freeze.json")
    require(result["status"] == "completed_json_only_semantic_transfer_placebo_probe" and result["policy_blocks"] == 8 and result["intervention_rows"] == 1920, "Incomplete result")
    require(result["live_rows"] == result["alias_rows"] == 960 and status["status"] == "completed" and status["results_sha256"] == sha(result_path) and receipt["results_sha256"] == sha(result_path), "Result/status mismatch")
    require(verification["status"] == "passed" and verification["max_abs_error"] == 0.0 and verification["probe"] == str(probe), "Audit gate failed")
    require(sha(probe / "plan.json") == freeze["plan_sha256"] and sha(probe / "prepared.json") == freeze["prepared_sha256"] and prepared["manifest_sha256"] == sha(probe / "manifest.json"), "Probe freeze mismatch")
    require(verification["rows_replayed"] == 1920 and verification["worlds"] == 69120 and verification["model_forward_samples"] == 414720, "Audit counts mismatch")
    policies = result["policies"]; require({(int(p["seed"]), p["condition"]) for p in policies} == {(s, c) for s in SEEDS for c in CONDITIONS}, "Policy grid mismatch")
    out.mkdir(parents=True); inputs = {str(path): sha(path) for path in (result_path, status_path, receipt_path, audit_path, audit_receipt, probe / "prepared.json", probe / "manifest.json", probe / "plan.json", probe / "freeze.json")}
    by_axis = {condition: {axis: summarize(policies, condition, axis) for axis in AXES} for condition in CONDITIONS}
    summary = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), probe=str(probe), audit=dict(path=str(audit), status=verification["status"], verification_sha256=sha(audit_path), max_abs_error=verification["max_abs_error"]), contract=dict(seeds=list(SEEDS), conditions=list(CONDITIONS), axes=list(AXES), directed_cases_per_policy=240, backgrounds_per_case=36, intervention_rows=1920, live_rows=960, alias_rows=960, worlds=69120, aligned_live_forwards=207360, placebo_live_forwards=207360, optimizer_updates=0, model_calls=0), axis_balanced={condition: balanced(policies, condition) for condition in CONDITIONS}, by_axis=by_axis, interpretation_boundary="The aligned-minus-misaligned contrast asks whether the source message is more directionally useful than a same-stratum message assigned to the wrong case. It remains a fixed-policy task sensitivity diagnostic, not lexical meaning or language-origin evidence.", input_sha256=inputs)
    write_new(out / "summary.json", summary)
    lines = ["# 对齐消息与错配 placebo 汇总", "", "aligned 是当前源→目标消息移植；placebo 在相同需求轴、发送者、听者和方向的案例层内循环错配源消息，保持消息边际但打破源端需求对齐。每案36个背景，先对轴内案例等权、再对三轴等权、最后对四个种子等权。", "", "| 条件 | aligned source_pull | placebo source_pull | aligned−placebo | aligned动作改变率 | placebo动作改变率 |", "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        cells = summary["axis_balanced"][condition]["metrics"]
        def fmt(prefix, field):
            value = cells[prefix][field]; return f"{100*value['mean']:+.4f} [{100*value['ci95_t3'][0]:+.4f}, {100*value['ci95_t3'][1]:+.4f}]"
        lines.append(f"| {condition} | {fmt('aligned','source_pull')} | {fmt('placebo','source_pull')} | {fmt('aligned_minus_placebo','source_pull')} | {fmt('aligned','delta_action_change_rate')} | {fmt('placebo','delta_action_change_rate')} |")
    lines += ["", "## PI-live 分轴", "", "| 轴 | aligned source_pull | placebo source_pull | aligned−placebo |", "|---|---:|---:|---:"]
    for axis in AXES:
        cells = summary["by_axis"]["PI_live"][axis]["metrics"]
        def fmt_axis(prefix):
            value = cells[prefix]["source_pull"]; return f"{100*value['mean']:+.4f} [{100*value['ci95_t3'][0]:+.4f}, {100*value['ci95_t3'][1]:+.4f}]"
        lines.append(f"| {axis} | {fmt_axis('aligned')} | {fmt_axis('placebo')} | {fmt_axis('aligned_minus_placebo')} |")
    lines += ["", "aligned−placebo 若为正，才支持正确源消息比同层错配消息更能把动作推向源端；区间为四种子的描述性 t(3) 区间。PI-silent 两种干预均逐行等于自然路由。", "", "[JSON](summary.json)；[执行结果](../execution/results.json)；[独立审计](../../audit_semantic_transfer_placebo_unique_001/verification.json)。"]
    (out / "汇总表.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    receipt_out = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(), script_sha256=sha(__file__), input_files=len(inputs), outputs={path.name: sha(path) for path in out.iterdir() if path.is_file()}, model_forward_samples=414720, optimizer_updates=0)
    write_new(out / "receipt.json", receipt_out)
    return dict(status="completed", output=str(out), summary_sha256=sha(out / "summary.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--probe", required=True); parser.add_argument("--audit", required=True); parser.add_argument("--out", required=True); args = parser.parse_args(); print(json.dumps(analyze(args.probe, args.audit, args.out), ensure_ascii=False))
