"""JSON-only role-wise summary for the receiver crossover study."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import design

UPDATES = np.asarray(design.CHECKPOINTS, dtype=np.float64)
T7 = 2.3646242510102993
METRICS = ("q_rate", "conditional_q_rate", "physical_execution_rate", "target_pair_legal_rate", "proposal_legal_rate", "engagement_rate", "neutral_rate")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    x = np.asarray(values, dtype=np.float64); require(x.ndim == 1 and len(x) == len(design.SEEDS), "Expected eight seed values")
    mean = float(x.mean()); sd = float(x.std(ddof=1)); half = T7 * sd / math.sqrt(len(x))
    return dict(n=len(x), mean=mean, sample_sd=sd, ci95_t7=[mean - half, mean + half], values=x.tolist(), positive_count=int((x > 0).sum()), negative_count=int((x < 0).sum()))


def centered_auc(values):
    x = np.asarray(values, dtype=np.float64); require(x.shape == (len(UPDATES),), "Checkpoint shape")
    return float(np.trapezoid(x - x[0], UPDATES) / UPDATES[-1])


def trajectory(run, metric):
    rows = sorted(run["trajectory"], key=lambda row: row["update"]); require([row["update"] for row in rows] == list(map(int, UPDATES)), "Checkpoint order")
    return np.asarray([row["target_trajectory"][metric] for row in rows], dtype=np.float64)


def endpoint(run, metric):
    return float(run["final"]["new_layouts"][metric])


def summarize(runs):
    require(len(runs) == 96, "Expected 96 role-crossover runs")
    by = {(r["seed"], r["role"], r["schedule"], bool(r["live"])): r for r in runs}; require(len(by) == 96, "Duplicate run identity")
    roles = {}
    for role in design.ROLES:
        roles[role] = {}
        for schedule in design.SCHEDULES:
            cell = {}
            for metric in METRICS:
                auc = []; end = []
                for seed in design.SEEDS:
                    live = by[seed, role, schedule, True]; silent = by[seed, role, schedule, False]
                    auc.append(centered_auc(trajectory(live, metric) - trajectory(silent, metric))); end.append(endpoint(live, metric) - endpoint(silent, metric))
                cell[metric] = dict(centered_AUC=stats(auc), endpoint=stats(end), centered_AUC_values=auc, endpoint_values=end)
            roles[role][schedule] = cell
    contrasts = {}
    for role in design.ROLES:
        if role == "A":
            continue
        contrasts[f"{role}_minus_A"] = {}
        for metric in METRICS:
            auc = [float(np.mean([roles[role][s][metric]["centered_AUC_values"][i] - roles["A"][s][metric]["centered_AUC_values"][i] for s in design.SCHEDULES])) for i in range(len(design.SEEDS))]
            end = [float(np.mean([roles[role][s][metric]["endpoint_values"][i] - roles["A"][s][metric]["endpoint_values"][i] for s in design.SCHEDULES])) for i in range(len(design.SEEDS))]
            contrasts[f"{role}_minus_A"][metric] = dict(centered_AUC=stats(auc), endpoint=stats(end))
    inherited = {}
    for role in design.ROLES:
        inherited[role] = {}
        for schedule in design.SCHEDULES:
            inherited[role][schedule] = {}
            for live in (True, False):
                vals = [endpoint(by[seed, role, schedule, live], "q_rate") for seed in design.SEEDS]
                source = [float(by[seed, role, schedule, live]["inherited_source"]["new_layouts"]["q_rate"]) for seed in design.SEEDS]
                inherited[role][schedule]["live" if live else "silent"] = dict(adapted=vals, source=source, adapted_minus_source=stats((np.asarray(vals) - np.asarray(source)).tolist()))
    return dict(schema="triadic_receiver_role_crossover_summary_v1", roles=roles, role_contrasts=contrasts, inherited_source_control=inherited, metrics=list(METRICS), seeds=list(design.SEEDS), schedules=list(design.SCHEDULES), inference="Student-t intervals across eight independent source initializations; static and rematched are paired strata averaged within seed.", claim_boundary="Role-wise live-minus-silent adaptation is a task-protocol diagnostic, not lexical meaning or language-origin evidence.")


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    result = json.loads((source / "execution/results.json").read_text()); summary = summarize(result["runs"])
    payload = dict(status="completed_after_json_only_aggregation", source=str(source), summary=summary, no_model_calls=True)
    path = output / "summary.json"; path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    fmt = lambda cell: f"{100*cell['mean']:+.3f} [{100*cell['ci95_t7'][0]:+.3f}, {100*cell['ci95_t7'][1]:+.3f}]"
    lines = ["# 接收者角色交叉：live−silent 适应", "", "单位为百分点；区间是 8 个种子的 t(7) 区间，static/rematched 分开报告。", "", "| 角色 | static Q AUC | rematched Q AUC | static 终点 Q | rematched 终点 Q |", "|---|---:|---:|---:|---:|"]
    for role in design.ROLES:
        q = summary["roles"][role]; lines.append(f"| {role} | {fmt(q['static']['q_rate']['centered_AUC'])} | {fmt(q['rematched']['q_rate']['centered_AUC'])} | {fmt(q['static']['q_rate']['endpoint'])} | {fmt(q['rematched']['q_rate']['endpoint'])} |")
    lines += ["", "## 角色差值（每个 seed 先平均两个 schedule）", "", "| 对比 | Q AUC | 终点 Q |", "|---|---:|---:|"]
    for key, value in summary["role_contrasts"].items():
        lines.append(f"| {key} | {fmt(value['q_rate']['centered_AUC'])} | {fmt(value['q_rate']['endpoint'])} |")
    lines += ["", "正值表示被替换角色相对 A 角色从 live 通道获得更大的适应增益。角色差异仍属于任务协议适应，不是词义或语言起源证据。", "", "[机器可读汇总](summary.json)", "[训练审计](../audit_role_crossover_004/verification.json)"]
    (output / "汇总表.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    receipt = dict(status="passed", input_runs=len(result["runs"]), model_forwards=0, output_summary_sha256=sha(path)); (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); main(args.source, args.output)

