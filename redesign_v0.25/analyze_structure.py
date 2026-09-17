"""v25 offline joint-protocol assay, reusing the frozen v24 algorithm.

Existing mean/attention scores are hash-bound and never re-evaluated here.
No checkpoint/model is loaded; no training or policy inference is performed.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
V24 = ROOT.parent / "redesign_v0.24"
sys.dont_write_bytecode = True
sys.path.insert(0, str(V24))
import structural_assay as prior
from test_structural_assay import slow_reference

ASSAY_SHA = "bda5004eed00c66558233658c6ed7bba06443be8853946c53cbf5e647830b4bf"
TEST_SHA = "76cc6023d6c4fd48750b2888b961ec359f5f512032fea7b2c089689157f96824"
BASELINES = {
    False: {"batch": "attention_001",
        "summary_sha": "fec2d85100cfbb7428fdd3199ed5982ad9a26a01f9b7bce1471dcafec9719fbb",
        "audit": "structural_assay_audit.json",
        "audit_sha": "cc1ec67e9187d22994111ab3a827c3c460cab8763d17968b3eb7463c2df43572"},
    True: {"batch": "smoke_001",
        "summary_sha": "daba5d9c2fe8ae827f71562544fd0afa6889f7ae801c074debe62e220bd5f460",
        "audit": "structural_assay_dev_audit.json",
        "audit_sha": "f1afc87dee13e926a1e1368e7cde79d2fd2ecdbf7c8f04b2a943a04bd4ab03eb"}}


def check_hash(path, expected):
    actual = prior.sha(path)
    if actual != expected:
        raise ValueError(f"Frozen source/result changed: {path}; SHA {actual}")
    return {"path": str(Path(path).resolve()), "sha256": actual}


def load_baseline(development):
    if Path(prior.__file__).resolve() != V24 / "structural_assay.py":
        raise ValueError("Wrong structural_assay module resolved")
    bindings = {"assay": check_hash(V24 / "structural_assay.py", ASSAY_SHA),
                "naive_reference": check_hash(V24 / "test_structural_assay.py", TEST_SHA)}
    entry = BASELINES[development]
    batch = V24 / "results" / entry["batch"]
    summary_path, audit_path = batch / "structural_assay.json", batch / entry["audit"]
    bindings["baseline_summary"] = check_hash(summary_path, entry["summary_sha"])
    bindings["baseline_audit"] = check_hash(audit_path, entry["audit_sha"])
    summary, audit = json.loads(summary_path.read_text()), json.loads(audit_path.read_text())
    if summary["status"] != "complete" or audit["status"] != "passed":
        raise ValueError("Baseline summary or its recorded audit did not complete")
    if bool(summary["development"]) != development:
        raise ValueError("Development/formal baseline mismatch")
    if not development and audit["summary_sha256"] != entry["summary_sha"]:
        raise ValueError("Formal baseline audit binds a different result")
    for path, expected in summary["source_fingerprints"].items():
        check_hash(path, expected)
    ref = summary["reference"]
    references = np.asarray(ref["permutations"], np.int64)
    if ref["seed"] != 2402401 or ref["count"] != 200:
        raise ValueError("Unexpected frozen reference bank")
    prior.validate_references(references)
    if not np.array_equal(references, prior.make_references()):
        raise ValueError("Stored reference permutations differ from the fixed generator")
    if prior.array_sha(references) != ref["sha256"]:
        raise ValueError("Reference bank fingerprint mismatch")
    return summary, references, bindings


def collect_joint(out, step, development):
    seeds = (99523,) if development else (33101, 33102, 33103, 33104)
    ps = (1,) if development else (1, 2, 3)
    if step != (40 if development else 2400):
        raise ValueError("Use fixed endpoint 0040 for development or 2400 for formal")
    expected = set(itertools.product(seeds, ps, (0, 1)))
    rows, actual = [], set()
    for path in sorted((out / "social").glob(f"*/protocol_{step:04d}_d*.npz")):
        match = re.fullmatch(r"s(\d+)_p([123])_joint", path.parent.name)
        dmatch = re.fullmatch(rf"protocol_{step:04d}_d([01])\.npz", path.name)
        if not match or not dmatch:
            raise ValueError(f"Unexpected joint protocol path: {path}")
        seed, p, d = int(match[1]), int(match[2]), int(dmatch[1])
        key = (seed, p, d)
        if key in actual:
            raise ValueError("Duplicate joint protocol identity")
        actual.add(key)
        config_path, result_path = path.parent / "config.json", path.parent / "result.json"
        config, result = json.loads(config_path.read_text()), json.loads(result_path.read_text())
        if result.get("status") != "complete":
            raise ValueError("Refusing an unfinished joint training run")
        if (config.get("seed"), config.get("partition"),
            config.get("arm", config.get("condition")), config.get("updates")) != (seed, p, "joint", step):
            raise ValueError("Joint config/directory identity mismatch")
        rows.append({"seed": seed, "partition": p, "condition": "joint", "direction": d,
                     "path": str(path.resolve()), "sha256": prior.sha(path),
                     "config_sha256": prior.sha(config_path)})
    if actual != expected:
        raise ValueError(f"Incomplete joint grid; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    return rows


def summarize_joint(rows):
    partition_cells = []
    for seed, p in sorted({(r["seed"], r["partition"]) for r in rows}):
        chosen = [r for r in rows if (r["seed"], r["partition"]) == (seed, p)]
        if sorted(r["direction"] for r in chosen) != [0, 1]:
            raise ValueError("Each joint pair requires both directions")
        partition_cells.append({"seed": seed, "partition": p, "condition": "joint",
            "scores": prior.average_scores([r["scores"] for r in chosen])})
    source_cells = []
    for seed in sorted({r["seed"] for r in rows}):
        chosen = [r for r in partition_cells if r["seed"] == seed]
        source_cells.append({"seed": seed, "condition": "joint",
            "partitions": [r["partition"] for r in chosen],
            "scores": prior.average_scores([r["scores"] for r in chosen])})
    return partition_cells, source_cells, prior.average_scores([r["scores"] for r in source_cells])


def compare_to_baselines(source_cells, baseline):
    output = {}
    for condition in ("mean", "attention"):
        comparison = {"role": "primary_auxiliary" if condition == "mean" else "descriptive",
                      "assignments": {}}
        old = {r["seed"]: r for r in baseline["source_cells"] if r["condition"] == condition}
        if set(old) != {r["seed"] for r in source_cells}:
            raise ValueError("Joint/baseline sources do not match")
        for assignment in prior.ASSIGNMENTS:
            comparison["assignments"][assignment] = {}
            for group in prior.GROUPS:
                comparison["assignments"][assignment][group] = {}
                for mask in prior.MASKS:
                    differences = []
                    for row in source_cells:
                        old_row = old[row["seed"]]
                        if row["partitions"] != old_row["partitions"]:
                            raise ValueError("Joint/baseline partition supports do not match")
                        a = row["scores"][assignment][group][mask]
                        b = old_row["scores"][assignment][group][mask]
                        differences.append({"seed": row["seed"],
                            "joint_recombined_J": a["recombined_J"],
                            "baseline_recombined_J": b["recombined_J"],
                            "recombined_J_difference": a["recombined_J"] - b["recombined_J"],
                            "reference_excess_difference": a["recoding"]["observed_minus_mean"]
                                - b["recoding"]["observed_minus_mean"],
                            "natural_J_difference": a["natural_J"] - b["natural_J"]})
                    comparison["assignments"][assignment][group][mask] = {
                        "source_differences": differences,
                        "mean_differences": {key: float(np.mean([r[key] for r in differences]))
                            for key in ("recombined_J_difference", "reference_excess_difference",
                                        "natural_J_difference")}}
        output[f"joint_minus_{condition}"] = comparison
    return output


def audit_fixed_protocol(rows, references, development):
    seed = 99523 if development else 33101
    row = next(r for r in rows if (r["seed"], r["partition"], r["direction"]) == (seed, 1, 0))
    raw = prior.read_npz(row["path"])
    if len(raw["map_id"]) != 960:
        raise ValueError("The prespecified bounded audit requires all 960 target worlds")
    _, fast = prior.evaluate_protocol(raw, 1, references[:3], return_details=True)
    slow = slow_reference(raw, 1, references[:3])
    count = 0
    for assignment in prior.ASSIGNMENTS:
        values = np.vstack([fast[assignment]["target_rates"], fast[assignment]["reference_target_rates"]])
        np.testing.assert_allclose(values, slow[assignment], atol=0, rtol=0)
        count += values.size
    # Same-algorithm saved-summary consistency, distinct from the naive check.
    if row["scores"] != prior.evaluate_protocol(raw, 1, references):
        raise AssertionError("Saved selected row differs from its full reference evaluation")
    return {"status": "passed", "development": development,
        "selection": {"seed": seed, "partition": 1, "direction": 0, "condition": "joint"},
        "protocol": row["path"], "protocol_sha256": row["sha256"],
        "target_worlds": 960, "assignments": list(prior.ASSIGNMENTS),
        "identity_included": True, "reference_indices": [0, 1, 2],
        "target_rate_comparisons": count, "max_absolute_error": 0.0,
        "saved_summary_all_200_references_exactly_equal": True,
        "limits": "Naive independent replay covers one fixed joint direction protocol, not every formal reference score.",
        "model_calls": 0, "training_updates": 0}


def report(payload):
    lines = ["# v25 joint 条件的辅助符号重组评估", "",
        "本页不改变主要 old18 自然双目标 J。旧 mean/attention 直接引用 v24 已审结构结果，新 joint 使用相同评价函数和200个整体码双射。",
        "两个槽位—资源分配全部保留；供体选择使用实验者地图信息，重组成功不等于主体自然产生新组合或句法。", "",
        "| 条件 | token0/token1供体 | 支持 | 自然J | 重组J | 参考均值 | 参考2.5%–97.5% |",
        "|---|---|---|---:|---:|---:|---:|"]
    for condition, assignment, group in itertools.product(("mean", "attention", "joint"), prior.ASSIGNMENTS, prior.GROUPS):
        c = payload["aggregate"][condition][assignment][group]["pooled"]
        ref = c["recoding"]
        lines.append(f"| {condition} | {assignment} | {group} | {100*c['natural_J']:.2f}% | "
            f"{100*c['recombined_J']:.2f}% | {100*ref['mean']:.2f}% | "
            f"{100*ref['quantiles']['0.025']:.2f}%–{100*ref['quantiles']['0.975']:.2f}% |")
    lines += ["", "下表为 joint−mean 的主要辅助重组差（百分点），保留每个来源。joint−attention 仅描述，完整结果在 JSON。", "",
              "| 分配 | 支持 | 来源及配对差 |", "|---|---|---|"]
    for assignment, group in itertools.product(prior.ASSIGNMENTS, prior.GROUPS):
        cells = payload["comparisons"]["joint_minus_mean"]["assignments"][assignment][group]["pooled"]["source_differences"]
        text = "; ".join(f"{r['seed']}: {r['recombined_J_difference']*100:+.3f}" for r in cells)
        lines.append(f"| {assignment} | {group} | {text} |")
    lines += ["", "分位数按同一参考索引先方向、分区、来源平均后计算，不是独立来源置信区间。",
        "该比较不能独立区分信息压缩、读出易学性与更一般的表达能力，也不能凭重组分数宣称因果发现。",
        "固定单份协议的朴素循环复核见 structural_assay_audit.json；旧基线不可变哈希及全部两mask结果见 structural_assay.json。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--step", type=int, default=2400)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    baseline, references, bindings = load_baseline(args.dev)
    rows = collect_joint(args.out, args.step, args.dev)
    for row in rows:
        raw = prior.read_npz(row["path"])
        world_sha = {k: prior.array_sha(raw[k]) for k in prior.WORLD_KEYS}
        if world_sha != baseline["evaluation_world_fingerprints"]:
            raise ValueError("New and old protocols do not share the frozen evaluation worlds")
        row["scores"] = prior.evaluate_protocol(raw, row["partition"], references)
    partition_cells, source_cells, aggregate = summarize_joint(rows)
    audit = audit_fixed_protocol(rows, references, args.dev)
    for binding in bindings.values():
        check_hash(binding["path"], binding["sha256"])
    payload = {"schema": "v25_joint_structural_assay_v1", "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(), "development": args.dev,
        "step": args.step, "source_count": len(source_cells), "new_runs": rows,
        "partition_cells": partition_cells, "source_cells": source_cells,
        "aggregate": {"mean": baseline["aggregate"]["mean"],
                      "attention": baseline["aggregate"]["attention"], "joint": aggregate},
        "comparisons": compare_to_baselines(source_cells, baseline),
        "reference": baseline["reference"], "baseline_bindings": bindings,
        "evaluation_world_fingerprints": baseline["evaluation_world_fingerprints"],
        "wrapper_sha256": prior.sha(__file__), "old_baselines_recomputed": False,
        "role": "Auxiliary structure only; the v25 old18 natural-J primary remains separate."}
    output = args.out / "structural_assay.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    audit.update(wrapper_sha256=prior.sha(__file__), baseline_bindings=bindings,
                 summary_sha256=prior.sha(output))
    (args.out / "structural_assay_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    (args.out / "结构重组评估.md").write_text(report(payload))
    print(json.dumps({"status": "complete", "new_direction_protocols": len(rows),
                      "source_count": len(source_cells), "output": str(output),
                      "naive_target_comparisons": audit["target_rate_comparisons"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
