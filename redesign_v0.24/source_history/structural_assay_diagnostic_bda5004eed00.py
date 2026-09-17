"""Fixed offline donor recombination assay for v24 saved greedy protocols.

No model is loaded and no policy is sampled. The only production dependency is
the already fixed map partition in v23/world.py. Both token-resource assignments
are reported; this module never selects the better assignment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
WORLD_PATH = ROOT.parent / "redesign_v0.23" / "world.py"
SPEC = importlib.util.spec_from_file_location("v24_assay_world", WORLD_PATH)
world = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(world)

VOCAB = 7
NCODES = VOCAB ** 2
N_REFERENCE = 200
REFERENCE_SEED = 2402401
ASSIGNMENTS = ("food_water", "water_food")
CONDITIONS = ("mean", "attention")
GROUPS = ("old18", "new12", "common30")
MASKS = ("pooled", "food_only", "water_only")
WORLD_KEYS = ("map_id", "photo_ids", "positions", "shown")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def read_npz(path):
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def partitions(p):
    split = world.partition(p)
    result = {"old18": np.asarray(split["old"], np.int64),
              "new12": np.sort(np.r_[split["added"], split["sealed"]]),
              "common30": np.arange(30, dtype=np.int64)}
    if len(result["old18"]) != 18 or len(result["new12"]) != 12:
        raise ValueError("Expected the frozen 18/12 layout split")
    if len(np.unique(np.r_[result["old18"], result["new12"]])) != 30:
        raise ValueError("Layout supports must partition the full 30 maps")
    return result


def make_references():
    """One common bank, intentionally shared across conditions and sources."""
    rng = np.random.default_rng(REFERENCE_SEED)
    references = np.asarray([rng.permutation(NCODES)
                             for _ in range(N_REFERENCE)], np.int64)
    validate_references(references)
    return references


def validate_references(references):
    references = np.asarray(references)
    if references.ndim != 2 or references.shape[1] != NCODES:
        raise ValueError("Expected full 49-code bijections")
    if not np.issubdtype(references.dtype, np.integer):
        raise ValueError("Bijections must be integer arrays")
    if not np.all(np.sort(references, axis=1) == np.arange(NCODES)):
        raise ValueError("Invalid complete-code bijection")


def validate_raw(raw):
    required = set(WORLD_KEYS) | {"tokens", "receiver_logits"}
    if not required.issubset(raw):
        raise ValueError(f"Missing protocol fields: {sorted(required - set(raw))}")
    n = len(raw["map_id"])
    shapes = {"map_id": (n,), "photo_ids": (n, 2), "positions": (n, 2),
              "shown": (n,), "tokens": (n, 2), "receiver_logits": (49, 2, 6)}
    for key, shape in shapes.items():
        if raw[key].shape != shape:
            raise ValueError(f"{key} shape {raw[key].shape}; expected {shape}")
        if not np.isfinite(raw[key]).all():
            raise ValueError(f"Nonfinite {key}")
        if key != "receiver_logits" and not np.issubdtype(raw[key].dtype, np.integer):
            raise ValueError(f"Expected integer {key}")
    if n == 0 or not np.all((raw["map_id"] >= 0) & (raw["map_id"] < 30)):
        raise ValueError("Invalid map IDs")
    if not np.array_equal(raw["positions"], world.MAPS[raw["map_id"]]):
        raise ValueError("Saved positions differ from fixed map table")
    if not np.all(np.isin(raw["shown"], (0, 1))):
        raise ValueError("Expected food-only or water-only final mask")
    if not np.all((raw["tokens"] >= 0) & (raw["tokens"] < VOCAB)):
        raise ValueError("Tokens outside the fixed seven-symbol vocabulary")
    if not np.all(raw["photo_ids"] >= 0):
        raise ValueError("Negative photo IDs")
    contexts = {}
    for i in range(n):
        context = (int(raw["shown"][i]), *map(int, raw["photo_ids"][i]))
        map_rows = contexts.setdefault(context, {})
        mid = int(raw["map_id"][i])
        if mid in map_rows:
            raise ValueError("Duplicate map in the same mask/photo context")
        map_rows[mid] = i
    expected = set(range(30))
    if any(set(rows) != expected for rows in contexts.values()):
        raise ValueError("Every mask/photo context must contain all 30 maps")
    photo_by_mask = [{key[1:] for key in contexts if key[0] == shown}
                     for shown in (0, 1)]
    if not photo_by_mask[0] or photo_by_mask[0] != photo_by_mask[1]:
        raise ValueError("The two masks must contain identical photo-pair supports")
    return contexts


def donor_trials(raw, p):
    """All ordered donor pairs per target, with no success-based filtering.

    Returned donor_food and donor_water are roles, not token positions. An old
    target has 2x2 pairs; a new target has 3x3 under this balanced partition.
    """
    contexts = validate_raw(raw)
    old = partitions(p)["old18"]
    food, water = world.MAPS[old].T
    targets, donor_food, donor_water = [], [], []
    for rows in contexts.values():
        for mid in range(30):
            tf, tw = world.MAPS[mid]
            fd = old[(food == tf) & (water != tw)]
            wd = old[(water == tw) & (food != tf)]
            if len(fd) == 0 or len(wd) == 0:
                raise ValueError("A target has no legal old-support donors")
            for fmid, wmid in itertools.product(fd, wd):
                targets.append(rows[mid])
                donor_food.append(rows[int(fmid)])
                donor_water.append(rows[int(wmid)])
    arrays = tuple(np.asarray(x, np.int64) for x in (targets, donor_food, donor_water))
    counts = np.bincount(arrays[0], minlength=len(raw["map_id"]))
    expected = np.where(np.isin(raw["map_id"], old), 4, 9)
    if not np.array_equal(counts, expected):
        raise ValueError("Donor counts differ from the fixed balanced map design")
    return (*arrays, counts)


def compose_codes(codes0, codes1, permutation=None):
    """Rename complete donors, splice the two digits, then undo the rename."""
    if permutation is None:
        return codes0 // VOCAB * VOCAB + codes1 % VOCAB
    renamed0, renamed1 = permutation[codes0], permutation[codes1]
    recombined = renamed0 // VOCAB * VOCAB + renamed1 % VOCAB
    return np.argsort(permutation)[recombined]


def reference_summary(rates, observed):
    rates = np.asarray(rates, np.float64)
    return {"count": int(len(rates)), "mean": float(rates.mean()),
            "quantiles": {str(q): float(np.quantile(rates, q))
                          for q in (0.025, 0.5, 0.975)},
            "min": float(rates.min()), "max": float(rates.max()),
            "observed_minus_mean": float(observed - rates.mean()),
            "rates": rates.tolist()}


def evaluate_protocol(raw, p, references=None, return_details=False):
    references = make_references() if references is None else references
    validate_references(references)
    ti, fd, wd, counts = donor_trials(raw, p)
    codes = raw["tokens"][:, 0] * VOCAB + raw["tokens"][:, 1]
    actions = raw["receiver_logits"].argmax(-1)
    target_positions = raw["positions"][ti]
    natural_correct = np.all(actions[codes] == raw["positions"], axis=-1)
    # Native communication is unchanged under any simultaneous S/R full rename.
    for permutation in references:
        if not np.array_equal(np.argsort(permutation)[permutation[codes]], codes):
            raise AssertionError("Recoding did not preserve complete messages")
    group_masks = {name: np.isin(raw["map_id"], ids)
                   for name, ids in partitions(p).items()}
    result, details = {}, {}
    for assignment, d0, d1 in (("food_water", fd, wd), ("water_food", wd, fd)):
        ordinary = compose_codes(codes[d0], codes[d1])
        correct = np.all(actions[ordinary] == target_positions, axis=-1)
        successes = np.bincount(ti, weights=correct, minlength=len(codes))
        target_rates = successes / counts
        null_target_rates = np.empty((len(references), len(codes)), np.float64)
        for r, permutation in enumerate(references):
            recoded = compose_codes(codes[d0], codes[d1], permutation)
            matched = np.all(actions[recoded] == target_positions, axis=-1)
            null_target_rates[r] = np.bincount(ti, weights=matched,
                                               minlength=len(codes)) / counts
        if return_details:
            details[assignment] = {"target_rates": target_rates,
                                   "reference_target_rates": null_target_rates,
                                   "donor_pairs_per_target": counts.copy()}
        result[assignment] = {}
        for group, eligible in group_masks.items():
            result[assignment][group] = {}
            for label in MASKS:
                selected = eligible.copy()
                if label != "pooled":
                    selected &= raw["shown"] == (0 if label == "food_only" else 1)
                observed = float(target_rates[selected].mean())
                rates = null_target_rates[:, selected].mean(axis=1)
                result[assignment][group][label] = {
                    "target_worlds": int(selected.sum()),
                    "donor_pairs": int(counts[selected].sum()),
                    "recombined_correct_pairs": int(successes[selected].sum()),
                    "recombined_J": observed,
                    "natural_J": float(natural_correct[selected].mean()),
                    "recoding": reference_summary(rates, observed)}
    return (result, details) if return_details else result


def average_scores(scores):
    """Equal-weight cells; average each paired reference BEFORE quantiles."""
    result = {}
    for assignment in ASSIGNMENTS:
        result[assignment] = {}
        for group in GROUPS:
            result[assignment][group] = {}
            for mask in MASKS:
                cells = [score[assignment][group][mask] for score in scores]
                observed = float(np.mean([c["recombined_J"] for c in cells]))
                refs = np.mean([c["recoding"]["rates"] for c in cells], axis=0)
                result[assignment][group][mask] = {
                    "recombined_J": observed,
                    "natural_J": float(np.mean([c["natural_J"] for c in cells])),
                    "recoding": reference_summary(refs, observed)}
    return result


def summarize_runs(rows):
    partition_cells = []
    keys = sorted({(r["seed"], r["partition"], r["condition"]) for r in rows})
    for seed, p, condition in keys:
        selected = [r for r in rows if (r["seed"], r["partition"], r["condition"])
                    == (seed, p, condition)]
        if sorted(r["direction"] for r in selected) != [0, 1]:
            raise ValueError("Each run requires both communication directions")
        partition_cells.append({"seed": seed, "partition": p, "condition": condition,
                                "scores": average_scores([r["scores"] for r in selected])})
    source_cells = []
    for seed, condition in sorted({(r["seed"], r["condition"]) for r in rows}):
        selected = [r for r in partition_cells if (r["seed"], r["condition"])
                    == (seed, condition)]
        source_cells.append({"seed": seed, "condition": condition,
                             "partitions": [r["partition"] for r in selected],
                             "scores": average_scores([r["scores"] for r in selected])})
    aggregate = {condition: average_scores([r["scores"] for r in source_cells
                                           if r["condition"] == condition])
                 for condition in CONDITIONS}
    comparisons = {}
    for assignment in ASSIGNMENTS:
        comparisons[assignment] = {}
        for group in GROUPS:
            per_source = []
            for seed in sorted({r["seed"] for r in source_cells}):
                by = {r["condition"]: r["scores"][assignment][group]["pooled"]
                      for r in source_cells if r["seed"] == seed}
                a, b = by["attention"], by["mean"]
                per_source.append({"seed": seed,
                    "recombined_J_difference": a["recombined_J"] - b["recombined_J"],
                    "reference_excess_difference": a["recoding"]["observed_minus_mean"]
                        - b["recoding"]["observed_minus_mean"],
                    "natural_J_difference": a["natural_J"] - b["natural_J"]})
            comparisons[assignment][group] = {"source_differences": per_source,
                "mean_differences": {key: float(np.mean([r[key] for r in per_source]))
                    for key in ("recombined_J_difference", "reference_excess_difference",
                                "natural_J_difference")}}
    return {"partition_cells": partition_cells, "source_cells": source_cells,
            "aggregate": aggregate, "attention_minus_mean": comparisons}


def collect_runs(out, step, development=False):
    paths = sorted((out / "social").glob(f"*/protocol_{step:04d}_d*.npz"))
    if not paths:
        raise ValueError("No saved protocols at the requested checkpoint")
    records = []
    for path in paths:
        match = re.fullmatch(r"s(\d+)_p([123])_(mean|attention)", path.parent.name)
        dmatch = re.fullmatch(rf"protocol_{step:04d}_d([01])\.npz", path.name)
        if not match or not dmatch:
            raise ValueError(f"Unexpected protocol path: {path}")
        seed, p, condition = int(match[1]), int(match[2]), match[3]
        cfg_path, result_path = path.parent / "config.json", path.parent / "result.json"
        cfg, result = json.loads(cfg_path.read_text()), json.loads(result_path.read_text())
        if result.get("status") != "complete":
            raise ValueError("Refusing an unfinished training run")
        if cfg.get("seed") != seed or cfg.get("partition") != p:
            raise ValueError("Directory/config identity mismatch")
        declared_condition = cfg.get("arm", cfg.get("condition"))
        if declared_condition != condition or cfg.get("updates") != step:
            raise ValueError("Condition or endpoint differs from config")
        records.append({"seed": seed, "partition": p, "condition": condition,
                        "direction": int(dmatch[1]), "path": str(path.resolve()),
                        "sha256": sha(path), "config_sha256": sha(cfg_path)})
    actual = {(r["seed"], r["partition"], r["condition"], r["direction"]) for r in records}
    seeds = sorted({r["seed"] for r in records})
    ps = sorted({r["partition"] for r in records})
    expected = set(itertools.product(seeds, ps, CONDITIONS, (0, 1)))
    if actual != expected or len(actual) != len(records):
        raise ValueError("Incomplete or duplicate source/partition/condition/direction grid")
    if not development and (len(seeds) != 4 or ps != [1, 2, 3] or step != 2400):
        raise ValueError("Formal assay requires four sources, all three partitions, step 2400")
    if development and step != 40:
        raise ValueError("This development assay is fixed to checkpoint 0040")
    return records


def render_report(payload):
    lines = ["# v24 保存协议的供体符号重组评估", "",
        "本评估不重新运行主体。供体均为 old18 中、与目标同末帧 mask 和照片对的自然贪心消息。",
        "两种槽位—资源分配都报告；不选择较好者。每目标内均匀平均全部合法供体对，再目标等权。",
        "重组使用实验者地图身份挑供体，不是主体自主生成新组合，不证明句法或因果发现。", "",
        f"输入：{len(payload['runs'])} 份方向协议；{payload['source_count']} 个来源。",
        "200 个完整49码双射使用同一固定随机流，跨臂、方向、分区和来源共用；不是200个独立主体。",
        "参考分位数先平均两个通信方向、三个分区，再平均来源后计算；不是均值的置信区间。", "",
        "| 条件 | token0/token1供体 | 目标支持 | 自然J | 重组J | 重编码均值 | 重编码2.5%–97.5% |",
        "|---|---|---|---:|---:|---:|---:|"]
    for condition, assignment, group in itertools.product(CONDITIONS, ASSIGNMENTS, GROUPS):
        c = payload["aggregate"][condition][assignment][group]["pooled"]
        null = c["recoding"]
        lines.append(f"| {condition} | {assignment} | {group} | {100*c['natural_J']:.2f}% | "
                     f"{100*c['recombined_J']:.2f}% | {100*null['mean']:.2f}% | "
                     f"{100*null['quantiles']['0.025']:.2f}%–{100*null['quantiles']['0.975']:.2f}% |")
    lines += ["", "完整逐来源配对差、两 mask、200 个参考率和每份输入哈希见 structural_assay.json。",
        "参考仅比较固定采样的整体重命名，未穷举49!种双射；高于参考说明当前符号坐标下的局部重组有用，不能替代自然 new12 表达成绩。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="Completed v24 batch directory")
    parser.add_argument("--step", type=int, default=2400)
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    rows = collect_runs(args.out, args.step, args.dev)
    references = make_references()
    reference_world = None
    for row in rows:
        raw = read_npz(row["path"])
        world_fingerprint = {k: array_sha(raw[k]) for k in WORLD_KEYS}
        if reference_world is None:
            reference_world = world_fingerprint
        elif world_fingerprint != reference_world:
            raise ValueError("Input protocols do not share the fixed evaluation worlds")
        row["scores"] = evaluate_protocol(raw, row["partition"], references)
    summary = summarize_runs(rows)
    payload = {"schema": "v24_old_donor_recombination_v1", "status": "complete",
        "development": args.dev, "step": args.step,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_count": len({r["seed"] for r in rows}), "runs": rows,
        "reference": {"seed": REFERENCE_SEED, "count": N_REFERENCE,
            "scope": "One common bank shared by every run/direction/assignment",
            "sha256": array_sha(references), "permutations": references.tolist()},
        "source_fingerprints": {str(Path(__file__).resolve()): sha(__file__),
            str(WORLD_PATH): sha(WORLD_PATH),
            str(ROOT.parent / "redesign_v0.20" / "temporal_world.py"):
                sha(ROOT.parent / "redesign_v0.20" / "temporal_world.py")},
        "evaluation_world_fingerprints": reference_world, **summary}
    output = args.out / "structural_assay.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (args.out / "结构重组评估.md").write_text(render_report(payload))
    print(json.dumps({"status": "complete", "protocols": len(rows),
                      "source_count": payload["source_count"], "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
