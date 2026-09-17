"""Read-only v0.5 analysis: independently recompute outcomes, then write new reports.

The script does not import the environment, runner, agents, or PyTorch. It cannot
train or replay policies. All final reward/transition checks use stored NPZ traces
and independent NumPy expressions; checkpoint means remain recorded runner data.

Usage (project root):
  .venv/bin/python redesign_v0.5/analyze_stages.py
  .venv/bin/python redesign_v0.5/analyze_stages.py --check-only
  .venv/bin/python redesign_v0.5/analyze_stages.py --input PATH --seeds 99001 --check-only
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONDITIONS = {
    "A": ("known_sequence", "hidden_sequence", "known_atomic", "hidden_atomic",
          "hidden_single", "hidden_blocked", "hidden_sequence_direct",
          "hidden_sequence_mixed", "hidden_sequence_holdout"),
    "B": ("immediate_continue", "delay_memory", "delay_reset", "delay_replay"),
    "C": ("persistent_communication", "persistent_channel_removed", "persistent_blocked",
          "persistent_known", "persistent_delayed"),
}
LABELS = {
    "known_sequence": "目标可见\n两符号",
    "hidden_sequence": "目标隐藏\n两符号",
    "known_atomic": "目标可见\n25选1",
    "hidden_atomic": "目标隐藏\n25选1",
    "hidden_single": "目标隐藏\n5选1",
    "hidden_blocked": "目标隐藏\n始终关通道",
    "hidden_sequence_direct": "目标隐藏\n直接四地点",
    "hidden_sequence_mixed": "目标隐藏\n相同批次混排",
    "hidden_sequence_holdout": "目标隐藏\n8图训练",
    "immediate_continue": "即时继续",
    "delay_memory": "延迟并保持",
    "delay_reset": "延迟后清空",
    "delay_replay": "延迟后回放",
    "persistent_communication": "持续通信",
    "persistent_channel_removed": "此阶段关通道",
    "persistent_blocked": "始终关通道",
    "persistent_known": "目标改为可见",
    "persistent_delayed": "加入延迟",
}
MODES = ("normal", "shuffle", "blank", "stochastic", "erase_memory")
MAPS = np.asarray(list(itertools.permutations(range(4), 2)), dtype=np.int64)
OUT4 = {(0, 1), (1, 2), (2, 3), (3, 0)}
TOLERANCE = 1e-7  # Subset means in result.json were stored as float32.


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(a, b, context):
    require(np.array_equal(a, b), f"{context}: arrays differ")


def near(a, b, context):
    require(np.allclose(a, b, atol=TOLERANCE, rtol=0), f"{context}: {a} != {b}")


def selected_mean(reward, mask):
    return {"n_steps": int(mask.sum()), "mean_reward": float(reward[mask].mean()) if mask.any() else None}


def trace_audit(path, plan, stated, photos):
    """Recompute collection, capacity/consumption, respawn, and legal history."""
    with np.load(path, allow_pickle=False) as stored:
        a = {key: stored[key] for key in stored.files}
    required = {"scout", "step", "episode", "positions", "inventory", "goals", "menu",
                "photo_ids", "sent", "delivered", "action", "place", "reward", "gathered",
                "overflow", "next_inventory", "next_positions", "refill_uniform", "history"}
    require(required.issubset(a), f"{path}: missing trace columns {required - a.keys()}")
    n, h = int(stated["episodes"]), int(stated["horizon"])
    require(n % 2 == 0 and h == plan.get("horizon", 1), f"{path}: episode/horizon mismatch")
    rows, n2 = n * h, n // 2
    require(all(len(v) == rows for v in a.values()), f"{path}: inconsistent row counts")
    equal(np.sort(a["menu"], axis=1), np.tile(np.arange(4), (rows, 1)), f"{path}: menu is not a permutation")
    require(np.all((a["positions"] >= 0) & (a["positions"] < 4)) and
            np.all(a["positions"][:, 0] != a["positions"][:, 1]), f"{path}: invalid map")
    require(np.all((a["action"] >= 0) & (a["action"] < 4)), f"{path}: invalid action")
    require(np.all((a["goals"] >= 0) & (a["goals"] <= 1)), f"{path}: invalid goal")
    require(np.all((a["inventory"] >= 0) & (a["inventory"] <= 2)), f"{path}: invalid inventory")
    require(a["sent"].shape == (rows, plan["length"]), f"{path}: invalid message shape")
    for column in ("sent", "delivered"):
        require(np.issubdtype(a[column].dtype, np.integer), f"{path}: noninteger channel")
        require(np.all((a[column] >= 0) & (a[column] < plan["vocab"])), f"{path}: token out of range")
    equal(a["place"], a["menu"][np.arange(rows), a["action"]], f"{path}: menu routing")

    # Deliberately separate from camp.collect: identify the item at each chosen
    # site, then increment, discard overflow, and consume the current demand.
    picked = np.zeros((rows, 2), np.int64)
    for resource in (0, 1):
        picked[:, resource] = (a["place"] == a["positions"][:, resource])
    before_consumption = a["inventory"].astype(np.int64) + picked
    overflow = np.where(before_consumption > 2, before_consumption - 2, 0)
    capped = before_consumption - overflow
    reward = (capped[np.arange(rows), a["goals"]] >= 1).astype(np.float64)
    after = capped.copy()
    after[np.arange(rows), a["goals"]] -= reward.astype(np.int64)
    equal(picked, a["gathered"], f"{path}: gathered")
    equal(overflow, a["overflow"], f"{path}: overflow")
    equal(reward, a["reward"], f"{path}: reward")
    equal(after, a["next_inventory"], f"{path}: next inventory")
    next_map = a["positions"].copy()
    if h > 1:
        require(np.all((a["refill_uniform"] >= 0) & (a["refill_uniform"] < 1)), f"{path}: respawn draws")
        for resource in (0, 1):
            mask = picked[:, resource].astype(bool)
            rank = (3 * a["refill_uniform"][mask]).astype(np.int64)
            excluded = a["positions"][mask, 1 - resource]
            # The rank-th member of [0,1,2,3] with excluded removed.
            next_map[mask, resource] = rank + (rank >= excluded)
    equal(next_map, a["next_positions"], f"{path}: respawn locations")

    mode = path.stem.removeprefix("final_")
    if plan.get("blocked") or mode == "blank":
        equal(a["delivered"], np.zeros_like(a["delivered"]), f"{path}: blocked delivery")
    elif mode != "shuffle":
        equal(a["sent"], a["delivered"], f"{path}: normal delivery")
    else:
        keys = np.column_stack((a["scout"], a["step"], a["goals"], a["inventory"]))
        for key in np.unique(keys, axis=0):
            mask = (keys == key).all(1)
            sent_values, sent_counts = np.unique(a["sent"][mask], axis=0, return_counts=True)
            recv_values, recv_counts = np.unique(a["delivered"][mask], axis=0, return_counts=True)
            equal(sent_values, recv_values, f"{path}: conditional message values {key.tolist()}")
            equal(sent_counts, recv_counts, f"{path}: conditional message counts {key.tolist()}")

    # Both modes and later stages must retain actual personal collection history,
    # never counterfactual events from another intervention's closed-loop path.
    recorded_world = hashlib.sha256()
    for scout in (0, 1):
        previous = None
        for step in range(h):
            indices = np.flatnonzero((a["scout"] == scout) & (a["step"] == step))
            require(len(indices) == n2, f"{path}: direction/step size")
            indices = indices[np.argsort(a["episode"][indices])]
            equal(a["episode"][indices], np.arange(n2), f"{path}: episode IDs")
            for column in ("positions", "photo_ids", "goals", "menu", "refill_uniform"):
                recorded_world.update(np.ascontiguousarray(a[column][indices]).tobytes())
            if previous is None:
                equal(a["inventory"][indices], np.zeros((n2, 2)), f"{path}: initial inventory")
                expected_history = np.zeros((n2, 14))
            else:
                equal(a["positions"][indices], next_map[previous], f"{path}: map continuity")
                equal(a["inventory"][indices], after[previous], f"{path}: inventory continuity")
                last_event = np.column_stack((np.eye(4)[a["place"][previous]], picked[previous], picked[previous].sum(1) == 0))
                expected_history = np.column_stack((a["history"][previous, 7:], last_event))
                for resource in (0, 1):
                    unchanged = picked[previous, resource] == 0
                    equal(a["photo_ids"][indices[unchanged], resource],
                          a["photo_ids"][previous[unchanged], resource], f"{path}: untouched photo continuity")
            equal(a["history"][indices], expected_history, f"{path}: personal history")
            previous = indices
    require(recorded_world.hexdigest() == stated["world_sha256"], f"{path}: world hash differs from stored trace")
    for resource, category in enumerate(("food", "water")):
        ids = a["photo_ids"][:, resource]
        require(np.all((ids >= 0) & (ids < len(photos))), f"{path}: invalid photo IDs")
        require(all(photos[int(i)]["category"] == category and photos[int(i)]["split"] == "test"
                    for i in np.unique(ids)), f"{path}: photo category/split")

    episode_id = a["scout"] * n2 + a["episode"]
    total_per_episode = np.bincount(episode_id, weights=reward, minlength=n)
    out4 = np.asarray([tuple(p) in OUT4 for p in a["positions"]])
    metrics = {
        "mean_reward": float(reward.mean()),
        "perfect_episode_rate": float((total_per_episode == h).mean()),
        "mean_total_episode_reward": float(total_per_episode.mean()),
        "episodes": n, "horizon": h, "n_steps": rows,
        "per_step": [float(reward[a["step"] == t].mean()) for t in range(h)],
        "direction_means": [float(reward[a["scout"] == s].mean()) for s in (0, 1)],
        "goal_means": [selected_mean(reward, a["goals"] == g) for g in (0, 1)],
        "direction_goal": [[selected_mean(reward, (a["scout"] == s) & (a["goals"] == g))
                             for g in (0, 1)] for s in (0, 1)],
        "map_subsets": {"in8": selected_mean(reward, ~out4), "out4": selected_mean(reward, out4)},
        "map_means": [{"food_site": int(p[0]), "water_site": int(p[1]),
                       **selected_mean(reward, (a["positions"] == p).all(1))} for p in MAPS],
        "maps_were_withheld": bool(plan.get("holdout", False)),
    }
    for key in ("mean_reward", "perfect_episode_rate", "per_step", "direction_means"):
        near(metrics[key], stated[key], f"{path}: stated {key}")
    for key, subset in (("seen_map_reward", "in8"), ("heldout_map_reward", "out4")):
        if key in stated:
            near(metrics["map_subsets"][subset]["mean_reward"], stated[key], f"{path}: {key}")
    return metrics, a, {"file": str(path), "sha256": sha(path), "steps_checked": rows,
                        "status": "passed", "checks": ["reward", "gathering", "capacity", "consumption", "respawn",
                         "continuity", "personal_history", "private_menu", "delivery", "photo_split", "reported_means"]}


def compare_exogenous(a, b, context):
    for key in ("scout", "step", "episode", "goals", "menu", "refill_uniform"):
        equal(a[key], b[key], f"{context}: exogenous {key}")
    initial = a["step"] == 0
    for key in ("positions", "photo_ids", "inventory", "history"):
        equal(a[key][initial], b[key][initial], f"{context}: initial {key}")
    # After intervention changes actions, subsequent maps/inventory/photos need
    # not match. Their own legal transitions were checked in trace_audit.


def read_run(directory, stage, photos):
    result = read_json(directory / "result.json")
    config = read_json(directory / "config.json")
    curve = read_json(directory / "curve.json")
    for key in ("seed", "condition", "plan", "updates", "batch", "initial_sha256"):
        require(result[key] == config[key], f"{directory}: config/result {key}")
    require(result["condition"] in CONDITIONS[stage], f"{directory}: unrecognized condition")
    require([row["update"] for row in curve] == config["checkpoints"], f"{directory}: incomplete checkpoints")
    require(curve[-1]["update"] == result["updates"], f"{directory}: terminal checkpoint missing")
    require(result["frozen_projection_verified"], f"{directory}: projection not verified")
    metric, audits, baseline = {}, {}, None
    for mode in MODES:
        metric[mode], trace, audits[mode] = trace_audit(directory / f"final_{mode}.npz", result["plan"], result["scores"][mode], photos)
        if baseline is None:
            baseline = trace
        else:
            compare_exogenous(baseline, trace, f"{directory}: normal vs {mode}")
    normal, shuffle = metric["normal"], metric["shuffle"]
    difference = {
        "mean_reward": normal["mean_reward"] - shuffle["mean_reward"],
        "direction_means": (np.asarray(normal["direction_means"]) - shuffle["direction_means"]).tolist(),
        "per_step": (np.asarray(normal["per_step"]) - shuffle["per_step"]).tolist(),
        "goal_means": [normal["goal_means"][g]["mean_reward"] - shuffle["goal_means"][g]["mean_reward"] for g in (0, 1)],
        "map_subsets": {subset: normal["map_subsets"][subset]["mean_reward"] - shuffle["map_subsets"][subset]["mean_reward"] for subset in ("in8", "out4")},
    }
    rows = [json.loads(line) for line in (directory / "training.jsonl").read_text().splitlines() if line.strip()]
    require([row["update"] for row in rows] == list(range(1, result["updates"] + 1)), f"{directory}: training record gap")
    require(all(np.isfinite(row["reward"]) for row in rows), f"{directory}: nonfinite training reward")
    schedule_path = directory / "training_schedule.json"
    schedule = None
    if schedule_path.exists():
        schedule = read_json(schedule_path)
        equal(schedule["batch_identities"], [row["batch_identity"] for row in rows], f"{directory}: batch order")
        equal(schedule["levels"], [row["active_sites"] for row in rows], f"{directory}: difficulty schedule")
        equal(sorted(schedule["batch_identities"]), list(range(result["updates"])), f"{directory}: each batch used once")
    compact_curve = [{"update": row["update"], "scores": {
        mode: {key: row["scores"][mode][key] for key in
               ("mean_reward", "direction_means", "per_step", "episodes", "horizon")}
        for mode in MODES}} for row in curve]
    return {"stage": stage, "seed": result["seed"], "condition": result["condition"],
            "directory": str(directory), "plan": result["plan"], "updates": result["updates"],
            "batch": result["batch"], "seconds": result["seconds"],
            "initial_sha256": result["initial_sha256"], "final_sha256": result["final_sha256"],
            "trainable_parameters": config["trainable_parameters"], "source_hashes": config["source_hashes"],
            "training_schedule": schedule,
            "training_worlds_by_batch_identity": {str(row.get("batch_identity", row["update"] - 1)): row["world_sha256"] for row in rows},
            "scores": metric, "normal_minus_shuffle": difference, "curve": compact_curve,
            "training_record_count": len(rows), "training_sha256": sha(directory / "training.jsonl"),
            "result_sha256": sha(directory / "result.json"), "trace_audit": audits}, baseline


def describe(values):
    values = np.asarray(values, dtype=float)
    return {"n_seeds": len(values), "mean": float(values.mean()),
            "min": float(values.min()), "max": float(values.max()),
            "values": values.tolist()}


def aggregate(runs):
    groups = {}
    for stage, names in CONDITIONS.items():
        groups[stage] = {}
        for name in names:
            selected = sorted((run for run in runs if run["stage"] == stage and run["condition"] == name), key=lambda r: r["seed"])
            if not selected:
                continue
            group = {"seeds": [run["seed"] for run in selected], "scores": {}, "normal_minus_shuffle": {}}
            for mode in MODES:
                group["scores"][mode] = {
                    key: describe([run["scores"][mode][key] for run in selected])
                    for key in ("mean_reward", "perfect_episode_rate", "mean_total_episode_reward")}
                for key, count in (("direction_means", 2), ("per_step", selected[0]["scores"][mode]["horizon"])):
                    group["scores"][mode][key] = [describe([run["scores"][mode][key][i] for run in selected]) for i in range(count)]
                group["scores"][mode]["map_subsets"] = {
                    subset: describe([run["scores"][mode]["map_subsets"][subset]["mean_reward"] for run in selected])
                    for subset in ("in8", "out4")}
            group["normal_minus_shuffle"]["mean_reward"] = describe([run["normal_minus_shuffle"]["mean_reward"] for run in selected])
            group["normal_minus_shuffle"]["direction_means"] = [describe([run["normal_minus_shuffle"]["direction_means"][i] for run in selected]) for i in (0, 1)]
            group["normal_minus_shuffle"]["map_subsets"] = {
                subset: describe([run["normal_minus_shuffle"]["map_subsets"][subset] for run in selected])
                for subset in ("in8", "out4")}
            group["updates"] = sorted({run["updates"] for run in selected})
            group["trainable_parameters"] = sorted({tuple(run["trainable_parameters"]) for run in selected})
            groups[stage][name] = group
    return groups


def paired_comparisons(runs):
    index = {(r["stage"], r["seed"], r["condition"]): r for r in runs}
    specs = {
        "A": [("hidden_sequence", "hidden_sequence_direct", "逐级课程减直接四地点，同起点及更新预算；难度暴露不同"),
              ("hidden_sequence", "hidden_sequence_mixed", "逐级课程减同批次混排，完整训练世界/照片/目标多重集匹配"),
              ("known_sequence", "hidden_sequence", "目标可见减隐藏，同两符号接口和逐级课程"),
              ("hidden_sequence", "hidden_atomic", "两符号减25选1，同名义容量；架构/参数不同"),
              ("hidden_sequence", "hidden_single", "两符号减5选1，容量与架构同时不同"),
              ("hidden_sequence", "hidden_blocked", "通信减从头关通道"),
              ("hidden_sequence_direct", "hidden_sequence_holdout", "直接12图训练减直接8图训练，训练地图分布不同")],
        "B": [("delay_memory", "immediate_continue", "延迟保持减即时继续，共同A隐藏双符号终点"),
              ("delay_memory", "delay_reset", "延迟保持减清空；后者丢失现场信息"),
              ("delay_replay", "delay_memory", "外部回放减内部保持，同源继续训练")],
        "C": [("persistent_communication", "persistent_channel_removed", "持续通信减此阶段关通道，共同A通信起点"),
              ("persistent_communication", "persistent_blocked", "持续通信减始终关通道；A阶段历史不同"),
              ("persistent_known", "persistent_communication", "改为目标可见减目标隐藏，共同A通信起点"),
              ("persistent_delayed", "persistent_communication", "加入延迟减即时，共同A通信起点")],
    }
    output = []
    seeds = sorted({r["seed"] for r in runs})
    for stage, comparisons in specs.items():
        for left, right, interpretation in comparisons:
            pairs = [(index[stage, seed, left], index[stage, seed, right]) for seed in seeds
                     if (stage, seed, left) in index and (stage, seed, right) in index]
            if not pairs:
                continue
            require(all(a["updates"] == b["updates"] and a["batch"] == b["batch"] for a, b in pairs),
                    f"{stage}/{left}-{right}: unequal current-stage budget")
            values = [a["scores"]["normal"]["mean_reward"] - b["scores"]["normal"]["mean_reward"] for a, b in pairs]
            gaps = [a["normal_minus_shuffle"]["mean_reward"] - b["normal_minus_shuffle"]["mean_reward"] for a, b in pairs]
            output.append({"stage": stage, "left": left, "right": right, "interpretation": interpretation,
                           "seeds": [a["seed"] for a, _ in pairs], "normal_difference": describe(values),
                           "message_gap_difference": describe(gaps),
                           "same_initial_parameters": [a["initial_sha256"] == b["initial_sha256"] for a, b in pairs]})
    return output


def provenance_audit(runs):
    index = {(r["stage"], r["seed"], r["condition"]): r for r in runs}
    records = []
    for run in runs:
        stage, seed, name = run["stage"], run["seed"], run["condition"]
        if stage == "A":
            continue
        origin_name = "hidden_blocked" if name == "persistent_blocked" else "hidden_sequence"
        origin = index.get(("A", seed, origin_name))
        require(origin is not None, f"{stage}/{seed}/{name}: missing completed A origin")
        require(run["initial_sha256"] == origin["final_sha256"], f"{stage}/{seed}/{name}: warmstart mismatch")
        records.append({"stage": stage, "seed": seed, "condition": name,
                        "origin": f"A/s{seed}_{origin_name}", "status": "passed",
                        "origin_updates": origin["updates"], "additional_updates": run["updates"],
                        "total_updates": origin["updates"] + run["updates"],
                        "initial_sha256": run["initial_sha256"]})
    for seed in sorted({run["seed"] for run in runs}):
        for names in (("known_sequence", "hidden_sequence", "hidden_blocked", "hidden_sequence_holdout",
                       "hidden_sequence_direct", "hidden_sequence_mixed"),
                      ("known_atomic", "hidden_atomic")):
            family = [index["A", seed, name] for name in names if ("A", seed, name) in index]
            if len(family) > 1:
                require(len({run["initial_sha256"] for run in family}) == 1, f"A/{seed}: matched interface start mismatch")
                records.append({"stage": "A", "seed": seed, "conditions": [run["condition"] for run in family],
                                "same_initial_parameters": True, "status": "passed"})
        course, mixed = index.get(("A", seed, "hidden_sequence")), index.get(("A", seed, "hidden_sequence_mixed"))
        if course and mixed:
            require(course["training_worlds_by_batch_identity"] == mixed["training_worlds_by_batch_identity"],
                    f"A/{seed}: course/mixed training world multiset differs")
            records.append({"stage": "A", "seed": seed, "conditions": ["hidden_sequence", "hidden_sequence_mixed"],
                            "training_batch_multiset_world_hashes_match": True, "status": "passed"})
        c_com, c_never = index.get(("C", seed, "persistent_communication")), index.get(("C", seed, "persistent_blocked"))
        if c_com and c_never:
            a_com, a_never = index["A", seed, "hidden_sequence"], index["A", seed, "hidden_blocked"]
            require(a_com["updates"] == a_never["updates"] and a_com["batch"] == a_never["batch"], f"C/{seed}: different A budgets")
        immediate, replay = index.get(("B", seed, "immediate_continue")), index.get(("B", seed, "delay_replay"))
        if immediate and replay:
            require(immediate["final_sha256"] == replay["final_sha256"], f"B/{seed}: replay/immediate final state fingerprints differ")
            require(immediate["curve"] == replay["curve"], f"B/{seed}: replay/immediate curves differ")
            require(immediate["scores"] == replay["scores"], f"B/{seed}: replay/immediate outcomes differ")
            records.append({"stage": "B", "seed": seed, "conditions": ["immediate_continue", "delay_replay"],
                            "identical_final_state_fingerprints_curves_and_scores": True, "status": "passed"})
    return records


def protocol_summary(source, runs):
    """Recompute descriptive summaries from the independent enumeration analysis.

    The policy probes themselves are NOT rerun here. Ratios, reference means,
    bijections, checkpoint hashes and original/extended consistency are checked.
    """
    original_path = source / "protocol_analysis.json"
    if not original_path.exists():
        return None
    original = read_json(original_path)
    path = source / "protocol_analysis_extended_null.json"
    if not path.exists():
        path = original_path
    data = read_json(path)
    require(data["status"] == "complete", "Protocol analysis is incomplete")
    require(data["synthetic_checks"]["status"] == "passed", "Protocol synthetic checks did not pass")
    originals = {(r["seed"], r["condition"]): r for r in original["runs"]}
    index = {(r["seed"], r["condition"]): r for r in runs if r["stage"] == "A"}

    def checked_ratio(item, context):
        n, d = item["numerator"], item["denominator"]
        require(0 <= n <= d, f"{context}: invalid count")
        if d:
            near(item["rate"], n / d, context)
        else:
            require(item["rate"] is None, f"{context}: empty denominator must be undefined")
        return item["rate"]

    rows = []
    for r in data["runs"]:
        key = r["seed"], r["condition"]
        require(key in index, f"Protocol checkpoint missing from completed A runs: {key}")
        require(sha(r["final_checkpoint"]) == r["final_sha256"], f"Protocol checkpoint changed: {key}")
        require(Path(r["final_checkpoint"]).resolve().parent == Path(index[key]["directory"]).resolve(), f"Protocol wrong run: {key}")
        require(len(r["directions"]) == 2, f"Protocol must include both directions: {key}")
        for d, previous in zip(r["directions"], originals[key]["directions"]):
            for k, value in previous.items():
                require(d[k] == value, f"Extended protocol changed original values: {key}/{k}")
            row = {"seed": key[0], "condition": key[1], "scout": d["scout"], "collector": d["collector"], "counts": {}}
            cross = d["phases"]["validation"]["cross_goal"]
            for short, metric in (("native", "native_goal"), ("switched", "goal_switched_with_message_fixed"),
                                  ("both_goals", "same_message_correct_for_both_goals")):
                row[short] = checked_ratio(cross[metric], f"{key}/{short}")
                row["counts"][short] = cross[metric]
            if "fragments" in d:
                frag = d["fragments"]["validation"]
                row["strict_fragment"] = checked_ratio(frag["strict_transfer_all"], f"{key}/fragment")
                conditional = frag["strict_transfer_when_both_maps_both_goals_correct"]
                row["conditional_fragment"] = checked_ratio(conditional, f"{key}/conditional fragment")
                row["eligible_fraction"] = conditional["denominator"] / frag["strict_transfer_all"]["denominator"]
                row["counts"]["strict_fragment"] = frag["strict_transfer_all"]
                row["counts"]["conditional_fragment"] = conditional
                row["selected_food_water_slots"] = d["fragments"]["selected_assignment_food_water_slots"]
            if "whole_message_recoding" in d:
                rec = d["whole_message_recoding"]
                require(rec["menu_invariance_checked_for_all_codes_and_goals"], f"{key}: unchecked menu collapse")
                require(rec["calibration_selection_repeated_for_every_recoding"], f"{key}: unmatched calibration")
                require(len(rec["records"]) == rec["replicates"] == 100, f"{key}: missing recodings")
                values = []
                for item in rec["records"]:
                    equal(sorted(item["old_to_new_code"]), list(range(25)), f"{key}: recoding bijection")
                    values.append(checked_ratio(item["validation_strict_transfer_all"], f"{key}: recoding rate"))
                stat = rec["strict_transfer_all"]
                near(row["strict_fragment"], stat["observed"], f"{key}: recoding observed")
                near(np.mean(values), stat["reference_mean"], f"{key}: recoding mean")
                near(stat["observed"] - np.mean(values), stat["observed_minus_reference_mean"], f"{key}: recoding difference")
                row["recoding_mean"] = float(np.mean(values))
                row["fragment_minus_recoding"] = row["strict_fragment"] - row["recoding_mean"]
            if "heldout_stitch_validation" in d:
                item = d["heldout_stitch_validation"]["both_goals_correct_all"]
                row["heldout_stitch"] = checked_ratio(item, f"{key}: heldout stitch")
                row["counts"]["heldout_stitch"] = item
            rows.append(row)
    groups = {}
    for name in CONDITIONS["A"]:
        selected = [row for row in rows if row["condition"] == name]
        if not selected:
            continue
        seeds = sorted({row["seed"] for row in selected})
        metrics = ("native", "switched", "both_goals", "strict_fragment", "eligible_fraction",
                   "recoding_mean", "fragment_minus_recoding", "heldout_stitch")
        group = {"seeds": seeds}
        for metric in metrics:
            if not all(metric in row for row in selected):
                continue
            group[metric] = describe([float(np.mean([row[metric] for row in selected if row["seed"] == seed])) for seed in seeds])
        groups[name] = group
    return {"source": str(path), "source_sha256": sha(path), "original_source": str(original_path),
            "original_source_sha256": sha(original_path), "evaluation_amendment": data["evaluation_amendment"],
            "exploratory_extension": data.get("exploratory_extension"), "photo_splits": data["photo_splits"],
            "audit": "passed: checkpoint file hashes, unchanged original values, integer ratios, 100 bijections/direction and recoding reference means",
            "aggregation": "mean of two directions within each seed, then equal mean of three trained seeds; conditional rates with zero denominator remain null in rows",
            "groups": groups, "direction_rows": rows}


def fixed_context_summary(source, runs):
    path = source / "fixed_context_probe.json"
    if not path.exists():
        return None
    data = read_json(path)
    require(data["status"] == "complete", "Fixed-context probe incomplete")
    index = {(r["stage"], r["seed"], r["condition"]): r for r in runs}
    for row in data["rows"]:
        run = index[row["stage"], row["seed"], row["condition"]]
        require(sha(Path(run["directory"]) / "final.pt") == row["checkpoint_sha256"], "Fixed-context checkpoint hash mismatch")
        require(sha(Path(run["directory"]) / "final_normal.npz") == row["trace_sha256"], "Fixed-context trace hash mismatch")
        near(row["normal_mean"], run["scores"]["normal"]["mean_reward"], "Fixed-context native reward")
        n = sum(r["cases"] for r in row["rows"])
        require(n == row["cases"], "Fixed-context case count mismatch")
        for column, numerator in (("normal_mean", "native_reward_sum"), ("intervened_mean", "intervened_reward_sum")):
            near(row[column], sum(r[numerator] for r in row["rows"]) / n, f"Fixed-context {column}")
        near(row["normal_mean"] - row["intervened_mean"], row["current_message_reward_drop"], "Fixed-context difference")
    require(sum(r["cases"] for r in data["rows"]) == data["cases_checked"], "Fixed-context total cases mismatch")
    groups = {}
    for stage in ("B", "C"):
        groups[stage] = {}
        for name in CONDITIONS[stage]:
            selected = sorted([r for r in data["rows"] if r["stage"] == stage and r["condition"] == name], key=lambda r: r["seed"])
            if selected:
                groups[stage][name] = {k: describe([r[k] for r in selected]) for k in
                                       ("normal_mean", "intervened_mean", "current_message_reward_drop")}
    return {"source": str(path), "source_sha256": sha(path), "cases_checked": data["cases_checked"],
            "runs": data["runs"], "groups": groups, "rows": data["rows"],
            "audit": "passed: checkpoint/trace hashes, native rewards and counterfactual sums independently checked; policy actions checked by fixed_context_probe.py"}


def make_plots(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    plt.rcParams.update({"font.family": ["PingFang SC", "DejaVu Sans"], "font.size": 11,
                         "axes.unicode_minus": False, "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    plots = []
    colors = ("#377EB8", "#D88D32", "#9566AA", "#198C82")
    for stage, all_names in CONDITIONS.items():
        names = [name for name in all_names if name in summary["groups"][stage]]
        if not names:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(max(8.5, len(names) * 1.4), 7), sharex=True,
                                 gridspec_kw={"height_ratios": [1.25, 1]}, constrained_layout=True)
        for i, name in enumerate(names):
            g = summary["groups"][stage][name]
            for row, values in enumerate((g["scores"]["normal"]["mean_reward"]["values"],
                                           g["normal_minus_shuffle"]["mean_reward"]["values"])):
                jitter = np.linspace(-.10, .10, len(values))
                axes[row].scatter(i + jitter, np.asarray(values) * 100, s=32,
                                  c=[colors[j % len(colors)] for j in range(len(values))], alpha=.8, zorder=3)
                axes[row].plot([i - .22, i + .22], [np.mean(values) * 100] * 2, color="#202A32", lw=2.3)
        axes[0].set_ylabel("正常模式每步奖励（%）")
        axes[1].set_ylabel("正常 − 打乱（百分点）")
        axes[0].set_ylim(0, 104)
        no_message = 100 * (summary["bounds"]["dynamic_no_message"] if stage == "C" else .25)
        axes[0].axhline(no_message, ls=":", color="#707880", lw=1, label="无通道理想期望上界")
        axes[0].legend(frameon=False, fontsize=10, loc="lower left")
        axes[1].axhline(0, color="#90979D", lw=.8)
        for ax in axes:
            ax.grid(axis="y", alpha=.15)
        axes[1].set_xticks(range(len(names)), [LABELS[name] for name in names])
        fig.suptitle(f"阶段 {stage}：探索种子终点分布；短黑线为均值", fontsize=13)
        stem = f"stage_{stage}_comparison"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=240)
        plt.close(fig)
        plots.append(stem + ".png")

        ncols = min(3, len(names)); nrows = int(np.ceil(len(names) / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.1 * ncols, 3.3 * nrows), squeeze=False,
                                 sharey=True, constrained_layout=True)
        for ax, name in zip(axes.flat, names):
            selected = sorted((run for run in summary["runs"] if run["stage"] == stage and run["condition"] == name), key=lambda r: r["seed"])
            for mode, color, style in (("normal", "#117D7C", "-"), ("shuffle", "#B87536", "--")):
                points = sorted(set.intersection(*({row["update"] for row in run["curve"]} for run in selected)))
                means = []
                for run in selected:
                    x = [row["update"] for row in run["curve"]]
                    y = [100 * row["scores"][mode]["mean_reward"] for row in run["curve"]]
                    ax.plot(x, y, color=color, ls=style, lw=.8, alpha=.22)
                    by_update = {row["update"]: row for row in run["curve"]}
                    means.append([100 * by_update[t]["scores"][mode]["mean_reward"] for t in points])
                ax.plot(points, np.mean(means, axis=0), color=color, ls=style, lw=2,
                        marker="o", ms=3, label="正常" if mode == "normal" else "打乱")
            ax.axhline(no_message, ls=":", color="#8B9299", lw=.8)
            ax.set(title=LABELS[name].replace("\n", " "), xlabel="本阶段训练更新", ylim=(0, 104))
            ax.xaxis.set_major_locator(MaxNLocator(4, integer=True))
            ax.grid(axis="y", alpha=.12)
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("每步奖励（%）")
        axes[0, 0].legend(frameon=False, fontsize=9)
        fig.suptitle(f"阶段 {stage}：记录检查点的过程评估；细线为种子，粗线为均值", fontsize=13)
        stem = f"stage_{stage}_process"
        for extension in ("png", "pdf"):
            fig.savefig(output / f"{stem}.{extension}", dpi=240)
        plt.close(fig)
        plots.append(stem + ".png")
    return plots


def label(name):
    return LABELS[name].replace("\n", " / ")


def pct(value):
    return f"{100 * value:.2f}%"


def pp(value):
    return f"{100 * value:+.2f}"


def report(summary, output):
    groups, runs = summary["groups"], summary["runs"]
    complete = summary["completed_runs"] == summary["expected_runs"]
    lines = ["# v0.5 从地点信息到延迟交流与持续采集的阶段实验报告", "",
        f"生成时间：{summary['generated_utc']}。状态：{'预定扫描范围全部完成' if complete else '运行中的阶段汇总，尚未完成全部条件'}。",
        "", f"已完成并纳入 {summary['completed_runs']}/{summary['expected_runs']} 个运行；种子范围为 " +
        "、".join(map(str, summary["expected_seeds"])) + "。仅扫描当前 progression 批次，不纳入 smoke 开发试验或 v0.4 训练结果。",
        "", "## 研究范围与分析口径", "",
        "这是一轮小规模工程探索。重复单位是主体对种子，不是照片、采集步骤或两种职责。"
        "报告保留每个种子、两个发信方向、食物/水需求和全部预定条件；组均值对种子等权，"
        "不利用三个种子的小 p 值或选择性显著结果宣称创新。百分比表示每步是否满足当轮需求的平均二元奖励；"
        "阶段 C 另报告三步全满足率，不能把平均奖励与整段全部成功混同。",
        "", "仍使用原有 44 张训练、16 张开发留出照片及冻结 DINOv2 ViT-L/14；这些照片已经用于既往探索，"
        "并非新确认集。个人视觉投影经新的独立资源后果准备后冻结；公共地点坐标、二维自身需求、"
        "库存、存在掩码和实际采集结果均由实验接口提供。主体不接收地图中资源的类别元数据或世界真值，"
        "侦察使用图片特征；实际采集后获得的资源计数是合法自身观察，不能笼统说主体不接收任何类别信息。",
        "", "接收者的行动网络按自身需求选用两组四地点输出，是实验明确提供的非语言条件行动能力；"
        "消息两位置没有资源或地点语义标签，但其后形成的解释受结构化地点接口与需求分支影响，仍允许整体码。"
        "完整地图码诊断两人达到 100%，只证实这一接口在给定正确编码时可执行任务；"
        "它不是自然通信成绩，也不能证明冻结 DINO 表征的独立贡献。原准备动作分支的资源能力验收，"
        "也不等于新通信/地点接口或延迟记忆已经单独验收。",
        "", "## 实施阶段与对照关系", "",
        "A 的主要条件采用从两地点、三地点到四地点的课程；在默认 1800 次预算下分别为 400、400、1000 次。"
        "每批次可到达地点随机选定，并对合法地点施加公共行动限制。直接条件始终使用四地点；混排条件打乱前 1500 个相同批次身份，"
        "末 300 次共同关闭熵正则并保持相同次序，"
        "保持世界、照片、目标和行动菜单的训练多重集。该比较检验固定优化日程下的训练安排，"
        "直接条件与课程的差异还包括难度暴露；留出条件直接在指定八张地图上训练。",
        "", "B 的即时继续、延迟保持、延迟清空、延迟回放都从同一种子的 A hidden_sequence 社会终点出发。"
        "延迟是观察后输入三次空白，再发送；清空直接删除现场潜在表示，回放重新呈现同一合法观察。"
        "这不是自然时间长度、事件衰退或人类工作记忆的完整模拟。清空使场景信息不可用，低分不能解释为缺乏语言能力。",
        "", "当前回放实现重新从零状态计算同一观察，因此函数和梯度在数学上等同即时继续。"
        "三个种子的最终参数逐位一致，检查点与全部终点评估也一致；这两个条件只计作工程等价核查，"
        "不能当作外部记忆优于内部保持的独立机制证据。",
        "", "C 的持续通信、此阶段移除通道、目标改为可见与延迟通信也从同一 A hidden_sequence 终点出发；"
        "始终关通道条件从 A hidden_blocked 出发。因此前四者是相同通信历史的继续训练比较，"
        "始终关通道与它们具有不同 A 阶段历史，虽然总预算匹配。所有 B/C 条件重新建立 Adam，"
        "不能解释为原优化器状态的无缝继续。",
        "", "C 每段三次采集，同一侦察者与采集者保持职责；先采集入库、截断至每类容量二，再消费当轮需求。"
        "被采资源在不占用另一资源的位置中等概率补充，库存跨步保留。侦察者每步重新观察现场，"
        "其 GRU 观察从零状态开始；采集者保存最近两次真实地点选择及食物/水/空采集事件。"
        "这包含持续环境和有限个人后果记录，不包含自发职业分化、长期事件记忆或抽象因果推理验收。",
        "", "本次实际完成 A 27 个运行×1800 更新、B 12 个×900 更新、C 15 个×1200 更新，"
        "合计 54 个运行、77,400 次社会训练更新，每次 512 段。A/B 为单步段，C 为三步段，"
        "因此更新数、段数和采集步数不能互换。共享的 A 热启动只实际训练一次，"
        "但每个 B 条件的经历是 1800+900 更新，每个 C 条件是 1800+1200 更新。",
        "", "## 上界与测量解释", "",
        "独立数学核查通过。单轮无通信的理想期望成绩为 25%；目标隐藏的一个五选一符号上界为 17/24≈70.83%。"
        "两次五选一与一次二十五选一均有 25 种码，可直接查表表示全部 12 张地图；同容量成绩差仍可能来自架构、参数或优化差异。"
        "高分、新组合泛化与消息变长均不能单独证明组合结构。四图子分布不直接受全十二图的 17/24 上界约束。",
        "", f"三轮持续条件允许理想无通信策略利用完整合法个人历史，其每步期望上界为 265/648={pct(summary['bounds']['dynamic_no_message'])}；"
        "全信息策略可逐步达到 100%。实际接口的两事件记录并不等于理想策略拥有的完整历史。"
        "有限测试集的经验均值可以略高于期望界限，不能逐例判为实现错误。",
        "", "正常模式选择最大概率消息与动作；按策略采样另列。打乱在同一侦察方向、步骤、当前需求和库存内重排消息，"
        "保留整条消息的条件频次，不固定每个采集者的具体过去历史。动态干预会改变行动和后续环境，"
        "正常减打乱是这项闭环干预的回报差，不是逐步冻结世界下的单一语义指标。blank 是测试时强制零消息，"
        "erase_memory 是测试时清除侦察现场表示，与训练中始终关通道、训练中延迟清空分别报告。",
        "", "## 主要发现", ""]
    if complete:
        def normal(stage, name):
            return pct(groups[stage][name]["scores"]["normal"]["mean_reward"]["mean"])
        lines += [
            f"A 中逐级课程的目标隐藏两符号条件为 {normal('A', 'hidden_sequence')}，低于直接四地点的 "
            f"{normal('A', 'hidden_sequence_direct')} 和同批次混排的 {normal('A', 'hidden_sequence_mixed')}。"
            "这是本轮的课程负结果；不能据此预设从简单到复杂一定帮助共同通信形成。"
            "课程与混排匹配完整训练批次多重集，因此这个差异支持继续检验训练顺序与优化过程的作用，"
            "尚不能外推所有课程设计。", "",
            f"B 中延迟保持为 {normal('B', 'delay_memory')}，清空为 {normal('B', 'delay_reset')}；"
            f"即时继续与等价回放均为 {normal('B', 'immediate_continue')}。"
            "结果表明当前接口能保留支持通信的现场信息；清空后信息损失具有行为后果，"
            "没有形成对语言或一般记忆能力的独立测量。", "",
            f"C 中持续通信为 {normal('C', 'persistent_communication')}，此阶段移除通道为 "
            f"{normal('C', 'persistent_channel_removed')}，始终关通道为 {normal('C', 'persistent_blocked')}。"
            f"目标改为可见为 {normal('C', 'persistent_known')}，加入延迟为 {normal('C', 'persistent_delayed')}。"
            "通信支持三轮库存与资源补充环境中的需求满足；后两个条件仍是共同 A 起点上的适应，"
            "本轮尚无工具链、自发分工或新人学习。", ""]
    lines += ["## 终点结果", ""]
    for stage, entries in groups.items():
        lines += [f"### 阶段 {stage}", ""]
        if not entries:
            lines += ["尚无完成运行，不推断结果。", ""]
            continue
        lines += ["| 条件 | n种子 | 正常 | 正常种子范围 | 打乱 | 正常减打乱（百分点） | 策略采样 | 全段满足率 |",
                  "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |"]
        for name in CONDITIONS[stage]:
            if name not in entries:
                continue
            g = entries[name]; normal = g["scores"]["normal"]["mean_reward"]
            lines.append(f"| {label(name)} | {len(g['seeds'])} | {pct(normal['mean'])} | {pct(normal['min'])}–{pct(normal['max'])} | "
                         f"{pct(g['scores']['shuffle']['mean_reward']['mean'])} | {pp(g['normal_minus_shuffle']['mean_reward']['mean'])} | "
                         f"{pct(g['scores']['stochastic']['mean_reward']['mean'])} | {pct(g['scores']['normal']['perfect_episode_rate']['mean'])} |")
        lines += ["", f"![阶段{stage}终点比较]({(output / f'stage_{stage}_comparison.png').resolve()})", "",
                  "点为种子，短黑线为均值；横线是该阶段无通道理想期望上界，不是所有通信条件的上界。无置信区间或假设检验。", "",
                  f"![阶段{stage}形成过程]({(output / f'stage_{stage}_process.png').resolve()})", "",
                  "过程仅连接实际记录检查点，不插值推断首次形成时刻。通常过程每项 1024 段，终点每项 8192 段；"
                  "实际样本数保存在 summary.json。B/C 横轴是继续训练预算，不包含此前 A 的更新数。", ""]
        lines += ["| 条件 | A侦察→B采集 | 该方向正常减打乱（百分点） | B侦察→A采集 | 该方向正常减打乱（百分点） |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name, g in entries.items():
            directions = g["scores"]["normal"]["direction_means"]; gap = g["normal_minus_shuffle"]["direction_means"]
            lines.append(f"| {label(name)} | {pct(directions[0]['mean'])} | {pp(gap[0]['mean'])} | {pct(directions[1]['mean'])} | {pp(gap[1]['mean'])} |")
        lines += [""]
    lines += ["## 相同种子的条件差", "",
              "以下先在同种子内相减，再对配对差取均值与最小/最大范围。没有配齐的种子不进入该项差值；"
              "不能把缺失条件当作零分。不同接口形状或既往通道历史会改变比较的解释范围。", "",
              "| 阶段与比较 | 配对种子数 | 正常均值差（百分点） | 各种子差值（百分点） | 消息落差的处理差（百分点） |",
              "| --- | ---: | ---: | --- | ---: |"]
    for comparison in summary["paired_comparisons"]:
        lines.append(f"| {comparison['stage']}：{comparison['interpretation']} | {len(comparison['seeds'])} | "
                     f"{pp(comparison['normal_difference']['mean'])} | " + "、".join(pp(x) for x in comparison['normal_difference']['values']) +
                     f" | {pp(comparison['message_gap_difference']['mean'])} |")
    lines += ["", "## 八图与四图的分布检查", "",
              "只有 A 的 hidden_sequence_holdout 在社会训练中保留四图未见，其余条件的 in8/out4 只是同一地图分组诊断。"
              "四图为 (食物地点,水地点)=(0,1)、(1,2)、(2,3)、(3,0)，各单独资源—地点值仍在八图训练集合出现。"
              "这只有一种划分，照片仍是旧开发留出；不构成广泛组合泛化或新视觉确认。", "",
              "| A条件 | 八图正常 | 四图正常 | 八图正常减打乱（百分点） | 四图正常减打乱（百分点） | 真正训练留四图 |",
              "| --- | ---: | ---: | ---: | ---: | --- |"]
    for name, g in groups["A"].items():
        nm, delta = g["scores"]["normal"]["map_subsets"], g["normal_minus_shuffle"]["map_subsets"]
        lines.append(f"| {label(name)} | {pct(nm['in8']['mean'])} | {pct(nm['out4']['mean'])} | {pp(delta['in8']['mean'])} | {pp(delta['out4']['mean'])} | {'是' if name == 'hidden_sequence_holdout' else '否'} |")
    lines += ["", "## 独立重算与完整性", "",
        f"已逐步重算 {summary['trace_steps_independently_checked']:,} 条终点评估记录，覆盖每个完成运行的五种模式；"
        f"核对 {sum(run['training_record_count'] for run in runs):,} 条训练更新记录的连续性，但未以训练日志代替轨迹奖励重算。"
        "重算不导入 camp.py、run_stages.py 或神经策略；从地点、选择、库存与需求自行恢复采集、截断、消费和奖励，"
        "再核对资源补充、库存衔接、真实个人历史、菜单映射、消息投递和照片类别/数据分割。",
        "", "每步奖励、整段全满足率、两个职责方向、各步及地图子集均重新计算并与 result.json 比较。"
        "不同干预共享外生目标、菜单、补充随机量和初始状态；后续地图和库存只要求各自物理合法，"
        "不强迫采取不同动作的轨迹保持一致。来源与终点参数指纹检验 B/C 的规定热启动，"
        "A 课程/混排按 batch_identity 核对完整训练世界哈希多重集。"
        "原始 NPZ、result、训练日志和代码哈希均纳入机器可读记录。该审计核对保存的数据与环境后果，"
        "不等于重新执行神经策略或重新训练。", ""]
    if summary["missing_runs"]:
        lines += ["尚未完成的运行：" + "、".join(summary["missing_runs"]) + "。", ""]
    execution = summary.get("execution_audit")
    if execution:
        lines += [f"另一独立执行审计 [{Path(execution['source']).name}]({execution['source']}) 检验 "
                  f"{execution['completed_runs']}/{execution['planned_runs']} 个运行，共 {execution['checks_count']:,} 项检查通过，"
                  f"自行重建 A 的 {execution['A_world_updates_verified']:,} 次更新所用世界；"
                  "新个人准备与既往准备未发现参数指纹重合。它同时逐位核对即时/回放参数等价，"
                  "与本脚本的轨迹奖励审计分工不同。", ""]
    fixed = summary.get("fixed_context")
    if fixed:
        lines += ["## 固定当前情境的消息作用", "",
                  "补充干预保持正常轨迹的当前世界、需求、库存、最近两次实际采集历史与菜单不变，"
                  "只从同方向、同一步数且具有相同需求/库存/历史的合法供体重排消息，再计算即时收益。"
                  "不模拟后续世界变化，因此与上文闭环正常减打乱的总回报差是不同口径。"
                  "单例层及重复消息可能保持不变，干预也不保证完全移除信息。", "",
                  "| 阶段 | 条件 | 正常即时收益 | 替换后即时收益 | 当前消息落差（百分点） | 各种子落差（百分点） |",
                  "| --- | --- | ---: | ---: | ---: | --- |"]
        for stage, entries in fixed["groups"].items():
            for name, g in entries.items():
                lines.append(f"| {stage} | {label(name)} | {pct(g['normal_mean']['mean'])} | {pct(g['intervened_mean']['mean'])} | "
                             f"{pp(g['current_message_reward_drop']['mean'])} | " + "、".join(pp(x) for x in g["current_message_reward_drop"]["values"]) + " |")
        lines += ["", f"[{Path(fixed['source']).name}]({fixed['source']}) 覆盖 {fixed['runs']} 个 B/C 运行、"
                  f"{fixed['cases_checked']:,} 条正常轨迹记录，模型动作和独立重算收益核查通过。"
                  "本汇总另核对其参数/轨迹文件哈希、正常收益与替换后计数。"
                  "该结果支持当前消息影响选择和收益，不能单独证明词义、组合性或长期总收益的单一因果效应。", ""]
    protocol = summary.get("protocol")
    if protocol:
        pg = protocol["groups"]
        lines += ["## 协议结构的独立分析", "",
                  "以下是固定接收情境的照片平衡枚举探针，区别于前文随机抽取 8192 段的终点回报。"
                  "校准与验证分别使用互不重叠的四张食物照片×四张水照片，即各 16 个照片组合；"
                  "两组仍来自旧照片库。跨目标探针在每方向遍历 16 照片组合×12 地图×2 发信目标×24 行动菜单 "
                  "=9216 个记录。均值先在每种子内对两方向平均，再对三个种子等权；"
                  "枚举情境和两方向不计作独立实验重复。", "",
                  "跨目标复用保持同一消息和现场，改变接收者的需求。双目标均正确要求同一消息在两个需求下都能选对资源；"
                  "目标隐藏条件中消息不依赖发信目标，因此原目标与切换目标总体准确率相等并不构成额外结构证据。", "",
                  "| A条件 | 原目标 | 固定消息切换目标 | 同一消息双目标均正确 |",
                  "| --- | ---: | ---: | ---: |"]
        for name, g in pg.items():
            lines.append(f"| {label(name)} | {pct(g['native']['mean'])} | {pct(g['switched']['mean'])} | {pct(g['both_goals']['mean'])} |")
        lines += ["", "局部成分替换以仅一种资源地点变化的源/目标地图成对，在校准照片选择食物/水对应的两个位置，"
                  "再在验证照片上替换一位。严格成功同时要求变化资源从原正确地点转到供体地点、另一资源仍正确。"
                  "下表分母包含每方向全部 36,864 个枚举记录；不先排除自然消息失败的情境。"
                  "另存的条件成功率仅在两地图两需求自然消息都正确的子集定义，零分母保留为未定义，"
                  "避免筛选子集或剔除失败造成表面高分。", "",
                  "整体重编码对 25 条完整消息作双射，同时逆向改写接收查表，精确保留自然行为、跨目标行为与码频率；"
                  "每次重编码重做校准位置选择，再评验证替换，共每方向 100 次。"
                  "它测量局部位结构相对任务行为等价的整体码表示是否突出，参考分布不等于训练种子的置信区间，"
                  "也不是语法显著性检验。先验证全部菜单等价后，参考计算折叠 24 个重复菜单，比例保持不变。", "",
                  "| 序列条件 | 严格替换成功 | 双地图双目标自然正确的合格比例 | 重编码参考均值 | 观察减参考（百分点） | 各种子观察减参考（百分点） |",
                  "| --- | ---: | ---: | ---: | ---: | --- |"]
        for name, g in pg.items():
            if "strict_fragment" not in g:
                continue
            reference = pct(g["recoding_mean"]["mean"]) if "recoding_mean" in g else "未评估"
            delta = pp(g["fragment_minus_recoding"]["mean"]) if "fragment_minus_recoding" in g else "未评估"
            seed_delta = "、".join(pp(x) for x in g["fragment_minus_recoding"]["values"]) if "fragment_minus_recoding" in g else "未评估"
            lines.append(f"| {label(name)} | {pct(g['strict_fragment']['mean'])} | {pct(g['eligible_fraction']['mean'])} | {reference} | {delta} | {seed_delta} |")
        visible, hidden = pg["known_sequence"], pg["hidden_sequence"]
        lines += ["", f"目标可见课程的严格替换率为 {pct(visible['strict_fragment']['mean'])}，目标隐藏课程为 "
                  f"{pct(hidden['strict_fragment']['mean'])}；观察减重编码参考分别为 "
                  f"{pp(visible['fragment_minus_recoding']['mean'])} 与 {pp(hidden['fragment_minus_recoding']['mean'])} 个百分点。"
                  "这三个种子中，隐藏对方目标没有提高所测结构；不能把任务直觉当作已获支持的机制。", "",
                  "初版重编码评估是在 A 训练期间、首种子完成后、正式协议分析及任何重编码结果产生前补充，"
                  "仅覆盖目标可见课程、目标隐藏课程及八图训练。直接、混排、关通道三项在看到全部 raw 分数和初版重编码结果后扩展；"
                  "本表采用独立保存的扩展文件，明确属于结果后的探索比较，初版未被替换。", "",
                  f"八图训练的留四图拼接探针中，将两条训练地图消息的校准位置成分组成留出地图消息，"
                  f"双目标均正确的全情境比例为 {pct(pg['hidden_sequence_holdout']['heldout_stitch']['mean'])}，"
                  "各种子为 " + "、".join(pct(x) for x in pg['hidden_sequence_holdout']['heldout_stitch']['values']) + "。"
                  "其每方向分母为 12,288 个记录，包含所有指定合法供体组合；这不同于表中自然消息的四图回报。"
                  "局部替换可行并不保证能系统拼出未训练地图。", "",
                  "一个可核查的局部例子来自混排条件种子 24003 的 B 侦察→A 采集方向："
                  "校准选择第一个符号对应水、第二个对应食物；在该验证集合中，原目标、切换目标和严格替换均为 100%。"
                  "地点编号为 0–3，消息 [2,4] 对应水在 2、食物在 0；[1,4] 对应水在 3、食物在 0；"
                  "[2,3] 对应水在 2、食物在 3。这展示该方向的局部成分功能，"
                  "只覆盖 12 张已训练地图及照片验证样本；其他方向和留出地图并不稳定，同义符号也可能存在。", "",
                  f"详见 [{Path(protocol['source']).name}]({protocol['source']})、"
                  f"[原始协议分析]({protocol['original_source']})。本汇总重新核对来源参数哈希、整数计数比例、"
                  "100 次双射与参考均值，并确认扩展文件原有方向和原有重编码结果逐值不变；没有重新运行神经策略探针。", ""]
    lines += ["## 逐种子终点记录", "",
              "| 阶段 | 种子 | 条件 | 正常 | 打乱 | blank | 策略采样 | 清空现场记忆 | 食物需求正常 | 水需求正常 |",
              "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in runs:
        values = " | ".join(pct(run["scores"][mode]["mean_reward"]) for mode in MODES)
        goals = run["scores"]["normal"]["goal_means"]
        lines.append(f"| {run['stage']} | {run['seed']} | {label(run['condition'])} | {values} | {pct(goals[0]['mean_reward'])} | {pct(goals[1]['mean_reward'])} |")
    lines += ["", "## 当前结论的边界", "",
        "本轮数据能检验规定视觉与行动接口、训练安排、信息保留方式及有限持续任务中的功能回报变化。"
        "结构结论需要独立协议干预，而新照片、更多因子值、多组平衡留出、直接复杂任务同预算对照以及新主体测试仍需后续验证。"
        "B/C 由 A 热启动，本报告不能仅比较各阶段最终成绩便归因于复杂程度或课程历史。"
        "无论分数高低，三探索种子和旧照片均不足以确认语言诞生的一般条件、完整组合语法或自发社会分工。", ""]
    (output / "阶段实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results" / "progression_001")
    parser.add_argument("--output", type=Path, help="Default: INPUT/analysis; only analysis outputs are written")
    parser.add_argument("--seeds", type=int, nargs="+", default=[24001, 24002, 24003])
    parser.add_argument("--check-only", action="store_true", help="Validate completed runs without writing reports or plots")
    args = parser.parse_args()
    source = args.input.resolve(); output = (args.output or source / "analysis").resolve()
    photos = read_json(ROOT.parent / "redesign_v0.4" / "data" / "manifest.json")["images"]
    bounds_path = ROOT / "bounds_audit_verified.json"; bounds = read_json(bounds_path)
    require(bounds["status"] == "passed", "Mathematical bounds audit did not pass")
    runs, missing, references = [], [], {}
    for stage, names in CONDITIONS.items():
        for seed in args.seeds:
            for name in names:
                path = source / stage / f"s{seed}_{name}"
                if not (path / "result.json").is_file():
                    missing.append(f"{stage}/s{seed}_{name}")
                    continue
                run, trace = read_run(path, stage, photos)
                key = stage, seed
                if key in references:
                    compare_exogenous(references[key], trace, f"{stage}/{seed}: across-condition final evaluation")
                else:
                    references[key] = trace
                runs.append(run)
                print(f"AUDITED {stage}/s{seed}_{name}: normal={run['scores']['normal']['mean_reward']:.6f}", flush=True)
    provenance = provenance_audit(runs)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "input": str(source),
        "analysis_script_sha256": sha(__file__), "status": "complete" if not missing else "incomplete",
        "expected_seeds": args.seeds, "expected_runs": len(args.seeds) * sum(len(v) for v in CONDITIONS.values()),
        "completed_runs": len(runs), "missing_runs": missing,
        "statistical_unit": "independent agent-pair seed; equal seed weighting; no p values",
        "bounds": {"source": str(bounds_path), "sha256": sha(bounds_path), "single_no_message": .25,
                   "hidden_single_5_messages": bounds["stage_a"]["goal_hidden"]["5"]["decimal"],
                   "dynamic_no_message": bounds["persistent_three_rounds"]["by_horizon"]["3"]["no_message_mean_reward"]["decimal"]},
        "trace_steps_independently_checked": sum(audit["steps_checked"] for run in runs for audit in run["trace_audit"].values()),
        "provenance_audit": provenance, "runs": runs, "groups": aggregate(runs),
        "paired_comparisons": paired_comparisons(runs), "protocol_artifacts": [],
    }
    summary["protocol"] = protocol_summary(source, runs)
    summary["fixed_context"] = fixed_context_summary(source, runs)
    execution_path = source / "audit_execution.json"
    if execution_path.exists():
        execution = read_json(execution_path)
        require(execution["status"] == "passed", "Execution audit did not pass")
        summary["execution_audit"] = {"source": str(execution_path), "source_sha256": sha(execution_path),
            "completed_runs": execution["completed_runs"], "planned_runs": execution["planned_runs"],
            "checks_count": sum(execution["checks"].values()), "A_world_updates_verified": execution["A_world_updates_verified"]}
    for name in ("protocol_analysis.json", "protocol_analysis.md", "protocol_analysis_extended_null.json", "protocol_analysis_extended_null.md"):
        for candidate in sorted(source.rglob(name)) if source.exists() else []:
            summary["protocol_artifacts"].append({"path": str(candidate.resolve()), "sha256": sha(candidate)})
    if not args.check_only:
        output.mkdir(parents=True, exist_ok=True)
        summary["plots"] = make_plots(summary, output)
        report(summary, output)
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (output / "per_seed_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["stage", "seed", "condition", "mode", "mean_reward", "perfect_episode_rate", "A_to_B", "B_to_A", "in8", "out4", "maps_withheld"])
            for run in runs:
                for mode in MODES:
                    m = run["scores"][mode]
                    writer.writerow([run["stage"], run["seed"], run["condition"], mode, m["mean_reward"],
                                     m["perfect_episode_rate"], *m["direction_means"],
                                     m["map_subsets"]["in8"]["mean_reward"], m["map_subsets"]["out4"]["mean_reward"], m["maps_were_withheld"]])
    print(json.dumps({"status": summary["status"], "completed_runs": len(runs), "missing_runs": len(missing),
                      "trace_steps_checked": summary["trace_steps_independently_checked"],
                      "outputs_written": not args.check_only, "output": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
