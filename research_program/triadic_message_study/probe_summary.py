"""Read-only numerical summary of completed, frozen endpoint probes.

No policy module imports, checkpoint loading, forward evaluation or training.
All original conditions, seeds, partitions and eligible private-fact pairs remain.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

SEEDS = (47101, 47102, 47103, 47104)
CONDITIONS = ("FI_silent", "FI_live", "PI_silent", "PI_live")
PARTITIONS = ("train", "new_needs", "new_layouts", "new_needs_and_layouts")
MODES = ("cross_channels_closed", "constant_full", "action_only_drop")
WORLD_COUNTS = dict(zip(PARTITIONS, (82836, 24732, 27612, 8244)))
PAIR_COUNTS_PER_LISTENER = dict(zip(PARTITIONS, (900, 372, 112, 56)))
AGENTS = ("A", "B", "C")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False,
                       allow_nan=False, separators=(",", ":")) + "\n").encode()


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json_bytes(value))


def approx(actual, expected, name, tolerance=2e-12):
    require(math.isfinite(float(actual)) and math.isfinite(float(expected))
            and abs(float(actual)-float(expected)) <= tolerance, name+" differs")


def validate_probabilities(probabilities, choices, name):
    require(probabilities.shape == choices.shape+(17,) and np.isfinite(probabilities).all()
        and ((probabilities >= 0) & (probabilities <= 1)).all(), name+" invalid")
    require(np.allclose(probabilities.sum(-1), 1, rtol=0, atol=2e-12), name+" sum differs")
    require(np.array_equal(probabilities.argmax(-1), choices), name+" argmax differs")


def summarize_deletions(row, data, natural):
    """Recompute all recorded score/action effects; original row defines no labels."""
    states, original_actions = natural["states"], natural["action_indices"]
    n = len(states)
    require(n == row["worlds"] and np.array_equal(data["states"], states)
            and np.array_equal(data["natural_action_indices"], original_actions), "Natural anchor differs")
    native = natural["greedy_reward"]
    require(native.shape == (n,) and np.isin(native, (0, .5, 1)).all(), "Invalid saved natural score")
    validate_probabilities(natural["action_probabilities"], original_actions, "natural")
    approx(native.mean(), row["natural_reward_mean"], "Natural reward mean")
    approx((native == 1).mean(), row["natural_full_success_rate"], "Natural full success")
    flat = []
    for mode in MODES:
        actual = data[mode+"_actions"]
        validate_probabilities(data[mode+"_probabilities"], actual, mode)
        reward = data[mode+"_reward"]
        require(reward.shape == (n,) and np.isin(reward, (0, .5, 1)).all(), "Invalid saved intervention score")
        expected = row["deletions"][mode]
        executed_transports = expected["executed_transports"]
        require(isinstance(executed_transports, int) and 0 <= executed_transports <= 2*n,
                "Invalid saved execution count")
        changed = actual != original_actions
        scores = dict(worlds=n, reward_sum=float(reward.sum()), full_successes=int((reward == 1).sum()),
            full_success_rate=float((reward == 1).mean()), reward_mean=float(reward.mean()),
            full_success_rate_change_from_natural=float((reward == 1).mean()-(native == 1).mean()),
            worlds_with_any_action_change=int(changed.any(1).sum()), agent_action_changes=changed.sum(0).tolist(),
            fullsuccess_lost=int(((native == 1) & (reward != 1)).sum()),
            fullsuccess_gained=int(((native != 1) & (reward == 1)).sum()), executed_transports=executed_transports)
        require(set(scores) == set(expected), "Intervention result schema changed")
        for key, value in scores.items():
            if isinstance(value, float):
                approx(value, expected[key], mode+"/"+key)
            else:
                require(value == expected[key], mode+"/"+key+" differs")
        if row["condition"].endswith("silent"):
            require(np.array_equal(actual, original_actions)
                and np.array_equal(data[mode+"_messages"], natural["messages"]), "Silent intervention changed action/messages")
            require(np.allclose(data[mode+"_probabilities"], natural["action_probabilities"], rtol=0, atol=2e-12),
                "Silent probabilities changed beyond saved replay precision")
        flat.append(dict(seed=row["seed"], condition=row["condition"], partition=row["partition"], mode=mode,
            natural_full_successes=int((native == 1).sum()), natural_full_success_rate=float((native == 1).mean()),
            natural_reward_mean=float(native.mean()), world_action_change_rate=float(changed.any(1).mean()),
            **scores))
    return flat


def summarize_content(records, definitions, transplant_probabilities, natural, silent):
    """Recompute local suitability and both transplant directions from saved arrays."""
    defs = {r["pair_id"]: r for r in definitions}
    require(len(defs) == len(definitions) == len(records), "Pair denominator differs")
    require(len({r["pair_id"] for r in records}) == len(records)
        and {r["pair_id"] for r in records} == set(defs), "Missing/duplicate/extra pair")
    probs = transplant_probabilities["probabilities"]
    require(probs.shape == (2*len(records), 17)
        and transplant_probabilities["pair_id"].tolist() == [r["pair_id"] for r in records], "Transplant array order differs")
    output = []
    for k, record in enumerate(records):
        definition = defs[record["pair_id"]]
        listener = definition["listener_index"]
        require(record["listener"] == AGENTS[listener], "Listener differs")
        low, high = definition["row_index_low"], definition["row_index_high"]
        require(np.array_equal(natural["states"][low], definition["state_low"])
            and np.array_equal(natural["states"][high], definition["state_high"]), "Pair states differ")
        low_set, high_set = set(definition["fullsuccess_actions_low"]), set(definition["fullsuccess_actions_high"])
        require(low_set and high_set and not low_set.intersection(high_set), "Invalid disjoint action sets")
        ai, aj = int(natural["action_indices"][low, listener]), int(natural["action_indices"][high, listener])
        computed = dict(natural_action_low=ai, natural_action_high=aj, natural_suitable_low=ai in low_set,
            natural_suitable_high=aj in high_set, natural_both_suitable=ai in low_set and aj in high_set,
            natural_action_changed=ai != aj)
        require(all(record[key] == value for key, value in computed.items()), "Recorded natural local suitability differs")
        if computed["natural_both_suitable"]:
            require(computed["natural_action_changed"], "Disjoint local solutions allow no common correct action")
        if silent:
            require(ai == aj and not computed["natural_both_suitable"], "Silent same-observation invariant failed")
        require([r["direction"] for r in record["directions"]] == ["low_from_high", "high_from_low"], "Direction order differs")
        directions = []
        for d, source, donor, own_set, donor_set in ((0, low, high, low_set, high_set), (1, high, low, high_set, low_set)):
            p = probs[2*k+d]
            action = int(p.argmax())
            validate_probabilities(p[None], np.array([action]), "Transplanted listener")
            donor_p = natural["action_probabilities"][donor, listener]
            own_p = natural["action_probabilities"][source, listener]
            require(np.allclose(p, donor_p, rtol=0, atol=2e-12), "Saved donor probability identity failed")
            require(action == int(natural["action_indices"][donor, listener]), "Donor action identity failed")
            actual = dict(transplanted_action=action,
                action_changed=action != int(natural["action_indices"][source, listener]),
                suitable_for_recipient_world=action in own_set, suitable_for_donor_world=action in donor_set,
                probability_total_variation=float(np.abs(p-own_p).sum()/2),
                donor_probability_saved_exact=bool(np.array_equal(p, donor_p)),
                donor_probability_saved_max_abs=float(np.abs(p-donor_p).max()),
                same_batch_donor_probability_exact=True)
            saved = record["directions"][d]
            for key, value in actual.items():
                if isinstance(value, float):
                    approx(value, saved[key], "Direction "+key)
                else:
                    require(value == saved[key], "Direction "+key+" differs")
            directions.append(dict(direction=saved["direction"], **actual))
        output.append(dict(pair_id=record["pair_id"], listener=record["listener"], **computed, directions=directions))
    return output


def content_counts(records):
    n = len(records)
    require(n > 0, "Empty official content stratum")
    both = sum(r["natural_both_suitable"] for r in records)
    neither = sum(not r["natural_suitable_low"] and not r["natural_suitable_high"] for r in records)
    directional = {}
    for i, direction in enumerate(("low_from_high", "high_from_low")):
        ds = [r["directions"][i] for r in records]
        directional[direction] = dict(directions=n, action_changes=sum(d["action_changed"] for d in ds),
            donor_suitable=sum(d["suitable_for_donor_world"] for d in ds),
            recipient_suitable=sum(d["suitable_for_recipient_world"] for d in ds),
            mean_probability_total_variation=sum(d["probability_total_variation"] for d in ds)/n,
            saved_donor_probability_exact=sum(d["donor_probability_saved_exact"] for d in ds),
            max_saved_donor_probability_difference=max(d["donor_probability_saved_max_abs"] for d in ds),
            same_batch_donor_probability_identity_reported=sum(d["same_batch_donor_probability_exact"] for d in ds))
    return dict(pairs=n, naturally_both_suitable=both, naturally_exactly_one_suitable=n-both-neither,
        naturally_neither_suitable=neither, natural_action_changes=sum(r["natural_action_changed"] for r in records),
        natural_both_suitable_rate=both/n,
        natural_action_change_rate=sum(r["natural_action_changed"] for r in records)/n,
        directions=directional)


def aggregate(flat, content):
    """Preserve four seed values before reporting their equal-weight mean."""
    grouped = []
    for condition, partition, mode in product(CONDITIONS, PARTITIONS, MODES):
        rows = sorted([r for r in flat if (r["condition"], r["partition"], r["mode"]) == (condition, partition, mode)], key=lambda r: r["seed"])
        require([r["seed"] for r in rows] == list(SEEDS), "Missing four-seed intervention stratum")
        means = {key: sum(r[key] for r in rows)/4 for key in ("natural_full_success_rate", "full_success_rate",
            "full_success_rate_change_from_natural", "natural_reward_mean", "reward_mean", "world_action_change_rate")}
        grouped.append(dict(condition=condition, partition=partition, mode=mode, independent_seed_count=4,
            seeds=[{k: r[k] for k in ("seed", "worlds", "natural_full_success_rate", "full_success_rate",
                "full_success_rate_change_from_natural", "reward_mean", "world_action_change_rate",
                "fullsuccess_lost", "fullsuccess_gained")} for r in rows], equal_seed_means=means))
    content_grouped = []
    for condition, partition in product(("PI_silent", "PI_live"), PARTITIONS):
        rows = sorted([r for r in content if (r["condition"], r["partition"]) == (condition, partition)], key=lambda r: r["seed"])
        require([r["seed"] for r in rows] == list(SEEDS), "Missing four-seed private-content stratum")
        content_grouped.append(dict(condition=condition, partition=partition, independent_seed_count=4,
            per_seed_pairs=PAIR_COUNTS_PER_LISTENER[partition]*3,
            seeds=[dict(seed=r["seed"], **r["all_listeners"]) for r in rows],
            equal_seed_mean_both_suitable_rate=sum(r["all_listeners"]["natural_both_suitable_rate"] for r in rows)/4,
            equal_seed_mean_action_change_rate=sum(r["all_listeners"]["natural_action_change_rate"] for r in rows)/4))
    return grouped, content_grouped


def report_text(result):
    lines = ["# 末点消息干预与私人事实配对汇总", "",
        "全部16个运行、64个分区单元、三种干预及PI的全部预定配对均保留。独立单位是4个训练种子；世界、听者和方向不是额外训练重复。该汇总只读已保存数组和JSON，未调用政策或训练。", "",
        "## 双留出格：四种条件与三种干预", "",
        "表内为四种子等权平均，差值=干预后−自然。逐种子及全部分区读数保留在JSON；变化为负并不单独证明词义，删除与常量替换可能造成分布变化。", "",
        "| 条件 | 干预 | 自然满分率 | 干预满分率 | 满分差/百分点 | 任一人动作改变率 |", "|---|---|---:|---:|---:|---:|"]
    for row in result["condition_partition_mode_summaries"]:
        if row["partition"] != "new_needs_and_layouts":
            continue
        m = row["equal_seed_means"]
        lines.append(f"| {row['condition']} | {row['mode']} | {m['natural_full_success_rate']:.2%} | {m['full_success_rate']:.2%} | {100*m['full_success_rate_change_from_natural']:+.3f} | {m['world_action_change_rate']:.2%} |")
    lines.extend(["", "## PI：自然消息与私人事实", "",
        "自然双方适切要求同一听者在两个同观察世界的行动各自属于该世界可满分行动集合。集合不相交，所以双方适切必然改变行动；仅改变而不适切不计成功。每条件、每种子固定4320对，双留出168对；未按成绩删选。", "",
        "| 条件 | 分区 | 每种子配对数 | 自然双方适切率（四种子） | 自然动作改变率（四种子） |", "|---|---|---:|---|---|"])
    for row in result["content_condition_partition_summaries"]:
        b = ", ".join(f"{s['seed']}:{s['natural_both_suitable_rate']:.2%}" for s in row["seeds"])
        a = ", ".join(f"{s['seed']}:{s['natural_action_change_rate']:.2%}" for s in row["seeds"])
        lines.append(f"| {row['condition']} | {row['partition']} | {row['per_seed_pairs']} | {b} | {a} |")
    lines.extend(["", "## 实现恒等式与内容证据分列", "",
        "将donor两窗的其他人消息送给相同观察的听者，并重生成听者自身第二窗，会按构造恢复donor输入。donor概率/动作重现是实现检查，不能作为额外的学会私人事实成绩；同batch概率恒等沿用测量器保存的检查，本汇总另从原终点与移植数组复核donor动作及概率。自然两端是否分别适切才提供行为证据。", "",
        "两方向的donor适切、recipient适切、动作改变、概率总变差和精确重现数全部按听者/种子/分区保存。移植只针对单个听者，未把其他人也换成新的协调政策，不能解释为完整团队反事实成功率。", "",
        "全窗关闭会重新生成第二窗；行动前删除保留自然本人第二窗，可能仍含伙伴第一窗信息。constant保持真实可见位，只固定跨人内容，帮助区分缺失接口与消息内容，但也可能远离训练消息分布。FI两世界的完整观察不同，因此不做同观察私人事实移植。", "",
        "这些都是冻结政策的末点测量。形成中的消息效应仍由预定检查点与从头silent/live比较界定；本文件不声称已证明组合语言或给出环境机制因果结论。", ""])
    return "\n".join(lines)


def run(probe, output):
    probe, output = Path(probe).resolve(), Path(output).resolve()
    require(not output.exists(), "Refuse to overwrite an existing summary")
    execution = probe/"execution"
    status, result = read(execution/"status.json"), read(execution/"results.json")
    require(status["status"] == result["status"] == "completed", "Probe must fully finish before summary")
    rows = result["rows"]
    expected_keys = set(product(SEEDS, CONDITIONS, PARTITIONS))
    keys = {(r["seed"], r["condition"], r["partition"]) for r in rows}
    require(len(rows) == len(keys) == result["run_partitions"] == 64 and keys == expected_keys, "Incomplete64-row matrix")
    plan = read(probe/"plan.json")
    freeze = read(execution/"input_freeze.json")
    require(sha(probe/"plan.json") == read(probe/"freeze.json")["plan_sha256"] == freeze["probe_plan_sha256"], "Probe plan anchor changed")
    pair_dir = Path(plan["pair_dir"])
    require(sha(pair_dir/"manifest.json") == plan["pair_manifest_sha256"], "Pair manifest anchor changed")
    pair_manifest = read(pair_dir/"manifest.json")
    require(sha(pair_dir/"pairs.json") == pair_manifest["files_sha256"]["pairs.json"], "Pair definitions changed")
    definitions = read(pair_dir/"pairs.json")
    require(len(definitions) == 4320, "Pair count changed")
    source = Path(plan["source_run"])/"execution"
    output.mkdir(parents=True)
    input_hashes = {str(p): sha(p) for p in (execution/"status.json", execution/"results.json", execution/"input_freeze.json",
        probe/"plan.json", pair_dir/"manifest.json", pair_dir/"pairs.json")}
    flat, content, implementations = [], [], []
    try:
        for row in sorted(rows, key=lambda r: (r["seed"], CONDITIONS.index(r["condition"]), PARTITIONS.index(r["partition"]))):
            seed, condition, partition = row["seed"], row["condition"], row["partition"]
            require(row["worlds"] == WORLD_COUNTS[partition], "World denominator changed")
            folder = execution/f"seed_{seed}_{condition}"/partition
            require(read(folder/"result.json") == row, "Local/collected probe result differs")
            for name, digest in row["files_sha256"].items():
                require(sha(folder/name) == digest, "Probe data hash differs")
                input_hashes[str(folder/name)] = digest
            original_path = source/f"seed_{seed}_{condition}"/f"final_{partition}.npz"
            require(sha(original_path) == freeze["inputs_sha256"][str(original_path)], "Original endpoint anchor changed")
            input_hashes[str(original_path)] = freeze["inputs_sha256"][str(original_path)]
            with np.load(original_path, allow_pickle=False) as f:
                natural = {k: f[k] for k in ("states", "messages", "action_indices", "action_probabilities", "greedy_reward", "executed")}
            with np.load(folder/"deletions.npz", allow_pickle=False) as f:
                wanted = ["states", "natural_action_indices"] + [mode+"_"+key
                    for mode in MODES for key in ("actions", "probabilities", "messages", "reward")]
                data = {k: f[k] for k in wanted}
            flat.extend(summarize_deletions(row, data, natural))
            implementations.append(dict(seed=seed, condition=condition, partition=partition, **row["checks"]))
            del data
            if condition.startswith("PI"):
                selected = [d for d in definitions if d["partition"] == partition]
                require(len(selected) == 3*PAIR_COUNTS_PER_LISTENER[partition], "Private pair stratum changed")
                with np.load(folder/"transplanted_listener_probabilities.npz", allow_pickle=False) as f:
                    transplant = {k: f[k] for k in f.files}
                records = summarize_content(read(folder/"content_pairs.json"), selected, transplant, natural, condition.endswith("silent"))
                strata = {a: content_counts([r for r in records if r["listener"] == a]) for a in AGENTS}
                for agent, s in strata.items():
                    require(s["pairs"] == PAIR_COUNTS_PER_LISTENER[partition], "Listener denominator changed")
                    old = row["content"][agent]
                    require(old == dict(pairs=s["pairs"], both_naturally_suitable=s["naturally_both_suitable"],
                        natural_action_changed=s["natural_action_changes"]), "Saved content summary differs")
                content.append(dict(seed=seed, condition=condition, partition=partition,
                    all_listeners=content_counts(records), listeners=strata))
        grouped, content_grouped = aggregate(flat, content)
        summary = dict(status="completed", created_at=datetime.now(timezone.utc).isoformat(),
            scope="pure numerical summary/recheck of saved endpoint data; no model import or forward",
            original_probe_rows=64, intervention_cells=len(flat), content_seed_condition_partitions=len(content),
            independent_paired_initializations=4, result_selected_pairs=False,
            actual_pair_observations=sum(r["all_listeners"]["pairs"] for r in content),
            directional_transplant_observations=2*sum(r["all_listeners"]["pairs"] for r in content),
            intervention_cells_all_seeds=flat, condition_partition_mode_summaries=grouped,
            content_all_seed_partition_listener_strata=content, content_condition_partition_summaries=content_grouped,
            implementation_replay_checks_separate=implementations,
            interpretation_boundaries=dict(donor_identity_is_implementation_check=True,
                natural_both_locally_suitable_is_behavior_readout=True, transplanted_listener_not_new_joint_policy=True,
                source_regeneration_identity_audit_is_not_reperformed_by_this_no_forward_summary=True,
                physical_rewards_are_source_arrays_independently_audited_by_root_not_resettled_here=True),
            source_sha256=sha(__file__), input_sha256=input_hashes)
        require(len(flat) == 192 and len(content) == 32
            and summary["actual_pair_observations"] == 8*4320, "Summary coverage differs")
        write_new(output/"summary.json", summary)
        with (output/"汇总.md").open("x") as f:
            f.write(report_text(summary))
        write_new(output/"status.json", dict(status="completed"))
    except BaseException as error:
        write_new(output/"failure.json", dict(error_type=type(error).__name__, error=str(error), input_sha256=input_hashes))
        write_new(output/"status.json", dict(status="failed"))
        raise
    return dict(status="completed", output=str(output), intervention_cells=192, content_pair_observations=8*4320)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.probe, args.out), ensure_ascii=False))
