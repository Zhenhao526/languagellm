"""Independent full-state/full-action audit; no policy or model is imported."""
from collections import Counter
import hashlib
import importlib.util
from itertools import combinations, permutations, product
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent


def require(value, reason):
    if not value:
        raise AssertionError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def wants(demand, item, destination):
    """Independent integer-factor definition, not the source resource sets."""
    factor = demand // 3
    material = np.where(factor < 2, item // 2 == factor, item % 2 == factor - 2)
    return material & ((demand % 3 == 2) | (demand % 3 == destination))


def pair_can_satisfy_both(table, pair):
    return any(wants(table[pair[0]], item, destination) and
               wants(table[pair[1]], item, destination)
               for item in range(4) for destination in range(2))


def matching_contributors(joint):
    """Physical D1 settlement, independent of source score/payoff helpers."""
    active = [i for i in range(3) if joint[i] is not None]
    if len(active) > 2:
        return []
    matched = []
    for i in active:
        position, destination, partner = joint[i]
        if joint[partner] == (position, destination, i):
            matched.append(i)
    return matched


def grouped_optimum(demands, layouts, values, focal, assignment, shared):
    """Sum over indistinguishable worlds before maximizing the focal action."""
    groups = {}
    for state, (table, layout) in enumerate(zip(demands, layouts)):
        prefix = tuple(map(int, table)) if shared else (int(table[focal]),)
        key = prefix + (int(layout[0]), int(layout[assignment[focal]]))
        if key not in groups:
            groups[key] = [0, np.zeros(17, dtype=np.int64)]
        groups[key][0] += 1
        groups[key][1] += values[state]
    keys = sorted(groups)
    counts = np.asarray([groups[k][0] for k in keys], dtype=np.int64)
    sums = np.asarray([groups[k][1] for k in keys], dtype=np.int64)
    winners = np.argmax(sums, axis=1)
    best = np.max(sums, axis=1)
    return keys, counts, sums, winners, best


def main():
    plan_path, freeze_path = HERE / "plan.json", HERE / "freeze.json"
    plan, freeze = json.loads(plan_path.read_bytes()), json.loads(freeze_path.read_bytes())
    require(sha(plan_path) == freeze["plan_sha256"], "Frozen plan digest mismatch")
    hashes = {str(p): sha(p) for p in (plan_path, freeze_path, Path(__file__).resolve())}
    for filename, expected in plan["source_sha256"].items():
        require(sha(filename) == sha(HERE / Path(filename).name) == expected, "Source/snapshot digest mismatch")
        hashes[filename] = expected
        hashes[str(HERE / Path(filename).name)] = expected
    source_file = next(Path(p) for p in plan["source_sha256"] if Path(p).name == "ecology_information_bounds.py")
    spec = importlib.util.spec_from_file_location("audited_information_bound", source_file)
    source = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(source)

    # Enumerate demand constraints independently as factors, not intersections of
    # the source's prewritten allowed-resource sets.
    classification = Counter()
    tables = []
    for table in product(range(12), repeat=3):
        edges = sum(pair_can_satisfy_both(table, pair) for pair in combinations(range(3), 2))
        classification[edges] += 1
        if edges >= 2:
            tables.append(table)
    physical_layouts = list(permutations(range(4)))
    demand_array = np.asarray([table for table in tables for _ in physical_layouts], dtype=np.int16)
    layout_array = np.asarray([layout for _ in tables for layout in physical_layouts], dtype=np.int8)
    table_ids = {table: i for i, table in enumerate(tables)}
    require(dict(classification) == {0: 120, 1: 612, 2: 576, 3: 420}, "Independent demand support differs")
    source_demands, source_layouts, source_table_ids = source.states()
    require(np.array_equal(demand_array, source_demands) and np.array_equal(layout_array, source_layouts)
            and np.array_equal(source_table_ids, np.repeat(np.arange(996), 24)), "State enumeration mismatch")

    action_tables = [[None] + [(position, destination, partner)
                               for position in range(4) for destination in range(2)
                               for partner in range(3) if partner != focal] for focal in range(3)]
    require(all(action_tables[i] == source.local_actions(i) for i in range(3)), "Local action enumeration differs")
    maxima = np.zeros((3, len(layout_array), 17), dtype=np.int8)
    response_counts = np.zeros((3, 17), dtype=np.int64)
    action_cases, matching_cases, zero_cases = 0, 0, 0
    for ids in product(range(17), repeat=3):
        joint = [action_tables[i][ids[i]] for i in range(3)]
        contributors = matching_contributors(joint)
        action_cases += 1
        for focal in range(3):
            response_counts[focal, ids[focal]] += 1
        if not contributors:
            # Independently proven zero under D1 for every demand/layout. The
            # initialized maxima already include these responses; none is sampled.
            zero_cases += 1
            continue
        require(len(contributors) == 2, "One-sided physical matching unexpectedly counted")
        matching_cases += 1
        score_twice = np.zeros(len(layout_array), dtype=np.int8)
        for actor in contributors:
            position, destination, _ = joint[actor]
            score_twice += wants(demand_array[:, actor], layout_array[:, position], destination).astype(np.int8)
        require(np.isin(score_twice, [0, 1, 2]).all(), "Independent score exceeds legal range")
        for focal in range(3):
            np.maximum(maxima[focal, :, ids[focal]], score_twice,
                       out=maxima[focal, :, ids[focal]])
    require(action_cases == 4913 and matching_cases == 24 and zero_cases == 4889
            and np.all(response_counts == 289), "Not all 17-squared oracle responses were covered")
    for focal in range(3):
        expected = source.other_oracle_payoff(demand_array, layout_array, focal)
        require(np.array_equal(maxima[focal], expected), f"Oracle payoff differs for focal {focal}")

    # Exact site relabeling proof on the entire finite state/action grid.
    # sigma sends the privately observed site of actor i to canonical site i+1,
    # keeps public site 0 fixed, and leaves demands/identities/destinations intact.
    layout_index = {layout: i for i, layout in enumerate(physical_layouts)}
    symmetry_checks = []
    for assignment in permutations((1, 2, 3)):
        sigma = [0, 0, 0, 0]
        for actor, original_site in enumerate(assignment):
            sigma[original_site] = actor + 1
        inverse = [sigma.index(i) for i in range(4)]
        mapping24 = [layout_index[tuple(layout[i] for i in inverse)] for layout in physical_layouts]
        mapping = np.repeat(np.arange(996), 24) * 24 + np.tile(mapping24, 996)
        require(len(set(mapping24)) == 24 and np.array_equal(demand_array, demand_array[mapping]),
                "Site relabeling is not a demand-preserving state bijection")
        require(np.array_equal(layout_array[:, 0], layout_array[mapping, 0]), "Public observation changed")
        for focal in range(3):
            require(np.array_equal(layout_array[:, assignment[focal]], layout_array[mapping, focal+1]),
                    "Private observation not preserved under site relabeling")
            action_map = [0] + [action_tables[focal].index((sigma[a[0]], a[1], a[2]))
                                for a in action_tables[focal][1:]]
            require(len(set(action_map)) == 17 and np.array_equal(
                maxima[focal], maxima[focal][mapping][:, action_map]), "Oracle payoffs change under relabeling")
        symmetry_checks.append({"assignment": list(assignment), "site_bijection": sigma,
                                "all_states_actions_preserved": True})

    result_file = HERE / "execution/results.json"
    results = json.loads(result_file.read_bytes())
    hashes[str(result_file)] = sha(result_file)
    aggregate_rows = []
    certificate_count = 0
    for assignment in permutations((1, 2, 3)):
        for focal in range(3):
            for shared in (True, False):
                keys, counts, sums, winners, best = grouped_optimum(
                    demand_array, layout_array, maxima[focal], focal, assignment, shared)
                numerator, denominator = int(best.sum()), 2 * len(layout_array)
                condition = "I0_D1" if shared else "I1_D1"
                original = next(r for r in results["results"]
                                if r["condition"] == condition and r["constrained_agent"] == focal)
                require((numerator, denominator) == (original["upper_bound_numerator"], original["upper_bound_denominator"])
                        and len(keys) == original["observation_classes"], "Independent conditional optimization differs")
                require(int((best < 2 * counts).sum()) == original["below_full_information_classes"],
                        "Deficient observation class count differs")
                if assignment == (1, 2, 3):
                    path = HERE / "execution" / original["certificate"]
                    require(sha(path) == original["certificate_sha256"], "Certificate digest differs")
                    hashes[str(path)] = sha(path)
                    certificate = np.load(path, allow_pickle=False)
                    translated = np.asarray([(table_ids[k[:3]], k[3], k[4]) if shared else k for k in keys])
                    for name, actual in (("observations", translated), ("counts", counts),
                                         ("action_reward_sums", sums), ("maximizing_action", winners),
                                         ("observation_numerator", best)):
                        require(np.array_equal(actual, certificate[name]), f"Certificate field mismatch: {name}")
                    certificate.close()
                    certificate_count += 1
                aggregate_rows.append({"assignment": list(assignment), "focal": focal, "condition": condition,
                    "observation_classes": len(keys), "hidden_states_per_class_min": int(counts.min()),
                    "hidden_states_per_class_max": int(counts.max()), "numerator": numerator,
                    "denominator": denominator, "bound": numerator / denominator,
                    "argmax_before_observation_averaging_would_give": int(maxima[focal].max(1).sum()) / denominator,
                    "waiting_is_maximizer_in_all_classes": bool(np.all(sums[:, 0] == best)),
                    "chosen_action_counts": {str(i): int((winners == i).sum()) for i in np.unique(winners)}})

    require("torch" not in sys.modules and not any(m == "mlx" or m.startswith("mlx.") for m in sys.modules),
            "Unexpected model runtime import")
    audit = {"status": "independently_verified", "frozen_source_files_sha256": hashes,
        "scope": "all 996 retained demand tables, all 24 layouts, all three focal agents, all 17 focal actions and 17^2 responses; all six public private-site assignments",
        "independent_demand_table_counts": dict(sorted(classification.items())),
        "representative_states": len(layout_array), "expanded_states": len(layout_array) * 6,
        "joint_action_structures_checked": action_cases,
        "identically_zero_structures_proven_and_pruned": zero_cases,
        "matching_structures_scored_in_every_state": matching_cases,
        "independent_state_dependent_score_cases": matching_cases * len(layout_array),
        "responses_per_focal_action": 289,
        "oracle_payoff_entries_compared": int(maxima.size),
        "equivalent_assignment_conditional_optimizations": len(aggregate_rows),
        "source_certificates_all_fields_exact": certificate_count,
        "symmetry_checks": symmetry_checks, "conditional_results": aggregate_rows,
        "new_model_calls": 0, "training_updates": 0,
        "bounds_are_relaxations_not_tight_optima": True}
    output = HERE / "独立核验.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    lines = ["# 单主体局部信息松弛：独立核验", "",
        "已核验冻结代码、完整执行结果和6个NPZ证书。没有调用模型或训练；这不是抽样检查。", "",
        "独立用整数因子定义需求和真实D1动作结算，枚举全部4913个联合动作。4889种动作结构按物理匹配/超载规则在所有状态恒为0，其余24种在全部996×24=23904个状态逐项评分。每个焦点动作对应的289种另两人响应全部覆盖，3×23904×17=1219104个最优响应值与源码公式完全一致。", "",
        "焦点等待时，其余两人可在需求相交时取1；不相交时仍可匹配运输并满足其中一人，取0.5。焦点运输时，三人一起运输会超载；只有它指名的伙伴互选、同位置、同目的地，才有非零可能，所以最优值为两人各自需求正确性的和除以2。若二者都不满足，最优为0，不存在另两人绕开焦点的非零三人方案。", "",
        "I0按完整三人需求表、公共物资、本人私有物资分组，共11952类；每类两个不可区分的隐藏布局。I1只按本人需求与两项可见物资分组，共144类；过滤后的需求分布保留原频数，没有错误地把每个观察类等权。每类先累加同一焦点动作的所有世界回报，再取最大值，最后按世界频数汇总；没有先在每个世界选最优动作。", "",
        "| 条件 | 独立重算上界 | 对三个焦点及六种观察归属的结果 |",
        "| --- | ---: | --- |",
        "| I0–D1 | 47408/47808 = 0.9916331995 | 全部一致 |",
        "| I1–D1 | 43200/47808 = 0.9036144578 | 全部一致 |", "",
        "六种私人位置归属的对称性成立：固定公共位置0，把每人所属私人位置重命名到规范位置1/2/3；这是24布局及17个本地动作的双射，保留个人观察、需求、伙伴身份、目的地与回报。已对全状态/动作验证该双射，并实际重算全部36种条件化优化。原结果可扩展到143424状态；六种归属不是六个独立样本。", "",
        "该值是合法无通信策略的乐观上界：两个响应者被赋予完整世界及焦点已选动作，拥有真实主体没有的信息。各焦点最大化动作不构成一套可同时部署的三人策略，数值也不宣称紧最优。共享或私有随机性只混合不超过该值的选择；在每轮新状态独立同分布、历史不含本轮隐藏变量的前提下，不突破此界。", "",
        "两界严格小于充分信息最优值1，因此在此离散观察、过滤需求分布、D1规则下，无通信不能达到满分期望。它们不表示实际无通信表现接近这两个值，也不证明任意有限符号协议足以满分、主体具备视觉能力或一定形成语言。I0上界距1很小，不能把这个严格性误写成很强的实际通信压力。D0及其他需求支持不在本核验范围内。", ""]
    with (HERE / "独立核验.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    print(json.dumps({k: audit[k] for k in ("status", "representative_states", "expanded_states",
        "joint_action_structures_checked", "oracle_payoff_entries_compared", "source_certificates_all_fields_exact",
        "equivalent_assignment_conditional_optimizations", "new_model_calls")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
