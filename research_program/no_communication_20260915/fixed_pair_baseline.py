"""Frozen finite-team fixed-AB reference. No model, training, or shared secrets.

First: python fixed_pair_baseline.py freeze
Then:  python fixed_pair_baseline.py execute
All outputs are exclusive new files; only this subtask's prefixed files are written.
"""
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import shutil
import sys


OUT = Path(__file__).resolve().parent
DRAFT = OUT.parent / "下一轮生态操纵草案.md"
RESOURCE_SETS = ({0, 1}, {2, 3}, {0, 2}, {1, 3})
DESTINATION_SETS = ({0}, {1}, {0, 1})
DEMANDS = tuple(product(range(4), range(3)))
ACTIONS = tuple(product(range(4), range(2)))
LAYOUTS = tuple(permutations(range(4)))
STRATEGIES = ("D0_complement", "I0_D1_exact_fixed_AB",
              "I1_D1_public_L", "I1_D1_public_own_first", "I1_D1_public_position_optimized")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def accepted(resource, destination, demand_id):
    resource_type, destinations = DEMANDS[demand_id]
    return resource in RESOURCE_SETS[resource_type] and destination in DESTINATION_SETS[destinations]


def compatible(a, b):
    ra, da = DEMANDS[a]
    rb, db = DEMANDS[b]
    return bool(RESOURCE_SETS[ra] & RESOURCE_SETS[rb]) and bool(DESTINATION_SETS[da] & DESTINATION_SETS[db])


def state_support():
    return [table for table in product(range(12), repeat=3)
            if sum(compatible(table[a], table[b]) for a, b in combinations(range(3), 2)) >= 2]


def observe(agent, table, layout, information):
    """Only public resource, own private resource and permitted demands leave here."""
    assert agent in (0, 1)
    return (layout[0], layout[agent + 1], table if information == "I0" else table[agent])


def policy(strategy, agent, obs, policies):
    """No state, partner private view or current shared random draw is available."""
    public, own_private, demand_info = obs
    own = demand_info[agent] if isinstance(demand_info, tuple) else demand_info
    if strategy == "D0_complement":
        allowed = RESOURCE_SETS[DEMANDS[own][0]]
        if public in allowed:
            site = 0
        elif own_private in allowed:
            site = agent + 1
        else:
            site = next(s for s in range(4) if s not in (0, agent + 1))
        return site, min(DESTINATION_SETS[DEMANDS[own][1]])
    if strategy == "I0_D1_exact_fixed_AB":
        a, b, _ = demand_info
        entry = policies["I0"][f"{public}/{a}/{b}"]
        private_index = [r for r in range(4) if r != public].index(own_private)
        return ACTIONS[entry["A" if agent == 0 else "B"][private_index]]
    if strategy == "I1_D1_public_L":
        return 0, 0
    if strategy == "I1_D1_public_own_first":
        return 0, min(DESTINATION_SETS[DEMANDS[own][1]])
    if strategy == "I1_D1_public_position_optimized":
        return 0, policies["I1_public"][str(public)]["A" if agent == 0 else "B"][own]
    raise ValueError(strategy)


def score_units(table, layout, a, b, matching):
    """Fixed A/B active, reciprocally partnered, C waits; score denominator is 2."""
    if matching and a != b:
        return 0
    return int(accepted(layout[a[0]], a[1], table[0])) + int(accepted(layout[b[0]], b[1], table[1]))


def solve_public_demands():
    """Exact fixed-pair I0: 8^3 A tables; pointwise best response for B."""
    solutions = {}
    for public, a, b in product(range(4), range(12), range(12)):
        private_types = [r for r in range(4) if r != public]
        worlds = [(private_types.index(rest[0]), private_types.index(rest[1]), (public,) + rest)
                  for rest in permutations(private_types)]
        payoff = [[int(accepted(layout[s], d, a)) + int(accepted(layout[s], d, b))
                   for s, d in ACTIONS] for _, _, layout in worlds]
        best, best_a, best_b = -1, None, None
        for pa in product(range(8), repeat=3):
            responses = [[0] * 8 for _ in range(3)]
            for w, (oa, ob, _) in enumerate(worlds):
                action = pa[oa]
                responses[ob][action] += payoff[w][action]
            pb = tuple(max(range(8), key=lambda x: responses[ob][x]) for ob in range(3))
            value = sum(responses[ob][pb[ob]] for ob in range(3))
            if value > best:
                best, best_a, best_b = value, pa, pb
        solutions[f"{public}/{a}/{b}"] = {"A": best_a, "B": best_b,
            "max_success_units_over_six_layouts": best, "score_denominator": 12}
    return solutions


def solve_private_public_position(tables):
    """Exact only in the declared restricted family: site0, own-demand destination."""
    weights = Counter((a, b) for a, b, _ in tables)
    solutions = {}
    for public in range(4):
        # Utility already integrates the actual filtered C-demand distribution.
        reward = [[[weights[a, b] * (int(accepted(public, d, a)) + int(accepted(public, d, b)))
                    for d in range(2)] for b in range(12)] for a in range(12)]
        best, best_a, best_b = -1, None, None
        for pa in product(range(2), repeat=12):
            response = [[sum(reward[a][b][d] for a in range(12) if pa[a] == d)
                         for d in range(2)] for b in range(12)]
            pb = tuple(max(range(2), key=lambda d: response[b][d]) for b in range(12))
            value = sum(response[b][pb[b]] for b in range(12))
            if value > best:
                best, best_a, best_b = value, pa, pb
        solutions[str(public)] = {"A": best_a, "B": best_b,
            "max_success_units_over_996_tables": best, "score_denominator": 1992}
    return solutions


def fraction_record(numerator, denominator):
    value = Fraction(numerator, denominator)
    return {"numerator": numerator, "denominator": denominator,
            "reduced_fraction": str(value), "value": float(value)}


def freeze():
    plan = {"created_at": datetime.now().astimezone().isoformat(), "status": "frozen_before_enumeration",
        "script_sha256": sha(__file__), "draft_sha256": sha(DRAFT),
        "distribution": "Uniform over all 996 demand tables with at least two compatible pairs, independently uniform over all 24 layouts.",
        "resource_order": ["short_wood", "long_wood", "short_fiber", "long_fiber"],
        "resource_demand_order": ["wood", "fiber", "short", "long"],
        "destination_demand_order": ["L", "R", "either"],
        "demand_index": "resource_demand_index * 3 + destination_demand_index",
        "fixed_pair": "A and B always active and choose each other; C always waits.",
        "observation_assignment": "canonical public=0,A=1,B=2,C=3; publicly rename private sites for the other six assignments",
        "I0": "all three demands shared; only public and own-private resource seen",
        "I1": "only own demand, public and own-private resource seen",
        "strategy_families": list(STRATEGIES),
        "D0_complement": "Choose visible acceptable resource; if both visible resources are unacceptable, either unseen site is acceptable; own first allowed destination.",
        "I0_D1_exact": "For each public resource and pair of public AB demands, enumerate all 8^3 deterministic A policies over three private resource observations; calculate B best response separately for each of its three observations. Deterministic lexicographic tie breaking.",
        "I1_D1_restricted": "Always choose public site0. For each public resource, enumerate all 2^12 A destination maps indexed only by own demand and compute B pointwise best response. Also evaluate constant L and own first acceptable destination.",
        "evaluation": "Strictly enumerate 996*24=23904 states for every frozen strategy. Assert identical legal observation produces identical action; prove public site-renaming equivariance for all six observation assignments.",
        "bounds": "D0 exact optimum<=1. I0-D1 exhaustive optimum only for fixed AB. I1-D1 restricted optimum is a lower bound on unrestricted fixed-AB policy; I0 fixed-AB optimum upper-bounds I1. Full-state fixed-AB oracle reported separately and never deployed as policy.",
        "randomness": "No randomness used. Shared randomness independent of current state is a convex mixture of deterministic policies and cannot improve the stated deterministic optima.",
        "new_model_calls": 0, "training_updates": 0}
    write_new(OUT / "fixed_pair_plan.json", plan)
    with (OUT / "fixed_pair_code_snapshot.py").open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    print("Frozen script, strategy classes, objective, observation and distribution; not executed.")


def execute():
    plan = json.loads((OUT / "fixed_pair_plan.json").read_text())
    assert sha(__file__) == plan["script_sha256"] == sha(OUT / "fixed_pair_code_snapshot.py")
    assert sha(DRAFT) == plan["draft_sha256"]
    assert not (OUT / "fixed_pair_results.json").exists()
    write_new(OUT / "fixed_pair_started.json", {"started_at": datetime.now().astimezone().isoformat(),
                                                "plan_sha256": sha(OUT / "fixed_pair_plan.json")})
    tables = state_support()
    assert len(tables) == 996 and len(LAYOUTS) == 24
    policies = {"I0": solve_public_demands(), "I1_public": solve_private_public_position(tables)}
    write_new(OUT / "fixed_pair_policies.json", policies)
    evaluations = {}
    for strategy in STRATEGIES:
        information = "I0" if strategy == "I0_D1_exact_fixed_AB" else "I1"
        histogram, action_tables, total, states = Counter(), ({}, {}), 0, 0
        for table, layout in product(tables, LAYOUTS):
            actions = []
            for agent in (0, 1):
                obs = observe(agent, table, layout, information)
                action = policy(strategy, agent, obs, policies)
                assert action in ACTIONS
                if obs in action_tables[agent]:
                    assert action_tables[agent][obs] == action
                action_tables[agent][obs] = action
                actions.append(action)
            units = score_units(table, layout, *actions, matching=strategy != "D0_complement")
            histogram[units] += 1
            total += units
            states += 1
        assert states == 23904
        evaluations[strategy] = {"score": fraction_record(total, states * 2),
            "state_count": states, "reward_histogram_success_units": dict(sorted(histogram.items())),
            "legal_observation_counts_AB": list(map(len, action_tables)),
            "identical_observation_identical_action": True}
    assert evaluations["D0_complement"]["score"]["value"] == 1
    # I0 use of a public demand table cannot change private-layout probabilities.
    exact_n = sum(policies["I0"][f"{p}/{a}/{b}"]["max_success_units_over_six_layouts"]
                  for a, b, _ in tables for p in range(4))
    assert exact_n == evaluations["I0_D1_exact_fixed_AB"]["score"]["numerator"]
    restricted_n = 6 * sum(entry["max_success_units_over_996_tables"] for entry in policies["I1_public"].values())
    assert restricted_n == evaluations["I1_D1_public_position_optimized"]["score"]["numerator"]
    # Public renaming preserves each legal observation and each transported item.
    symmetry_cases = 0
    for assignment, layout in product(permutations((1, 2, 3)), LAYOUTS):
        renaming = (0,) + assignment
        renamed_layout = [None] * 4
        for canonical, physical in enumerate(renaming):
            renamed_layout[physical] = layout[canonical]
        for agent in (0, 1):
            assert (renamed_layout[0], renamed_layout[assignment[agent]]) == (layout[0], layout[agent + 1])
            for site, destination in ACTIONS:
                assert renamed_layout[renaming[site]] == layout[site]
                symmetry_cases += 1
    fixed_pair_compatible = sum(compatible(table[0], table[1]) for table in tables)
    oracle = fraction_record(2 * fixed_pair_compatible + (996 - fixed_pair_compatible), 1992)
    result = {"status": "completed_mathematical_reference", "finished_at": datetime.now().astimezone().isoformat(),
        "plan_sha256": sha(OUT / "fixed_pair_plan.json"), "script_sha256": sha(__file__),
        "policies_sha256": sha(OUT / "fixed_pair_policies.json"),
        "distribution": {"demand_tables": 996, "layouts": 24, "canonical_states": 23904,
            "public_observation_assignments": 6, "represented_full_states": 143424,
            "fixed_AB_compatible_demand_tables": fixed_pair_compatible},
        "evaluations": evaluations,
        "D0_I0_and_I1_fixed_AB_exact_optimum": evaluations["D0_complement"]["score"],
        "D1_I0_fixed_AB_exact_optimum": evaluations["I0_D1_exact_fixed_AB"]["score"],
        "D1_I1_fixed_AB_lower_bound": evaluations["I1_D1_public_position_optimized"]["score"],
        "D1_I1_fixed_AB_upper_bound": evaluations["I0_D1_exact_fixed_AB"]["score"],
        "D1_fixed_AB_full_state_oracle_upper_bound": oracle,
        "exact_search_counts": {"I0_public_blocks": 576, "I0_A_policies_per_block": 512,
            "I1_public_position_blocks": 4, "I1_A_destination_policies_per_block": 4096},
        "public_renaming_item_checks": symmetry_cases,
        "new_model_calls": 0, "training_updates": 0, "I1_unrestricted_optimum_claimed": False,
        "fixed_AB_result_is_general_three_agent_optimum": False}
    write_new(OUT / "fixed_pair_results.json", result)
    print(json.dumps({key: result[key] for key in ("status", "D0_I0_and_I1_fixed_AB_exact_optimum",
        "D1_I0_fixed_AB_exact_optimum", "D1_I1_fixed_AB_lower_bound", "D1_I1_fixed_AB_upper_bound",
        "D1_fixed_AB_full_state_oracle_upper_bound")}, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("freeze", "execute"):
        raise SystemExit("Usage: fixed_pair_baseline.py {freeze|execute}")
    freeze() if sys.argv[1] == "freeze" else execute()
