"""Independent exact enumeration and no-communication certificates for the draft.

No training/model modules or optimization libraries. A state's winning actions
are only used inside a CSP that enforces identical actions at identical local
observations. They are never mistaken for a decentralized policy by themselves.
"""
from collections import Counter
from datetime import datetime
from fractions import Fraction
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
RESOURCES = ({0, 1}, {2, 3}, {0, 2}, {1, 3})
DESTINATIONS = ({0}, {1}, {0, 1})
DEMANDS = tuple(product(range(4), range(3)))
PAIRS = tuple(combinations(range(3), 2))


def acceptable(need, material, destination):
    return material in RESOURCES[need[0]] and destination in DESTINATIONS[need[1]]


def compatible(need1, need2):
    return bool(RESOURCES[need1[0]] & RESOURCES[need2[0]]) and bool(
        DESTINATIONS[need1[1]] & DESTINATIONS[need2[1]])


def reward_twice(needs, layout, actions, matching):
    active = [i for i, action in enumerate(actions) if action is not None]
    if len(active) > 2:
        return 0
    total = 0
    for i in active:
        site, destination, partner = actions[i]
        assert 0 <= site < 4 and destination in (0, 1) and partner in range(3) and partner != i
        physical = not matching or actions[partner] == (site, destination, i)
        total += int(physical and acceptable(needs[i], layout[site], destination))
    return total


def d0_local_action(agent, need, public_material, own_site, own_material):
    if agent == 2:
        return None
    if public_material in RESOURCES[need[0]]:
        site = 0
    elif own_material in RESOURCES[need[0]]:
        site = own_site
    else:
        site = min(set(range(4)) - {0, own_site})
    return site, min(DESTINATIONS[need[1]]), 1 - agent


def block_layouts(public_material):
    return tuple((public_material,) + perm
                 for perm in permutations(set(range(4)) - {public_material}))


def pair_actions(pair, site, destination):
    i, j = pair
    actions = [None] * 3
    actions[i] = site, destination, j
    actions[j] = site, destination, i
    return tuple(actions)


def winning_actions(needs, layout):
    out = []
    for i, j in PAIRS:
        for material in sorted(RESOURCES[needs[i][0]] & RESOURCES[needs[j][0]]):
            for destination in sorted(DESTINATIONS[needs[i][1]] & DESTINATIONS[needs[j][1]]):
                actions = pair_actions((i, j), layout.index(material), destination)
                assert reward_twice(needs, layout, actions, True) == 2
                out.append(actions)
    return out


def successful_subset_satisfiable(layouts, options, subset):
    """Exact CSP for one fixed public block; three observation values per agent.

    A variable is (agent, its private material). Each of its occurrences must
    receive the same full action, including wait/site/destination/partner.
    """
    keys = [[(agent, layout[agent + 1]) for agent in range(3)] for layout in layouts]
    visited = 0

    def solve(remaining, assigned):
        nonlocal visited
        visited += 1
        if not remaining:
            return True
        best_state, best_legal = None, None
        for state in remaining:
            legal = [actions for actions in options[state]
                     if all(key not in assigned or assigned[key] == action
                            for key, action in zip(keys[state], actions))]
            if not legal:
                return False
            if best_legal is None or len(legal) < len(best_legal):
                best_state, best_legal = state, legal
        rest = tuple(s for s in remaining if s != best_state)
        for actions in best_legal:
            extended = dict(assigned)
            extended.update(zip(keys[best_state], actions))
            if solve(rest, extended):
                return True
        return False

    return solve(tuple(subset), {}), visited


def connected(vertices, edges):
    found = {min(vertices)}
    while True:
        grown = found | {v for u, v, _ in edges if u in found} | {u for u, v, _ in edges if v in found}
        if grown == found:
            return found == set(vertices)
        found = grown


def graph_certificates():
    checks = 0
    for public in range(4):
        layouts = block_layouts(public)
        for allocation in permutations((1, 2, 3)):
            edges = []
            for u, v in combinations(range(6), 2):
                same = [agent for agent in range(3)
                        if layouts[u][allocation[agent]] == layouts[v][allocation[agent]]]
                assert len(same) <= 1
                if same:
                    edges.append((u, v, same[0]))
            assert len(edges) == 9
            assert all(sum(node in (u, v) for u, v, _ in edges) == 3 for node in range(6))
            for omitted in (None, *range(6)):
                vertices = set(range(6)) - ({omitted} if omitted is not None else set())
                present = [(u, v, agent) for u, v, agent in edges if u in vertices and v in vertices]
                assert connected(vertices, present)
                for pair in PAIRS:
                    active_edges = [(u, v, agent) for u, v, agent in present if agent in pair]
                    assert connected(vertices, active_edges)
                    assert len(active_edges) == (6 if omitted is None else 4)
                    checks += 1
    return checks


def frac(value):
    value = Fraction(value)
    return {"numerator": value.numerator, "denominator": value.denominator, "value": float(value)}


def main():
    counts = Counter()
    support = []
    for needs in product(DEMANDS, repeat=3):
        edges = sum(compatible(needs[i], needs[j]) for i, j in PAIRS)
        counts[edges] += 1
        if edges >= 2:
            support.append(needs)
    assert counts == {0: 120, 1: 612, 2: 576, 3: 420} and len(support) == 996

    # Exact legal I1 policy, hence legal also in I0. No partner demand appears.
    d0_worlds = 0
    for needs in support:
        for layout in permutations(range(4)):
            for allocation in permutations((1, 2, 3)):
                actions = tuple(d0_local_action(i, needs[i], layout[0], allocation[i], layout[allocation[i]])
                                for i in range(3))
                assert reward_twice(needs, layout, actions, False) == 2
                d0_worlds += 1
    assert d0_worlds == 143424

    bad_blocks, good_blocks, identical_bad = [], [], []
    csp_checks = csp_nodes = 0
    public_strategy_reward_sum = Fraction(0)
    i1_fixed_public_reward_sum = Fraction(0)
    witness = None
    for table_id, needs in enumerate(support):
        for public in range(4):
            layouts = block_layouts(public)
            options = [winning_actions(needs, layout) for layout in layouts]
            has_public_pair = any(acceptable(needs[i], public, dest) and acceptable(needs[j], public, dest)
                                  for i, j in PAIRS for dest in (0, 1))
            key = [table_id, public]
            if has_public_pair:
                good_blocks.append(key)
                # A single public-information action tuple works on all six.
                witness_actions = next(actions for actions in options[0]
                                       if any(a is not None and a[0] == 0 for a in actions))
                assert all(reward_twice(needs, layout, witness_actions, True) == 2 for layout in layouts)
            else:
                bad_blocks.append(key)
                if needs[0] == needs[1] == needs[2]:
                    identical_bad.append(key)
                for omitted in range(6):
                    satisfiable, nodes = successful_subset_satisfiable(layouts, options,
                                                                     set(range(6)) - {omitted})
                    assert not satisfiable, (needs, public, omitted)
                    csp_checks += 1
                    csp_nodes += nodes
                if witness is None:
                    max_successes = 0
                    for size in range(6, -1, -1):
                        if any(successful_subset_satisfiable(layouts, options, subset)[0]
                               for subset in combinations(range(6), size)):
                            max_successes = size
                            break
                    assert max_successes == 4 and needs[0] == needs[1] == needs[2]
                    witness = {"table_id": table_id, "demands": needs, "public_material": public,
                               "layouts": layouts, "max_full_reward_layouts_exact": 4,
                               "optimal_expected_reward": frac(Fraction(2, 3))}

            # Restricted but executable I0 baseline: choose a single joint action
            # from shared needs/public material, then use it in all six layouts.
            # max E[reward], never E[max over complete-state actions].
            fixed_candidates = [pair_actions(pair, site, dest)
                                for pair in PAIRS for site in range(4) for dest in (0, 1)]
            best_twice = max(sum(reward_twice(needs, layout, actions, True) for layout in layouts)
                             for actions in fixed_candidates)
            public_strategy_reward_sum += Fraction(best_twice, 12)
            # Legal in I1: AB always take the public pile to L and select each
            # other; C waits. Own demand does not change this fixed action.
            i1_actions = pair_actions((0, 1), 0, 0)
            i1_fixed_public_reward_sum += Fraction(sum(reward_twice(needs, layout, i1_actions, True)
                                                      for layout in layouts), 12)

    blocks = len(support) * 4
    assert (len(good_blocks), len(bad_blocks), len(identical_bad), csp_checks) == (1944, 2040, 24, 12240)
    # General bad-block <=5/6; identical demands have only 0/1 reward, so <=2/3.
    simple_upper = 1 - Fraction(len(bad_blocks), 6 * blocks)
    refined_upper = simple_upper - Fraction(len(identical_bad), 6 * blocks)
    assert simple_upper == Fraction(911, 996) and refined_upper == Fraction(455, 498)
    graph_checks = graph_certificates()
    output = {
        "status": "exact_enumeration_and_strict_bounds_verified",
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "draft_sha256": hashlib.sha256((ROOT.parent / "下一轮生态操纵草案.md").read_bytes()).hexdigest(),
        "scope": "Unfrozen 996-table semantic task only, uniform layouts/observation allocations; no trained agents.",
        "resource_order": ["short wood", "long wood", "short fiber", "long fiber"],
        "demand_filter_order": ["wood", "fiber", "short", "long"],
        "destination_order": ["L", "R", "L or R"],
        "support_tables": support,
        "demand_counts_by_compatible_pairs": dict(sorted(counts.items())),
        "d0": {"legal_i1_policy_verified_worlds": d0_worlds, "i0_exact_optimum": frac(1),
               "i1_exact_optimum": frac(1), "uses_partner_demand": False},
        "d1": {"public_blocks": blocks, "public_pair_good_blocks": good_blocks,
               "no_public_pair_bad_blocks": bad_blocks, "identical_demand_bad_blocks": identical_bad,
               "five_layout_csp_infeasibility_checks": csp_checks, "csp_search_nodes": csp_nodes,
               "graph_connectivity_checks": graph_checks,
               "simple_global_upper_both_i0_i1": frac(simple_upper),
               "refined_global_upper_both_i0_i1": frac(refined_upper),
               "i0_public_information_only_strategy_lower": frac(public_strategy_reward_sum / blocks),
               "i1_fixed_ab_public_l_strategy_lower": frac(i1_fixed_public_reward_sum / blocks),
               "exact_example": witness},
        "shared_randomness": "Fix all public/private random coins independently of the current hidden state. Each resulting deterministic team obeys the same bound, so mixtures cannot exceed it.",
        "limits": ["D1 global bound is certified but not claimed tight.",
                   "The I0 public-only policy is a restricted-class achievable lower bound, not the optimum among private-observation policies.",
                   "I1 inherits the I0 upper bound by information inclusion, not by giving I1 agents the shared demand table.",
                   "Fresh layouts must remain uniform conditional on public information/history; layout-correlated randomness or metadata invalidates this support argument.",
                   "Visual instance/menus may not leak the unseen layout. No perception or learned-policy performance is measured."],
        "model_calls": 0, "training_updates": 0}
    path = ROOT / "theory_probe.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"path": str(path), "d0": output["d0"],
                      "d1": {k: v for k, v in output["d1"].items() if not k.endswith("blocks")},
                      "good_blocks": len(good_blocks), "bad_blocks": len(bad_blocks)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
