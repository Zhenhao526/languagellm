"""Static payoff and coordination audit. No networks, parameters, or training.

Independent truth implementation, with direct old-environment comparisons.
The state support and split are read from the frozen context study.
"""
from collections import Counter
from fractions import Fraction
from itertools import combinations, product
from pathlib import Path
import hashlib
import json
import math
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
PRIOR = ROOT / 'research_program/triadic_action_dependency_study/results/context_001/prepared.json'
RESOURCE = ((0, 1), (2, 3), (0, 2), (1, 3), (0,), (1,), (2,), (3,))
DESTINATION = ((0,), (1,), (0, 1))
PAIRS = tuple(combinations(range(3), 2))
PLANS = tuple((i, j, m, d) for i, j in PAIRS for m, d in product(range(4), range(2)))


def accepts(need, material, destination):
    resource, dest = divmod(need, 3)
    return material in RESOURCE[resource] and destination in DESTINATION[dest]


def joint(plan):
    i, j, material, dest = plan
    ans = [0, 0, 0]
    for a, b in ((i, j), (j, i)):
        ans[a] = 1 + 4 * material + 2 * dest + tuple(k for k in range(3) if k != a).index(b)
    return tuple(ans)


def units(needs, choices):
    active = [a for a, c in enumerate(choices) if c]
    if len(active) != 2:
        return 0
    i, j = active
    decoded = {}
    for a in active:
        c = choices[a] - 1
        decoded[a] = (c // 4, (c // 2) % 2, tuple(k for k in range(3) if k != a)[c % 2])
    if decoded[i][2] != j or decoded[j][2] != i or decoded[i][:2] != decoded[j][:2]:
        return 0
    m, d = decoded[i][:2]
    return int(accepts(needs[i], m, d)) + int(accepts(needs[j], m, d))


def frac(v):
    return str(Fraction(v))


def partition_bounds(needs):
    own = [dict() for _ in range(3)]
    pair_counts = Counter()
    positive_hist = Counter()
    full_correct = {}
    fixed_pair_max_units = Counter()
    for ns in needs:
        scores = [units(ns, joint(p)) for p in PLANS]
        assert scores.count(2) == 1
        correct = PLANS[scores.index(2)]
        full_correct[ns] = joint(correct)
        pair_counts[correct[:2]] += 1
        positive_hist[sum(u > 0 for u in scores)] += 1
        for pair in PAIRS:
            fixed_pair_max_units[pair, max(scores[8 * PAIRS.index(pair):8 * (PAIRS.index(pair) + 1)])] += 1
        for a in range(3):
            role = correct[1] if a == correct[0] else correct[0] if a == correct[1] else -1
            own[a].setdefault(ns[a], Counter())[role] += 1
    n = len(needs)
    rho = [Fraction(sum(max(c.values()) for c in rows.values()), n) for rows in own]
    assert rho[0] == rho[1] == rho[2]
    assert list(pair_counts.values()) == [n // 3] * 3
    by_alpha = {}
    for alpha in (Fraction(1, 2), Fraction(1, 10)):
        by_alpha[str(alpha)] = dict(
            full_information_optimal_utility='1', full_information_optimal_full_success='1',
            global_fixed_pair_utility_oracle=frac(Fraction(1, 3) + Fraction(2, 3) * alpha),
            global_fixed_pair_mean_log_expected_utility_oracle=float(Fraction(2, 3)) * math.log(float(alpha)),
            no_communication_full_success_upper_bound=frac(rho[0]),
            no_communication_expected_utility_upper_bound=frac(alpha + (1-alpha) * rho[0]),
            at_most_two_pairs_full_success_upper_bound='2/3',
            at_most_two_pairs_utility_upper_bound=frac(Fraction(2, 3) + alpha / 3),
            no_communication_mean_log_expected_utility_jensen_upper_bound=math.log(float(alpha + (1-alpha) * rho[0])),
            native_reward_bound_unchanged=True)
    return dict(need_count=n, unique_success_pair_counts={str(k):v for k,v in sorted(pair_counts.items())},
        positive_plan_count_histogram={str(k):v for k,v in sorted(positive_hist.items())},
        fixed_pair_max_satisfied_units={str(k):v for k,v in sorted(fixed_pair_max_units.items())},
        no_communication_role_lower_bound='1/3',
        no_communication_role_bayes_upper_bound=frac(rho[0]),
        no_communication_native_reward_upper_bound=frac(Fraction(1, 2) + rho[0] / 2),
        global_fixed_pair_full_success_upper_bound='1/3',
        global_fixed_pair_native_reward_oracle='2/3',
        no_communication_bound_is_not_achieved_policy_or_exact_optimum=True,
        by_alpha=by_alpha), full_correct


def main():
    started = time.time()
    from research_program.triadic_action_dependency_study import environment as original
    prepared = json.loads(PRIOR.read_text())
    split = {k: [tuple(ns) for ns in prepared['partitions'][part]['needs']]
             for k, part in (('train', 'train'), ('heldout', 'new_needs'))}
    split['all'] = sorted(split['train'] + split['heldout'])
    assert len(split['all']) == len(set(split['all'])) == 5376
    reports = {}
    all_correct = None
    for name, needs in split.items():
        reports[name], correct = partition_bounds(needs)
        if name == 'all':
            all_correct = correct
    old_full_plan_checks = proper_mixture_checks = unilateral_checks = wrong_pair_worlds = 0
    example = None
    for ns in split['all']:
        full_plans = [p for p in PLANS if units(ns, joint(p)) == 2]
        assert tuple(full_plans) == original.full_success_plans(ns)
        old_full_plan_checks += 1
        target = all_correct[ns]
        full_pair = full_plans[0][:2]
        for pair in PAIRS:
            if pair == full_pair:
                continue
            candidates = [joint(p) for p in PLANS if p[:2] == pair]
            best_units = max(units(ns, action) for action in candidates)
            assert best_units == 1
            old = next(a for a in candidates if units(ns, a) == 1)
            assert all(a != b for a, b in zip(old, target))
            wrong_pair_worlds += 1
            for bits in product((0, 1), repeat=3):
                if sum(bits) in (0, 3):
                    continue
                mixed = tuple(target[a] if bits[a] else old[a] for a in range(3))
                assert units(ns, mixed) == 0
                proper_mixture_checks += 1
            for actor in range(3):
                for action in range(17):
                    if action == old[actor]:
                        continue
                    alternate = list(old); alternate[actor] = action
                    assert units(ns, alternate) == 0
                    unilateral_checks += 1
            if example is None:
                example = dict(needs=list(ns), old_actions=list(old), correct_actions=list(target),
                               wrong_pair=list(pair), correct_pair=list(full_pair))
    analytic = {}
    for alpha in (0.5, 0.1):
        root = math.sqrt(alpha)
        epsilon = root / (1 + root)
        minimum = alpha / ((1 + root) ** 2)
        analytic[str(alpha)] = dict(
            expected_utility_polynomial='alpha * (1-epsilon)^3 + epsilon^3',
            stationary_minimum_epsilon=epsilon, minimum_expected_utility=minimum,
            initial_log_utility_directional_derivative=-3.0,
            log_utility_barrier_relative_to_old=math.log(minimum / alpha),
            old_expected_utility=alpha, correct_expected_utility=1.0,
            scope='Illustrative synchronous product-policy interpolation in one wrong-pair state; no entropy, no neural parameterization.')
    sources = [Path(__file__), PRIOR,
        ROOT/'research_program/triadic_action_dependency_study/environment.py',
        ROOT/'research_program/triadic_action_dependency_study/dataset.py',
        ROOT/'research_program/triadic_message_study/runner.py',
        ROOT/'research_program/triadic_coordination_study/runner.py']
    result = dict(status='static_complete', partitions=reports, analytic_barrier=analytic,
        example=example, verified_counts=dict(old_environment_plan_checks=old_full_plan_checks,
            wrong_pair_worlds=wrong_pair_worlds, proper_old_new_mixture_checks=proper_mixture_checks,
            unilateral_alternative_actions=unilateral_checks),
        computation_scope=dict(neural_forward_calls=0, parameter_loads=0, training_updates=0,
            additional_model_results_read=0, canonical_layouts_per_need=1,
            reason_layouts_not_enumerated='Site relabeling is a bijection of the full action menu. Owners do not enter physical settlement.'),
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        elapsed_seconds=time.time()-started)
    output = HERE/'payoff_static_001.json'
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(dict(status=result['status'], output=str(output), verified_counts=result['verified_counts'],
                         elapsed_seconds=result['elapsed_seconds']), ensure_ascii=False))


if __name__ == '__main__':
    main()
