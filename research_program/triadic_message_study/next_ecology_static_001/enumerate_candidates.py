"""Read-only finite ecology enumeration; no policies, training or environment edits.

Every ordered demand triple in its stated domain has equal mass. Results are
conditional on edge count, not on any trained policy. Layouts contain all four
materials; including all 24 layouts and six owner permutations repeats each
demand row 144 times and does not change these structural proportions.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
AGENTS = ('A', 'B', 'C')
PAIR_IDS = ((0, 1), (0, 2), (1, 2))
RESOURCE_NAMES = ('wood', 'fiber', 'short', 'long')
DESTINATION_NAMES = ('L', 'R', 'L_or_R')
RESOURCE_SETS = ({0, 1}, {2, 3}, {0, 2}, {1, 3})
DESTINATION_SETS = ({0}, {1}, {0, 1})


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def rate(n, d):
    return {'numerator': n, 'denominator': d, 'fraction': str(Fraction(n, d)),
            'value': n / d}


def pair_name(pair):
    return ''.join(AGENTS[i] for i in pair)


def configurations(needs, pair):
    """Independent intersection formula; resource/destination, not action IDs."""
    i, j = pair
    resources = RESOURCE_SETS[needs[i] // 3] & RESOURCE_SETS[needs[j] // 3]
    destinations = DESTINATION_SETS[needs[i] % 3] & DESTINATION_SETS[needs[j] % 3]
    return [(m, d) for m in sorted(resources) for d in sorted(destinations)]


def record(needs):
    counts = {pair_name(p): len(configurations(needs, p)) for p in PAIR_IDS}
    return {'needs': list(needs), 'compatible_pairs': [p for p, n in counts.items() if n],
            'edge_count': sum(n > 0 for n in counts.values()),
            'full_success_material_destination_configurations': counts}


def distribution(values, categories, n):
    c = Counter(values)
    return {str(k): rate(c[k], n) for k in categories}


def summarize(rows):
    n = len(rows)
    if not n:
        return {'ordered_demand_triples': 0}
    result = {'ordered_demand_triples': n, 'per_agent': {}, 'pairs': {},
              'edge_count': distribution((r['edge_count'] for r in rows), range(4), n)}
    for i, agent in enumerate(AGENTS):
        needs = [r['needs'][i] for r in rows]
        result['per_agent'][agent] = {
            'need_id': distribution(needs, range(12), n),
            'resource_requirement': distribution((RESOURCE_NAMES[k // 3] for k in needs), RESOURCE_NAMES, n),
            'factor': distribution(('kind' if k // 3 < 2 else 'length' for k in needs), ('kind', 'length'), n),
            'destination_constraint': distribution((DESTINATION_NAMES[k % 3] for k in needs), DESTINATION_NAMES, n),
            'accepted_destination_count': distribution((len(DESTINATION_SETS[k % 3]) for k in needs), (1, 2), n),
            'compatible_partner_count': distribution((sum(agent in p for p in r['compatible_pairs']) for r in rows), (0, 1, 2), n),
        }
    for pair in PAIR_IDS:
        key = pair_name(pair)
        ok = sum(key in r['compatible_pairs'] for r in rows)
        # Full-information fixed-pair oracle: compatible -> 1, otherwise -> .5.
        result['pairs'][key] = {
            'can_achieve_full_success': rate(ok, n),
            'fixed_pair_mean_reward_oracle': rate(n + ok, 2 * n),
            'same_resource_requirement': rate(sum(r['needs'][pair[0]] // 3 == r['needs'][pair[1]] // 3 for r in rows), n),
            'same_requirement_factor': rate(sum((r['needs'][pair[0]] // 3 < 2) == (r['needs'][pair[1]] // 3 < 2) for r in rows), n),
            'disjoint_resource_acceptance': rate(sum(not (RESOURCE_SETS[r['needs'][pair[0]] // 3] & RESOURCE_SETS[r['needs'][pair[1]] // 3]) for r in rows), n),
            'disjoint_destination_acceptance': rate(sum(not (DESTINATION_SETS[r['needs'][pair[0]] % 3] & DESTINATION_SETS[r['needs'][pair[1]] % 3]) for r in rows), n),
        }
    two_pairs = {}
    for allowed in combinations([pair_name(p) for p in PAIR_IDS], 2):
        ok = sum(any(p in r['compatible_pairs'] for p in allowed) for r in rows)
        two_pairs['+'.join(allowed)] = {'full_information_select_between_two_fixed_pairs_full_success': rate(ok, n),
                                       'mean_reward_oracle': rate(n + ok, 2 * n)}
    result['two_fixed_pairs'] = two_pairs
    ok = sum(r['edge_count'] > 0 for r in rows)
    result['all_three_pairs_oracle'] = {'full_success': rate(ok, n), 'mean_reward': rate(n + ok, 2 * n)}
    total_configs = sum(sum(r['full_success_material_destination_configurations'].values()) for r in rows)
    result['mean_full_success_matching_configurations'] = rate(total_configs, n)
    result['within_team'] = {
        'all_same_factor': rate(sum(len({k // 6 for k in r['needs']}) == 1 for r in rows), n),
        'all_same_destination_constraint': rate(sum(len({k % 3 for k in r['needs']}) == 1 for r in rows), n),
        'resource_requirement_distinct_count': distribution((len({k // 3 for k in r['needs']}) for r in rows), (1, 2, 3), n),
        'destination_constraint_distinct_count': distribution((len({k % 3 for k in r['needs']}) for r in rows), (1, 2, 3), n),
    }
    singles = [r for r in rows if r['edge_count'] == 1]
    if singles:
        equal = 0
        for r in singles:
            pair = [i for i, a in enumerate(AGENTS) if a in r['compatible_pairs'][0]]
            equal += r['needs'][pair[0]] // 3 == r['needs'][pair[1]] // 3
        result['unique_pair_same_resource_requirement'] = rate(equal, len(singles))
    return result


def compare_marginals(one, multi):
    keys = ('need_id', 'resource_requirement', 'factor', 'destination_constraint', 'accepted_destination_count')
    result = {}
    for agent in AGENTS:
        result[agent] = {}
        for key in keys:
            a, b = one['per_agent'][agent][key], multi['per_agent'][agent][key]
            delta = {c: Fraction(v['numerator'], v['denominator']) - Fraction(b[c]['numerator'], b[c]['denominator']) for c, v in a.items()}
            result[agent][key] = {'all_exactly_equal': all(v == 0 for v in delta.values()),
                                  'one_minus_at_least_two': {c: {'fraction': str(v), 'value': float(v)} for c, v in delta.items()},
                                  'total_variation': {'fraction': str(sum(abs(v) for v in delta.values()) / 2),
                                                     'value': float(sum(abs(v) for v in delta.values()) / 2)}}
    return result


def main():
    targets = [OUT / 'enumerated_tables.json', OUT / 'results.json']
    if any(p.exists() for p in targets):
        raise FileExistsError('Refuse to overwrite prior enumeration output')
    environment = ROOT / 'research_program/triadic_task/environment.py'
    env_hash = digest(environment)
    script_hash = digest(Path(__file__))
    original = [record(t) for t in product(range(12), repeat=3)]
    narrow = [record(tuple(3 * r + destination for r in resources))
              for destination in (0, 1) for resources in product(range(4), repeat=3)]
    # Only import the pure environment for a separate exact cross-check.
    sys.path.insert(0, str(ROOT))
    from research_program.triadic_task import environment as env
    for row in original + narrow:
        official = [pair_name(p) for p in env.compatible_pairs(row['needs'])]
        assert official == row['compatible_pairs']
    assert len(original) == 1728 and len(narrow) == 128
    assert len({tuple(r['needs']) for r in original}) == 1728
    assert len({tuple(r['needs']) for r in narrow}) == 128
    assert sum(r['edge_count'] >= 2 for r in original) == len(env.support()) == 996
    assert digest(environment) == env_hash and digest(Path(__file__)) == script_hash
    results = {}
    for name, rows in [('original_12_needs', original), ('common_single_destination_4_resource_needs', narrow)]:
        groups = {'all': summarize(rows)}
        groups.update({f'exactly_{k}': summarize([r for r in rows if r['edge_count'] == k]) for k in range(4)})
        groups['at_least_2'] = summarize([r for r in rows if r['edge_count'] >= 2])
        groups['marginal_comparison_exactly_1_vs_at_least_2'] = compare_marginals(groups['exactly_1'], groups['at_least_2'])
        results[name] = groups
    targets[0].write_text(json.dumps({'original_12_needs': original,
        'common_single_destination_4_resource_needs': narrow}, ensure_ascii=False, indent=2) + '\n')
    receipt = {'status': 'passed', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {str(environment): env_hash, str(Path(__file__)): script_hash},
        'enumerated_tables_sha256': digest(targets[0]), 'enumerated_rows': 1856,
        'official_compatible_pairs_exact_checks': 1856,
        'distribution': 'Uniform ordered demand triples within each named domain, then conditional on edge count. Narrow domain pools common L and common R equally.',
        'narrow_resource_categories': list(RESOURCE_NAMES),
        'need_id_mapping': [{'id': k, 'resource_requirement': RESOURCE_NAMES[k // 3], 'destination_constraint': DESTINATION_NAMES[k % 3]} for k in range(12)],
        'fixed_pair_bound_scope': 'Chosen pair fixed before demands; site and destination may use full state. Exactly attainable with full information, only an upper bound for a legal PI no-communication policy. All four materials exist once in every layout.',
        'model_forward_calls': 0, 'training_updates': 0, 'environment_modified': False,
        'new_training_or_ecology_selected': False, 'results': results}
    targets[1].write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'output': str(targets[1]), 'domain_sizes': [1728, 128],
                      'results_sha256': digest(targets[1])}, ensure_ascii=False))


if __name__ == '__main__':
    main()
