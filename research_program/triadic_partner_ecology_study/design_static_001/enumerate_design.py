"""Exact finite destination-stratified ecology audit. No model dependencies."""
from collections import Counter, defaultdict
from fractions import Fraction as F
from itertools import combinations, permutations, product
from pathlib import Path
import hashlib
import json

from research_program.triadic_task import environment as env

HERE = Path(__file__).resolve().parent
PAIRS = ((0, 1), (0, 2), (1, 2))
NAMES = ('AB', 'AC', 'BC')
RES = ((0, 1), (2, 3), (0, 2), (1, 3))
DEST = ((0,), (1,), (0, 1))


def fraction(x):
    x = F(x)
    return {'fraction': str(x), 'numerator': x.numerator, 'denominator': x.denominator, 'decimal': float(x)}


def encoded(value):
    if isinstance(value, F):
        return fraction(value)
    if isinstance(value, dict):
        return {str(k): encoded(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [encoded(v) for v in value]
    return value


def compatible(r, d):
    return tuple(k for k, (a, b) in enumerate(PAIRS)
                 if set(RES[r[a]]) & set(RES[r[b]]) and set(DEST[d[a]]) & set(DEST[d[b]]))


def successful_plans(r, d):
    return tuple((k, material, destination) for k, (a, b) in enumerate(PAIRS)
        for material, destination in product(range(4), range(2))
        if material in RES[r[a]] and material in RES[r[b]]
        and destination in DEST[d[a]] and destination in DEST[d[b]])


def tv(left, right):
    return sum(abs(left.get(k, F())-right.get(k, F())) for k in set(left)|set(right))/2


def run():
    env_sha = hashlib.sha256(Path(env.__file__).read_bytes()).hexdigest()
    assert env_sha == 'b2d49ab829edc92e77df527f5b470602c289cdf3ad7728d5e71f411c53582ef6'
    all_rows, layers = [], []
    for d in product(range(3), repeat=3):
        groups = {'zero': [], 'unique': [], 'multiple': []}
        for r in product(range(4), repeat=3):
            edges = compatible(r, d)
            needs = tuple(3*r[a]+d[a] for a in range(3))
            assert tuple(PAIRS[k] for k in edges) == env.compatible_pairs(needs)
            row = {'destinations': d, 'resources': r, 'needs': needs,
                   'compatible_pairs': [NAMES[k] for k in edges], 'successful_plan_count': len(successful_plans(r, d))}
            groups['zero' if not edges else 'unique' if len(edges) == 1 else 'multiple'].append(row)
            all_rows.append(row)
        common = bool(groups['unique'] and groups['multiple'])
        layers.append({'D': d, 'common': common,
            'destination_compatible_pairs': [NAMES[k] for k, (a, b) in enumerate(PAIRS) if set(DEST[d[a]]) & set(DEST[d[b]])],
            'resource_counts': {name: len(rows) for name, rows in groups.items()},
            '_groups': groups})
    selected = [layer for layer in layers if layer['common']]
    assert len(selected) == 21
    weighted = {'unique': [], 'multiple': []}
    for layer in selected:
        for ecology in weighted:
            rows = layer['_groups'][ecology]
            for row in rows:
                weighted[ecology].append({**row, 'semantic_world_weight': F(1, 21*len(rows))})
    layouts, owners = list(permutations(range(4))), list(permutations((1, 2, 3)))
    result, marginal_output = {}, {}
    native_checks = 0
    for ecology, rows in weighted.items():
        assert sum(r['semantic_world_weight'] for r in rows) == 1
        need_marginal = [defaultdict(F) for _ in range(3)]
        resource_joint, full_joint, D_joint = defaultdict(F), defaultdict(F), defaultdict(F)
        need_pairs = [defaultdict(F) for _ in PAIRS]
        resource_pairs = [defaultdict(F) for _ in PAIRS]
        gamma, number_of_solutions, edge_hist, solution_hist = [F()]*3, F(), defaultdict(F), defaultdict(F)
        destination_compatible_mean = F()
        observations = [defaultdict(F) for _ in range(3)]
        for layer in selected:
            group = layer['_groups'][ecology]
            # Integer local-observation frequencies within D; exact layer weight applied last.
            local_counts = [Counter() for _ in range(3)]
            for row in group:
                r, d, needs = row['resources'], row['destinations'], row['needs']
                w = F(1, 21*len(group)); edges = compatible(r, d)
                plans = successful_plans(r, d)
                number_of_solutions += w*len(plans)
                solution_hist[len(plans)] += w; edge_hist[len(edges)] += w
                resource_joint[r] += w; full_joint[needs] += w; D_joint[d] += w
                destination_compatible_mean += w*len(layer['destination_compatible_pairs'])
                for k in range(3):
                    gamma[k] += w*(k in edges)
                    need_marginal[k][needs[k]] += w
                for k, (a, b) in enumerate(PAIRS):
                    need_pairs[k][(needs[a], needs[b])] += w
                    resource_pairs[k][(r[a], r[b])] += w
                # Full24 native plans on canonical layout: other layouts permute site labels.
                state = env.State(needs, (0, 1, 2, 3))
                native_full = 0
                for k, (a, b) in enumerate(PAIRS):
                    for material, destination in product(range(4), range(2)):
                        actions = {agent: {'kind': 'wait'} for agent in env.AGENTS}
                        for who, other in ((a, b), (b, a)):
                            actions[env.AGENTS[who]] = dict(kind='transport', site=env.SITES[material],
                                destination=env.DESTINATIONS[destination], partner=env.AGENTS[other])
                        out = env.settle(state, actions, require_match=True)
                        native_full += out['full_success']; native_checks += 1
                assert native_full == len(plans)
                for layout, assignment in product(layouts, owners):
                    for a in range(3):
                        key = (needs[a], *assignment, layout[0], layout[assignment[a]])
                        local_counts[a][key] += 1
            for a in range(3):
                for key, count in local_counts[a].items():
                    observations[a][key] += F(count, 21*len(group)*144)
        assert all(sum(m.values()) == 1 for m in observations)
        assert all(len(m) == 864 for m in observations)
        marginal_output[ecology] = {'own_need': need_marginal, 'full_destination_table': D_joint,
            'resource_triple': resource_joint, 'need_triple': full_joint,
            'need_pairs': need_pairs, 'resource_pairs': resource_pairs,
            'PI_observation': [{str(k): v for k, v in m.items()} for m in observations]}
        adaptive = F()
        for layer in selected:
            group = layer['_groups'][ecology]
            adaptive += max(F(sum(NAMES[k] in r['compatible_pairs'] for r in group), len(group)) for k in range(3))/21
        result[ecology] = dict(semantic_world_count=len(rows), physical_world_count=len(rows)*144,
            destination_strata=21, compatible_edge_count_distribution=edge_hist,
            full_success_plan_count_distribution=solution_hist, expected_full_success_plan_count=number_of_solutions,
            fixed_pair_full_success_oracle=dict(zip(NAMES, gamma)),
            fixed_pair_native_reward_oracle={n: (1+g)/2 for n, g in zip(NAMES, gamma)},
            fixed_two_pair_full_success_oracle={NAMES[i]+'+'+NAMES[j]:sum(r['semantic_world_weight']*
                bool(set((NAMES[i], NAMES[j])) & set(r['compatible_pairs'])) for r in rows)
                for i, j in combinations(range(3), 2)},
            full_information_D_adaptive_pair_oracle=adaptive,
            mean_destination_compatible_pairs=destination_compatible_mean,
            own_need_marginal=need_marginal, PI_observations_per_actor=864,
            full_information_observation_note='Contains all needs; ecological supports disjoint, not marginally matched.')
    for layer in layers:
        layer['ecologies'] = {}
        for ecology in ('unique', 'multiple'):
            group = layer['_groups'][ecology]
            if not group:
                layer['ecologies'][ecology] = None
                continue
            hist = Counter(r['successful_plan_count'] for r in group)
            gamma = {name: F(sum(name in r['compatible_pairs'] for r in group), len(group)) for name in NAMES}
            layer['ecologies'][ecology] = dict(resource_count=len(group),
                full_success_plan_count_histogram=dict(sorted(hist.items())),
                mean_full_success_plans=F(sum(r['successful_plan_count'] for r in group), len(group)),
                fixed_pair_full_success_oracle=gamma,
                fixed_pair_native_reward_oracle={k: (1+v)/2 for k, v in gamma.items()})
        del layer['_groups']
    u, m = marginal_output['unique'], marginal_output['multiple']
    checks = dict(all_1728_compatibility_graphs_match_original=True,
        native_environment_canonical_plan_checks=native_checks,
        individual_12_need_marginals_equal=u['own_need'] == m['own_need'],
        full_destination_joint_distribution_equal=u['full_destination_table'] == m['full_destination_table'],
        PI_observation_marginals_equal=u['PI_observation'] == m['PI_observation'],
        all_resource_values_uniform_given_each_D=True,
        no_models_loaded_no_training=True)
    for layer in selected:
        for ecology in ('unique', 'multiple'):
            group = [r for r in weighted[ecology] if tuple(r['destinations']) == tuple(layer['D'])]
            assert all(sum(row['resources'][a] == v for row in group)*4 == len(group)
                       for a, v in product(range(3), range(4)))
    assert checks['individual_12_need_marginals_equal'] and checks['full_destination_joint_distribution_equal'] and checks['PI_observation_marginals_equal']
    dependencies = dict(need_triple_TV=tv(u['need_triple'], m['need_triple']),
        resource_triple_TV=tv(u['resource_triple'], m['resource_triple']),
        pair_need_TV={NAMES[k]: tv(u['need_pairs'][k], m['need_pairs'][k]) for k in range(3)},
        pair_resource_TV={NAMES[k]: tv(u['resource_pairs'][k], m['resource_pairs'][k]) for k in range(3)},
        destination_conflict_probability=F(sum(len(l['destination_compatible_pairs']) < 3 for l in selected), 21),
        weighted_multiple_vs_old_uniform996_need_TV=tv(m['need_triple'], {needs: F(1, 996) for needs in m['need_triple']}))
    report = dict(schema='destination_stratified_partner_ecology_static_v1',
        distribution='D uniform over21 shared strata; resource triple conditionally uniform within each ecology; independent uniform24 layouts and6 private-site assignments',
        shared_destination_strata=[l['D'] for l in layers if l['common']],
        excluded_destination_strata=[l['D'] for l in layers if not l['common']],
        per_destination=layers, ecologies=result, unavoidable_joint_differences=dependencies,
        checks=checks, source_sha256={str(Path(env.__file__).resolve()):env_sha,
            str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    for filename, value in (('results.json', report), ('weighted_semantic_support.json', weighted), ('marginals.json', marginal_output)):
        with (HERE/filename).open('x') as f:
            json.dump(encoded(value), f, ensure_ascii=False, indent=2); f.write('\n')
    print(json.dumps(encoded({'ecologies': result, 'differences': dependencies, 'checks': checks}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
