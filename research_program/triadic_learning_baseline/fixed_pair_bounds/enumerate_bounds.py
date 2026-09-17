"""Static fixed-pair oracle bounds; never reads actor checkpoints or scores.

The prepared split is obtained from runner.make_prepared only. Acceptance and
all upper bounds are independently enumerated using explicit semantic sets.
"""
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from hashlib import sha256
from itertools import combinations, permutations, product
import json
from pathlib import Path
from random import Random

from research_program.triadic_learning_baseline import runner
from research_program.triadic_task import environment as env


HERE = Path(__file__).resolve().parent
PAIRS = {'AB': (0, 1), 'AC': (0, 2), 'BC': (1, 2)}
MATERIALS = (('wood', 'short'), ('wood', 'long'), ('fiber', 'short'), ('fiber', 'long'))
PREDICATES = (('kind', 'wood'), ('kind', 'fiber'), ('length', 'short'), ('length', 'long'))
DESTINATIONS = (('L',), ('R',), ('L', 'R'))


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def fraction(value):
    return {'numerator': value.numerator, 'denominator': value.denominator,
            'value': float(value), 'percentage': 100 * float(value)}


def write_new(path, value):
    with path.open('xb') as stream:
        stream.write(encoded(value))


def semantic_sets():
    rows = []
    for (factor, value), allowed_destinations in product(PREDICATES, DESTINATIONS):
        index = 0 if factor == 'kind' else 1
        accepted = {(m, d) for m, attrs in enumerate(MATERIALS)
                    for d in ('L', 'R') if attrs[index] == value and d in allowed_destinations}
        assert len(accepted) in (2, 4)
        rows.append({'need_id': len(rows), 'factor': factor, 'value': value,
                     'destinations': list(allowed_destinations), 'accepted': accepted})
    assert len(rows) == 12
    return rows


def main():
    out = HERE / 'enumeration_001'
    if out.exists():
        raise FileExistsError('Refuse to overwrite a prior enumeration')
    source_paths = [Path(__file__), Path(runner.__file__), Path(env.__file__), HERE.parent / 'plan.md']
    source_hashes = {str(p): file_hash(p) for p in source_paths}
    needs = semantic_sets()
    # Consistency check only: counts below do not call env.accepts/support.
    for n, m, d in product(range(12), range(4), range(2)):
        assert ((m, ('L', 'R')[d]) in needs[n]['accepted']) == env.accepts(n, m, d)
    classifications = []
    for triple in product(range(12), repeat=3):
        pairs = {}
        for label, (i, j) in PAIRS.items():
            left, right = needs[triple[i]]['accepted'], needs[triple[j]]['accepted']
            # Enumerate every physical material/destination choice. The third
            # actor waits, both active actors choose each other and same site.
            best_units = max(int((m, d) in left) + int((m, d) in right)
                             for m, d in product(range(4), ('L', 'R')))
            compatible = bool(left & right)
            assert best_units == (2 if compatible else 1)
            pairs[label] = {'compatible': compatible, 'max_satisfied_units': best_units,
                            'max_native_reward': best_units / 2}
        edge_count = sum(v['compatible'] for v in pairs.values())
        if edge_count >= 2:
            classifications.append({'needs': list(triple), 'compatible_pair_count': edge_count, 'pairs': pairs})
    assert len(classifications) == 996
    by_need = {tuple(row['needs']): row for row in classifications}
    prepared = runner.make_prepared()
    # Independently reconstruct the fixed selection and compare exact order.
    multisets = sorted({tuple(sorted(n)) for n in by_need})
    assert len(multisets) == 212
    Random(2026091901).shuffle(multisets)
    assert multisets[:159] == prepared['train_need_multisets']
    assert multisets[159:] == prepared['heldout_need_multisets']
    train_set = set(multisets[:159])
    train_needs = [n for n in by_need if tuple(sorted(n)) in train_set]
    held_needs = [n for n in by_need if tuple(sorted(n)) not in train_set]
    layouts = list(permutations(range(4)))
    Random(2026091902).shuffle(layouts)
    assignments = list(permutations((1, 2, 3)))
    independent = {
        'train': (train_needs, layouts[:18]), 'new_needs': (held_needs, layouts[:18]),
        'new_layouts': (train_needs, layouts[18:]), 'new_needs_and_layouts': (held_needs, layouts[18:]),
    }
    summaries = {}
    partitions = {}
    for name, (triples, maps) in independent.items():
        spec = prepared['partitions'][name]
        assert triples == spec['needs'] and maps == spec['layouts'] and assignments == spec['private_sites']
        for t in triples:
            assert set(permutations(t)) <= set(triples), 'Actor permutations cross the split'
        multiplier = len(maps) * len(assignments)
        assert len(triples) * multiplier == spec['world_count']
        pair_stats = {}
        for pair in PAIRS:
            good = [list(t) for t in triples if by_need[t]['pairs'][pair]['compatible']]
            bad = [list(t) for t in triples if not by_need[t]['pairs'][pair]['compatible']]
            c, b, total = len(good), len(bad), len(triples)
            assert c + b == total and b > 0
            pair_stats[pair] = {
                'compatible_demand_count': c, 'incompatible_demand_count': b,
                'compatible_world_count': c * multiplier, 'incompatible_world_count': b * multiplier,
                'fullsuccess_upper_bound': fraction(Fraction(c, total)),
                'native_R_optimal_upper_bound': fraction(Fraction(2*c+b, 2*total)),
                'compatible_demand_tables': good, 'incompatible_demand_tables': bad,
            }
        assert len({p['compatible_demand_count'] for p in pair_stats.values()}) == 1
        edge_hist = Counter(by_need[t]['compatible_pair_count'] for t in triples)
        two_pair = {}
        for pair_1, pair_2 in combinations(PAIRS, 2):
            covered = sum(by_need[t]['pairs'][pair_1]['compatible'] or by_need[t]['pairs'][pair_2]['compatible'] for t in triples)
            assert covered == len(triples)
            common = next(a for a in pair_1 if a in pair_2)
            two_pair[pair_1 + '+' + pair_2] = {'compatible_demand_count': covered,
                'compatible_world_count': covered * multiplier, 'fullsuccess_oracle': 1,
                'native_R_oracle': 1, 'always_active_hub_possible': common}
        fixed_full = Fraction(pair_stats['AB']['compatible_demand_count'], len(triples))
        summaries[name] = {
            'demand_count': len(triples), 'layout_count': len(maps), 'private_assignment_count': len(assignments),
            'world_count': len(triples) * multiplier,
            'compatible_pair_count_histogram_demands': dict(edge_hist),
            'compatible_pair_count_histogram_worlds': {str(k): v * multiplier for k, v in edge_hist.items()},
            'fixed_pair_bounds': pair_stats, 'best_single_pair_fullsuccess': fraction(fixed_full),
            'best_single_pair_native_R': fraction((1+fixed_full)/2),
            'any_two_preselected_pairs_oracle': two_pair,
            'min_nondefault_pair_fullsuccess_fraction_to_reach_99percent': fraction(Fraction(99, 100)-fixed_full),
        }
        partitions[name] = {'needs': triples, 'layouts': maps, 'private_sites': assignments}
    assert sum(s['world_count'] for s in summaries.values()) == 143424
    assert {str(p): file_hash(p) for p in source_paths} == source_hashes
    result = {
        'status': 'completed_static_enumeration_only', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': source_hashes, 'runner_make_prepared_sha256': sha256(encoded(prepared)).hexdigest(),
        'scope': 'Full-information optimal site/destination choice with one fixed mutual transport pair; third actor waits; matching D1.',
        'counts_do_not_use_env_accepts': True, 'semantic_consistency_checks': 96,
        'demand_tables_all': 1728, 'supported_demand_tables': 996,
        'partitions': summaries,
        'no_trained_policy_or_future_results_read': True, 'model_calls': 0, 'parameter_updates': 0,
        'actor_labels_or_action_inputs_changed': False,
    }
    serial_needs = [{**r, 'accepted': sorted(r['accepted'])} for r in needs]
    out.mkdir()
    write_new(out / 'bounds.json', result)
    write_new(out / 'demand_classification.json', {'need_semantics': serial_needs, 'supported_tables': classifications})
    write_new(out / 'partition_definitions.json', partitions)
    write_new(out / 'manifest.json', {'source_sha256': source_hashes, 'outputs_sha256': {
        name: file_hash(out / name) for name in ('bounds.json', 'demand_classification.json', 'partition_definitions.json')},
        'scope': 'Static task reference only, not learned behavior or language evidence.'})
    print(json.dumps({name: {k: row[k] for k in ('world_count', 'best_single_pair_fullsuccess', 'best_single_pair_native_R')}
                      for name, row in summaries.items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
