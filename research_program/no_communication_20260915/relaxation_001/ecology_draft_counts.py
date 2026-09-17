"""Independent combinatorial check of the unfrozen three-person task draft.

No agents, policies, image models or natural/symbolic conversations are run.
The witness knows the full state; it is not a no-communication policy.
"""
from collections import Counter
from datetime import datetime
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parent
RESOURCES = (frozenset((0, 1)), frozenset((2, 3)),
             frozenset((0, 2)), frozenset((1, 3)))
DESTINATIONS = (frozenset((0,)), frozenset((1,)), frozenset((0, 1)))
DEMANDS = tuple(product(RESOURCES, DESTINATIONS))
PAIRS = tuple(combinations(range(3), 2))


def compatible(demands, i, j):
    return bool(demands[i][0] & demands[j][0]) and bool(demands[i][1] & demands[j][1])


def score(demands, layout, actions, matching):
    active = [i for i, action in enumerate(actions) if action is not None]
    if len(active) > 2:
        return 0.0
    successes = 0
    for i in active:
        site, destination, partner = actions[i]
        assert site in range(4) and destination in (0, 1) and partner in range(3) and partner != i
        physical = not matching or actions[partner] == (site, destination, i)
        correct = layout[site] in demands[i][0] and destination in demands[i][1]
        successes += int(physical and correct)
    assert successes <= 2
    return successes / 2


def main():
    counts = Counter()
    centers = Counter()
    tables = []
    for demands in product(DEMANDS, repeat=3):
        edges = [(i, j) for i, j in PAIRS if compatible(demands, i, j)]
        counts[len(edges)] += 1
        if len(edges) == 2:
            degree = Counter(i for pair in edges for i in pair)
            centers[next(i for i, n in degree.items() if n == 2)] += 1
        if len(edges) >= 2:
            tables.append((demands, edges))
    assert dict(counts) == {0: 120, 1: 612, 2: 576, 3: 420}
    assert len(tables) == 996 and dict(centers) == {0: 192, 1: 192, 2: 192}
    states = 0
    scoring_cases = 0
    for demands, edges in tables:
        # A full-information witness only. It is not delivered to a learner.
        i, j = edges[0]
        resource = min(demands[i][0] & demands[j][0])
        destination = min(demands[i][1] & demands[j][1])
        for layout in permutations(range(4)):
            site = layout.index(resource)
            actions = [None] * 3
            actions[i], actions[j] = (site, destination, j), (site, destination, i)
            for private_assignment in permutations((1, 2, 3)):
                views = [{0, private_assignment[a]} for a in range(3)]
                assert all(len(v) == 2 for v in views) and set.union(*views) == set(range(4))
                states += 1
                for shared_demands, matching in product((False, True), repeat=2):
                    # Shared/private information changes policy input, not the
                    # fully informed researcher's physical witness or score.
                    assert score(demands, layout, actions, matching) == 1.0
                    scoring_cases += 1
    assert states == 143424 and scoring_cases == 573696
    result = {
        'status': 'verified_combinatorial_draft_only',
        'checked_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'raw_demand_tables': 1728,
        'tables_by_compatible_pair_count': dict(sorted(counts.items())),
        'two_edge_centers': dict(sorted(centers.items())),
        'selected_demand_tables': len(tables),
        'semantic_states': states,
        'four_cell_full_information_witness_checks': scoring_cases,
        'full_information_optimum': 1.0,
        'upper_bound_reason': 'At most two agents execute transport, each binary success is at most one, team score is successes/2; overload scores zero.',
        'local_actions': 17, 'joint_actions': 17**3,
        'overloaded_joint_actions': 16**3,
        'nonoverloaded_joint_actions': 1 + 3*16 + 3*16**2,
        'model_calls': 0, 'training_updates': 0,
        'no_communication_optimum_computed': False,
        'limits': [
            'The task remains an unfrozen proposal; this is not the implemented training environment.',
            'A full-state witness does not obey local observation constraints and proves no communication necessity.',
            'No image/attribute recognition, dialogue sufficiency, symbol budget or learned policy is tested.',
            'D0 retains the common two-person capacity constraint; the switch adds matching dependence, not all dependence.',
            'Information filtering correlations, public-object and fixed-participant shortcuts remain untested.',
        ],
    }
    output = ROOT / 'ecology_draft_counts_20260915.json'
    with output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
