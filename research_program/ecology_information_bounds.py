"""Freeze and compute a one-local-agent relaxation of the no-message task.

The other two agents are granted full current state and the constrained agent's
action. This is an upper bound, not a feasible decentralized policy or training.
"""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
from itertools import combinations, permutations, product
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'ecology_draft_counts.py'
DEFAULT = ROOT / 'no_communication_20260915/relaxation_001'
RESOURCES = ({0, 1}, {2, 3}, {0, 2}, {1, 3})
DESTINATIONS = ({0}, {1}, {0, 1})
DEMANDS = tuple(product(RESOURCES, DESTINATIONS))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def new_json(path, data):
    with Path(path).open('x', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def states():
    tables = [t for t in product(range(12), repeat=3)
              if sum(bool(DEMANDS[t[i]][0] & DEMANDS[t[j]][0]) and
                     bool(DEMANDS[t[i]][1] & DEMANDS[t[j]][1]) for i, j in combinations(range(3), 2)) >= 2]
    layouts = list(permutations(range(4)))
    assert len(tables) == 996 and len(layouts) == 24
    return (np.repeat(np.asarray(tables, dtype=np.int16), 24, axis=0),
            np.tile(np.asarray(layouts, dtype=np.int8), (996, 1)),
            np.repeat(np.arange(996, dtype=np.int16), 24))


def local_actions(agent):
    return [None] + list(product(range(4), range(2), [a for a in range(3) if a != agent]))


def other_oracle_payoff(demand_ids, layouts, agent):
    """Return twice-reward for each constrained action; others see full state.

    A waiting focal agent permits the other pair to attain 1 if their needs
    intersect in both factors, otherwise 1/2. An active focal agent can receive
    a matching response from its named partner, while the third waits.
    """
    resource = np.asarray([[c in d[0] for c in range(4)] for d in DEMANDS])
    dest = np.asarray([[c in d[1] for c in range(2)] for d in DEMANDS])
    left, right = [a for a in range(3) if a != agent]
    compatible = ((resource[demand_ids[:, left]] & resource[demand_ids[:, right]]).any(1) &
                  (dest[demand_ids[:, left]] & dest[demand_ids[:, right]]).any(1))
    payoff = np.empty((len(layouts), 17), dtype=np.int8)
    payoff[:, 0] = 1 + compatible.astype(np.int8)
    for index, (site, destination, partner) in enumerate(local_actions(agent)[1:], 1):
        own = resource[demand_ids[:, agent], layouts[:, site]] & dest[demand_ids[:, agent], destination]
        other = resource[demand_ids[:, partner], layouts[:, site]] & dest[demand_ids[:, partner], destination]
        payoff[:, index] = own.astype(np.int8) + other.astype(np.int8)
    return payoff


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__).resolve(), SOURCE]
    sources = {str(f): sha(f) for f in files}
    for f in files:
        shutil.copyfile(f, out / f.name)
    plan = {
        'status': 'prepared_not_executed',
        'created': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        'source_sha256': sources, 'demand_support': 'at_least_two_compatible_pairs',
        'demand_tables': 996, 'layouts': 24, 'fixed_private_assignment': [1, 2, 3],
        'representative_states': 23904, 'expanded_states': 143424,
        'expansion_rule': 'All six publicly known private-site assignments are related by a bijection of the three nonpublic sites. The map distribution is uniform and menus contain all semantic actions; each context permits its own policy. Values are equal, not independent replicates.',
        'conditions': ['I0_D1', 'I1_D1'], 'constrained_agents': [0, 1, 2],
        'reward': 'half the number of individually satisfied needs after physical matching; overload zero',
        'relaxation': 'Keep one agent local and grant the other two all current state and knowledge of its chosen action; maximize their response per state, then sum within the local observation classes, then maximize the focal action.',
        'observation_I0': ['all_three_demands', 'public_material', 'own_private_material'],
        'observation_I1': ['own_demand', 'public_material', 'own_private_material'],
        'no_message': True, 'new_model_calls': 0, 'training_updates': 0,
        'randomness': 'Independent new state each round; past histories and shared/private randomness carry no current private variables. A mixture of deterministic policies cannot exceed this expected-reward upper bound.',
        'labels': 'Post-draft exact enumeration of a relaxation, not the exact unrestricted no-communication optimum.',
    }
    new_json(out / 'plan.json', plan)
    new_json(out / 'freeze.json', {'plan_sha256': sha(out / 'plan.json')})
    return plan


def execute(out):
    plan = json.loads((out / 'plan.json').read_text())
    freeze = json.loads((out / 'freeze.json').read_text())
    assert sha(out / 'plan.json') == freeze['plan_sha256']
    for filename, expected in plan['source_sha256'].items():
        assert sha(filename) == expected == sha(out / Path(filename).name)
    execution = out / 'execution'
    execution.mkdir(exist_ok=False)
    demand_ids, layouts, table_ids = states()
    results = []
    for agent in (0, 1, 2):
        payoff = other_oracle_payoff(demand_ids, layouts, agent)
        for shared in (True, False):
            keys = np.column_stack((table_ids if shared else demand_ids[:, agent], layouts[:, 0], layouts[:, agent + 1]))
            obs, inverse = np.unique(keys, axis=0, return_inverse=True)
            sums = np.zeros((len(obs), 17), dtype=np.int64)
            np.add.at(sums, inverse, payoff)
            counts = np.bincount(inverse)
            winners = sums.argmax(1)
            optimum = sums[np.arange(len(obs)), winners]
            numerator = int(optimum.sum())
            denominator = 2 * len(layouts)
            tag = f'I{0 if shared else 1}_D1_agent{agent}'
            np.savez_compressed(execution / f'{tag}.npz', observations=obs, counts=counts,
                                action_reward_sums=sums, maximizing_action=winners,
                                observation_numerator=optimum)
            result = {'condition': f'I{0 if shared else 1}_D1', 'constrained_agent': agent,
                      'representative_states': len(layouts), 'observation_classes': len(obs),
                      'upper_bound_numerator': numerator, 'upper_bound_denominator': denominator,
                      'upper_bound': numerator / denominator, 'strictly_below_one': numerator < denominator,
                      'certificate': f'{tag}.npz', 'certificate_sha256': sha(execution / f'{tag}.npz'),
                      'below_full_information_classes': int((optimum < 2*counts).sum())}
            results.append(result)
            print(json.dumps(result), flush=True)
    for condition in ('I0_D1', 'I1_D1'):
        assert len({r['upper_bound_numerator'] for r in results if r['condition'] == condition}) == 1
    data = {'status': 'completed', 'plan_sha256': freeze['plan_sha256'], 'results': results,
            'exact_unrestricted_optimum_claimed': False, 'new_model_calls': 0, 'training_updates': 0,
            'finished': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            'limits': ['The two oracle agents have strictly more information than legal policies.',
                       'The focal maximizing action is not a deployable three-agent policy certificate.',
                       'The bound assumes the draft iid state distribution, not temporally correlated worlds.',
                       'No actor perception, language understanding, communication budget or learning is verified.']}
    new_json(execution / 'results.json', data)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'execute'))
    parser.add_argument('--out', type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = prepare(args.out) if args.command == 'prepare' else execute(args.out)
    print(json.dumps({'command': args.command, 'status': result['status']}))
