"""Frozen design for the bidirectional cross-split confirmation.

The task, observation interface, payoff table and optimizer are inherited from
the already frozen factorial-payoff study.  This package changes only the
initialization block and makes the seen-to-heldout and heldout-to-seen edge
measurements part of the training-time protocol.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from research_program.triadic_action_dependency_study import environment as native
from research_program.triadic_factorial_payoff_study import design as base
from research_program.triadic_need_response_study import cases

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(65101, 65117))
PAYOFFS = base.PAYOFFS
RULES = base.RULES
LIVES = base.LIVES
PARTS = ('train', 'heldout_resource')
UPDATES = base.UPDATES
CONDITIONS = base.CONDITIONS
STATE_ORDER = base.STATE_ORDER
BACKGROUND_COUNT = 18 * 6


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    return base.parse_condition(condition)


def payoff_rewards(native_rewards, payoff):
    return base.payoff_rewards(native_rewards, payoff)


def sample_indices(spec, uniforms):
    return base.sample_indices(spec, uniforms)


def _target_pair(need):
    plans = native.full_success_plans(tuple(need))
    require(len(plans) == 1, 'Each need must have one unique plan')
    return ((0, 1), (0, 2), (1, 2)).index(tuple(plans[0][:2]))


def _edge_list(source_spec, destination_spec):
    source_lookup = {tuple(map(int, row)): i for i, row in enumerate(source_spec['needs'])}
    destination_lookup = {tuple(map(int, row)): i for i, row in enumerate(destination_spec['needs'])}
    rows = []
    for before in source_lookup:
        for actor in range(3):
            for axis, changed_id in cases.flips(before[actor]):
                after = list(before)
                after[actor] = changed_id
                after = tuple(after)
                if after not in destination_lookup:
                    continue
                before_pair = _target_pair(before)
                after_pair = _target_pair(after)
                if before_pair == after_pair:
                    continue
                rows.append(
                    dict(
                        before_need=list(before),
                        after_need=list(after),
                        before_index=source_lookup[before],
                        after_index=destination_lookup[after],
                        changed_person=actor,
                        axis=axis,
                        axis_index=cases.AXES.index(axis),
                        target_pairs=[before_pair, after_pair],
                    )
                )
    rows.sort(key=lambda row: (row['changed_person'], row['axis_index'], row['before_need'], row['after_need']))
    groups = {f'{actor}_{axis}': [] for actor in range(3) for axis in range(3)}
    for index, row in enumerate(rows):
        row['edge_index'] = index
        groups[f"{row['changed_person']}_{row['axis_index']}"].append(index)
    require(len(rows) == 576, 'Unexpected cross-split edge count')
    require(all(groups[f'{actor}_{axis}'] for actor in range(3) for axis in (0, 1)),
            'Missing kind/length cross-split stratum')
    require(all(not groups[f'{actor}_2'] for actor in range(3)),
            'Destination cross-split edges should be empty')
    return rows, groups


def build_cross_splits(partitions):
    train = partitions['train']
    held = partitions['heldout_resource']
    forward_edges, forward_groups = _edge_list(train, held)
    reverse_edges, reverse_groups = _edge_list(held, train)
    return {
        'train_to_heldout': dict(source='train', destination='heldout_resource', edges=forward_edges, groups=forward_groups),
        'heldout_to_train': dict(source='heldout_resource', destination='train', edges=reverse_edges, groups=reverse_groups),
        'background_count': BACKGROUND_COUNT,
        'weighting': 'Equal six changed-person×{kind,length} strata; within each stratum equal edges and backgrounds.',
    }


def make_prepared():
    value = deepcopy(base.make_prepared())
    value['schema'] = 'triadic_cross_split_confirmation_v1'
    value['seeds'] = list(SEEDS)
    value['parts'] = list(PARTS)
    value['runs'] = [
        dict(seed=seed, payoff=payoff, rule=rule, live=live,
             condition=f'{payoff}_{rule}_PL_{"live" if live else "silent"}',
             directory=f'seed_{seed}_{payoff}_{rule}_PL_{"live" if live else "silent"}')
        for seed in SEEDS for payoff in PAYOFFS for rule in RULES for live in (True, False)
    ]
    value['cross_splits'] = build_cross_splits(value['partitions'])
    value['scientific_question'] = (
        'Does a payoff ecology increase bidirectional behavioral response across a '
        'seen-to-heldout object×attribute semantic edge?'
    )
    value['primary'] = (
        'centered time-AUC of the payoff×communication interaction on bidirectional '
        'cross-split changed-actor Q, averaged over six nonempty strata, rules and directions'
    )
    value['evaluation'] = 'Compact bidirectional cross-split Q and partner metrics at six checkpoints; final checkpoints retained for audit.'
    value['selection'] = 'fixed16 new seeds ×2 payoff modes ×2 rules ×2 communication channels ×6000 updates; no early stopping'
    value['claim_boundary'] = (
        'Cross-split behavioral response is not by itself evidence for words, compositionality, '
        'intergenerational transmission or human language origin.'
    )
    value['development_status'] = 'New independent seed block; bidirectional cross-split target is frozen before training.'
    value['training_updates'] = len(value['runs']) * 6000
    value['training_world_samples'] = len(value['runs']) * 6000 * 256
    return value


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
