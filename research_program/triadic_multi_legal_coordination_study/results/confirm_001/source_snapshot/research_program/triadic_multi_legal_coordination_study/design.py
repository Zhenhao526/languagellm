"""Frozen design for the multiple-legal-action coordination study."""
from __future__ import annotations

from pathlib import Path

from research_program.triadic_action_dependency_study import dataset, environment

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(66401, 66417))
PAYOFFS = ('partial_multi',)
RULES = ('strict',)
LIVES = (False, True)
PARTS = ('train', 'new_layouts')
UPDATES = (0, 100, 500, 1500, 3000, 6000)
PAIR_NAMES = ('AB', 'AC', 'BC')
PAIR_TUPLES = ((0, 1), (0, 2), (1, 2))
STATE_ORDER = 'need-major, then layout, then owner'
CONDITIONS = tuple(f'partial_multi_strict_PL_{"live" if live else "silent"}' for live in (True, False))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    return 'partial_multi', 'strict', condition.endswith('_live')


def pair_index(need):
    plans = environment.full_success_plans(tuple(need))
    require(len(plans) == 2 and len({tuple(p[:2]) for p in plans}) == 1, 'Need is not a two-plan same-pair need')
    pair = tuple(plans[0][:2]); require(pair in PAIR_TUPLES, 'Unknown target pair')
    return PAIR_TUPLES.index(pair)


def multi_needs():
    support = [tuple(map(int, row)) for row in __import__('itertools').product(range(24), repeat=3)]
    needs = [row for row in support if len(environment.full_success_plans(row)) == 2 and
             len({tuple(p[:2]) for p in environment.full_success_plans(row)}) == 1]
    require(len(needs) == 1452, 'Unexpected multi-legal need count')
    counts = [sum(pair_index(row) == p for row in needs) for p in range(3)]
    require(counts == [484, 484, 484], 'Multi-legal pair counts are not balanced')
    require(all(len({row[a] for row in needs}) == 24 for a in range(3)), 'Individual need coverage is incomplete')
    return needs


def _source_splits():
    source = dataset.make_prepared()
    return source['partitions']['train']['layouts'], source['partitions']['new_layouts']['layouts'], source['partitions']['train']['private_sites']


def _spec(partition, needs, layouts, owners, role):
    needs = [list(map(int, row)) for row in sorted(needs)]
    layouts = [list(map(int, row)) for row in layouts]; owners = [list(map(int, row)) for row in owners]
    return dict(partition=partition, role=role, needs=needs, layouts=layouts, private_sites=owners,
                world_count=len(needs) * len(layouts) * len(owners), state_order=STATE_ORDER,
                weighting='uniform needs × layouts × owners')


def make_prepared():
    train_layouts, new_layouts, owners = _source_splits(); needs = multi_needs()
    specs = {'train': _spec('train', needs, train_layouts, owners, 'two_legal_plans_train'),
             'new_layouts': _spec('new_layouts', needs, new_layouts, owners, 'two_legal_plans_new_layout')}
    runs = [dict(seed=seed, condition=condition, payoff='partial_multi', rule='strict', live=live,
                 directory=f'seed_{seed}_partial_multi_strict_PL_{"live" if live else "silent"}')
            for seed in SEEDS for condition, live in ((CONDITIONS[0], True), (CONDITIONS[1], False))]
    return dict(schema='triadic_multi_legal_coordination_v1', seeds=list(SEEDS), payoffs=list(PAYOFFS),
                rules=list(RULES), lives=['live', 'silent'], conditions=list(CONDITIONS), parts=list(PARTS),
                checkpoints=list(UPDATES), partitions=specs, runs=runs, independent_seed_count=len(SEEDS),
                need_count=len(needs), pair_counts=[484, 484, 484], train_layout_count=18, new_layout_count=6,
                owner_count=6, training_updates=len(runs) * 6000, training_world_samples=len(runs) * 6000 * 256,
                target_trajectory_files=len(runs) * len(UPDATES),
                target_trajectory_worlds=len(runs) * len(UPDATES) * specs['train']['world_count'],
                final_files=len(runs),
                scientific_question='Does live communication help agents coordinate on one of two equivalent full-success plans?',
                primary='centered time-AUC of live−silent team Q over the two-legal-plan world set',
                secondary='legal target-actor rate, physical execution, third-actor wait and selected legal plan',
                native_settlement='Unchanged strict physical settlement; all full-success plans remain physically legal.',
                observations='PL own need and public layout only; other private need blocks and full-information flag are absent.',
                claim_boundary='Multiple-legal-action coordination is not evidence for words, compositionality, intergenerational transmission or human language origin.',
                development_status='New independent seed block; legal-plan multiplicity and live/silent primary frozen before training.')


def sample_indices(spec, uniforms):
    import numpy as np
    u = np.asarray(uniforms, dtype=np.float64)
    require(u.ndim == 2 and u.shape[1] == 3 and np.isfinite(u).all() and ((u >= 0) & (u < 1)).all(), 'Expected B×3 uniforms')
    n = np.floor(u[:, 0] * len(spec['needs'])).astype(np.int64); l = np.floor(u[:, 1] * len(spec['layouts'])).astype(np.int64); o = np.floor(u[:, 2] * len(spec['private_sites'])).astype(np.int64)
    return (n * len(spec['layouts']) + l) * len(spec['private_sites']) + o


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
