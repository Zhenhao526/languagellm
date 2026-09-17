"""Frozen design for multi-legal coordination with random need rematching."""
from __future__ import annotations

from itertools import product

from research_program.triadic_action_dependency_study import dataset, environment
from research_program.triadic_multi_legal_coordination_study import design as base_design

SEEDS = tuple(range(66501, 66517))
SCHEDULES = ('static', 'rematched')
LIVES = (False, True)
PARTS = ('train', 'new_layouts')
UPDATES = (0, 100, 500, 1500, 3000, 6000)
PAIR_NAMES = ('AB', 'AC', 'BC')
PAIR_TUPLES = ((0, 1), (0, 2), (1, 2))
STATE_ORDER = 'need-major, then layout, then owner'
CONDITIONS = tuple(
    f'{schedule}_partial_multi_strict_PL_{"live" if live else "silent"}'
    for schedule in SCHEDULES for live in (True, False)
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    schedule, suffix = condition.split('_', 1)
    require(schedule in SCHEDULES and suffix in ('partial_multi_strict_PL_live', 'partial_multi_strict_PL_silent'),
            'Malformed condition')
    return schedule, 'partial_multi', 'strict', condition.endswith('_live')


def pair_index(need):
    plans = environment.full_success_plans(tuple(need))
    require(len(plans) == 2 and len({tuple(p[:2]) for p in plans}) == 1,
            'Need is not a two-plan same-pair need')
    pair = tuple(plans[0][:2])
    require(pair in PAIR_TUPLES, 'Unknown target pair')
    return PAIR_TUPLES.index(pair)


def multi_needs():
    """Return the balanced two-legal-plan need set used by the prior confirmation."""
    needs = base_design.multi_needs()
    require(len(needs) == 1452 and all(pair_index(row) in range(3) for row in needs),
            'Unexpected multi-legal support')
    require([sum(pair_index(row) == p for row in needs) for p in range(3)] == [484, 484, 484],
            'Multi-legal pair counts are not balanced')
    require(all(len({row[a] for row in needs}) == 24 for a in range(3)),
            'Individual need coverage is incomplete')
    return needs


def _spec(partition, needs, layouts, owners, role):
    needs = [list(map(int, row)) for row in sorted(tuple(map(int, row)) for row in needs)]
    layouts = [list(map(int, row)) for row in layouts]
    owners = [list(map(int, row)) for row in owners]
    return dict(partition=partition, role=role, needs=needs, layouts=layouts,
                private_sites=owners, world_count=len(needs) * len(layouts) * len(owners),
                state_order=STATE_ORDER, weighting='uniform needs × layouts × owners')


def _source_splits():
    source = dataset.make_prepared()
    return source['partitions']['train']['layouts'], source['partitions']['new_layouts']['layouts'], source['partitions']['train']['private_sites']


def make_prepared():
    train_layouts, new_layouts, owners = _source_splits()
    needs = multi_needs()
    partitions = {
        'train': _spec('train', needs, train_layouts, owners, 'two_legal_plans_train'),
        'new_layouts': _spec('new_layouts', needs, new_layouts, owners, 'two_legal_plans_new_layout'),
    }
    runs = [dict(seed=seed, schedule=schedule, payoff='partial_multi', rule='strict', live=live,
                 condition=f'{schedule}_partial_multi_strict_PL_{"live" if live else "silent"}',
                 directory=f'seed_{seed}_{schedule}_partial_multi_strict_PL_{"live" if live else "silent"}')
            for seed in SEEDS for schedule in SCHEDULES for live in (True, False)]
    return dict(
        schema='triadic_multi_legal_rematched_confirmation_v1', seeds=list(SEEDS),
        schedules=list(SCHEDULES), lives=['live', 'silent'], conditions=list(CONDITIONS),
        parts=list(PARTS), checkpoints=list(UPDATES), pair_names=list(PAIR_NAMES),
        partitions=partitions, runs=runs, independent_seed_count=len(SEEDS),
        need_count=len(needs), pair_counts=[484, 484, 484], train_layout_count=len(train_layouts),
        new_layout_count=len(new_layouts), owner_count=len(owners),
        legal_plan_multiplicity=2, training_updates=len(runs) * 6000,
        training_world_samples=len(runs) * 6000 * 256,
        target_trajectory_files=len(runs) * len(UPDATES),
        target_trajectory_worlds=len(runs) * len(UPDATES) * partitions['train']['world_count'],
        final_files=len(runs),
        scientific_question='Does communication improve plan selection after execution when private needs are rematched across physical agents?',
        primary='centered time-AUC of the schedule-by-communication interaction on Q conditional on any physical execution',
        secondary='unconditioned team Q, target-pair legal action, physical execution, third-agent wait and legal-plan selection',
        rematching='For rematched training, each sampled world assigns its three private needs through an independent uniform six-way permutation; layouts and private-site ownership are unchanged.',
        native_settlement='Unchanged strict physical settlement; both full-success plans remain physically legal after need relabeling.',
        observations='PL own need and public layout only; other private need blocks and full-information flag are absent.',
        pairing='Same initial parameters, world uniforms and message uniforms within every seed; rematched live and silent share permutation assignments.',
        evaluation='Canonical two-legal-plan worlds on train layouts at six checkpoints and on new layouts at the final checkpoint.',
        claim_boundary='This measures conditional task coordination and plan choice. It is not evidence for words, compositionality, intergenerational transmission or human language origin.',
        development_status='Independent seed block; conditional Q primary and rematching schedule frozen before training.',
    )


def sample_indices(spec, uniforms):
    import numpy as np
    u = np.asarray(uniforms, dtype=np.float64)
    require(u.ndim == 2 and u.shape[1] == 3 and np.isfinite(u).all() and ((u >= 0) & (u < 1)).all(),
            'Expected B×3 world uniforms')
    n = np.floor(u[:, 0] * len(spec['needs'])).astype(np.int64)
    l = np.floor(u[:, 1] * len(spec['layouts'])).astype(np.int64)
    o = np.floor(u[:, 2] * len(spec['private_sites'])).astype(np.int64)
    return (n * len(spec['layouts']) + l) * len(spec['private_sites']) + o


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
