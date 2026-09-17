"""Frozen design for factorized neutral-action coordination with partner choice.

The task keeps the public D1 resource world, but selects need triples with
exactly two full-success plans that use different partner pairs.  The policy
has an explicit binary ``neutral/engage`` head and a 16-way proposal head;
the proposal is ignored when the neutral action is selected.
"""
from __future__ import annotations

from itertools import product

from research_program.triadic_action_dependency_study import dataset, environment

SEEDS = tuple(range(66701, 66717))
SCHEDULES = ('static', 'rematched')
LIVES = (False, True)
PARTS = ('train', 'new_layouts')
UPDATES = (0, 100, 500, 1500, 3000, 6000)
PAIR_NAMES = ('AB', 'AC', 'BC')
PAIR_TUPLES = ((0, 1), (0, 2), (1, 2))
STATE_ORDER = 'need-major, then layout, then owner'
CONDITIONS = tuple(
    f'{schedule}_altpair_factorized_strict_PL_{"live" if live else "silent"}'
    for schedule in SCHEDULES for live in (True, False)
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    schedule, suffix = condition.split('_', 1)
    require(schedule in SCHEDULES and suffix in ('altpair_factorized_strict_PL_live',
                                                  'altpair_factorized_strict_PL_silent'),
            'Malformed condition')
    return schedule, 'altpair_factorized', 'strict', condition.endswith('_live')


def pair_mask_from_plans(plans):
    mask = 0
    for plan in plans:
        i, j = plan[:2]
        mask |= 1 << PAIR_TUPLES.index(tuple(sorted((i, j))))
    return mask


def altpair_needs():
    """All need triples with exactly two full plans on distinct partner pairs."""
    rows = []
    for need in product(range(24), repeat=3):
        plans = environment.full_success_plans(tuple(need))
        pairs = {tuple(p[:2]) for p in plans}
        if len(plans) == 2 and len(pairs) == 2:
            rows.append(tuple(map(int, need)))
    require(len(rows) == 1560, 'Unexpected alternative-partner need count')
    counts = [sum(pair_mask_from_plans(environment.full_success_plans(row)) & (1 << p) != 0
                  for row in rows) for p in range(3)]
    require(counts == [1040, 1040, 1040], 'Alternative-partner pair support is not balanced')
    pair_set_counts = {tuple(sorted(tuple(p[:2]) for p in environment.full_success_plans(row))): 0
                       for row in rows}
    for row in rows:
        key = tuple(sorted(tuple(p[:2]) for p in environment.full_success_plans(row)))
        pair_set_counts[key] += 1
    require(sorted(pair_set_counts.values()) == [520, 520, 520],
            'Alternative-partner pair-set counts are not balanced')
    require(all(len({row[a] for row in rows}) == 24 for a in range(3)),
            'Individual need coverage is incomplete')
    return rows


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
    needs = altpair_needs()
    partitions = {
        'train': _spec('train', needs, train_layouts, owners, 'two_alternative_partner_plans_train'),
        'new_layouts': _spec('new_layouts', needs, new_layouts, owners, 'two_alternative_partner_plans_new_layout'),
    }
    runs = [dict(seed=seed, schedule=schedule, payoff='altpair_factorized', rule='strict', live=live,
                 condition=f'{schedule}_altpair_factorized_strict_PL_{"live" if live else "silent"}',
                 directory=f'seed_{seed}_altpair_factorized_strict_PL_{"live" if live else "silent"}')
            for seed in SEEDS for schedule in SCHEDULES for live in (True, False)]
    return dict(
        schema='triadic_factorized_neutral_altpartner_v1', seeds=list(SEEDS),
        schedules=list(SCHEDULES), lives=['live', 'silent'], conditions=list(CONDITIONS),
        parts=list(PARTS), checkpoints=list(UPDATES), pair_names=list(PAIR_NAMES),
        partitions=partitions, runs=runs, independent_seed_count=len(SEEDS),
        need_count=len(needs), pair_set_counts={'AB_AC': 520, 'AB_BC': 520, 'AC_BC': 520},
        train_layout_count=len(train_layouts), new_layout_count=len(new_layouts), owner_count=len(owners),
        legal_plan_multiplicity=2, legal_pair_multiplicity=2,
        training_updates=len(runs) * 6000, training_world_samples=len(runs) * 6000 * 256,
        target_trajectory_files=len(runs) * len(UPDATES),
        target_trajectory_worlds=len(runs) * len(UPDATES) * partitions['train']['world_count'],
        final_files=len(runs),
        scientific_question='Does communication affect partner and site choice when two different partner pairs are equally successful, after neutral participation is separately represented?',
        primary='centered time-AUC of rematching-by-communication interaction on target-pair choice conditional on engagement',
        secondary='team Q, physical execution, engagement, proposal content conditional on engagement, partner/site selection and third-agent neutral rate',
        rematching='For rematched training, each sampled world assigns its three private needs through an independent uniform six-way permutation; static uses identity assignment.',
        native_settlement='Strict mutual proposal settlement; exactly two full-success plans remain legal for every layout and use distinct partner pairs.',
        observations='PL own need and public layout only; other private need blocks and full-information flag are absent.',
        action_factorization='Two-way intent: neutral=0 or engage=1. If engage, a 16-way proposal chooses site, destination and partner. Neutral ignores the proposal head.',
        pairing='Same initial parameters, world uniforms, batch indices and message uniforms within every seed; rematched live/silent share permutation assignments.',
        evaluation='Canonical two-alternative-partner worlds on train layouts at six checkpoints and new layouts at final checkpoint.',
        claim_boundary='This measures factorized task coordination and partner/content choice. It is not evidence for words, compositionality, intergenerational transmission or human language origin.',
        development_status='Independent seed block; alternative-partner support, explicit neutral head and primary decomposition frozen before training.',
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
