"""Frozen design for a rule-by-information-by-channel coordination study.

The worlds contain exactly two full-success plans that use different partner
pairs.  The experiment varies the settlement rule (``strict`` versus
``reciprocal``), what each actor can observe (private-local ``PL`` versus
full-information ``FI``), partner rematching, and whether messages are routed
to the other actors.  All four factors are crossed on the same seeds and
random streams.
"""
from __future__ import annotations

from itertools import product

from research_program.triadic_action_dependency_study import dataset, environment

SEEDS = tuple(range(66801, 66809))
RULES = ('strict', 'reciprocal')
INFORMATIONS = ('PL', 'FI')
SCHEDULES = ('static', 'rematched')
LIVES = (True, False)
PARTS = ('train', 'new_layouts')
UPDATES = (0, 100, 500, 1500, 3000, 6000)
PAIR_NAMES = ('AB', 'AC', 'BC')
PAIR_TUPLES = ((0, 1), (0, 2), (1, 2))
STATE_ORDER = 'need-major, then layout, then owner'

CONDITIONS = tuple(
    f'{rule}_{information}_{schedule}_{"live" if live else "silent"}'
    for rule in RULES for information in INFORMATIONS
    for schedule in SCHEDULES for live in (True, False)
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    rule, information, schedule, channel = condition.split('_')
    require(rule in RULES and information in INFORMATIONS and schedule in SCHEDULES
            and channel in ('live', 'silent'), 'Malformed condition')
    return rule, information, schedule, channel == 'live'


def pair_mask_from_plans(plans):
    mask = 0
    for plan in plans:
        i, j = plan[:2]
        mask |= 1 << PAIR_TUPLES.index(tuple(sorted((i, j))))
    return mask


def altpair_needs():
    """All need triples with exactly two full plans on distinct pairs."""
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
    return (source['partitions']['train']['layouts'],
            source['partitions']['new_layouts']['layouts'],
            source['partitions']['train']['private_sites'])


def make_prepared():
    train_layouts, new_layouts, owners = _source_splits()
    needs = altpair_needs()
    partitions = {
        'train': _spec('train', needs, train_layouts, owners,
                       'two_alternative_partner_plans_train'),
        'new_layouts': _spec('new_layouts', needs, new_layouts, owners,
                             'two_alternative_partner_plans_new_layout'),
    }
    runs = [dict(seed=seed, rule=rule, information=information, schedule=schedule,
                 live=live, condition=f'{rule}_{information}_{schedule}_{"live" if live else "silent"}',
                 directory=f'seed_{seed}_{rule}_{information}_{schedule}_{"live" if live else "silent"}')
            for seed in SEEDS for rule in RULES for information in INFORMATIONS
            for schedule in SCHEDULES for live in (True, False)]
    return dict(
        schema='triadic_rule_communication_v1', seeds=list(SEEDS), rules=list(RULES),
        information=list(INFORMATIONS), schedules=list(SCHEDULES),
        lives=['live', 'silent'], conditions=list(CONDITIONS), parts=list(PARTS),
        checkpoints=list(UPDATES), pair_names=list(PAIR_NAMES), partitions=partitions,
        runs=runs, independent_seed_count=len(SEEDS), need_count=len(needs),
        pair_set_counts={'AB_AC': 520, 'AB_BC': 520, 'AC_BC': 520},
        train_layout_count=len(train_layouts), new_layout_count=len(new_layouts),
        owner_count=len(owners), legal_plan_multiplicity=2, legal_pair_multiplicity=2,
        training_updates=len(runs) * 6000,
        training_world_samples=len(runs) * 6000 * 256,
        target_trajectory_files=len(runs) * len(UPDATES),
        target_trajectory_worlds=len(runs) * len(UPDATES) * partitions['train']['world_count'],
        final_files=len(runs),
        scientific_question=(
            'Does allowing the unmatched third actor to be ignored change the causal effect '
            'of routed messages on partner selection, and how does that effect depend on '
            'private versus full information?'),
        primary=('rule × routed-communication interaction on target-pair legal rate '
                 'conditional on physical execution, centered time-AUC'),
        secondary=('team Q, physical execution, conditional Q, engage rate, proposal legality, '
                   'third-agent neutrality, ignored and unexecuted proposals, pair/site/destination rates'),
        rematching=('For rematched training, each sampled world assigns its three private '
                     'needs through an independent uniform six-way permutation; static uses identity.'),
        settlement=(
            'Strict requires a matching pair and a neutral third actor. Reciprocal requires '
            'only the two matching proposals; the third proposal is ignored.'),
        observations=('PL exposes each actor own need and public layout/owners. FI exposes all '
                      'three needs and the same public layout; the live channel is otherwise identical.'),
        action_factorization=('Two-way intent: neutral=0 or engage=1. If engage, a 16-way '
                              'proposal chooses site, destination and partner.'),
        pairing=('Within each seed, every condition shares initial parameters, world uniforms, '
                 'batch indices and message uniforms. Live/silent and strict/reciprocal therefore '
                 'differ only through their declared factor and resulting updates.'),
        evaluation=('Canonical two-alternative-partner worlds on train layouts at six checkpoints '
                    'and on unseen layouts at the final checkpoint.'),
        claim_boundary=('This measures task-level coordination and a protocol-level rule effect. '
                        'It is not evidence for lexical meaning, grammar, cultural transmission, '
                        'or human language origin.'),
        development_status=('Independent seed block 66801–66808; full factorial rule/information/'
                             'schedule/channel grid frozen before training.'),
    )


def sample_indices(spec, uniforms):
    import numpy as np
    u = np.asarray(uniforms, dtype=np.float64)
    require(u.ndim == 2 and u.shape[1] == 3 and np.isfinite(u).all()
            and ((u >= 0) & (u < 1)).all(), 'Expected B×3 world uniforms')
    n = np.floor(u[:, 0] * len(spec['needs'])).astype(np.int64)
    l = np.floor(u[:, 1] * len(spec['layouts'])).astype(np.int64)
    o = np.floor(u[:, 2] * len(spec['private_sites'])).astype(np.int64)
    return (n * len(spec['layouts']) + l) * len(spec['private_sites']) + o


if __name__ == '__main__':
    import json
    print(json.dumps(make_prepared(), ensure_ascii=False, indent=2))
