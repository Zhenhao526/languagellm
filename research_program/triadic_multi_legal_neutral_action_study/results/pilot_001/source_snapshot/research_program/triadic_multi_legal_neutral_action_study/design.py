"""Frozen task design for the explicit neutral/cancel pilot.

The pilot keeps the two legal transport plans and adds a third, symmetric
all-cancel outcome.  ``wait`` remains the passive third-agent action; ``cancel``
is an explicit shared opt-out and is rewarded only when all three agents select
it.  This makes a safe neutral outcome observable without adding a teacher
label, a hidden plan, or a new object-level demand.
"""
from __future__ import annotations

from itertools import product

from research_program.triadic_action_dependency_study import dataset, environment
from research_program.triadic_multi_legal_rematched_confirmation_study import design as base_design

# Four paired seeds are an exploratory pilot.  A later confirmation must use a
# fresh 16-seed block after the pilot is frozen and read without seed selection.
SEEDS = tuple(range(66701, 66705))
SCHEDULES = ('static', 'rematched')
CANCEL_REWARDS = (0.0, 0.1)
LIVES = (False, True)
PARTS = ('train', 'new_layouts')
UPDATES = (0, 100, 500, 1500, 3000, 6000)
PAIR_NAMES = ('AB', 'AC', 'BC')
PAIR_TUPLES = ((0, 1), (0, 2), (1, 2))
CHECKPOINT = 6000
CHUNK_SIZE = 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def cancel_code(value):
    """Stable string for reward values in condition names."""
    require(float(value) in CANCEL_REWARDS, 'Unknown cancel reward')
    return f'{int(round(float(value) * 100)):02d}'


CONDITIONS = tuple(
    f'{schedule}_cancel{cancel_code(cancel)}_partial_multi_strict_PL_{"live" if live else "silent"}'
    for schedule in SCHEDULES for cancel in CANCEL_REWARDS for live in (True, False)
)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    schedule, cancel_name, suffix = condition.split('_', 2)
    require(schedule in SCHEDULES and cancel_name.startswith('cancel')
            and suffix in ('partial_multi_strict_PL_live', 'partial_multi_strict_PL_silent'),
            'Malformed condition')
    reward = int(cancel_name.removeprefix('cancel')) / 100.0
    require(reward in CANCEL_REWARDS, 'Invalid cancel reward code')
    return schedule, reward, 'partial_multi', 'strict', condition.endswith('_live')


def pair_index(need):
    plans = environment.full_success_plans(tuple(need))
    require(len(plans) == 2 and len({tuple(p[:2]) for p in plans}) == 1,
            'Need is not a two-plan same-pair need')
    pair = tuple(plans[0][:2])
    require(pair in PAIR_TUPLES, 'Unknown target pair')
    return PAIR_TUPLES.index(pair)


def multi_needs():
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
                state_order='need-major, then layout, then owner',
                weighting='uniform needs × layouts × owners')


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
    runs = [dict(seed=seed, schedule=schedule, cancel_reward=cancel, payoff='partial_multi', rule='strict', live=live,
                 condition=f'{schedule}_cancel{cancel_code(cancel)}_partial_multi_strict_PL_{"live" if live else "silent"}',
                 directory=f'seed_{seed}_{schedule}_cancel{cancel_code(cancel)}_partial_multi_strict_PL_{"live" if live else "silent"}')
            for seed in SEEDS for schedule in SCHEDULES for cancel in CANCEL_REWARDS for live in (True, False)]
    return dict(
        schema='triadic_multi_legal_neutral_action_v1', seeds=list(SEEDS), schedules=list(SCHEDULES),
        cancel_rewards=list(CANCEL_REWARDS), lives=['live', 'silent'], conditions=list(CONDITIONS),
        parts=list(PARTS), checkpoints=list(UPDATES), pair_names=list(PAIR_NAMES), partitions=partitions,
        runs=runs, independent_seed_count=len(SEEDS), need_count=len(needs), pair_counts=[484, 484, 484],
        train_layout_count=len(train_layouts), new_layout_count=len(new_layouts), owner_count=len(owners),
        legal_plan_multiplicity=2, explicit_neutral_action={'index': 17, 'name': 'cancel',
            'settlement': 'all three cancel gives the configured safe-neutral reward; mixed cancel blocks transport and gives zero',
            'wait_semantics': 'passive no-proposal action remains index 0 and is required for the strict third actor'},
        training_updates=len(runs) * 6000, training_world_samples=len(runs) * 6000 * 256,
        target_trajectory_files=len(runs) * len(UPDATES),
        target_trajectory_worlds=len(runs) * len(UPDATES) * partitions['train']['world_count'], final_files=len(runs),
        scientific_question='Does an explicit safe-neutral outcome change the communication effect on execution versus post-execution selection?',
        primary='cancel-reward-by-communication interaction on conditional Q, averaged across static and rematched schedules, centered time-AUC',
        secondary='schedule-by-communication conditional-Q interaction at each cancel reward, physical execution, cancel protocol and transport plan selection',
        rematching='For rematched training, each sampled world assigns its three private needs through an independent uniform six-way permutation; static uses identity assignment.',
        native_settlement='Strict transport matching is unchanged when no cancel is selected; all-cancel is a separate symmetric neutral outcome.',
        observations='PL own need and public layout only; other private need blocks and full-information flag are absent.',
        pairing='Same initial parameters, world uniforms, message uniforms and rematch permutations across paired reward/channel arms within each seed and schedule.',
        evaluation='Canonical two-legal-plan worlds on train layouts at six checkpoints and new layouts at final.',
        claim_boundary='This tests a task-level neutral alternative and conditional coordination. It is not evidence for words, compositionality, intergenerational transmission or human language origin.',
        development_status='Exploratory four-seed pilot; no seed or checkpoint selection for claims.')


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
