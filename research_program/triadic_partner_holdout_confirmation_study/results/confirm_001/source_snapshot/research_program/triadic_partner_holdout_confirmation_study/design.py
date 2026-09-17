"""Frozen design for social partner-pair holdout confirmation."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from research_program.triadic_action_dependency_study import dataset, environment
from research_program.triadic_factorial_payoff_study import design as payoff_design

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(66101, 66117))
PAYOFFS = payoff_design.PAYOFFS
RULES = payoff_design.RULES
LIVES = payoff_design.LIVES
PARTS = ('train', 'heldout_pair', 'heldout_pair_new_layout', 'seen_control_new_layout')
UPDATES = payoff_design.UPDATES
PAIR_NAMES = ('AB', 'AC', 'BC')
PAIR_TUPLES = ((0, 1), (0, 2), (1, 2))
CONDITIONS = payoff_design.CONDITIONS
STATE_ORDER = payoff_design.STATE_ORDER


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    return payoff_design.parse_condition(condition)


def payoff_rewards(native_rewards, payoff):
    return payoff_design.payoff_rewards(native_rewards, payoff)


def pair_index(need):
    plans = environment.full_success_plans(tuple(need))
    require(len(plans) == 1, 'Each support need must have one unique plan')
    pair = tuple(plans[0][:2])
    require(pair in PAIR_TUPLES, 'Unknown target pair')
    return PAIR_TUPLES.index(pair)


def _spec(partition, needs, layouts, owners, role, heldout_pair=None):
    needs = [list(map(int, row)) for row in sorted(tuple(map(int, row)) for row in needs)]
    layouts = [list(map(int, row)) for row in layouts]
    owners = [list(map(int, row)) for row in owners]
    return dict(partition=partition, role=role, needs=needs, layouts=layouts,
                private_sites=owners, world_count=len(needs) * len(layouts) * len(owners),
                heldout_pair=heldout_pair, state_order=STATE_ORDER,
                weighting='uniform needs × layouts × owners')


def _source_splits():
    source = dataset.make_prepared()
    return (source['partitions']['train']['layouts'],
            source['partitions']['new_layouts']['layouts'],
            source['partitions']['train']['private_sites'])


def heldout_pair_for_seed(seed):
    require(seed in SEEDS, 'Unknown seed')
    return (seed - SEEDS[0]) % 3


def build_seed_partitions(seed):
    train_layouts, heldout_layouts, owners = _source_splits()
    support = [tuple(map(int, row)) for row in environment.support()]
    require(len(support) == 5376 and len(set(support)) == 5376, 'Unexpected support')
    held = heldout_pair_for_seed(seed)
    held_needs = [row for row in support if pair_index(row) == held]
    train_needs = [row for row in support if pair_index(row) != held]
    counts = [sum(pair_index(row) == p for row in support) for p in range(3)]
    require(counts == [1792, 1792, 1792], 'Support pair counts are not balanced')
    require(len(held_needs) == 1792 and len(train_needs) == 3584, 'Pair split count mismatch')
    return {
        'train': _spec('train', train_needs, train_layouts, owners, 'seen_pairs_train'),
        'heldout_pair': _spec('heldout_pair', held_needs, train_layouts, owners, 'unseen_pair_same_layout', held),
        'heldout_pair_new_layout': _spec('heldout_pair_new_layout', held_needs, heldout_layouts, owners,
                                         'unseen_pair_new_layout', held),
        'seen_control_new_layout': _spec('seen_control_new_layout', train_needs, heldout_layouts, owners,
                                         'seen_pair_new_layout', held),
    }


def make_prepared():
    seed_partitions = {str(seed): build_seed_partitions(seed) for seed in SEEDS}
    runs = [dict(seed=seed, heldout_pair=heldout_pair_for_seed(seed),
                 heldout_pair_name=PAIR_NAMES[heldout_pair_for_seed(seed)], payoff=payoff,
                 rule=rule, live=live,
                 condition=f'{payoff}_{rule}_PL_{"live" if live else "silent"}',
                 directory=f'seed_{seed}_{payoff}_{rule}_PL_{"live" if live else "silent"}')
            for seed in SEEDS for payoff in PAYOFFS for rule in RULES for live in (True, False)]
    return dict(
        schema='triadic_partner_holdout_confirmation_v1', seeds=list(SEEDS), payoffs=list(PAYOFFS),
        rules=list(RULES), lives=['live', 'silent'], conditions=list(CONDITIONS), parts=list(PARTS),
        checkpoints=list(UPDATES), pair_names=list(PAIR_NAMES), heldout_pair_by_seed={str(s): heldout_pair_for_seed(s) for s in SEEDS},
        seed_partitions=seed_partitions, runs=runs, independent_seed_count=len(SEEDS),
        support_count=5376, train_need_count=3584, heldout_need_count=1792, train_layout_count=18,
        heldout_layout_count=6, owner_count=6, training_updates=len(runs) * 6000,
        training_world_samples=len(runs) * 6000 * 256,
        target_trajectory_files=len(runs) * len(UPDATES),
        target_trajectory_worlds=len(runs) * len(UPDATES) * 1792 * 18 * 6,
        scientific_question='Does communication recover an unseen target partner pair when all individual need values remain trained?',
        primary='centered time-AUC of payoff×communication interaction on heldout-pair target-actor exact action rate',
        secondary='heldout-pair Q, physical execution, third-actor wait, and seen-pair new-layout control',
        native_settlement='Unchanged strict/reciprocal physical settlement; payoff intervention affects only training objective.',
        observations='PL own need and public layout only; other private need blocks and full-information flag are absent.',
        split='One complete pair AB/AC/BC absent from training; heldout pair rotates across seeds; all personal need values remain in both splits.',
        claim_boundary='Partner-pair behavioral generalization is not evidence for words, compositionality, intergenerational transmission or human language origin.',
        development_status='New independent seed block; social-pair holdout is frozen before training.',
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
