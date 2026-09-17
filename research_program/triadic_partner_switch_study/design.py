"""Static design for the partner-switching payoff ecology.

The physical task and observations are inherited from the audited D1 learner.
The only training-ecology intervention is whether a half-success is useful:
``partial`` keeps native 0/.5/1 utility, while ``all_or_nothing`` maps .5 to
zero for the learning objective.  Native settlement remains unchanged.
"""
from copy import deepcopy
from itertools import permutations
from pathlib import Path
import hashlib
import json
import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment

HERE = Path(__file__).resolve().parent
SEEDS = (63101, 63102, 63103, 63104)  # developmental pilot; not independent confirmation
PAYOFFS = ('partial', 'all_or_nothing')
RULES = ('strict', 'reciprocal')
LIVES = (False, True)
PARTS = ('train', 'heldout_layout')
UPDATES = (0, 100, 500, 1500, 3000, 6000)
STATE_ORDER = 'need-major, then layout, then owner'
CONDITIONS = tuple(
    f'{payoff}_{rule}_PL_{"live" if live else "silent"}'
    for payoff in PAYOFFS for rule in RULES for live in (True, False))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    for payoff in PAYOFFS:
        prefix = payoff + '_'
        if condition.startswith(prefix):
            rest = condition[len(prefix):]
            rule, channel = rest.split('_PL_')
            require(rule in RULES and channel in ('live', 'silent'), 'Malformed condition')
            return payoff, rule, channel == 'live'
    raise ValueError('Malformed condition')


def _source_splits():
    source = dataset.make_prepared()
    return source['partitions']['train']['layouts'], source['partitions']['new_layouts']['layouts']


def _spec(partition, needs, layouts, owners):
    needs = [list(map(int, row)) for row in sorted(tuple(map(int, row)) for row in needs)]
    layouts = [list(map(int, row)) for row in layouts]
    owners = [list(map(int, row)) for row in owners]
    return dict(partition=partition, needs=needs, layouts=layouts, private_sites=owners,
                world_count=len(needs) * len(layouts) * len(owners),
                state_order=STATE_ORDER, weighting='uniform needs × layouts × owners')


def pair_index_for_need(need):
    plans = environment.full_success_plans(tuple(need))
    require(len(plans) == 1, 'Every selected need must have one full plan')
    pair = plans[0][:2]
    return ((0, 1), (0, 2), (1, 2)).index(tuple(pair))


def payoff_rewards(native_rewards, payoff):
    native = np.asarray(native_rewards, dtype=np.float64)
    require(payoff in PAYOFFS and native.ndim == 2 and native.shape[1] == 24,
            'Invalid payoff or reward table')
    require(np.isin(native, (0., .5, 1.)).all() and np.all((native == 1).sum(axis=1) == 1),
            'Native reward table must have one full plan')
    if payoff == 'partial':
        return native.copy()
    return np.where(native == .5, 0., native)


def make_prepared():
    train_layouts, heldout_layouts = _source_splits()
    owners = [list(p) for p in permutations((1, 2, 3))]
    needs = [list(map(int, row)) for row in environment.support()]
    require(len(needs) == 5376 and len({tuple(row) for row in needs}) == 5376,
            'Complete unique-plan support required')
    pair_counts = [0, 0, 0]
    for need in needs:
        pair_counts[pair_index_for_need(need)] += 1
    require(pair_counts == [1792, 1792, 1792], 'Support must balance the three compatible pairs')
    partitions = {
        'train': _spec('train', needs, train_layouts, owners),
        'heldout_layout': _spec('heldout_layout', needs, heldout_layouts, owners),
    }
    for spec in partitions.values():
        require(spec['world_count'] == 5376 * len(spec['layouts']) * 6, 'Cartesian support mismatch')
    runs = [dict(seed=seed, payoff=payoff, rule=rule, live=live,
                 condition=f'{payoff}_{rule}_PL_{"live" if live else "silent"}',
                 directory=f'seed_{seed}_{payoff}_{rule}_PL_{"live" if live else "silent"}')
            for seed in SEEDS for payoff in PAYOFFS for rule in RULES for live in (True, False)]
    return dict(
        schema='triadic_partner_switch_payoff_ecology_v1',
        seeds=list(SEEDS), payoffs=list(PAYOFFS), rules=list(RULES), lives=['live', 'silent'],
        conditions=list(CONDITIONS), parts=list(PARTS), checkpoints=list(UPDATES),
        partitions=partitions, pair_counts=pair_counts,
        needs_support_count=len(needs), layouts_train_count=len(train_layouts),
        layouts_heldout_count=len(heldout_layouts), private_owner_count=len(owners),
        runs=runs, independent_pilot_seed_count=len(SEEDS),
        training_updates=len(runs) * 6000, training_world_samples=len(runs) * 6000 * 256,
        target_trajectory_files=len(runs) * len(UPDATES),
        target_trajectory_worlds=len(runs) * len(UPDATES) * partitions['heldout_layout']['world_count'],
        final_files=len(runs) * len(PARTS),
        scientific_question='Does removing partial-success utility make communication select the state-appropriate partner and support content response?',
        primary='centered time-AUC of (all_or_nothing live−silent)−(partial live−silent) on correct executed-pair rate',
        native_settlement='Unchanged strict/reciprocal physical settlement; payoff intervention affects only training objective.',
        observations='PL own need and public layout; other private needs remain hidden; object, attribute, partner and destination demands retained.',
        claim_boundary='Partner selection and content response are behavioral task measures; neither alone establishes words, compositionality or human language origin.',
        development_status='Four-seed pilot; no confirmation claim.',
    )


def sample_indices(spec, uniforms):
    u = np.asarray(uniforms, dtype=np.float64)
    require(u.ndim == 2 and u.shape[1] == 3 and np.isfinite(u).all() and ((u >= 0) & (u < 1)).all(),
            'Expected B×3 world uniforms')
    n = np.floor(u[:, 0] * len(spec['needs'])).astype(np.int64)
    l = np.floor(u[:, 1] * len(spec['layouts'])).astype(np.int64)
    o = np.floor(u[:, 2] * len(spec['private_sites'])).astype(np.int64)
    return (n * len(spec['layouts']) + l) * len(spec['private_sites']) + o


def prepare_manifest():
    value = make_prepared()
    return dict(status='passed_static_only', prepared=value,
                source_sha256={str(Path(dataset.__file__).resolve()): sha(dataset.__file__),
                               str(Path(environment.__file__).resolve()): sha(environment.__file__),
                               str(HERE / 'design.py'): sha(HERE / 'design.py')},
                no_model_calls=True, no_training_updates=True)


if __name__ == '__main__':
    print(json.dumps(prepare_manifest(), ensure_ascii=False, indent=2))
