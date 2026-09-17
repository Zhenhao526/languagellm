"""Frozen design for the payoff-by-semantic-generalization experiment.

The physical D1 task is unchanged.  Training excludes two object×attribute
resource predicates while retaining every individual object, attribute and
destination value in the marginal support.  The payoff intervention changes
only the learner's utility table; the native settlement and researcher-side
truth remain fixed.
"""
from itertools import permutations
from pathlib import Path
import hashlib
import json
import numpy as np

from research_program.triadic_action_dependency_study import dataset, environment
from research_program.triadic_factorial_formation_study import factor_cases
from research_program.triadic_need_response_study import cases

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(64101, 64117))
PAYOFFS = ('partial', 'all_or_nothing')
RULES = ('strict', 'reciprocal')
LIVES = (False, True)
PARTS = ('train', 'heldout_resource', 'heldout_layout', 'heldout_both')
TARGET = 'heldout_resource'
UPDATES = (0, 100, 500, 1500, 3000, 6000)
HELDOUT_RESOURCES = (5, 6)
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
    return source['partitions']['train']['layouts'], source['partitions']['new_layouts']['layouts'], source['partitions']['train']['private_sites']


def _spec(partition, needs, layouts, owners, role):
    needs = [list(map(int, row)) for row in sorted(tuple(map(int, row)) for row in needs)]
    layouts = [list(map(int, row)) for row in layouts]
    owners = [list(map(int, row)) for row in owners]
    return dict(partition=partition, role=role, needs=needs, layouts=layouts,
                private_sites=owners, world_count=len(needs) * len(layouts) * len(owners),
                state_order=STATE_ORDER, weighting='uniform needs × layouts × owners')


def _build_specs():
    train_layouts, heldout_layouts, owners = _source_splits()
    support = [tuple(map(int, row)) for row in environment.support()]
    factorial_needs = [row for row in support
                       if all(int(row[a]) // 3 not in HELDOUT_RESOURCES for a in range(3))]
    heldout_needs = [row for row in support
                     if any(int(row[a]) // 3 in HELDOUT_RESOURCES for a in range(3))]
    require(len(factorial_needs) == 2088 and len(heldout_needs) == 3288 and len(support) == 5376,
            'Unexpected factorial support counts')
    specs = {
        'train': _spec('train', factorial_needs, train_layouts, owners, 'factorial_train'),
        'heldout_resource': _spec('heldout_resource', heldout_needs, train_layouts, owners, 'unseen_resource'),
        'heldout_layout': _spec('heldout_layout', factorial_needs, heldout_layouts, owners, 'unseen_layout'),
        'heldout_both': _spec('heldout_both', heldout_needs, heldout_layouts, owners, 'unseen_resource_and_layout'),
    }
    return specs, dict(train_resources=[0, 1, 2, 3, 4, 7], heldout_resources=list(HELDOUT_RESOURCES),
                       train_need_count=len(factorial_needs), heldout_need_count=len(heldout_needs),
                       all_support_count=len(support), train_layout_count=len(train_layouts),
                       heldout_layout_count=len(heldout_layouts), owner_count=len(owners))


def payoff_rewards(native_rewards, payoff):
    native = np.asarray(native_rewards, dtype=np.float64)
    require(payoff in PAYOFFS and native.ndim == 2 and native.shape[1] == 24,
            'Invalid payoff or reward table')
    require(np.isin(native, (0., .5, 1.)).all() and np.all((native == 1).sum(axis=1) == 1),
            'Native reward table must have one full plan')
    return native.copy() if payoff == 'partial' else np.where(native == .5, 0., native)


def make_prepared():
    specs, split = _build_specs()
    response_cases = {part: cases.build_cases(specs[part]) for part in PARTS}
    groups = {part: factor_cases.heldout_edge_indices(response_cases[part], specs[part]['needs'], HELDOUT_RESOURCES)
              for part in PARTS}
    target = response_cases[TARGET]
    require(len(target['strata']) == 9 and not target['empty_strata'], 'Target needs all nine response strata')
    require(all(all(groups[TARGET][key]) for key in ('heldout_changed_actor', 'seen_changed_actor')),
            'Target heldout and seen edge groups must be populated')
    runs = [dict(seed=seed, payoff=payoff, rule=rule, live=live,
                 condition=f'{payoff}_{rule}_PL_{"live" if live else "silent"}',
                 directory=f'seed_{seed}_{payoff}_{rule}_PL_{"live" if live else "silent"}')
            for seed in SEEDS for payoff in PAYOFFS for rule in RULES for live in (True, False)]
    return dict(
        schema='triadic_factorial_payoff_v1', seeds=list(SEEDS), payoffs=list(PAYOFFS),
        rules=list(RULES), lives=['live', 'silent'], conditions=list(CONDITIONS),
        parts=list(PARTS), target=TARGET, checkpoints=list(UPDATES), heldout_resources=list(HELDOUT_RESOURCES),
        partitions=specs, split=split, need_response_cases=response_cases, factor_edge_groups=groups,
        runs=runs, independent_seed_count=len(SEEDS),
        training_updates=len(runs) * 6000, training_world_samples=len(runs) * 6000 * 256,
        target_trajectory_files=len(runs) * len(UPDATES),
        target_trajectory_worlds=len(runs) * len(UPDATES) * specs[TARGET]['world_count'],
        final_files=len(runs) * (len(PARTS) - 1),
        scientific_question='Does a payoff ecology that removes partial success make live communication support semantic response to unseen object×attribute combinations?',
        primary='centered time-AUC of [(all_or_nothing live−silent)−(partial live−silent)] on heldout changed-actor response Q',
        native_settlement='Unchanged strict/reciprocal physical settlement; payoff intervention affects only training objective.',
        observations='PL own need and public layout only; other private need blocks and full-information flag are absent.',
        semantic_holdout='Resource predicates 5=wood-long and 6=fiber-short are absent from factorial training; all individual values and the opposite singleton corners remain present.',
        claim_boundary='A heldout response interaction is evidence about behavioral generalization under this task; it does not by itself establish words, compositionality, intergenerational transmission, or human language origin.',
        development_status='Sixteen new seeds; confirmatory-scale within this task but still a model-based study, not a human-origin claim.',
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
