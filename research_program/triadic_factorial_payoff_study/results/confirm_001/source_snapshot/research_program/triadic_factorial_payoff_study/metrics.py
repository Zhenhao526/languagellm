"""Predeclared semantic-response and partner-adaptivity estimands."""
from itertools import product
import math
import numpy as np

from research_program.triadic_reciprocal_execution_study import environment

SEEDS = tuple(range(64101, 64117))
PAYOFFS = ('partial', 'all_or_nothing')
RULES = ('strict', 'reciprocal')
LIVES = (False, True)
UPDATES = (0, 100, 500, 1500, 3000, 6000)
HORIZON = 6000
PAIRS = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ('AB', 'AC', 'BC')
T15_975 = 2.1314495455597715


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _entropy(values):
    values = np.asarray(values, dtype=np.int64)
    active = values[values >= 0]
    if len(active) == 0:
        return 0.0
    _, counts = np.unique(active, return_counts=True)
    p = counts.astype(float) / len(active)
    return float(-np.sum(p * np.log2(p)))


def behavioral(data, truth, actions, rule):
    base = environment.metrics(data, truth, actions, rule)
    actual = np.asarray(data['actual_pair_index'], dtype=np.int8)
    target = np.asarray(truth['correct_pair_index'], dtype=np.int8)
    physical = actual >= 0
    matched = physical & (actual == target)
    base.update(correct_executed_pair_rate=float(matched.mean()), physical_pair_rate=float(physical.mean()),
                actual_pair_entropy_bits=_entropy(actual), target_pair_entropy_bits=_entropy(target),
                actual_pair_counts={name: int(np.sum(actual == i)) for i, name in enumerate(PAIR_NAMES)},
                no_execution_count=int(np.sum(actual < 0)),
                metric_scope='correct_executed_pair_rate requires an actual mutual pair and the researcher-side unique target pair.')
    return base


def interaction_curve(cells, key):
    expected = {(p, live) for p in PAYOFFS for live in LIVES}
    require(set(cells) == expected, 'Complete payoff/channel cells required')
    curves = {k: np.asarray(v, dtype=np.float64) for k, v in cells.items()}
    require(all(v.shape == (len(UPDATES),) and np.isfinite(v).all() for v in curves.values()),
            'Six finite checkpoints required')
    interaction = (curves['all_or_nothing', True] - curves['all_or_nothing', False]) - \
                  (curves['partial', True] - curves['partial', False])
    centered = interaction - interaction[0]
    widths = np.diff(np.asarray(UPDATES, dtype=np.float64))
    auc = float(np.sum(widths * (centered[:-1] + centered[1:]) * .5) / HORIZON)
    return dict(measure=key,
                all_or_nothing_live_minus_silent=(curves['all_or_nothing', True] - curves['all_or_nothing', False]).tolist(),
                partial_live_minus_silent=(curves['partial', True] - curves['partial', False]).tolist(),
                interaction=interaction.tolist(), centered_interaction=centered.tolist(),
                baseline_interaction=float(interaction[0]), centered_AUC=auc,
                endpoint_interaction=float(interaction[-1]))


def society_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (len(SEEDS),) and np.isfinite(values).all(), 'Sixteen finite seed values required')
    mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd / math.sqrt(len(values))
    half = T15_975 * se
    return dict(n=len(values), mean=mean, sample_sd=sd, standard_error=se, df=15,
                t_critical=T15_975, interval_level=.95, ci95_lower=mean-half, ci95_upper=mean+half,
                interval_method='Approximate two-sided Student-t interval across sixteen paired initializations; not a world-level interval or equivalence proof.')


def _trajectory_values(run, key):
    result = []
    for row in run['trajectory']:
        factor = row['evaluation']['factor_response']['heldout_changed_actor']
        require(factor is not None and key in factor, 'Target factor response missing')
        result.append(float(factor[key]))
    return result


def primary(runs):
    require(isinstance(runs, list) and len(runs) == 128, 'Exactly 128 runs required')
    by = {(r['seed'], r['payoff'], r['rule'], r['live']): r for r in runs}
    expected = {(s, p, rule, live) for s in SEEDS for p in PAYOFFS for rule in RULES for live in LIVES}
    require(set(by) == expected and len(by) == len(runs), 'Incomplete 128-cell grid')
    rows = []
    for seed in SEEDS:
        by_rule = {}; partner_by_rule = {}
        for rule in RULES:
            qcells = {(payoff, live): _trajectory_values(by[seed, payoff, rule, live], 'Q')
                      for payoff, live in product(PAYOFFS, LIVES)}
            pcells = {(payoff, live): [float(row['evaluation']['behavioral']['correct_executed_pair_rate'])
                                       for row in by[seed, payoff, rule, live]['trajectory']]
                      for payoff, live in product(PAYOFFS, LIVES)}
            by_rule[rule] = interaction_curve(qcells, 'heldout_changed_actor_Q')
            partner_by_rule[rule] = interaction_curve(pcells, 'correct_executed_pair_rate')
        rows.append(dict(seed=seed, by_rule=by_rule, partner_by_rule=partner_by_rule,
                         rule_mean_centered_AUC=float(np.mean([by_rule[r]['centered_AUC'] for r in RULES])),
                         partner_rule_mean_centered_AUC=float(np.mean([partner_by_rule[r]['centered_AUC'] for r in RULES]))))
    values = np.asarray([row['rule_mean_centered_AUC'] for row in rows])
    partner_values = np.asarray([row['partner_rule_mean_centered_AUC'] for row in rows])
    return dict(name='mean_payoff_communication_interaction_centered_heldout_Q_time_AUC',
                target='heldout_resource', edge_group='heldout_changed_actor', independent_seeds=len(SEEDS),
                mean_centered_AUC=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                statistics=society_statistics(values), partner_secondary=dict(mean_centered_AUC=float(partner_values.mean()),
                                                                               sample_sd=float(partner_values.std(ddof=1)),
                                                                               statistics=society_statistics(partner_values)),
                by_seed=rows,
                definition='For each seed and rule, [(all_or_nothing live−silent)−(partial live−silent)] on heldout changed-actor Q; subtract update0, trapezoid-integrate over six checkpoints /6000, then average strict and reciprocal and the16 seeds.',
                unit='Probability difference averaged over training time; multiply by100 for percentage points.',
                inference='Sixteen new seeds with an approximate Student-t interval across paired initializations; world-level observations are not independent societies.',
                scope='Q is a researcher-side equal-stratum behavioral semantic-response measure. It is not a lexical or compositionality score.')
