"""Behavioral metrics and the predeclared payoff×communication interaction."""
from itertools import product
import math
import numpy as np

from research_program.triadic_reciprocal_execution_study import environment

SEEDS = (63101, 63102, 63103, 63104)
PAYOFFS = ('partial', 'all_or_nothing')
RULES = ('strict', 'reciprocal')
LIVES = (False, True)
UPDATES = (0, 100, 500, 1500, 3000, 6000)
HORIZON = 6000
PAIRS = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ('AB', 'AC', 'BC')


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
    """Settlement metrics plus partner adaptivity, preserving all worlds."""
    base = environment.metrics(data, truth, actions, rule)
    actual = np.asarray(data['actual_pair_index'], dtype=np.int8)
    target = np.asarray(truth['correct_pair_index'], dtype=np.int8)
    physical = actual >= 0
    matched = physical & (actual == target)
    base.update(
        correct_executed_pair_rate=float(matched.mean()),
        physical_pair_rate=float(physical.mean()),
        actual_pair_entropy_bits=_entropy(actual),
        target_pair_entropy_bits=_entropy(target),
        actual_pair_counts={name: int(np.sum(actual == i)) for i, name in enumerate(PAIR_NAMES)},
        no_execution_count=int(np.sum(actual < 0)),
        metric_scope='correct_executed_pair_rate requires an actual mutual pair and the researcher-side unique target pair; it is a behavioral partner-adaptivity measure.',
    )
    return base


def interaction_curve(cells, key='correct_executed_pair_rate'):
    expected = {(p, live) for p in PAYOFFS for live in LIVES}
    require(set(cells) == expected, 'Complete payoff/channel cells required')
    curves = {k: np.asarray(v, dtype=np.float64) for k, v in cells.items()}
    require(all(v.shape == (len(UPDATES),) and np.isfinite(v).all() for v in curves.values()), 'Six finite checkpoints required')
    interaction = (curves['all_or_nothing', True] - curves['all_or_nothing', False]) - (curves['partial', True] - curves['partial', False])
    centered = interaction - interaction[0]
    widths = np.diff(np.asarray(UPDATES, dtype=np.float64))
    auc = float(np.sum(widths * (centered[:-1] + centered[1:]) * .5) / HORIZON)
    return dict(payoff_key=key,
                all_or_nothing_live_minus_silent=(curves['all_or_nothing', True] - curves['all_or_nothing', False]).tolist(),
                partial_live_minus_silent=(curves['partial', True] - curves['partial', False]).tolist(),
                interaction=interaction.tolist(), centered_interaction=centered.tolist(),
                baseline_interaction=float(interaction[0]), centered_AUC=auc,
                endpoint_interaction=float(interaction[-1]))


def primary(runs):
    require(isinstance(runs, list) and len(runs) == 32, 'Exactly four seeds × two payoffs × two rules × two channels required')
    by = {(r['seed'], r['payoff'], r['rule'], r['live']): r for r in runs}
    expected = {(s, p, rule, live) for s in SEEDS for p in PAYOFFS for rule in RULES for live in LIVES}
    require(set(by) == expected and len(by) == len(runs), 'Incomplete pilot grid')
    rows = []
    for seed in SEEDS:
        by_rule = {}
        for rule in RULES:
            cells = {}
            for payoff, live in product(PAYOFFS, LIVES):
                run = by[seed, payoff, rule, live]
                values = [row['evaluation'].get('behavioral', row['evaluation'].get('behavior'))
                          ['correct_executed_pair_rate'] for row in run['trajectory']]
                cells[payoff, live] = values
            by_rule[rule] = interaction_curve(cells)
        rows.append(dict(seed=seed, by_rule=by_rule,
                         rule_mean_centered_AUC=float(np.mean([by_rule[r]['centered_AUC'] for r in RULES]))))
    values = np.asarray([r['rule_mean_centered_AUC'] for r in rows], dtype=np.float64)
    return dict(name='mean_payoff_communication_interaction_centered_pair_rate_time_AUC',
                target='heldout_layout', independent_pilot_seeds=len(SEEDS),
                mean_centered_AUC=float(values.mean()), sample_sd=float(values.std(ddof=1)),
                by_seed=rows,
                definition='For each seed and rule, [(all_or_nothing live−silent)−(partial live−silent)] on correct executed-pair rate; subtract update0, trapezoid-integrate over six checkpoints /6000, then average strict and reciprocal.',
                unit='Probability difference averaged over training time; multiply by100 for percentage points.',
                inference='Developmental four-seed pilot; no confirmatory interval or automatic continuation.',
                scope='Pair labels are researcher-side unique-plan targets. This is an adaptive partner-selection statistic, not a language or compositionality score.')
