"""Metrics for partner-pair holdout trajectories."""
from __future__ import annotations

import math
import numpy as np

UPDATES = np.array([0, 100, 500, 1500, 3000, 6000], dtype=np.float64)
T15_975 = 2.1314495455597715


def require(ok, message):
    if not ok:
        raise ValueError(message)


def endpoint_interaction(cells):
    return float((cells['all_or_nothing', True] - cells['all_or_nothing', False]) -
                 (cells['partial', True] - cells['partial', False]))


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (16,), 'Sixteen seed values required')
    mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd / math.sqrt(16)
    half = T15_975 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15,
                ci95_lower=mean-half, ci95_upper=mean+half, t_critical=T15_975)


def centered_auc(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (6,), 'Six checkpoints required')
    return float(np.trapezoid(values - values[0], UPDATES) / 6000.0)


def _cells_for_seed(by, seed, rule, key):
    values = {}
    for payoff in ('partial', 'all_or_nothing'):
        for live in (False, True):
            rows = [r['target_trajectory'][key] for r in by[seed, payoff, rule, live]['trajectory']]
            rows = sorted(rows, key=lambda row: row['update'])
            values[payoff, live] = np.array([row['value'] for row in rows], dtype=np.float64)
    return values


def summarize(runs):
    by = {(r['seed'], r['payoff'], r['rule'], r['live']): r for r in runs}
    require(len(by) == 128, 'Complete run grid required')
    seed_rows = []
    for seed in sorted({r['seed'] for r in runs}):
        rule_rows = {}
        for rule in ('strict', 'reciprocal'):
            metric_rows = {}
            for key in ('target_pair_actor_action_rate', 'q_rate', 'physical_execution_rate', 'third_actor_wait_rate'):
                values = _cells_for_seed(by, seed, rule, key)
                interaction = np.array([endpoint_interaction({k: v[i] for k, v in values.items()}) for i in range(6)])
                metric_rows[key] = dict(interaction=interaction.tolist(), centered_AUC=centered_auc(interaction),
                                         endpoint_interaction=float(interaction[-1]), measure=key)
            rule_rows[rule] = metric_rows
        for key in ('target_pair_actor_action_rate', 'q_rate', 'physical_execution_rate', 'third_actor_wait_rate'):
            vals = [rule_rows[rule][key]['centered_AUC'] for rule in ('strict', 'reciprocal')]
            end = [rule_rows[rule][key]['endpoint_interaction'] for rule in ('strict', 'reciprocal')]
            if key == 'target_pair_actor_action_rate':
                seed_primary_auc = float(np.mean(vals)); seed_primary_endpoint = float(np.mean(end))
        seed_rows.append(dict(seed=seed, heldout_pair=by[seed, 'partial', 'strict', False]['heldout_pair'],
                              heldout_pair_name=by[seed, 'partial', 'strict', False]['heldout_pair_name'],
                              by_rule=rule_rows, primary_centered_AUC=seed_primary_auc,
                              primary_endpoint_interaction=seed_primary_endpoint))
    primary = np.array([row['primary_centered_AUC'] for row in seed_rows])
    endpoint = np.array([row['primary_endpoint_interaction'] for row in seed_rows])
    secondary = {}
    for key in ('q_rate', 'physical_execution_rate', 'third_actor_wait_rate'):
        arr = np.array([np.mean([row['by_rule'][rule][key]['centered_AUC'] for rule in ('strict', 'reciprocal')]) for row in seed_rows])
        end = np.array([np.mean([row['by_rule'][rule][key]['endpoint_interaction'] for rule in ('strict', 'reciprocal')]) for row in seed_rows])
        secondary[key] = dict(mean_centered_AUC=float(arr.mean()), statistics=stats(arr), endpoint_interaction=float(end.mean()), endpoint_statistics=stats(end))
    return dict(
        name='partner_holdout_target_actor_action_payoff_communication_interaction_centered_time_AUC',
        primary=dict(mean_centered_AUC=float(primary.mean()), statistics=stats(primary),
                     endpoint_interaction=float(endpoint.mean()), endpoint_statistics=stats(endpoint), by_seed=seed_rows,
                     scope='Heldout target-pair actors exact action rate; equal strict/reciprocal rules and 16 paired seeds.',
                     no_posthoc_selection=True),
        secondary=secondary, independent_seeds=16,
        inference='Approximate Student-t intervals across sixteen paired initializations; world-level observations are not independent societies.',
        unit='Probability difference averaged over training time; multiply by100 for percentage points.',
        direction_note='The primary averages the two target-pair actors; actor-specific rates are stored per trajectory for follow-up, while Q is not directional.',
    )
