"""Metrics for multi-legal-plan live/silent trajectories."""
from __future__ import annotations

import math

import numpy as np

UPDATES = np.array([0, 100, 500, 1500, 3000, 6000], dtype=np.float64)
T15_975 = 2.1314495455597715


def require(ok, message):
    if not ok: raise ValueError(message)


def stats(values):
    values = np.asarray(values, dtype=np.float64); require(values.shape == (16,), 'Sixteen seed values required')
    mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd / math.sqrt(16); half = T15_975 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15, ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T15_975)


def centered_auc(values):
    values = np.asarray(values, dtype=np.float64); require(values.shape == (6,), 'Six checkpoints required')
    return float(np.trapezoid(values - values[0], UPDATES) / 6000.0)


def _values(by, seed, live, key, part='trajectory'):
    run = by[seed, live]
    if part == 'trajectory':
        rows = sorted(run['trajectory'], key=lambda row: row['update']); return np.asarray([row['target_trajectory'][key] for row in rows], dtype=np.float64)
    return np.asarray([run['final']['new_layouts'][key]], dtype=np.float64)


def summarize(runs):
    by = {(r['seed'], r['live']): r for r in runs}; require(len(by) == 32, 'Complete 32-run grid required')
    seeds = sorted({r['seed'] for r in runs}); seed_rows = []; primary = []; primary_end = []; secondary = {k: [] for k in ('target_pair_actor_legal_action_rate', 'physical_execution_rate', 'third_actor_wait_rate')}; secondary_end = {k: [] for k in secondary}
    for seed in seeds:
        live_rows = {}; endpoint_rows = {}
        for key in ('q_rate', 'target_pair_actor_legal_action_rate', 'physical_execution_rate', 'third_actor_wait_rate'):
            silent = _values(by, seed, False, key); live = _values(by, seed, True, key); interaction = live - silent
            live_rows[key] = dict(interaction=interaction.tolist(), centered_AUC=centered_auc(interaction), endpoint_interaction=float(interaction[-1]), measure=key)
            if key == 'q_rate': primary.append(centered_auc(interaction)); primary_end.append(float(interaction[-1]))
            else:
                secondary[key].append(centered_auc(interaction)); secondary_end[key].append(float(interaction[-1]))
        for live in (False, True):
            for key in ('q_rate', 'target_pair_actor_legal_action_rate', 'physical_execution_rate', 'third_actor_wait_rate'):
                endpoint_rows[live, key] = float(_values(by, seed, live, key, part='final')[0])
        seed_rows.append(dict(seed=seed, by_metric=live_rows, endpoint=endpoint_rows,
                              q_centered_AUC=primary[-1], q_endpoint_interaction=primary_end[-1]))
    p = np.asarray(primary); pe = np.asarray(primary_end)
    sec = {key: dict(mean_centered_AUC=float(np.mean(secondary[key])), statistics=stats(secondary[key]), endpoint_interaction=float(np.mean(secondary_end[key])), endpoint_statistics=stats(secondary_end[key])) for key in secondary}
    return dict(name='live_minus_silent_team_Q_centered_time_AUC_two_legal_plans',
                primary=dict(mean_centered_AUC=float(p.mean()), statistics=stats(p), endpoint_interaction=float(pe.mean()), endpoint_statistics=stats(pe), by_seed=seed_rows, scope='Two-legal-plan worlds; team Q accepts either full-success plan.', no_posthoc_selection=True),
                secondary=sec, independent_seeds=16,
                inference='Approximate Student-t intervals across sixteen paired initializations; world-level observations are not independent societies.',
                unit='Probability difference averaged over training time; multiply by 100 for percentage points.',
                direction_note='Q is a team coordination metric over either legal plan; actor legal-action rates are separate diagnostics.')
