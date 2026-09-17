"""Pre-registered paired metrics for the rematching confirmation."""
from __future__ import annotations

import math

import numpy as np

UPDATES = np.array([0, 100, 500, 1500, 3000, 6000], dtype=np.float64)
T15_975 = 2.1314495455597715


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (16,), 'Sixteen seed values required')
    mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd / math.sqrt(16); half = T15_975 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15,
                ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T15_975)


def centered_auc(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (6,), 'Six checkpoints required')
    return float(np.trapezoid(values - values[0], UPDATES) / 6000.0)


def endpoint_interaction(cells):
    return float((cells['rematched', True] - cells['rematched', False]) -
                 (cells['static', True] - cells['static', False]))


def _values(by, seed, schedule, live, key):
    run = by[seed, schedule, live]
    rows = sorted(run['trajectory'], key=lambda row: row['update'])
    return np.asarray([row['target_trajectory'][key] for row in rows], dtype=np.float64)


def _summary(values, endpoint):
    return dict(mean_centered_AUC=float(values.mean()), statistics=stats(values),
                endpoint_interaction=float(endpoint.mean()), endpoint_statistics=stats(endpoint))


def summarize(runs):
    by = {(r['seed'], r['schedule'], r['live']): r for r in runs}
    require(len(by) == 64, 'Complete 64-run grid required')
    seed_rows = []
    primary_values = []; primary_endpoints = []
    secondary = {key: [] for key in ('q_rate', 'physical_execution_rate', 'third_actor_wait_rate')}
    secondary_end = {key: [] for key in secondary}
    schedule_live = []; schedule_silent = []; schedule_live_end = []; schedule_silent_end = []
    for seed in sorted({r['seed'] for r in runs}):
        rule_rows = {}
        for key in ('target_pair_actor_action_rate', 'q_rate', 'physical_execution_rate', 'third_actor_wait_rate'):
            trajectories = {(schedule, live): _values(by, seed, schedule, live, key)
                            for schedule in ('static', 'rematched') for live in (False, True)}
            interaction = ((trajectories['rematched', True] - trajectories['rematched', False]) -
                           (trajectories['static', True] - trajectories['static', False]))
            rematch_live = trajectories['rematched', True] - trajectories['static', True]
            rematch_silent = trajectories['rematched', False] - trajectories['static', False]
            rule_rows[key] = dict(interaction=interaction.tolist(), centered_AUC=centered_auc(interaction),
                                  endpoint_interaction=float(interaction[-1]),
                                  rematched_minus_static_live=rematch_live.tolist(),
                                  rematched_minus_static_silent=rematch_silent.tolist(),
                                  live_centered_AUC=centered_auc(rematch_live), silent_centered_AUC=centered_auc(rematch_silent),
                                  measure=key)
            if key == 'target_pair_actor_action_rate':
                primary_values.append(centered_auc(interaction)); primary_endpoints.append(float(interaction[-1]))
                schedule_live.append(centered_auc(rematch_live)); schedule_silent.append(centered_auc(rematch_silent))
                schedule_live_end.append(float(rematch_live[-1])); schedule_silent_end.append(float(rematch_silent[-1]))
            else:
                secondary[key].append(centered_auc(interaction)); secondary_end[key].append(float(interaction[-1]))
        seed_rows.append(dict(seed=seed, heldout_pair=by[seed, 'static', False]['heldout_pair'],
                              heldout_pair_name=by[seed, 'static', False]['heldout_pair_name'],
                              rematched_histogram=by[seed, 'rematched', False]['rematch_histogram'], by_metric=rule_rows,
                              primary_centered_AUC=primary_values[-1], primary_endpoint_interaction=primary_endpoints[-1],
                              rematched_minus_static_live_AUC=schedule_live[-1],
                              rematched_minus_static_silent_AUC=schedule_silent[-1]))
    # Keep the seed arrays aligned with seed_rows for readable summaries.
    primary = np.asarray(primary_values, dtype=np.float64); primary_end = np.asarray(primary_endpoints, dtype=np.float64)
    result_secondary = {key: _summary(np.asarray(secondary[key]), np.asarray(secondary_end[key])) for key in secondary}
    result_secondary['rematched_minus_static_live'] = _summary(np.asarray(schedule_live), np.asarray(schedule_live_end))
    result_secondary['rematched_minus_static_silent'] = _summary(np.asarray(schedule_silent), np.asarray(schedule_silent_end))
    return dict(
        name='rematched_by_communication_interaction_on_heldout_pair_target_actor_exact_action_centered_time_AUC',
        primary=dict(mean_centered_AUC=float(primary.mean()), statistics=stats(primary),
                     endpoint_interaction=float(primary_end.mean()), endpoint_statistics=stats(primary_end),
                     by_seed=seed_rows,
                     scope='Heldout target-pair actor exact action rate; rematched×communication interaction, 16 paired seeds.',
                     no_posthoc_selection=True),
        secondary=result_secondary, independent_seeds=16,
        inference='Approximate Student-t intervals across sixteen paired initializations; world-level observations are not independent societies.',
        unit='Probability difference averaged over training time; multiply by 100 for percentage points.',
        direction_note='Primary averages the two target-pair actors; Q is a secondary AND metric. Rematch histograms are diagnostics only.',
    )
