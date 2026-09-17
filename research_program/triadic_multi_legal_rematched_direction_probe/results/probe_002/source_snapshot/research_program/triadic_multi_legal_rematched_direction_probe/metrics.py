"""Paired summaries for the rematched directional message probe."""
from __future__ import annotations

import math

import numpy as np

T15_975 = 2.1314495455597715


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (16,), 'Sixteen seed values required')
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(16)
    half = T15_975 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15,
                ci95_lower=mean - half, ci95_upper=mean + half,
                t_critical=T15_975)


def _mean(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and np.isfinite(values).all(), 'Nonfinite probe values')
    return float(values.mean())


def _weighted_mean(values):
    total = sum(weight for _, weight in values)
    require(total > 0, 'Empty weighted probe values')
    return float(sum(value * weight for value, weight in values) / total)


def summarize_sender(sender, sums, n):
    """Collapse one sender's cases while retaining weighted physical rates."""
    require(sender in (0, 1, 2) and n > 0, 'Invalid sender summary')
    result = dict(sender=int(sender), rows=int(n),
                  plan_transfer=float(sum(sums['plan_transfer']) / n),
                  partner_transfer=float(sum(sums['partner_transfer']) / n))
    for key in ('natural_physical', 'natural_q', 'natural_conditional_q',
                'intervention_physical', 'intervention_q', 'intervention_conditional_q'):
        result[key] = _weighted_mean(sums[key])
    result['plan_transfer_pp'] = 100.0 * result['plan_transfer']
    result['partner_transfer_pp'] = 100.0 * result['partner_transfer']
    require(all(math.isfinite(result[key]) for key in result if key.endswith('transfer') or key.endswith('_q') or key.endswith('physical')),
            'Nonfinite sender summary')
    return result


def _grid(rows):
    by = {}
    for row in rows:
        seed = int(row['seed'])
        schedule = row['condition'].split('_', 1)[0]
        live = bool(row['live'])
        by[seed, schedule, live] = _mean([float(item['plan_transfer']) for item in row['by_sender']])
    return by


def _metric_grid(rows, key):
    by = {}
    for row in rows:
        seed = int(row['seed'])
        schedule = row['condition'].split('_', 1)[0]
        live = bool(row['live'])
        by[seed, schedule, live] = _mean([float(item[key]) for item in row['by_sender']])
    return by


def _paired_summary(values, name):
    arr = np.asarray(values, dtype=np.float64)
    require(arr.shape == (16,), 'Expected one value per seed')
    return dict(name=name, mean=float(arr.mean()), statistics=stats(arr), values_by_seed=arr.tolist())


def summarize(rows):
    require(len(rows) == 64, 'Expected 64 policy rows')
    by_row = {(int(row['seed']), row['condition']): row for row in rows}
    require(len(by_row) == 64, 'Duplicate policy rows')
    seeds = sorted({int(row['seed']) for row in rows})
    conditions = sorted({row['condition'] for row in rows})
    require(len(seeds) == 16 and len(conditions) == 4, 'Expected 16 seeds and four conditions')
    require(set((seed, condition) for seed in seeds for condition in conditions) == set(by_row), 'Missing condition rows')

    measure_names = ('plan_transfer', 'partner_transfer', 'natural_physical', 'natural_q',
                     'natural_conditional_q', 'intervention_physical', 'intervention_q',
                     'intervention_conditional_q')
    measures = {}
    for key in measure_names:
        grid = _metric_grid(rows, key)
        static = np.asarray([grid[seed, 'static', True] - grid[seed, 'static', False] for seed in seeds])
        rematched = np.asarray([grid[seed, 'rematched', True] - grid[seed, 'rematched', False] for seed in seeds])
        interaction = rematched - static
        measures[key] = dict(
            static_live_minus_silent=_paired_summary(static, f'static live−silent {key}'),
            rematched_live_minus_silent=_paired_summary(rematched, f'rematched live−silent {key}'),
            rematched_minus_static_interaction=_paired_summary(interaction, f'rematched×communication interaction {key}'),
        )

    # Per-seed rows retain the repeated-measures structure for auditing. The
    # inferential unit is the paired training seed, not an individual world.
    seed_rows = []
    grids = {key: _metric_grid(rows, key) for key in measure_names}
    for seed in seeds:
        row = {'seed': seed}
        for schedule in ('static', 'rematched'):
            for key in measure_names:
                row[f'{schedule}_live_{key}'] = grids[key][seed, schedule, True]
                row[f'{schedule}_silent_{key}'] = grids[key][seed, schedule, False]
                row[f'{schedule}_live_minus_silent_{key}'] = grids[key][seed, schedule, True] - grids[key][seed, schedule, False]
        row['rematched_minus_static_plan_transfer_interaction'] = (
            row['rematched_live_minus_silent_plan_transfer'] - row['static_live_minus_silent_plan_transfer'])
        row['rematched_minus_static_partner_interaction'] = (
            row['rematched_live_minus_silent_partner_transfer'] - row['static_live_minus_silent_partner_transfer'])
        seed_rows.append(row)

    return dict(
        name='directional_W1_plan_transfer_with_rematching_interaction',
        primary=measures['plan_transfer']['rematched_minus_static_interaction'],
        measures=measures,
        by_seed=seed_rows,
        independent_seeds=16,
        scope='Same-layout same-owner one-sender W1 donor intervention on 1404 two-plan needs and six new layouts; sender-level values are averaged within seed.',
        inference='Approximate Student-t intervals across sixteen paired training seeds; sender and world cases are repeated measures.',
        unit='Probability transfer; multiply by 100 for percentage points.',
        interpretation='Positive plan_transfer means the receiver action distribution moves toward the donor-only legal plan. The primary interaction is rematched live−silent minus static live−silent.',
        no_posthoc_selection=True,
    )
