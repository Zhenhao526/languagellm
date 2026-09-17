"""Pre-specified JSON-only summaries for the factorized directional probe."""
from __future__ import annotations

import math

import numpy as np

T15_975 = 2.1314495455597715


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (16,) and np.isfinite(x).all(), 'Expected sixteen finite seed values')
    mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / math.sqrt(16); half = T15_975 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15,
                ci95_lower=mean - half, ci95_upper=mean + half,
                t_critical=T15_975, positive_count=int((x > 0).sum()),
                zero_count=int((x == 0).sum()), negative_count=int((x < 0).sum()))


def _weighted_mean(values):
    total = sum(weight for _, weight in values)
    require(total > 0, 'Empty weighted probe values')
    return float(sum(value * weight for value, weight in values) / total)


def summarize_sender(sender, sums, n):
    require(sender in (0, 1, 2) and n > 0, 'Invalid sender summary')
    result = dict(sender=int(sender), rows=int(n),
                  plan_transfer=float(sum(sums['plan_transfer']) / n),
                  partner_transfer=float(sum(sums['partner_transfer']) / n))
    for key in ('natural_physical', 'natural_q', 'natural_conditional_q',
                'intervention_physical', 'intervention_q', 'intervention_conditional_q'):
        result[key] = _weighted_mean(sums[key])
    result['plan_transfer_pp'] = 100.0 * result['plan_transfer']
    result['partner_transfer_pp'] = 100.0 * result['partner_transfer']
    require(all(math.isfinite(result[key]) for key in result
                 if key.endswith('transfer') or key.endswith('_q') or key.endswith('physical')),
            'Nonfinite sender summary')
    return result


def _mean_sender(row, key):
    return float(np.mean([sender[key] for sender in row['by_sender']]))


def summarize(rows):
    require(len(rows) == 64, 'Expected 64 policy rows')
    by = {(row['seed'], row['schedule'], row['live']): row for row in rows}
    require(len(by) == 64, 'Duplicate policy rows')
    seeds = sorted({row['seed'] for row in rows}); require(len(seeds) == 16, 'Expected sixteen seeds')
    keys = ('plan_transfer', 'partner_transfer', 'natural_physical', 'natural_q', 'natural_conditional_q',
            'intervention_physical', 'intervention_q', 'intervention_conditional_q')
    primary = []; secondary = {key: [] for key in keys}; per_seed = []
    for seed in seeds:
        cell = {(schedule, live): {key: _mean_sender(by[seed, schedule, live], key)
                                   for key in keys + ('plan_transfer',)}
                for schedule in ('static', 'rematched') for live in (False, True)}
        interaction = ((cell['rematched', True]['partner_transfer'] - cell['rematched', False]['partner_transfer'])
                       - (cell['static', True]['partner_transfer'] - cell['static', False]['partner_transfer']))
        primary.append(interaction)
        row = dict(seed=seed, partner_transfer_by_condition={
            f'{schedule}_{"live" if live else "silent"}': cell[schedule, live]['partner_transfer']
            for schedule in ('static', 'rematched') for live in (False, True)},
                   partner_transfer_interaction=interaction,
                   plan_transfer_by_condition={
                       f'{schedule}_{"live" if live else "silent"}': cell[schedule, live]['plan_transfer']
                       for schedule in ('static', 'rematched') for live in (False, True)})
        for key in keys:
            value = ((cell['rematched', True][key] - cell['rematched', False][key])
                     - (cell['static', True][key] - cell['static', False][key]))
            secondary[key].append(value); row[f'{key}_interaction'] = value
        per_seed.append(row)
    primary = np.asarray(primary, dtype=np.float64)
    return dict(
        name='rematching_by_communication_signed_directional_partner_edge_transfer_factorized',
        primary=dict(mean=float(primary.mean()), statistics=stats(primary), by_seed=per_seed,
                     scope='At final checkpoint, average three senders within seed, then compute rematched×communication difference in signed donor-only minus receiver-only partner-edge transfer.',
                     no_posthoc_selection=True),
        secondary={key: dict(mean=float(np.mean(values)), statistics=stats(values))
                   for key, values in secondary.items()},
        independent_seeds=16,
        inference='Approximate Student-t intervals across sixteen paired frozen-policy seeds; probe cases are repeated measures.',
        unit='Probability transfer; multiply by 100 for percentage points.', posthoc=True,
        direction_note='A positive primary means the live-minus-silent directional transfer of the selected sender toward donor-only partner edges is larger after rematched training than after static training.',
    )
