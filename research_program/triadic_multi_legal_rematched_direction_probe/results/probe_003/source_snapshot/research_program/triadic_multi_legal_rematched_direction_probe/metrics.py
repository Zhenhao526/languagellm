"""JSON-only aggregation for the four-arm directional probe."""
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
                ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T15_975,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()), negative_count=int((x < 0).sum()))


def _mean_sender(row, key):
    return float(np.mean([sender[key] for sender in row['by_sender']]))


def summarize(rows):
    require(len(rows) == 64, 'Expected 64 policy rows')
    by = {(row['seed'], row['schedule'], row['live']): row for row in rows}
    require(len(by) == 64, 'Duplicate policy rows')
    seeds = sorted({row['seed'] for row in rows}); require(len(seeds) == 16, 'Expected sixteen seeds')
    per_seed = []; primary = []
    secondary_keys = ('partner_transfer', 'natural_physical', 'natural_q', 'natural_conditional_q',
                      'intervention_physical', 'intervention_q', 'intervention_conditional_q')
    secondary = {key: [] for key in secondary_keys}
    for seed in seeds:
        cell = {(schedule, live): {key: _mean_sender(by[seed, schedule, live], key) for key in secondary_keys + ('plan_transfer',)}
                for schedule in ('static', 'rematched') for live in (False, True)}
        interaction = (cell['rematched', True]['plan_transfer'] - cell['rematched', False]['plan_transfer']) - \
                      (cell['static', True]['plan_transfer'] - cell['static', False]['plan_transfer'])
        primary.append(interaction)
        for key in secondary_keys:
            secondary[key].append((cell['rematched', True][key] - cell['rematched', False][key]) -
                                  (cell['static', True][key] - cell['static', False][key]))
        per_seed.append(dict(
            seed=seed,
            plan_transfer_by_condition={f'{s}_{"live" if live else "silent"}': cell[s, live]['plan_transfer'] for s in ('static', 'rematched') for live in (False, True)},
            plan_transfer_interaction=interaction,
            partner_transfer_interaction=secondary['partner_transfer'][-1],
            intervention_conditional_q_interaction=secondary['intervention_conditional_q'][-1],
        ))

    def secondary_summary(key):
        values = np.asarray(secondary[key], dtype=np.float64)
        return dict(mean=float(values.mean()), statistics=stats(values))

    primary = np.asarray(primary, dtype=np.float64)
    return dict(
        name='rematching_by_communication_signed_directional_plan_transfer',
        primary=dict(mean=float(primary.mean()), statistics=stats(primary), by_seed=per_seed,
                     scope='At final checkpoint, average three senders within seed, then compute rematched×communication difference in signed donor-only minus receiver-only plan transfer.',
                     no_posthoc_selection=True),
        secondary={key: secondary_summary(key) for key in secondary_keys},
        independent_seeds=16,
        inference='Approximate Student-t intervals across sixteen paired training initializations; probe cases are repeated measures.',
        unit='Probability transfer; multiply by 100 for percentage points.',
        posthoc=True,
        direction_note='A positive primary means the live-minus-silent directional transfer is larger after rematched training than after static training.',
    )
