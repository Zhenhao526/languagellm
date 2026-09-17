"""Pre-registered paired summaries for the neutral/engage study."""
from __future__ import annotations

import math
import numpy as np

UPDATES = np.array([0, 100, 500, 1500, 3000, 6000], dtype=np.float64)
T15_975 = 2.1314495455597715


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(values):
    x = np.asarray(values, dtype=np.float64); require(x.shape == (16,) and np.isfinite(x).all(), 'Expected sixteen finite seed values')
    mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / math.sqrt(16); half = T15_975 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15,
                ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T15_975,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()), negative_count=int((x < 0).sum()))


def centered_auc(values):
    x = np.asarray(values, dtype=np.float64); require(x.shape == (6,), 'Six checkpoints required')
    return float(np.trapezoid(x - x[0], UPDATES) / 6000.0)


def _values(by, seed, schedule, live, key):
    rows = sorted(by[seed, schedule, live]['trajectory'], key=lambda row: row['update'])
    return np.asarray([row['target_trajectory'][key] for row in rows], dtype=np.float64)


def _summ(values, endpoints):
    x = np.asarray(values, dtype=np.float64); e = np.asarray(endpoints, dtype=np.float64)
    return dict(mean_centered_AUC=float(x.mean()), statistics=stats(x), endpoint_interaction=float(e.mean()), endpoint_statistics=stats(e))


def summarize(runs):
    by = {(r['seed'], r['schedule'], r['live']): r for r in runs}; require(len(by) == 64, 'Complete 64-run grid required')
    seeds = sorted({r['seed'] for r in runs}); require(len(seeds) == 16, 'Sixteen independent seeds required')
    secondary_keys = ('q_rate', 'conditional_q_rate', 'physical_execution_rate', 'engagement_rate',
                      'proposal_legal_rate', 'third_agent_neutral_rate')
    primary = []; primary_end = []; secondary = {k: [] for k in secondary_keys}; secondary_end = {k: [] for k in secondary_keys}
    seed_rows = []
    for seed in seeds:
        cells = {}
        for schedule in ('static', 'rematched'):
            for live in (False, True):
                cells[schedule, live] = {key: _values(by, seed, schedule, live, key)
                                         for key in ('target_pair_legal_rate',) + secondary_keys}
        interactions = {}
        for key in ('target_pair_legal_rate',) + secondary_keys:
            interactions[key] = ((cells['rematched', True][key] - cells['rematched', False][key]) -
                                 (cells['static', True][key] - cells['static', False][key]))
        primary.append(centered_auc(interactions['target_pair_legal_rate']))
        primary_end.append(float(interactions['target_pair_legal_rate'][-1]))
        for key in secondary_keys:
            secondary[key].append(centered_auc(interactions[key])); secondary_end[key].append(float(interactions[key][-1]))
        seed_rows.append(dict(
            seed=seed, primary_centered_AUC=primary[-1], primary_endpoint_interaction=primary_end[-1],
            rematched_live_minus_silent_AUC=centered_auc(cells['rematched', True]['target_pair_legal_rate'] - cells['rematched', False]['target_pair_legal_rate']),
            static_live_minus_silent_AUC=centered_auc(cells['static', True]['target_pair_legal_rate'] - cells['static', False]['target_pair_legal_rate']),
            endpoint_target_pair_legal_rate={f'{s}_{"live" if live else "silent"}': float(cells[s, live]['target_pair_legal_rate'][-1]) for s in ('static', 'rematched') for live in (False, True)},
            endpoint_engagement_rate={f'{s}_{"live" if live else "silent"}': float(cells[s, live]['engagement_rate'][-1]) for s in ('static', 'rematched') for live in (False, True)}))
    return dict(
        name='rematching_by_communication_interaction_on_target_pair_legal_rate_conditional_on_physical_execution',
        primary=dict(mean_centered_AUC=float(np.mean(primary)), statistics=stats(primary), endpoint_interaction=float(np.mean(primary_end)), endpoint_statistics=stats(primary_end), by_seed=seed_rows,
                     scope='Canonical worlds with exactly two full plans on distinct partner pairs; target pair rate is conditional on an actual strict physical pair; neutral/engage is a separate policy head.', no_posthoc_selection=True),
        secondary={key: _summ(secondary[key], secondary_end[key]) for key in secondary_keys}, independent_seeds=16,
        inference='Approximate Student-t intervals across sixteen paired initializations; world-level observations are repeated measures, not independent societies.',
        unit='Probability difference averaged over training time; multiply by 100 for percentage points.',
        denominator='Physical execution, engagement and proposal denominators are reported explicitly; zero denominators are encoded as 0.',
        direction_note='A positive primary means rematched live−silent target-pair selection is larger than static live−silent target-pair selection.',
    )


if __name__ == '__main__':
    print('neutral/engage metrics')
