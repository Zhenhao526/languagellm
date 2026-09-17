"""Paired metrics for the neutral-outcome pilot."""
from __future__ import annotations

import math

import numpy as np

from . import design

UPDATES = np.asarray(design.UPDATES, dtype=np.float64)
T_CRITICAL = {4: 3.182446305284263, 16: 2.1314495455597715}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def stats(values):
    values = np.asarray(values, dtype=np.float64); n = len(values)
    require(n in T_CRITICAL and values.shape == (n,) and np.isfinite(values).all(), 'Unexpected finite seed vector')
    mean = float(values.mean()); sd = float(values.std(ddof=1)) if n > 1 else 0.0; se = sd / math.sqrt(n)
    half = T_CRITICAL[n] * se
    return dict(n=n, mean=mean, sample_sd=sd, standard_error=se, df=n - 1,
                ci95_lower=mean - half, ci95_upper=mean + half, t_critical=T_CRITICAL[n],
                positive_count=int((values > 0).sum()), zero_count=int((values == 0).sum()),
                negative_count=int((values < 0).sum()))


def centered_auc(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (len(UPDATES),) and np.isfinite(values).all(), 'Six finite checkpoints required')
    return float(np.trapezoid(values - values[0], UPDATES) / 6000.0)


def _values(by, seed, schedule, cancel_reward, live, key, final=False):
    run = by[seed, schedule, float(cancel_reward), live]
    if final:
        return float(run['final']['new_layouts'][key])
    rows = sorted(run['trajectory'], key=lambda row: row['update'])
    return np.asarray([row['target_trajectory'][key] for row in rows], dtype=np.float64)


def _summary(values, endpoints):
    values = np.asarray(values, dtype=np.float64); endpoints = np.asarray(endpoints, dtype=np.float64)
    return dict(mean_centered_AUC=float(values.mean()), statistics=stats(values), endpoint_interaction=float(endpoints.mean()), endpoint_statistics=stats(endpoints))


def summarize(runs):
    by = {(r['seed'], r['schedule'], float(r['cancel_reward']), bool(r['live'])): r for r in runs}
    expected = {(seed, schedule, float(cancel), live) for seed in design.SEEDS for schedule in design.SCHEDULES for cancel in design.CANCEL_REWARDS for live in design.LIVES}
    require(set(by) == expected and len(by) == len(expected), 'Complete neutral pilot grid required')
    seeds = sorted(design.SEEDS); require(len(seeds) in T_CRITICAL, 'Unsupported seed count')
    primary_auc = []; primary_end = []; seed_rows = []
    secondary_keys = ('q_rate', 'physical_execution_rate', 'cancel_protocol_rate',
                      'cancel_any_rate', 'mixed_cancel_rate', 'wait_action_rate', 'transport_action_rate',
                      'target_pair_actor_legal_action_rate')
    secondary = {key: {'cancel_by_communication': {str(cancel): [] for cancel in design.CANCEL_REWARDS},
                       'rematching_by_communication_by_cancel': {str(cancel): [] for cancel in design.CANCEL_REWARDS}}
                 for key in secondary_keys}
    endpoint_secondary = {
        key: {
            'cancel_by_communication': {str(cancel): [] for cancel in design.CANCEL_REWARDS},
            'rematching_by_communication_by_cancel': {str(cancel): [] for cancel in design.CANCEL_REWARDS},
        }
        for key in secondary_keys
    }
    for seed in seeds:
        cells = {(schedule, float(cancel), live): _values(by, seed, schedule, cancel, live, 'conditional_q_rate')
                 for schedule in design.SCHEDULES for cancel in design.CANCEL_REWARDS for live in design.LIVES}
        comm = {float(cancel): np.mean([cells[schedule, float(cancel), True] - cells[schedule, float(cancel), False] for schedule in design.SCHEDULES], axis=0)
                for cancel in design.CANCEL_REWARDS}
        primary = comm[float(design.CANCEL_REWARDS[-1])] - comm[float(design.CANCEL_REWARDS[0])]
        primary_auc.append(centered_auc(primary)); primary_end.append(float(primary[-1]))
        by_metric = {}
        for key in ('conditional_q_rate',) + secondary_keys:
            trajectory_cells = {(schedule, float(cancel), live): _values(by, seed, schedule, cancel, live, key)
                                for schedule in design.SCHEDULES for cancel in design.CANCEL_REWARDS for live in design.LIVES}
            cancel_effects = {}
            rematch_effects = {}
            for cancel in design.CANCEL_REWARDS:
                c = float(cancel)
                cancel_effects[str(c)] = np.mean([trajectory_cells[s, c, True] - trajectory_cells[s, c, False] for s in design.SCHEDULES], axis=0)
                rematch_effects[str(c)] = ((trajectory_cells['rematched', c, True] - trajectory_cells['rematched', c, False]) -
                                             (trajectory_cells['static', c, True] - trajectory_cells['static', c, False]))
                if key != 'conditional_q_rate':
                    secondary[key]['cancel_by_communication'][str(c)].append(centered_auc(cancel_effects[str(c)]))
                    secondary[key]['rematching_by_communication_by_cancel'][str(c)].append(centered_auc(rematch_effects[str(c)]))
                    endpoint_secondary[key]['cancel_by_communication'][str(c)].append(float(cancel_effects[str(c)][-1]))
                    endpoint_secondary[key]['rematching_by_communication_by_cancel'][str(c)].append(float(rematch_effects[str(c)][-1]))
            by_metric[key] = dict(
                cancel_by_communication={c: v.tolist() for c, v in cancel_effects.items()},
                rematching_by_communication_by_cancel={c: v.tolist() for c, v in rematch_effects.items()},
                cancel_reward_interaction=(cancel_effects[str(float(design.CANCEL_REWARDS[-1]))] - cancel_effects[str(float(design.CANCEL_REWARDS[0]))]).tolist(),
                cancel_reward_interaction_AUC=centered_auc(cancel_effects[str(float(design.CANCEL_REWARDS[-1]))] - cancel_effects[str(float(design.CANCEL_REWARDS[0]))]),
                cancel_reward_interaction_endpoint=float((cancel_effects[str(float(design.CANCEL_REWARDS[-1]))] - cancel_effects[str(float(design.CANCEL_REWARDS[0]))])[-1]),
            )
        seed_rows.append(dict(seed=seed, primary_centered_AUC=primary_auc[-1], primary_endpoint_interaction=primary_end[-1], by_metric=by_metric,
                              rematch_histogram=by[seed, 'rematched', 0.0, False]['rematch_histogram']))
    primary = np.asarray(primary_auc); endpoint = np.asarray(primary_end)
    secondary_summary = {}
    for key in secondary_keys:
        secondary_summary[key] = {'cancel_by_communication': {c: _summary(secondary[key]['cancel_by_communication'][c], endpoint_secondary[key]['cancel_by_communication'][c])
                                                               for c in secondary[key]['cancel_by_communication']},
                                  'rematching_by_communication_by_cancel': {c: _summary(secondary[key]['rematching_by_communication_by_cancel'][c], endpoint_secondary[key]['rematching_by_communication_by_cancel'][c])
                                                                              for c in secondary[key]['rematching_by_communication_by_cancel']}}
    return dict(
        name='cancel_reward_by_communication_interaction_on_conditional_Q_centered_time_AUC',
        primary=dict(mean_centered_AUC=float(primary.mean()), statistics=stats(primary), endpoint_interaction=float(endpoint.mean()),
                     endpoint_statistics=stats(endpoint), by_seed=seed_rows,
                     scope='Within each seed, average live-minus-silent conditional-Q effects across static/rematched schedules, then subtract cancel-reward 0.00 from 0.10.',
                     no_posthoc_selection=True),
        secondary=secondary_summary, independent_seeds=len(seeds), cancel_rewards=list(design.CANCEL_REWARDS),
        inference=f'Approximate Student-t intervals across {len(seeds)} paired pilot initializations; worlds are repeated measures, not independent societies.',
        unit='Probability difference averaged over training time; multiply by 100 for percentage points.',
        denominator='Conditional Q is legal transport-plan execution divided by physical pair execution; all-cancel worlds have no physical denominator.',
        direction_note='A positive primary means adding a rewarded all-cancel outcome increases the live-minus-silent conditional-Q effect.')
