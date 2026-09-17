"""Compact metrics and across-seed summaries for the cross-split study."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

T15_975 = 2.1314495455597715
UPDATES = np.array([0, 100, 500, 1500, 3000, 6000], dtype=np.float64)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def endpoint_interaction(cells):
    return float((cells['all_or_nothing', True] - cells['all_or_nothing', False]) -
                 (cells['partial', True] - cells['partial', False]))


def edge_metrics(source_actual, source_target, destination_actual, destination_target, edges, groups, background_count):
    source_actual = np.asarray(source_actual, dtype=np.int8)
    source_target = np.asarray(source_target, dtype=np.int8)
    destination_actual = np.asarray(destination_actual, dtype=np.int8)
    destination_target = np.asarray(destination_target, dtype=np.int8)
    active = {key: value for key, value in groups.items() if value}
    strata_q = []
    strata_partner = []
    for key in sorted(active, key=lambda text: tuple(map(int, text.split('_')))):
        q_edges = []
        p_edges = []
        for edge_index in active[key]:
            edge = edges[edge_index]
            source_slice = slice(edge['before_index'] * background_count,
                                 (edge['before_index'] + 1) * background_count)
            destination_slice = slice(edge['after_index'] * background_count,
                                      (edge['after_index'] + 1) * background_count)
            source_ok = source_actual[source_slice] == source_target[source_slice]
            destination_ok = destination_actual[destination_slice] == destination_target[destination_slice]
            q_edges.append(float(np.mean(source_ok & destination_ok)))
            p_edges.append(float(np.mean(destination_ok)))
        strata_q.append(float(np.mean(q_edges)))
        strata_partner.append(float(np.mean(p_edges)))
    return dict(Q=float(np.mean(strata_q)), partner_endpoint_rate=float(np.mean(strata_partner)),
                strata_Q=strata_q, strata_partner_endpoint_rate=strata_partner,
                edge_count=len(edges), stratum_count=len(active),
                weighting='Equal six changed-person×{kind,length} strata; within each stratum equal edges and backgrounds.')


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (16,), 'Sixteen seed values required')
    mean = float(values.mean())
    sd = float(values.std(ddof=1))
    se = sd / math.sqrt(len(values))
    half = T15_975 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15,
                ci95_lower=mean-half, ci95_upper=mean+half, t_critical=T15_975)


def centered_auc(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (6,), 'Six checkpoints required')
    centered = values - values[0]
    return float(np.trapezoid(centered, UPDATES) / 6000.0)


def summarize(runs):
    """Summarize compact per-run trajectories without reading checkpoints."""
    by = {(r['seed'], r['payoff'], r['rule'], r['live']): r for r in runs}
    require(len(by) == 128, 'Complete run grid required')
    seed_rows = []
    for seed in sorted({r['seed'] for r in runs}):
        by_direction = {}
        for direction in ('train_to_heldout', 'heldout_to_train'):
            q_rules = {}; p_rules = {}
            for rule in ('strict', 'reciprocal'):
                q_values = {}; p_values = {}
                for payoff in ('partial', 'all_or_nothing'):
                    for live in (False, True):
                        rows = [r['cross_trajectory'][direction] for r in by[seed, payoff, rule, live]['trajectory']]
                        rows = sorted(rows, key=lambda row: row['update'])
                        q_values[payoff, live] = np.array([row['Q'] for row in rows])
                        p_values[payoff, live] = np.array([row['partner_endpoint_rate'] for row in rows])
                q_interaction = np.array([endpoint_interaction({k: v[i] for k, v in q_values.items()}) for i in range(6)])
                p_interaction = np.array([endpoint_interaction({k: v[i] for k, v in p_values.items()}) for i in range(6)])
                q_rules[rule] = dict(interaction=q_interaction.tolist(), centered_AUC=centered_auc(q_interaction), endpoint_interaction=float(q_interaction[-1]), measure='cross_split_Q')
                p_rules[rule] = dict(interaction=p_interaction.tolist(), centered_AUC=centered_auc(p_interaction), endpoint_interaction=float(p_interaction[-1]), measure='partner_endpoint_rate')
            by_direction[direction] = dict(Q=q_rules, partner=p_rules)
        q_auc = float(np.mean([by_direction[d]['Q'][rule]['centered_AUC'] for d in by_direction for rule in ('strict', 'reciprocal')]))
        p_auc = float(np.mean([by_direction[d]['partner'][rule]['centered_AUC'] for d in by_direction for rule in ('strict', 'reciprocal')]))
        q_endpoint = float(np.mean([by_direction[d]['Q'][rule]['endpoint_interaction'] for d in by_direction for rule in ('strict', 'reciprocal')]))
        p_endpoint = float(np.mean([by_direction[d]['partner'][rule]['endpoint_interaction'] for d in by_direction for rule in ('strict', 'reciprocal')]))
        direction_q = {d: float(np.mean([by_direction[d]['Q'][rule]['centered_AUC'] for rule in ('strict', 'reciprocal')])) for d in by_direction}
        seed_rows.append(dict(seed=seed, directions=by_direction, q_centered_AUC=q_auc, partner_centered_AUC=p_auc,
                              q_endpoint_interaction=q_endpoint, partner_endpoint_interaction=p_endpoint,
                              direction_q_centered_AUC=direction_q))
    q_auc = np.array([r['q_centered_AUC'] for r in seed_rows])
    p_auc = np.array([r['partner_centered_AUC'] for r in seed_rows])
    q_endpoint = np.array([r['q_endpoint_interaction'] for r in seed_rows])
    p_endpoint = np.array([r['partner_endpoint_interaction'] for r in seed_rows])
    direction_contrast = np.array([r['direction_q_centered_AUC']['train_to_heldout'] - r['direction_q_centered_AUC']['heldout_to_train'] for r in seed_rows])
    return dict(
        name='bidirectional_cross_split_payoff_communication_interaction_centered_time_AUC',
        target='bidirectional train_to_heldout and heldout_to_train',
        primary=dict(mean_centered_AUC=float(q_auc.mean()), statistics=stats(q_auc), by_seed=seed_rows,
                     scope='Cross-split seen↔heldout object×attribute edge Q; equal six changed-person×{kind,length} strata, two directions and two rules.',
                     no_posthoc_selection=True),
        partner_secondary=dict(mean_centered_AUC=float(p_auc.mean()), statistics=stats(p_auc)),
        endpoint=dict(q_interaction=float(q_endpoint.mean()), q_statistics=stats(q_endpoint),
                      partner_interaction=float(p_endpoint.mean()), partner_statistics=stats(p_endpoint)),
        direction_contrast=dict(name='train_to_heldout_minus_heldout_to_train_Q_AUC', statistics=stats(direction_contrast)),
        independent_seeds=16,
        inference='Approximate Student-t intervals across sixteen paired initializations; world-level observations are not independent societies.',
        unit='Probability difference averaged over training time; multiply by100 for percentage points.',
    )
