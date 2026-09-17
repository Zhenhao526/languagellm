"""Pure aggregation helpers for the directional probe."""
from __future__ import annotations

import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _weighted_mean(values):
    total = sum(weight for _, weight in values)
    return float(sum(value * weight for value, weight in values) / total) if total else 0.0


def summarize_sender(sender, sums, n):
    require(n > 0 and sender in (0, 1, 2), 'Invalid sender summary')
    result = dict(sender=int(sender), rows=int(n),
                  plan_transfer=float(sum(sums['plan_transfer']) / n),
                  partner_transfer=float(sum(sums['partner_transfer']) / n))
    for key in ('natural_physical', 'natural_q', 'natural_conditional_q', 'intervention_physical', 'intervention_q', 'intervention_conditional_q'):
        result[key] = _weighted_mean(sums[key])
    result['plan_transfer_pp'] = 100.0 * result['plan_transfer']
    result['partner_transfer_pp'] = 100.0 * result['partner_transfer']
    for key in ('natural_physical', 'natural_q', 'natural_conditional_q', 'intervention_physical', 'intervention_q', 'intervention_conditional_q'):
        require(math.isfinite(result[key]), 'Nonfinite sender summary')
    return result


def summarize(rows):
    require(len(rows) == 32, 'Expected 32 policy rows')
    by = {(row['seed'], row['condition']): row for row in rows}
    require(len(by) == 32, 'Duplicate policy rows')
    seeds = sorted({row['seed'] for row in rows})
    live = [row for row in rows if row['live']]
    silent = [row for row in rows if not row['live']]
    require(len(live) == len(silent) == len(seeds) == 16, 'Incomplete paired grid')
    seed_rows = []
    for seed in seeds:
        l = by[seed, 'partial_multi_strict_PL_live']; s = by[seed, 'partial_multi_strict_PL_silent']
        require(l['checkpoint_sha256'] != s['checkpoint_sha256'], 'Live and silent checkpoint unexpectedly identical')
        for sender in range(3):
            lp = l['by_sender'][sender]; sp = s['by_sender'][sender]
            seed_rows.append(dict(seed=seed, sender=sender,
                plan_transfer_live=lp['plan_transfer'], plan_transfer_silent=sp['plan_transfer'],
                plan_transfer_difference=lp['plan_transfer'] - sp['plan_transfer'],
                partner_transfer_live=lp['partner_transfer'], partner_transfer_silent=sp['partner_transfer'],
                partner_transfer_difference=lp['partner_transfer'] - sp['partner_transfer'],
                conditional_q_live=lp['intervention_conditional_q'], conditional_q_silent=sp['intervention_conditional_q']))
    def mean(values): return float(sum(values) / len(values)) if values else 0.0
    all_plan = [row['plan_transfer_difference'] for row in seed_rows]
    all_partner = [row['partner_transfer_difference'] for row in seed_rows]
    return dict(status='completed_json_only_summary', independent_seeds=len(seeds), rows=len(rows),
        plan_transfer_live_mean=mean([r['plan_transfer'] for row in live for r in row['by_sender']]),
        plan_transfer_silent_mean=mean([r['plan_transfer'] for row in silent for r in row['by_sender']]),
        plan_transfer_live_minus_silent_mean=mean(all_plan),
        partner_transfer_live_minus_silent_mean=mean(all_partner), seed_sender_rows=seed_rows,
        scope=dict(primary='Mean signed donor-only minus receiver-only legal-plan probability transfer, live minus silent, across three senders and sixteen paired seeds.',
                   intervention='Only the selected sender W1 outward packet is replaced; W2 and actions are recomputed.',
                   units='Probabilities; multiply transfer values by 100 for percentage points.',
                   posthoc='This probe is posthoc relative to the completed multi-legal training block and is exploratory, not a new confirmation training experiment.'))
