"""JSON-only aggregation for the completed directional probe."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import metrics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def stats(values):
    x = np.asarray(values, dtype=np.float64); require(len(x) == 16 and np.isfinite(x).all(), 'Expected 16 finite seed values')
    mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / math.sqrt(len(x)); t = 2.1314495455597715
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15, ci95_lower=mean - t * se, ci95_upper=mean + t * se,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()), negative_count=int((x < 0).sum()))


def aggregate(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); require(not output.exists(), 'Never overwrite aggregation output')
    result = read(source / 'execution/results.json'); require(result.get('status') == 'completed' and result.get('posthoc') is True, 'Incomplete posthoc execution')
    rows = result['rows']; require(len(rows) == 32, 'Expected 32 policy rows')
    summary = metrics.summarize(rows)
    by = {(row['seed'], row['condition']): row for row in rows}; seeds = sorted({row['seed'] for row in rows})
    per_seed = []
    for seed in seeds:
        live = by[seed, 'partial_multi_strict_PL_live']; silent = by[seed, 'partial_multi_strict_PL_silent']
        live_sender = {r['sender']: r for r in live['by_sender']}; silent_sender = {r['sender']: r for r in silent['by_sender']}
        def avg(key, source_rows): return float(np.mean([source_rows[s][key] for s in range(3)]))
        per_seed.append(dict(seed=seed,
            plan_transfer_live=avg('plan_transfer', live_sender), plan_transfer_silent=avg('plan_transfer', silent_sender),
            plan_transfer_live_minus_silent=avg('plan_transfer', live_sender) - avg('plan_transfer', silent_sender),
            partner_transfer_live=avg('partner_transfer', live_sender), partner_transfer_silent=avg('partner_transfer', silent_sender),
            partner_transfer_live_minus_silent=avg('partner_transfer', live_sender) - avg('partner_transfer', silent_sender),
            sender_rows=[dict(sender=s, plan_transfer_live=live_sender[s]['plan_transfer'], plan_transfer_silent=silent_sender[s]['plan_transfer'],
                              plan_transfer_difference=live_sender[s]['plan_transfer'] - silent_sender[s]['plan_transfer'],
                              partner_transfer_live=live_sender[s]['partner_transfer'], partner_transfer_silent=silent_sender[s]['partner_transfer'],
                              partner_transfer_difference=live_sender[s]['partner_transfer'] - silent_sender[s]['partner_transfer']) for s in range(3)]))
    def condition_means(live):
        selected = [row for row in rows if row['live'] is live]
        result = {}
        for key in ('natural_physical', 'natural_q', 'natural_conditional_q', 'intervention_physical', 'intervention_q', 'intervention_conditional_q'):
            result[key] = float(np.mean([r['by_sender'][s][key] for r in selected for s in range(3)]))
        return result
    seed_plan = [r['plan_transfer_live_minus_silent'] for r in per_seed]
    seed_partner = [r['partner_transfer_live_minus_silent'] for r in per_seed]
    sender_stats = []
    for sender in range(3):
        values = [r['sender_rows'][sender]['plan_transfer_difference'] for r in per_seed]
        sender_stats.append(dict(sender=sender, plan_transfer=stats(values), mean_percentage_points=100 * float(np.mean(values))))
    sham = [r['sham'][s] for r in rows for s in range(3)]
    aggregate_result = dict(status='completed_json_only_posthoc_aggregation', source=str(source), source_results_sha256=sha(source / 'execution/results.json'),
        primary=dict(name='live_minus_silent_signed_donor_plan_transfer', unit='percentage points', seed_statistics=stats(seed_plan),
                     seed_rows=per_seed, sender_statistics=sender_stats),
        secondary=dict(partner_transfer=stats(seed_partner), live_condition=condition_means(True), silent_condition=condition_means(False)),
        sham=dict(rows=len(sham), all_message_equal=all(r['message_equal'] for r in sham), all_action_equal=all(r['action_equal'] for r in sham), max_probability_error=max(r['max_probability_error'] for r in sham)),
        source_summary=summary,
        scope=dict(independent_units='Sixteen final policy initializations; worlds, cases and recipients are repeated measures.',
                   intervention='Only selected sender W1 outward packet is replaced; live recomputes W2/actions, silent is exact closed-channel replay.',
                   primary='Mean over three senders within seed, then paired live minus silent across seeds.',
                   posthoc=True, no_training_updates=True))
    output.mkdir(parents=True); (output / 'results.json').write_text(json.dumps(aggregate_result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    receipt = dict(status='passed', input_results_sha256=sha(source / 'execution/results.json'), output_results_sha256=sha(output / 'results.json'), model_forwards=0, optimizer_updates=0, posthoc=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); return aggregate_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True); args = parser.parse_args()
    print(json.dumps(aggregate(args.source, args.output), ensure_ascii=False))
