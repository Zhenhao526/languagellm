"""JSON-only recovery of the completed run grid.

The training process finished all 128 runs, but its final in-process reducer
looked for the old ``trajectory`` key.  This reducer reads the saved
``cross_trajectory`` records, does not load checkpoints, and never trains or
performs a model forward.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design, metrics, runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def corrected_summary(runs):
    by = {(r['seed'], r['payoff'], r['rule'], r['live']): r for r in runs}
    require(len(by) == 128, 'Complete run grid required')
    seed_rows = []
    for seed in sorted({r['seed'] for r in runs}):
        by_direction = {}
        for direction in ('train_to_heldout', 'heldout_to_train'):
            q_rules = {}; p_rules = {}
            for rule in design.RULES:
                q_values = {}; p_values = {}
                for payoff in design.PAYOFFS:
                    for live in design.LIVES:
                        rows = list(by[seed, payoff, rule, live]['cross_trajectory'])
                        rows.sort(key=lambda row: row['update'])
                        require([row['update'] for row in rows] == list(design.UPDATES), 'Checkpoint order mismatch')
                        q_values[payoff, live] = np.array([row['cross_trajectory'][direction]['Q'] for row in rows], dtype=np.float64)
                        p_values[payoff, live] = np.array([row['cross_trajectory'][direction]['partner_endpoint_rate'] for row in rows], dtype=np.float64)
                q_interaction = np.array([metrics.endpoint_interaction({key: value[i] for key, value in q_values.items()}) for i in range(6)])
                p_interaction = np.array([metrics.endpoint_interaction({key: value[i] for key, value in p_values.items()}) for i in range(6)])
                q_rules[rule] = dict(interaction=q_interaction.tolist(), centered_AUC=metrics.centered_auc(q_interaction), endpoint_interaction=float(q_interaction[-1]), measure='cross_split_Q')
                p_rules[rule] = dict(interaction=p_interaction.tolist(), centered_AUC=metrics.centered_auc(p_interaction), endpoint_interaction=float(p_interaction[-1]), measure='partner_endpoint_rate')
            by_direction[direction] = dict(Q=q_rules, partner=p_rules)
        q_auc = float(np.mean([by_direction[d]['Q'][rule]['centered_AUC'] for d in by_direction for rule in design.RULES]))
        p_auc = float(np.mean([by_direction[d]['partner'][rule]['centered_AUC'] for d in by_direction for rule in design.RULES]))
        q_endpoint = float(np.mean([by_direction[d]['Q'][rule]['endpoint_interaction'] for d in by_direction for rule in design.RULES]))
        p_endpoint = float(np.mean([by_direction[d]['partner'][rule]['endpoint_interaction'] for d in by_direction for rule in design.RULES]))
        direction_q = {d: float(np.mean([by_direction[d]['Q'][rule]['centered_AUC'] for rule in design.RULES])) for d in by_direction}
        seed_rows.append(dict(seed=seed, directions=by_direction, q_centered_AUC=q_auc, partner_centered_AUC=p_auc,
                              q_endpoint_interaction=q_endpoint, partner_endpoint_interaction=p_endpoint,
                              direction_q_centered_AUC=direction_q))
    q_auc = np.array([row['q_centered_AUC'] for row in seed_rows])
    p_auc = np.array([row['partner_centered_AUC'] for row in seed_rows])
    q_endpoint = np.array([row['q_endpoint_interaction'] for row in seed_rows])
    p_endpoint = np.array([row['partner_endpoint_interaction'] for row in seed_rows])
    direction_contrast = np.array([row['direction_q_centered_AUC']['train_to_heldout'] - row['direction_q_centered_AUC']['heldout_to_train'] for row in seed_rows])
    return dict(name='bidirectional_cross_split_payoff_communication_interaction_centered_time_AUC',
                target='bidirectional train_to_heldout and heldout_to_train',
                primary=dict(mean_centered_AUC=float(q_auc.mean()), statistics=metrics.stats(q_auc), by_seed=seed_rows,
                             scope='Cross-split seen↔heldout object×attribute edge Q; equal six changed-person×{kind,length} strata, two directions and two rules.',
                             no_posthoc_selection=True),
                partner_secondary=dict(mean_centered_AUC=float(p_auc.mean()), statistics=metrics.stats(p_auc)),
                endpoint=dict(q_interaction=float(q_endpoint.mean()), q_statistics=metrics.stats(q_endpoint),
                              partner_interaction=float(p_endpoint.mean()), partner_statistics=metrics.stats(p_endpoint)),
                direction_contrast=dict(name='train_to_heldout_minus_heldout_to_train_Q_AUC', statistics=metrics.stats(direction_contrast)),
                independent_seeds=16,
                inference='Approximate Student-t intervals across sixteen paired initializations; world-level observations are not independent societies.',
                unit='Probability difference averaged over training time; multiply by100 for percentage points.')


def aggregate(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite aggregation')
    plan, static = runner.verify(source)
    execution = source / 'execution'
    runs = []
    input_hashes = {}
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            path = execution / runner.name(seed, condition) / 'result.json'
            require(path.is_file(), f'Missing run result: {path}')
            input_hashes[str(path.relative_to(source))] = runner.sha(path)
            result = read(path)
            require(result['seed'] == seed and result['condition'] == condition, 'Run identity mismatch')
            require(len(result['cross_trajectory']) == len(design.UPDATES), 'Incomplete compact trajectory')
            runs.append(result)
    require(len(runs) == 128, 'Expected 128 saved runs')
    primary = corrected_summary(runs)
    result = dict(status='completed_after_json_only_aggregation', completed_at=runner.now(),
                  plan_sha256=runner.sha(source / 'plan.json'), budget=static['budget'],
                  measured_budget=runner.measured_budget(runs), runs=runs,
                  input_result_sha256=input_hashes, primary=primary,
                  experiment_type='bidirectional_cross_split_payoff_ecology',
                  language_claim_automatically_supported=False,
                  recovery_note='Training and checkpoint generation completed before the reducer field-name failure; this file was produced without model calls, forwards or optimizer updates.')
    execution_result = execution / 'results.json'
    write(execution_result, result)
    write(execution / 'status.json', dict(status='completed_after_json_only_aggregation', at=runner.now(), results_sha256=runner.sha(execution_result)))
    output.mkdir(parents=True)
    write(output / 'receipt.json', dict(status='passed', input_runs=128, input_result_sha256=input_hashes,
                                        output_results_sha256=runner.sha(execution_result), model_forwards=0,
                                        optimizer_updates=0, no_checkpoint_reads=True))
    return dict(status='passed', output=str(output), primary=primary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.source, args.output), ensure_ascii=False, indent=2))
