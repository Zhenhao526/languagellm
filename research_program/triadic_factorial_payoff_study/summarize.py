"""JSON-only summary for the audited factorial-payoff confirmation."""
from __future__ import annotations

from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from . import design, metrics

HERE = Path(__file__).resolve().parent


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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def behavior(evaluation):
    value = evaluation.get('behavioral', evaluation.get('behavior'))
    require(value is not None, 'Behavior record missing')
    return value


def mean_behavior(evaluations):
    values = [behavior(e) for e in evaluations]
    fields = ('reward_mean', 'full_success_rate', 'physical_execution_rate',
              'executed_partner_correct_rate', 'correct_executed_pair_rate',
              'physical_pair_rate', 'material_identity_correct_rate')
    result = {field: float(np.mean([value[field] for value in values])) for field in fields}
    factor = [e['factor_response']['heldout_changed_actor'] for e in evaluations]
    result['heldout_changed_actor_Q'] = None if any(value is None for value in factor) else float(np.mean([value['Q'] for value in factor]))
    result['heldout_changed_actor_Q_excess'] = None if any(value is None for value in factor) else float(np.mean([value['Q_excess'] for value in factor]))
    return result


def summarize(results, verification):
    require(results['status'] == 'completed' and verification['status'] == 'passed', 'Completed result and passed audit required')
    runs = results['runs']; require(len(runs) == 128, 'Expected 128 runs')
    expected = {(seed, payoff, rule, live) for seed in design.SEEDS for payoff in design.PAYOFFS for rule in design.RULES for live in design.LIVES}
    by = {(r['seed'], r['payoff'], r['rule'], r['live']): r for r in runs}
    require(set(by) == expected and len(by) == len(runs), 'Incomplete run grid')
    computed = metrics.primary(runs); require(computed == results['primary'], 'Primary mismatch with execution result')
    trajectories = []
    for payoff, rule, live in product(design.PAYOFFS, design.RULES, design.LIVES):
        condition = f'{payoff}_{rule}_PL_{"live" if live else "silent"}'
        for i, update in enumerate(design.UPDATES):
            evaluations = [by[seed, payoff, rule, live]['trajectory'][i]['evaluation'] for seed in design.SEEDS]
            row = mean_behavior(evaluations)
            row.update(payoff=payoff, rule=rule, live=live, update=update, partition=design.TARGET, condition=condition)
            trajectories.append(row)
    endpoints = []
    for payoff, rule, live, part in product(design.PAYOFFS, design.RULES, design.LIVES, design.PARTS):
        if part == design.TARGET:
            evaluations = [by[seed, payoff, rule, live]['trajectory'][-1]['evaluation'] for seed in design.SEEDS]
        else:
            evaluations = [by[seed, payoff, rule, live]['final'][part] for seed in design.SEEDS]
        row = mean_behavior(evaluations)
        row.update(payoff=payoff, rule=rule, live=live, partition=part,
                   condition=f'{payoff}_{rule}_PL_{"live" if live else "silent"}')
        endpoints.append(row)
    interaction_curves = []
    for rule in design.RULES:
        qcells = {(payoff, live): [float(np.mean([by[seed, payoff, rule, live]['trajectory'][i]['evaluation']['factor_response']['heldout_changed_actor']['Q'] for seed in design.SEEDS])) for i in range(len(design.UPDATES))]
                  for payoff, live in product(design.PAYOFFS, design.LIVES)}
        pcells = {(payoff, live): [float(np.mean([behavior(by[seed, payoff, rule, live]['trajectory'][i]['evaluation'])['correct_executed_pair_rate'] for seed in design.SEEDS])) for i in range(len(design.UPDATES))]
                  for payoff, live in product(design.PAYOFFS, design.LIVES)}
        interaction_curves.append(dict(rule=rule, Q=metrics.interaction_curve(qcells, 'heldout_changed_actor_Q'),
                                       partner=metrics.interaction_curve(pcells, 'correct_executed_pair_rate')))
    return dict(status='completed_json_only_summary', audit_status='passed', primary=computed,
                counts=dict(independent_seeds=len(design.SEEDS), runs=len(runs), trajectory_records=128 * len(design.UPDATES),
                            other_final_records=128 * (len(design.PARTS) - 1), evaluations=verification['evaluations'], worlds=verification['worlds']),
                trajectory_summaries=trajectories, endpoint_summaries=endpoints, interaction_curves=interaction_curves,
                budget=results['budget'], measured_budget=results['measured_budget'],
                scope=dict(input='Completed execution JSON plus passed full physical replay audit.', primary=computed['definition'],
                           semantic_holdout='Resource predicates 5 and 6 are absent from factorial training while all individual values remain present.',
                           inference='Sixteen new paired seeds; Student-t interval across seed-level society summaries. World rows are not independent societies.',
                           boundary='A Q interaction tests behavioral response to unseen combinations under this task. It does not identify words, compositionality, cultural transmission, or human language origin.'))


def execute(results_path, audit_path, output):
    results_path = Path(results_path).resolve(); audit_path = Path(audit_path).resolve(); output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite summary')
    value = summarize(read(results_path), read(audit_path))
    output.mkdir(parents=True); write(output / 'summary.json', value)
    receipt = dict(status='passed', at=datetime.now(timezone.utc).isoformat(),
                   inputs_sha256={str(results_path): sha(results_path), str(audit_path): sha(audit_path), str(HERE / 'summarize.py'): sha(HERE / 'summarize.py'), str(HERE / 'metrics.py'): sha(HERE / 'metrics.py')},
                   outputs_sha256={str(output / 'summary.json'): sha(output / 'summary.json')}, model_forwards=0, optimizer_updates=0)
    write(output / 'receipt.json', receipt); return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--results', required=True); parser.add_argument('--audit', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(execute(args.results, args.audit, args.output), ensure_ascii=False, indent=2))
