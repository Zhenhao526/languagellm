"""JSON-only descriptive summary for the completed partner-switch pilot."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

from . import design, metrics, runner

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
              'executed_partner_correct_rate', 'proposal_role_success_rate',
              'correct_executed_pair_rate', 'physical_pair_rate',
              'material_identity_correct_rate')
    return {field: float(np.mean([value[field] for value in values])) for field in fields}


def load_runs(results):
    runs = results['runs']
    require(len(runs) == 32, 'Expected 32 runs')
    expected = {(seed, payoff, rule, live) for seed in design.SEEDS for payoff in design.PAYOFFS
                for rule in design.RULES for live in design.LIVES}
    by = {(r['seed'], r['payoff'], r['rule'], r['live']): r for r in runs}
    require(set(by) == expected and len(by) == len(runs), 'Incomplete run grid')
    return by


def summarize(results, verification):
    require(results['status'] == 'completed' and verification['status'] == 'passed',
            'Completed aggregate and passed audit required')
    by = load_runs(results)
    corrected = deepcopy(results['runs'])
    primary = metrics.primary(corrected)
    trajectories = []
    for payoff, rule, live in product(design.PAYOFFS, design.RULES, design.LIVES):
        condition = design.parse_condition(f'{payoff}_{rule}_PL_{"live" if live else "silent"}')
        for i, update in enumerate(design.UPDATES):
            evaluations = [by[seed, payoff, rule, live]['trajectory'][i]['evaluation'] for seed in design.SEEDS]
            row = mean_behavior(evaluations)
            row.update(payoff=payoff, rule=rule, live=live, update=update, partition='heldout_layout',
                       condition=f'{payoff}_{rule}_PL_{"live" if live else "silent"}')
            trajectories.append(row)
    endpoints = []
    for payoff, rule, live, part in product(design.PAYOFFS, design.RULES, design.LIVES, design.PARTS):
        evaluations = [by[seed, payoff, rule, live]['final'][part] for seed in design.SEEDS]
        row = mean_behavior(evaluations)
        row.update(payoff=payoff, rule=rule, live=live, partition=part,
                   condition=f'{payoff}_{rule}_PL_{"live" if live else "silent"}')
        endpoints.append(row)
    interaction_curves = []
    for rule in design.RULES:
        cells = {}
        for payoff, live in product(design.PAYOFFS, design.LIVES):
            cells[payoff, live] = [float(np.mean([
                behavior(by[seed, payoff, rule, live]['trajectory'][i]['evaluation'])['correct_executed_pair_rate']
                for seed in design.SEEDS])) for i in range(len(design.UPDATES))]
        curve = metrics.interaction_curve(cells)
        curve.update(rule=rule, target='heldout_layout')
        interaction_curves.append(curve)
    return dict(
        status='completed_json_only_summary', audit_status='passed', primary=primary,
        primary_exact_match=True,
        counts=dict(independent_pilot_seeds=len(design.SEEDS), payoffs=len(design.PAYOFFS),
                    rules=len(design.RULES), channels=len(design.LIVES), runs=len(results['runs']),
                    trajectory_records=32 * len(design.UPDATES), final_records=32 * len(design.PARTS),
                    evaluations=256, worlds=verification['worlds']),
        trajectory_summaries=trajectories, endpoint_summaries=endpoints,
        interaction_curves=interaction_curves, budget=results['budget'],
        scope=dict(input='Completed aggregate JSON plus passed read-only physical replay audit.',
                   primary=primary['definition'],
                   semantics='Correct executed pair is matched to the researcher-side unique full-plan pair; all worlds remain in the denominator.',
                   inference='Four-seed developmental pilot; no confirmatory interval or automatic continuation.',
                   boundary='Partner adaptivity and communication gains are behavioral task measures. They do not establish word meanings, compositionality, intergenerational transmission, or human language origin.'))


def execute(aggregate_path, audit_path, output):
    aggregate_path = Path(aggregate_path).resolve(); audit_path = Path(audit_path).resolve()
    output = Path(output).resolve(); require(not output.exists(), 'Never overwrite summary')
    aggregate = read(aggregate_path / 'results.json'); verification = read(audit_path / 'verification.json')
    value = summarize(aggregate, verification)
    output.mkdir(parents=True)
    write(output / 'summary.json', value)
    receipt = dict(status='passed', at=datetime.now(timezone.utc).isoformat(),
                   inputs_sha256={str(aggregate_path / 'results.json'): sha(aggregate_path / 'results.json'),
                                  str(audit_path / 'verification.json'): sha(audit_path / 'verification.json'),
                                  str(HERE / 'summarize.py'): sha(HERE / 'summarize.py'),
                                  str(HERE / 'metrics.py'): sha(HERE / 'metrics.py')},
                   outputs_sha256={str(output / 'summary.json'): sha(output / 'summary.json')},
                   model_forwards=0, optimizer_updates=0)
    write(output / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--aggregate', required=True)
    parser.add_argument('--audit', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(execute(args.aggregate, args.audit, args.output), ensure_ascii=False, indent=2))
