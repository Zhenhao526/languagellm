"""Read-only aggregation for a completed partner-switch execution.

The first execution wrote valid run-level records but the final reducer looked
for ``behavioral`` while the evaluator stored ``behavior``.  This reducer
copies records in memory, verifies all immutable artefacts, aliases the field
only for the reducer input, and writes a separate correction directory.  It
does not load a model or update any parameters.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import argparse
import json
import numpy as np

from . import design, metrics, runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def load_runs(execution):
    execution = Path(execution)
    runs = []
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            path = execution / runner.name(seed, condition) / 'result.json'
            require(path.is_file(), f'Missing run result: {path}')
            value = read(path)
            require(value['seed'] == seed and value['condition'] == condition and
                    value['payoff'] == design.parse_condition(condition)[0] and
                    value['rule'] == design.parse_condition(condition)[1],
                    f'Run identity mismatch: {path}')
            runs.append(value)
    require(len(runs) == 32, 'Expected 32 completed run records')
    return runs


def verify_evaluations(runs):
    records = []
    for run in runs:
        records.extend(row['evaluation'] for row in run['trajectory'])
        records.extend(run['final'].values())
    require(len(records) == 256, 'Expected 256 evaluation records')
    paths = []
    total_worlds = 0
    for record in records:
        path = Path(record['path'])
        require(path.is_file(), f'Missing evaluation file: {path}')
        require(runner.sha(path) == record['data_sha256'], f'Evaluation hash mismatch: {path}')
        require(record['scope'] == 'complete_partition' and record['information'] == 'PL',
                f'Unexpected evaluation scope: {path}')
        with np.load(path, allow_pickle=False) as saved:
            require(saved['states'].shape[0] == record['worlds'] and
                    saved['action_indices'].shape == (record['worlds'], 3) and
                    saved['messages'].shape == (record['worlds'], 2, 3, 4),
                    f'Evaluation array shape mismatch: {path}')
        paths.append(str(path.resolve()))
        total_worlds += int(record['worlds'])
    require(len(set(paths)) == len(paths), 'Duplicate evaluation paths')
    return dict(records=len(records), worlds=total_worlds,
                trajectory_records=32 * len(design.UPDATES), final_records=32 * len(design.PARTS),
                unique_paths=len(set(paths)))


def reducer_input(runs):
    value = deepcopy(runs)
    aliased = 0
    for run in value:
        for row in run['trajectory']:
            evaluation = row['evaluation']
            require('behavior' in evaluation and 'behavioral' not in evaluation,
                    'Unexpected behavior field state')
            evaluation['behavioral'] = evaluation['behavior']
            aliased += 1
    require(aliased == 192, 'Expected one corrected field per trajectory checkpoint')
    return value


def aggregate(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite aggregation correction')
    plan, static = runner.verify(source)
    execution = source / 'execution'
    runs = load_runs(execution)
    runner.verify_pairing(execution, runs)
    evaluation_audit = verify_evaluations(runs)
    corrected_runs = reducer_input(runs)
    arrays = [read(execution / f'seed_{seed}_arrays.json')['array_hashes'] for seed in design.SEEDS]
    require(all(item == arrays[0] for item in arrays), 'Worker array hashes differ')
    primary = metrics.primary(corrected_runs)
    result = dict(status='completed', correction='read_only_field_alias_behavior_to_behavioral',
                  source=str(source), source_plan_sha256=runner.sha(source / 'plan.json'),
                  source_prepared_sha256=runner.sha(source / 'prepared.json'),
                  budget=static['budget'], runs=runs, array_hashes=arrays[0], primary=primary,
                  evaluation_audit=evaluation_audit,
                  experiment_type='discrete_symbol_message_co_learning_partner_switch_payoff_ecology',
                  language_claim_automatically_supported=False,
                  no_model_calls=True, no_training_updates=True)
    output.mkdir(parents=True)
    write(output / 'results.json', result)
    receipt = dict(status='passed', source=str(source), source_plan_sha256=result['source_plan_sha256'],
                   source_prepared_sha256=result['source_prepared_sha256'],
                   correction='Only in-memory reducer input was aliased; source run JSON/NPZ/log files were not changed.',
                   pairing_verified=True, evaluation_audit=evaluation_audit,
                   no_model_calls=True, no_training_updates=True,
                   primary=primary, result_sha256=runner.sha(output / 'results.json'))
    write(output / 'receipt.json', receipt)
    write(output / 'status.json', dict(status='completed', result_sha256=runner.sha(output / 'results.json')))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(aggregate(args.source, args.output), ensure_ascii=False, indent=2))
