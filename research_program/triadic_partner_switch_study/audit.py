"""Read-only physical replay audit for the partner-switching pilot.

Every saved evaluation is reopened, its settlement and researcher-side truth
are recomputed from states/actions/native rewards, and the stored behavioral
record is compared.  No network, optimizer, or training log generation is
invoked.  This audit intentionally accepts the post-run reducer-only source
edits by binding the run to the frozen plan/prepared hashes recorded at
execution time.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from research_program.triadic_reciprocal_execution_study import environment
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


def compare(left, right, label='value', tolerance=2e-12):
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), label + ' keys')
        for key in left:
            compare(left[key], right[key], label + '/' + str(key), tolerance)
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), label + ' list')
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, label + f'[{i}]', tolerance)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        require(isinstance(right, (int, float)) and not isinstance(right, bool) and
                math.isfinite(float(right)) and abs(float(left) - float(right)) <= tolerance,
                label + ' numbers')
    else:
        require(left == right, label)


def evaluation_records(execution):
    records = []
    run_values = []
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            directory = execution / runner.name(seed, condition)
            result_path = directory / 'result.json'
            status_path = directory / 'status.json'
            require(result_path.is_file() and status_path.is_file(), f'Missing run metadata: {directory}')
            result = read(result_path); status = read(status_path)
            require(status['status'] == 'completed' and status['seed'] == seed and
                    status['result_sha256'] == runner.sha(result_path), f'Run status mismatch: {directory}')
            require(result['seed'] == seed and result['condition'] == condition and
                    result['updates'] == 6000 and len(result['trajectory']) == len(design.UPDATES),
                    f'Run identity/trajectory mismatch: {directory}')
            require(result['final_checkpoint_sha256'] == runner.sha(directory / 'checkpoint_6000.npz'),
                    f'Final checkpoint hash mismatch: {directory}')
            require(result['training_log_sha256'] == runner.sha(directory / 'training.jsonl'),
                    f'Training log hash mismatch: {directory}')
            run_values.append(result)
            records.extend((result, row['evaluation']) for row in result['trajectory'])
            records.extend((result, evaluation) for evaluation in result['final'].values())
    require(len(run_values) == 32 and len(records) == 256, 'Unexpected run/evaluation count')
    return run_values, records


def replay_one(result, record, index):
    path = Path(record['path'])
    require(path.is_file() and runner.sha(path) == record['data_sha256'], f'Hash mismatch at evaluation {index}: {path}')
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    n = int(record['worlds'])
    require(data['states'].shape == (n, 10) and data['action_indices'].shape == (n, 3) and
            data['messages'].shape == (n, 2, 3, 4) and data['native_rewards'].shape == (n, 24),
            f'Array shape mismatch at evaluation {index}: {path}')
    payoff, rule, live = design.parse_condition(result['condition'])
    require(record['payoff'] == payoff and record['rule'] == rule and record['live'] == live,
            f'Evaluation identity mismatch at {path}')
    settled = environment.settle(data['states'], data['action_indices'], rule)
    truth = environment.truth_from_rewards(data['states'], data['native_rewards'])
    for key, value in settled.items():
        require(key in data and np.array_equal(data[key], value), f'Settlement mismatch {key}: {path}')
    require(np.array_equal(data['target_pair_index'], truth['correct_pair_index']), f'Target mismatch: {path}')
    expected_training = design.payoff_rewards(data['native_rewards'], payoff)
    require(np.array_equal(data['payoff_training_rewards'], expected_training), f'Payoff table mismatch: {path}')
    recomputed = metrics.behavioral(data, truth, data['action_indices'], rule)
    stored = record.get('behavioral', record.get('behavior'))
    require(stored is not None, f'Missing behavioral record: {path}')
    compare(stored, recomputed, f'Behavioral replay {path}')
    return n


def audit(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite audit')
    plan = read(source / 'plan.json'); prepared = read(source / 'prepared.json'); freeze = read(source / 'freeze.json')
    require(runner.sha(source / 'plan.json') == freeze['plan_sha256'] and
            runner.sha(source / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'],
            'Frozen plan/prepared hash mismatch')
    execution = source / 'execution'
    runs, records = evaluation_records(execution)
    # Pairing checks use only the completed log files and do not generate any
    # samples or invoke a model.
    runner.verify_pairing(execution, runs)
    paths = []
    total_worlds = 0
    for index, (result, record) in enumerate(records):
        total_worlds += replay_one(result, record, index)
        paths.append(str(Path(record['path']).resolve()))
    require(len(set(paths)) == len(paths), 'Duplicate evaluation paths')
    output.mkdir(parents=True)
    value = dict(status='passed', source=str(source), frozen_plan_sha256=runner.sha(source / 'plan.json'),
                 frozen_prepared_sha256=runner.sha(source / 'prepared.json'), runs=32,
                 evaluations=len(records), unique_evaluation_paths=len(set(paths)), worlds=total_worlds,
                 training_updates=32 * 6000, pairing_verified=True,
                 physical_settlement_replayed=True, truth_replayed=True, behavioral_metrics_replayed=True,
                 post_run_source_edit_note='Plan binding is to the frozen hashes. After execution, only reducer compatibility edits were made: evaluator field behavior was renamed behavioral and the metric reader accepts both.',
                 no_model_calls=True, no_training_updates=True)
    write(output / 'verification.json', value)
    receipt = dict(status='passed', verification_sha256=runner.sha(output / 'verification.json'),
                   source=str(source), artifacts_sha256={str(source / 'plan.json'): runner.sha(source / 'plan.json'),
                                                         str(source / 'prepared.json'): runner.sha(source / 'prepared.json'),
                                                         str(output / 'verification.json'): runner.sha(output / 'verification.json')},
                   evaluations=len(records), worlds=total_worlds, neural_forwards=0, optimizer_updates=0)
    write(output / 'receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.source, args.output), ensure_ascii=False, indent=2))
