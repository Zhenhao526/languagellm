"""Full read-only replay audit for the factorial payoff confirmation."""
from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import numpy as np

from research_program.triadic_reciprocal_execution_study import environment
from research_program.triadic_factorial_formation_study import factor_cases
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


def load_runs(execution):
    runs = []
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            directory = execution / runner.name(seed, condition)
            result_path = directory / 'result.json'; status_path = directory / 'status.json'
            require(result_path.is_file() and status_path.is_file(), f'Missing run metadata: {directory}')
            result = read(result_path); status = read(status_path)
            require(status['status'] == 'completed' and status['seed'] == seed and
                    status['result_sha256'] == runner.sha(result_path), f'Run status mismatch: {directory}')
            require(result['seed'] == seed and result['condition'] == condition and
                    result['updates'] == 6000 and [row['update'] for row in result['trajectory']] == list(design.UPDATES),
                    f'Run identity/trajectory mismatch: {directory}')
            require(result['final_checkpoint_sha256'] == runner.sha(directory / 'checkpoint_6000.npz') and
                    result['training_log_sha256'] == runner.sha(directory / 'training.jsonl'), f'Run artifact hash mismatch: {directory}')
            alias = result['final'][design.TARGET]; endpoint = result['trajectory'][-1]['evaluation']
            require(alias['alias_of'] == 'trajectory_update_6000' and alias['additional_forward_module_samples'] == 0 and
                    {k: v for k, v in alias.items() if k not in ('alias_of', 'additional_forward_module_samples')} == endpoint,
                    f'Target alias mismatch: {directory}')
            runs.append(result)
    require(len(runs) == 128, 'Expected 128 completed runs')
    return runs


def replay_one(result, record, static, index):
    path = Path(record['path']); require(path.is_file() and runner.sha(path) == record['data_sha256'], f'Evaluation hash mismatch: {path}')
    with np.load(path, allow_pickle=False) as saved:
        data = {key: saved[key] for key in saved.files}
    n = int(record['worlds']); require(data['states'].shape == (n, 10) and data['action_indices'].shape == (n, 3) and
                                        data['messages'].shape == (n, 2, 3, 4) and data['native_rewards'].shape == (n, 24),
                                        f'Array shape mismatch: {path}')
    payoff, rule, live = design.parse_condition(result['condition']); require(record['payoff'] == payoff and record['rule'] == rule and record['live'] == live, f'Identity mismatch: {path}')
    physical = environment.settle(data['states'], data['action_indices'], rule); truth = environment.truth_from_rewards(data['states'], data['native_rewards'])
    for key, value in physical.items(): require(key in data and np.array_equal(data[key], value), f'Settlement mismatch {key}: {path}')
    require(np.array_equal(data['target_pair_index'], truth['correct_pair_index']), f'Target mismatch: {path}')
    require(np.array_equal(data['payoff_training_rewards'], design.payoff_rewards(data['native_rewards'], payoff)), f'Payoff mismatch: {path}')
    strict = environment.settle(data['states'], data['action_indices'], 'strict')
    reciprocal = environment.settle(data['states'], data['action_indices'], 'reciprocal')
    behavior = metrics.behavioral(data, truth, data['action_indices'], rule)
    compare(record['behavioral'], behavior, f'Behavioral replay {path}')
    compare(record['native'], environment.metrics(physical, truth, data['action_indices'], rule), f'Native replay {path}')
    compare(record['strict'], environment.metrics(strict, truth, data['action_indices'], 'strict'), f'Strict replay {path}')
    compare(record['common_reciprocal'], environment.metrics(reciprocal, truth, data['action_indices'], 'reciprocal'), f'Reciprocal replay {path}')
    part = record['partition']; case_spec = static['need_response_cases'][part]; groups = static['factor_edge_groups'][part]
    expected_factor = {}
    for group in ('heldout_changed_actor', 'seen_changed_actor'):
        selected = groups[group]
        expected_factor[group] = factor_cases.subset_metrics(case_spec, physical['actual_pair_index'], selected) if all(selected) else None
    compare(record['factor_response'], expected_factor, f'Factor replay {path}')
    return n


def audit(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); require(not output.exists(), 'Never overwrite audit')
    plan, static = runner.verify(source); execution = source / 'execution'; runs = load_runs(execution)
    runner.verify_pairing(execution, runs)
    records = []
    for result in runs:
        records.extend((result, row['evaluation']) for row in result['trajectory'])
        records.extend((result, result['final'][part]) for part in design.PARTS if part != design.TARGET)
    require(len(records) == 1152, 'Expected 1152 evaluation records')
    paths = []; worlds = 0
    for index, (result, record) in enumerate(records):
        worlds += replay_one(result, record, static, index); paths.append(str(Path(record['path']).resolve()))
    require(len(set(paths)) == len(paths), 'Duplicate evaluation paths')
    output.mkdir(parents=True)
    verification = dict(status='passed', source=str(source), plan_sha256=runner.sha(source / 'plan.json'), prepared_sha256=runner.sha(source / 'prepared.json'),
                         runs=len(runs), evaluations=len(records), unique_evaluation_paths=len(set(paths)), worlds=worlds,
                         pairing_verified=True, physical_settlement_replayed=True, truth_replayed=True,
                         behavioral_metrics_replayed=True, factor_metrics_replayed=True,
                         no_model_calls=True, no_training_updates=True)
    write(output / 'verification.json', verification)
    receipt = dict(status='passed', verification_sha256=runner.sha(output / 'verification.json'),
                   artifacts_sha256={str(source / 'plan.json'): runner.sha(source / 'plan.json'), str(source / 'prepared.json'): runner.sha(source / 'prepared.json'), str(output / 'verification.json'): runner.sha(output / 'verification.json')},
                   evaluations=len(records), worlds=worlds, neural_forwards=0, optimizer_updates=0)
    write(output / 'receipt.json', receipt); return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); print(json.dumps(audit(args.source, args.output), ensure_ascii=False, indent=2))
