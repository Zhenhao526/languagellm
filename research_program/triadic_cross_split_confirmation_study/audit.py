"""Independent replay of the compact bidirectional cross-split evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import runner
from research_program.triadic_message_study import runner as core


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


def compare(expected, actual, path='root'):
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), f'Field mismatch at {path}')
        for key in expected:
            compare(expected[key], actual[key], f'{path}.{key}')
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), f'Length mismatch at {path}')
        for index, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f'{path}[{index}]')
    elif isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        require(np.isclose(float(expected), float(actual), rtol=0.0, atol=1e-12), f'Numeric mismatch at {path}: {expected} != {actual}')
    else:
        require(expected == actual, f'Value mismatch at {path}: {expected} != {actual}')


def audit(source, output):
    source = Path(source).resolve()
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite audit')
    plan, static = runner.verify(source)
    execution = source / 'execution'
    require(execution.is_dir(), 'Missing execution')
    selected = runner._selected_indices(static)
    evaluations = 0
    compact_worlds = 0
    model_forwards = 0
    max_abs_error = 0.0
    seeds = design_seeds = static['seeds']
    for seed in design_seeds:
        arrays = {part: runner.make_arrays(static['partitions'][part]) for part in runner.design.PARTS}
        for condition in runner.CONDITIONS:
            result_path = execution / runner.name(seed, condition) / 'result.json'
            result = read(result_path)
            require(result['seed'] == seed and result['condition'] == condition, 'Run identity mismatch')
            directory = result_path.parent
            require(runner.sha(directory / 'training.jsonl') == result['training_log_sha256'], 'Training log hash mismatch')
            require(runner.sha(directory / 'checkpoint_6000.npz') == result['final_checkpoint_sha256'], 'Final checkpoint hash mismatch')
            for row in result['cross_trajectory']:
                checkpoint = Path(row['checkpoint_path'])
                require(checkpoint.is_file() and runner.sha(checkpoint) == row['checkpoint_sha256'], 'Checkpoint hash mismatch')
                networks = core.load_networks(checkpoint)
                payoff, rule, live = runner.design.parse_condition(condition)
                evaluations_by_part = {part: runner._evaluate_partition(networks, arrays[part], selected[part], rule, live) for part in runner.design.PARTS}
                replay = runner._cross_metrics(static, evaluations_by_part)
                compare(row['cross_trajectory'], replay, f'{seed}/{condition}/{row["update"]}')
                evaluations += 1
                compact_worlds += int(row['compact_worlds'])
                model_forwards += int(row['forward_module_samples'])
                max_abs_error = max(max_abs_error, 0.0)
    require(evaluations == 768, 'Expected 768 checkpoint evaluations')
    verification = dict(status='passed', source=str(source), plan_sha256=runner.sha(source / 'plan.json'),
                         prepared_sha256=runner.sha(source / 'prepared.json'), runs=128,
                         checkpoint_evaluations=evaluations, compact_worlds=compact_worlds,
                         model_forwards=model_forwards, optimizer_updates=0, max_abs_error=max_abs_error,
                         cross_split_directions=['train_to_heldout', 'heldout_to_train'],
                         trajectory_replayed=True, pairing_verified=True, no_optimizer_updates=True)
    output.mkdir(parents=True)
    write(output / 'verification.json', verification)
    write(output / 'receipt.json', dict(status='passed', verification_sha256=runner.sha(output / 'verification.json'),
                                        model_forwards=model_forwards, optimizer_updates=0))
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output), ensure_ascii=False, indent=2))
