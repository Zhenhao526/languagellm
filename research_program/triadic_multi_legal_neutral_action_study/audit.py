"""Independent checkpoint replay audit for the neutral-action pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from . import runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compare(expected, actual, errors=None):
    errors = [] if errors is None else errors
    require(set(expected) == set(actual), 'Dictionary keys differ') if isinstance(expected, dict) else None
    if isinstance(expected, dict):
        for key in expected:
            compare(expected[key], actual[key], errors)
    elif isinstance(expected, list):
        require(len(expected) == len(actual), 'List lengths differ')
        for left, right in zip(expected, actual):
            compare(left, right, errors)
    elif isinstance(expected, float):
        require(math.isfinite(float(actual)), 'Nonfinite replay value')
        errors.append(abs(expected - float(actual)))
    else:
        require(expected == actual, f'Value differs: {expected!r} != {actual!r}')
    return errors


COMPARE_KEYS = ('value', 'target_pair_actor_legal_action_rate', 'actor_legal_action_rates', 'q_rate',
                'conditional_q_rate', 'physical_execution_rate', 'third_actor_wait_rate', 'mean_reward',
                'cancel_any_rate', 'cancel_protocol_rate', 'mixed_cancel_rate', 'cancel_action_rate',
                'wait_action_rate', 'transport_action_rate', 'action_counts', 'legal_plan_selection_counts',
                'legal_plan_selection_total', 'conditional_q_denominator_worlds', 'conditional_q_numerator_worlds',
                'legal_plan_count', 'worlds', 'partition', 'payoff', 'rule', 'cancel_reward', 'live')


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    runner.verify(source); raw = json.loads((source / 'execution/results.json').read_text())
    require(raw.get('status') == 'completed', 'Pilot execution incomplete')
    runs = raw['runs']; require(len(runs) == len(runner.SEEDS) * len(runner.CONDITIONS), 'Unexpected run count')
    static = json.loads((source / 'prepared.json').read_text())
    arrays = {part: runner.make_arrays(static['partitions'][part]) for part in runner.PARTS}
    max_error = 0.0; checkpoint_rows = final_rows = worlds = forward_samples = 0
    by = {(run['seed'], run['condition']): run for run in runs}; require(len(by) == len(runs), 'Duplicate run')
    for seed in runner.SEEDS:
        for condition in runner.CONDITIONS:
            run = by[seed, condition]; directory = source / 'execution' / runner.name(seed, condition)
            result_path = directory / 'result.json'; checkpoint_path = directory / 'checkpoint_6000.npz'
            require(result_path.is_file() and checkpoint_path.is_file(), 'Missing run output')
            require(runner.core18.sha(checkpoint_path) == run['final_checkpoint_sha256'], 'Final checkpoint hash mismatch')
            for row in sorted(run['trajectory'], key=lambda item: item['update']):
                trajectory_checkpoint = Path(row['checkpoint_path'])
                require(trajectory_checkpoint.is_file() and runner.core18.sha(trajectory_checkpoint) == row['checkpoint_sha256'], 'Trajectory checkpoint hash mismatch')
                networks = runner.core18.load_networks(trajectory_checkpoint)
                replay = runner.evaluate_multi(networks, arrays['train'], static['partitions']['train'], run['cancel_reward'], run['live'])
                expected = {key: row['target_trajectory'][key] for key in COMPARE_KEYS}
                actual = {key: replay[key] for key in COMPARE_KEYS}
                errors = compare(expected, actual); max_error = max(max_error, max(errors, default=0.0)); checkpoint_rows += 1
                worlds += replay['worlds']; forward_samples += 9 * replay['worlds']
            networks = runner.core18.load_networks(checkpoint_path)
            replay = runner.evaluate_multi(networks, arrays['new_layouts'], static['partitions']['new_layouts'], run['cancel_reward'], run['live'])
            expected = {key: run['final']['new_layouts'][key] for key in COMPARE_KEYS}; actual = {key: replay[key] for key in COMPARE_KEYS}
            errors = compare(expected, actual); max_error = max(max_error, max(errors, default=0.0)); final_rows += 1
            worlds += replay['worlds']; forward_samples += 9 * replay['worlds']
    require(checkpoint_rows == len(runs) * len(runner.STEPS) and final_rows == len(runs) and max_error <= 2e-14, 'Checkpoint replay mismatch')
    verification = dict(status='passed', source=str(source), policy_rows=len(runs), checkpoint_rows=checkpoint_rows,
                         final_rows=final_rows, worlds=worlds, model_forwards=forward_samples, optimizer_updates=0,
                         max_abs_error=max_error, all_cancel_settlement_checked=True, action_count=18,
                         cancel_rewards=list(runner.design.CANCEL_REWARDS), rematched_assignments=sum(run['rematch_assignments'] for run in runs))
    path = output / 'verification.json'; path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_results_sha256=hashlib.sha256((source / 'execution/results.json').read_bytes()).hexdigest(),
                   verification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_forwards=forward_samples,
                   optimizer_updates=0, no_training_updates=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True); args = parser.parse_args(); main(args.source, args.output)
