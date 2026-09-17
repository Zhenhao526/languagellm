"""Independent replay audit for the rule × information × channel experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from itertools import zip_longest
from pathlib import Path

import numpy as np

from . import design, runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


METRIC_FLOATS = (
    'value', 'q_rate', 'conditional_q_rate', 'target_pair_legal_rate',
    'proposal_legal_rate', 'executed_pair_proposal_legal_rate',
    'physical_execution_rate', 'engagement_rate', 'neutral_rate',
    'third_agent_neutral_rate', 'ignored_proposal_rate',
    'ignored_proposal_world_rate', 'unexecuted_proposal_rate',
    'exact_expected_reward_mean', 'exact_full_success_probability_mean',
    'exact_partial_success_probability_mean',
)
METRIC_INTS = (
    'worlds', 'physical_worlds', 'proposal_legal_denominator',
    'executed_pair_proposal_legal_denominator', 'conditional_q_denominator_worlds',
    'conditional_q_numerator_worlds', 'target_pair_denominator_worlds',
    'target_pair_numerator_worlds', 'legal_plan_count', 'legal_pair_count',
)
METRIC_VECTORS = ('actor_engagement_rates', 'legal_plan_selection_counts',
                  'legal_pair_selection_counts')


def compare_metric(actual, saved):
    require(actual['partition'] == saved['partition'], 'Metric partition mismatch')
    require(actual['information'] == saved['information'] == saved.get('information'),
            'Metric information mismatch')
    require(actual['rule'] == saved['rule'], 'Metric rule mismatch')
    require(bool(actual['live']) == bool(saved['live']), 'Metric channel mismatch')
    error = 0.0
    for key in METRIC_FLOATS:
        error = max(error, abs(float(actual[key]) - float(saved[key])))
    for key in METRIC_INTS:
        require(int(actual[key]) == int(saved[key]), 'Metric integer mismatch: ' + key)
    for key in METRIC_VECTORS:
        require(len(actual[key]) == len(saved[key]), 'Metric vector length mismatch: ' + key)
        for x, y in zip(actual[key], saved[key]):
            error = max(error, abs(float(x) - float(y)))
    require(error <= 1e-12, f'Metric replay discrepancy {error}')
    return error


def verify_array_receipts(source, static, execution):
    expected_by_info = {}
    for information in design.INFORMATIONS:
        expected_by_info[information] = {}
        arrays = {part: runner.make_arrays(static['partitions'][part], information)
                  for part in design.PARTS}
        keys = ('packed_states', 'native_rewards', 'x_' + information)
        expected_by_info[information] = {
            part: {key: runner.core.base.array_sha(arrays[part][key]) for key in keys}
            for part in design.PARTS}
        del arrays
    for seed in design.SEEDS:
        path = execution / f'seed_{seed}_arrays.json'
        require(path.is_file(), 'Missing array receipt: ' + path.name)
        saved = json.loads(path.read_text()).get('array_hashes')
        require(saved == expected_by_info, 'Array receipt mismatch: ' + path.name)
    return expected_by_info


def _trajectory_row(run, update):
    rows = sorted(run['trajectory'], key=lambda row: int(row['update']))
    require([int(row['update']) for row in rows] == list(design.UPDATES),
            'Checkpoint grid mismatch')
    return next(row for row in rows if int(row['update']) == int(update))


def audit_seed(payload):
    source, seed = Path(payload[0]), int(payload[1])
    static = json.loads((source / 'prepared.json').read_text())
    execution = source / 'execution'
    aggregate = json.loads((execution / 'results.json').read_text())
    rows = [row for row in aggregate['runs'] if int(row['seed']) == seed]
    require(len(rows) == len(design.CONDITIONS), f'Seed {seed} run count mismatch')
    by_condition = {row['condition']: row for row in rows}
    require(set(by_condition) == set(design.CONDITIONS), f'Seed {seed} condition grid mismatch')
    max_error = 0.0
    trajectory_worlds = final_worlds = 0
    checkpoints = finals = log_rows = 0
    for information in design.INFORMATIONS:
        arrays = {part: runner.make_arrays(static['partitions'][part], information)
                  for part in design.PARTS}
        for condition in design.CONDITIONS:
            rule, info, schedule, live = design.parse_condition(condition)
            if info != information:
                continue
            row = by_condition[condition]
            directory = execution / runner.name(seed, condition)
            require(directory.is_dir(), 'Missing run directory: ' + str(directory))
            saved_result = json.loads((directory / 'result.json').read_text())
            require(saved_result == row, 'Per-run result differs from aggregate')
            require(row['seed'] == seed and row['condition'] == condition
                    and row['rule'] == rule and row['information'] == information
                    and row['schedule'] == schedule and bool(row['live']) == bool(live),
                    'Run identity mismatch')
            require(row['updates'] == runner.CONFIG['updates'], 'Update count mismatch')
            trajectories = sorted(row['trajectory'], key=lambda value: int(value['update']))
            require([int(value['update']) for value in trajectories] == list(design.UPDATES),
                    'Saved trajectory grid mismatch')
            for trajectory in trajectories:
                update = int(trajectory['update'])
                checkpoint = directory / f'checkpoint_{update:04d}.npz'
                require(checkpoint.is_file() and runner.core.base.sha(checkpoint)
                        == trajectory['checkpoint_sha256'], 'Checkpoint hash mismatch')
                networks = runner.load_networks(checkpoint)
                parameter_digest = runner.parameter_hash(networks)
                require(parameter_digest == trajectory['parameter_sha256'],
                        'Checkpoint parameter hash mismatch')
                if update == 0:
                    require(parameter_digest == row['initial_parameter_sha256'],
                            'Initial checkpoint parameter mismatch')
                actual = runner.evaluate_factorized(networks, arrays['train'],
                                                    static['partitions']['train'], information,
                                                    live, rule)
                actual['update'] = update
                saved = trajectory['target_trajectory']
                max_error = max(max_error, compare_metric(actual, saved))
                require(trajectory['compact_worlds'] == actual['worlds'],
                        'Trajectory world count mismatch')
                require(trajectory['forward_module_samples'] == 9 * actual['worlds'],
                        'Trajectory forward count mismatch')
                checkpoints += 1
                trajectory_worlds += actual['worlds']
            final_checkpoint = directory / 'checkpoint_6000.npz'
            require(runner.core.base.sha(final_checkpoint) == row['final_checkpoint_sha256'],
                    'Final checkpoint hash mismatch')
            networks = runner.load_networks(final_checkpoint)
            require(runner.parameter_hash(networks) == row['final_parameter_sha256'],
                    'Final parameter hash mismatch')
            actual_final = runner.evaluate_factorized(
                networks, arrays['new_layouts'], static['partitions']['new_layouts'],
                information, live, rule)
            max_error = max(max_error, compare_metric(actual_final, row['final']['new_layouts']))
            final_worlds += actual_final['worlds']
            finals += 1
            log_path = directory / 'training.jsonl'
            require(log_path.is_file() and runner.core.base.sha(log_path)
                    == row['training_log_sha256'], 'Training log hash mismatch')
            with log_path.open(encoding='utf8') as stream:
                log = [json.loads(line) for line in stream]
            require(len(log) == runner.CONFIG['updates']
                    and all(int(item['update']) == index for index, item in enumerate(log, 1)),
                    'Training log sequence mismatch')
            expected_assignments = runner.CONFIG['updates'] * runner.CONFIG['batch_size'] \
                if schedule == 'rematched' else 0
            require(row['rematch_assignments'] == expected_assignments,
                    'Rematch assignment count mismatch')
            require(sum(row['rematch_histogram']) == expected_assignments,
                    'Rematch histogram mismatch')
            log_rows += len(log)
        del arrays
    return dict(seed=seed, max_error=max_error, trajectory_worlds=trajectory_worlds,
                final_worlds=final_worlds, checkpoints=checkpoints, finals=finals,
                log_rows=log_rows)


def audit_pairing(execution):
    paired_rows = 0
    for seed in design.SEEDS:
        paths = [(execution / runner.name(seed, condition) / 'training.jsonl').open()
                 for condition in design.CONDITIONS]
        try:
            count = 0
            for lines in zip_longest(*paths):
                require(all(line is not None for line in lines), 'Paired logs unequal length')
                rows = [json.loads(line) for line in lines]
                count += 1
                require(all(int(row['update']) == count and int(row['seed']) == seed
                            for row in rows), 'Paired update/seed mismatch')
                for key in ('world_uniforms_sha256', 'sample_uniforms_sha256',
                            'batch_indices_sha256', 'canonical_batch_states_sha256',
                            'entropy_coefficient'):
                    require(len({row[key] for row in rows}) == 1,
                            'Paired stream differs: ' + key)
                for schedule in design.SCHEDULES:
                    group = [row for row in rows if row['schedule'] == schedule]
                    require(len({row['effective_batch_states_sha256'] for row in group}) == 1
                            and len({row['permutation_indices_sha256'] for row in group}) == 1,
                            'Rematched stream differs')
            require(count == runner.CONFIG['updates'], 'Paired log row count mismatch')
            paired_rows += count
        finally:
            for path in paths:
                path.close()
    return paired_rows


def main(source, output, workers=2):
    source = Path(source).resolve()
    output = Path(output).resolve()
    require(not output.exists(), 'Never overwrite audit output')
    output.mkdir(parents=False)
    runner.verify(source)
    execution = source / 'execution'
    result_path = execution / 'results.json'
    status_path = execution / 'status.json'
    require(result_path.is_file() and status_path.is_file(), 'Execution result/status missing')
    result = json.loads(result_path.read_text())
    status = json.loads(status_path.read_text())
    require(status.get('status') == 'completed'
            and status.get('results_sha256') == runner.core.base.sha(result_path),
            'Execution status/hash mismatch')
    expected = len(design.SEEDS) * len(design.CONDITIONS)
    runs = result.get('runs', [])
    require(len(runs) == expected, 'Incomplete aggregate run grid')
    require(result['plan_sha256'] == runner.core.base.sha(source / 'plan.json'),
            'Execution plan hash mismatch')
    require(result['measured_budget'] == runner.measured_budget(runs),
            'Measured budget mismatch')
    static = json.loads((source / 'prepared.json').read_text())
    array_hashes = verify_array_receipts(source, static, execution)
    with multiprocessing.get_context('spawn').Pool(int(workers)) as pool:
        reports = pool.map(audit_seed, [(str(source), seed) for seed in design.SEEDS])
    pairing_rows = audit_pairing(execution)
    max_error = max(report['max_error'] for report in reports)
    verification = dict(
        status='passed', source=str(source), plan_sha256=runner.core.base.sha(source / 'plan.json'),
        prepared_sha256=runner.core.base.sha(source / 'prepared.json'), runs=len(runs),
        checkpoint_evaluations=sum(report['checkpoints'] for report in reports),
        final_evaluations=sum(report['finals'] for report in reports),
        trajectory_worlds=sum(report['trajectory_worlds'] for report in reports),
        final_worlds=sum(report['final_worlds'] for report in reports),
        model_forwards=9 * (sum(report['trajectory_worlds'] for report in reports)
                            + sum(report['final_worlds'] for report in reports)),
        optimizer_updates=len(runs) * runner.CONFIG['updates'], paired_training_log_rows=pairing_rows,
        max_abs_error=float(max_error), checkpoint_replayed=True, final_replayed=True,
        array_hashes_verified=True, paired_streams_verified=True, rematch_histograms_verified=True,
        no_external_model=True, no_llm=True, no_vision_model=True,
        independent_seed_reports=reports,
    )
    path = output / 'verification.json'
    path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', verification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   model_forwards=verification['model_forwards'],
                   optimizer_updates=verification['optimizer_updates'], max_abs_error=max_error)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    main(args.source, args.output, args.workers)
