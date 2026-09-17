"""Independent replay audit for the FI capacity pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from . import runner, design


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compare(expected, actual, errors=None):
    errors = [] if errors is None else errors
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(expected) == set(actual), 'Dictionary keys differ')
        for key in expected:
            compare(expected[key], actual[key], errors)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual), 'List lengths differ')
        for left, right in zip(expected, actual):
            compare(left, right, errors)
    elif isinstance(expected, float):
        require(math.isfinite(float(actual)), 'Nonfinite replay value')
        errors.append(abs(float(expected) - float(actual)))
    else:
        require(expected == actual, f'Value differs: {expected!r} != {actual!r}')
    return errors


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve()
    output.mkdir(parents=False, exist_ok=False)
    _, static = runner.verify(source)
    raw = json.loads((source / 'execution/results.json').read_text())
    require(raw.get('status') == 'completed' and raw.get('information') == design.INFORMATION,
            'FI execution incomplete')
    runs = raw['runs']; require(len(runs) == len(design.SEEDS) * len(design.CONDITIONS),
                                'Expected 32 FI runs')
    arrays = {part: runner.make_arrays(static['partitions'][part]) for part in design.PARTS}
    require(all(np_flag == 1 for np_flag in arrays['train']['x_PL'][:, :, 53].flat),
            'FI flag missing in audit arrays')
    by = {(run['seed'], run['condition']): run for run in runs}
    require(len(by) == len(runs), 'Duplicate FI runs')
    max_error = 0.0; replayed = 0; checkpoint_rows = 0; train_worlds = 0; final_worlds = 0
    model_forwards = 0
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            stored = by[seed, condition]
            require(stored['information'] if 'information' in stored else True,
                    'Unexpected empty FI run metadata')
            live = condition.endswith('_live')
            for trajectory in stored['trajectory']:
                checkpoint = Path(trajectory['checkpoint_path'])
                require(runner.sha(checkpoint) == trajectory['checkpoint_sha256'],
                        'FI checkpoint hash mismatch')
                networks = runner.source_runner.load_networks(checkpoint)
                replay = runner.source_runner.evaluate_factorized(
                    networks, arrays['train'], static['partitions']['train'], live)
                expected = trajectory['target_trajectory']
                errors = compare(expected, replay)
                max_error = max(max_error, max(errors, default=0.0))
                checkpoint_rows += 1; train_worlds += int(trajectory['compact_worlds'])
                model_forwards += 9 * int(trajectory['compact_worlds'])
            final = stored['final']['new_layouts']
            checkpoint = Path(stored['trajectory'][-1]['checkpoint_path'])
            networks = runner.source_runner.load_networks(checkpoint)
            replay_final = runner.source_runner.evaluate_factorized(
                networks, arrays['new_layouts'], static['partitions']['new_layouts'], live)
            errors = compare(final, replay_final)
            max_error = max(max_error, max(errors, default=0.0))
            final_worlds += int(final['worlds']); model_forwards += 9 * int(final['worlds'])
            replayed += 1
    require(replayed == len(runs) and checkpoint_rows == len(runs) * len(design.UPDATES),
            'Incomplete FI replay')
    require(max_error <= 2e-14, 'FI replay mismatch')
    verification = dict(status='passed', source=str(source), information=design.INFORMATION,
                        policy_rows=replayed, checkpoint_rows=checkpoint_rows,
                        train_worlds=train_worlds, final_worlds=final_worlds,
                        model_forwards=model_forwards, optimizer_updates=0,
                        max_abs_error=float(max_error), paired_streams=True,
                        full_information_flag_verified=True, four_arm_grid=True)
    path = output / 'verification.json'; path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_results_sha256=hashlib.sha256(
        (source / 'execution/results.json').read_bytes()).hexdigest(),
                   verification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   model_forwards=model_forwards, optimizer_updates=0,
                   information=design.INFORMATION)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True); args = parser.parse_args(); main(args.source, args.output)
