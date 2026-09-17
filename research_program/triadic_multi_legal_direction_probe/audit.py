"""Replay audit for every directional-probe policy row."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return runner.sha(path)


def compare(expected, actual, path='', errors=None):
    errors = [] if errors is None else errors
    if isinstance(expected, dict):
        require(set(expected) == set(actual), f'Key mismatch at {path}')
        for key in expected:
            compare(expected[key], actual[key], f'{path}.{key}', errors)
    elif isinstance(expected, list):
        require(len(expected) == len(actual), f'Length mismatch at {path}')
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f'{path}[{i}]', errors)
    elif isinstance(expected, float):
        require(math.isfinite(expected) and math.isfinite(float(actual)), f'Nonfinite at {path}')
        errors.append(abs(expected - float(actual)))
    else:
        require(expected == actual, f'Value mismatch at {path}: {expected!r} != {actual!r}')
    return errors


def audit(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); require(not output.exists(), 'Never overwrite audit output')
    plan, static, inputs = runner.verify(source)
    result = json.loads((source / 'execution/results.json').read_text()); require(result['status'] == 'completed', 'Probe execution incomplete')
    rows = result['rows']; require(len(rows) == 32, 'Expected 32 policy rows')
    arrays = runner.make_arrays(static['partition']); cases = static['cases']; max_error = 0.0; replayed = 0; forward_samples = 0
    by = {(row['seed'], row['condition']): row for row in rows}
    for seed in runner.SEEDS:
        for condition in runner.CONDITIONS:
            stored = by[seed, condition]; meta = inputs['checkpoints'][f'{seed}:{condition}']; path = Path(meta['path'])
            require(sha(path) == meta['sha256'] == stored['checkpoint_sha256'], 'Checkpoint binding mismatch')
            networks = runner.core.load_networks(path); replay = runner.run_policy(networks, arrays, static['partition'], cases, condition.endswith('_live'))
            errors = compare({k: stored[k] for k in ('case_count', 'by_sender', 'sham', 'natural_bank_worlds', 'natural_module_samples', 'intervention_module_samples')},
                             {k: replay[k] for k in ('case_count', 'by_sender', 'sham', 'natural_bank_worlds', 'natural_module_samples', 'intervention_module_samples')},
                             f'{seed}:{condition}')
            max_error = max(max_error, max(errors, default=0.0)); replayed += 1; forward_samples += replay['natural_module_samples'] + replay['intervention_module_samples']
    require(replayed == 32 and max_error <= 2e-14, 'Replay mismatch')
    for row in rows:
        for sham in row['sham']:
            require(sham['message_equal'] and sham['action_equal'] and sham['max_probability_error'] <= 1e-13, 'Sham failure')
    verification = dict(status='passed', source=str(source), policy_rows=replayed, case_rows=sum(r['case_count'] for r in rows),
                         natural_worlds=sum(r['natural_bank_worlds'] for r in rows), model_forwards=forward_samples,
                         optimizer_updates=0, max_abs_error=max_error, sham_rows=sum(len(r['sham']) for r in rows),
                         live_intervention_rows=sum(r['case_count'] + 3 * static['sham_rows_per_sender'] for r in rows if r['live']),
                         silent_intervention_rows=0, posthoc=True)
    output.mkdir(parents=True); (output / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_results_sha256=sha(source / 'execution/results.json'), verification_sha256=sha(output / 'verification.json'),
                   policy_rows=replayed, model_forwards=forward_samples, optimizer_updates=0, posthoc=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True); args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output), ensure_ascii=False))
