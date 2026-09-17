"""Independent replay audit for every policy row and sham case."""
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
    if isinstance(expected, dict):
        require(set(expected) == set(actual), 'Dictionary keys differ')
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


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    plan, static, inputs = runner.verify(source)
    result = json.loads((source / 'execution/results.json').read_text()); require(result['status'] == 'completed', 'Probe execution incomplete')
    rows = result['rows']; require(len(rows) == 64, 'Expected 64 rows')
    arrays = runner.make_arrays(static['partition']); cases = static['cases']; by = {(r['seed'], r['condition']): r for r in rows}
    max_error = 0.0; replayed = 0; forward_samples = 0
    for seed in runner.design.SEEDS:
        for condition in runner.design.CONDITIONS:
            stored = by[seed, condition]; meta = inputs['checkpoints'][f'{seed}:{condition}']; path = Path(meta['path'])
            require(runner.sha(path) == meta['sha256'] == stored['checkpoint_sha256'], 'Checkpoint binding mismatch')
            networks = runner.core.load_networks(path)
            replay = runner.run_policy(networks, arrays, static['partition'], cases, condition.endswith('_live'))
            selected = ('case_count', 'by_sender', 'sham', 'natural_bank_worlds', 'natural_module_samples', 'intervention_module_samples')
            errors = compare({key: stored[key] for key in selected}, {key: replay[key] for key in selected})
            max_error = max(max_error, max(errors, default=0.0)); replayed += 1
            forward_samples += replay['natural_module_samples'] + replay['intervention_module_samples']
    require(replayed == 64 and max_error <= 2e-14, 'Replay mismatch')
    verification = dict(status='passed', source=str(source), policy_rows=replayed,
                         case_rows=sum(r['case_count'] for r in rows), natural_worlds=sum(r['natural_bank_worlds'] for r in rows),
                         model_forwards=forward_samples, optimizer_updates=0, max_abs_error=max_error,
                         sham_rows=sum(len(r['sham']) for r in rows),
                         live_intervention_rows=sum(r['case_count'] + 3 * static['sham_rows_per_sender'] for r in rows if r['live']),
                         silent_intervention_rows=0, posthoc=True, four_arm_grid=True)
    path = output / 'verification.json'; path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_results_sha256=hashlib.sha256((source / 'execution/results.json').read_bytes()).hexdigest(),
                   verification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_forwards=forward_samples, optimizer_updates=0, posthoc=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n'); print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True); args = parser.parse_args(); main(args.source, args.output)
