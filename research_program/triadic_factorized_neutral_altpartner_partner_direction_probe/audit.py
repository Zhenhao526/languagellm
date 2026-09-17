"""Independent replay audit for every frozen-policy probe row and sham."""
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
    _, static, inputs = runner.verify(source)
    raw = json.loads((source / 'execution/results.json').read_text())
    require(raw.get('status') == 'completed' and raw.get('posthoc') is True,
            'Probe execution incomplete')
    rows = raw['rows']; require(len(rows) == 64, 'Expected 64 policy rows')
    arrays = runner.make_arrays(static['partition']); cases = static['cases']
    by = {(row['seed'], row['condition']): row for row in rows}; require(len(by) == 64, 'Duplicate policy rows')
    max_error = 0.0; replayed = 0; natural_worlds = 0; model_forwards = 0
    for seed in runner.design.SEEDS:
        for condition in runner.design.CONDITIONS:
            stored = by[seed, condition]; meta = inputs['checkpoints'][f'{seed}:{condition}']
            checkpoint = Path(meta['path'])
            require(runner.sha(checkpoint) == meta['sha256'] == stored['checkpoint_sha256'],
                    'Checkpoint binding mismatch')
            networks = runner.source_runner.load_networks(checkpoint)
            replay = runner.run_policy(networks, arrays, static['partition'], cases, condition.endswith('_live'))
            selected = ('case_count', 'by_sender', 'sham', 'natural_bank_worlds',
                        'natural_module_samples', 'intervention_module_samples')
            errors = compare({key: stored[key] for key in selected},
                             {key: replay[key] for key in selected})
            max_error = max(max_error, max(errors, default=0.0)); replayed += 1
            natural_worlds += int(replay['natural_bank_worlds'])
            model_forwards += int(replay['natural_module_samples'] + replay['intervention_module_samples'])
    require(replayed == 64 and max_error <= 2e-14, 'Replay mismatch')
    verification = dict(
        status='passed', source=str(source), policy_rows=replayed,
        case_rows=sum(int(row['case_count']) for row in rows), natural_worlds=natural_worlds,
        model_forwards=model_forwards, optimizer_updates=0, max_abs_error=float(max_error),
        sham_rows=sum(len(row['sham']) for row in rows),
        live_intervention_rows=sum(row['case_count'] + 3 * runner.design.SHAM_ROWS_PER_SENDER
                                   for row in rows if row['live']),
        silent_intervention_rows=0, posthoc=True, four_arm_grid=True,
        factorized_action_replayed=True, alternative_partner_cases=True,
    )
    verification_path = output / 'verification.json'
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    receipt = dict(status='passed', input_results_sha256=hashlib.sha256(
        (source / 'execution/results.json').read_bytes()).hexdigest(),
                   verification_sha256=hashlib.sha256(verification_path.read_bytes()).hexdigest(),
                   model_forwards=model_forwards, optimizer_updates=0, posthoc=True)
    (output / 'receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--source', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); main(args.source, args.output)
