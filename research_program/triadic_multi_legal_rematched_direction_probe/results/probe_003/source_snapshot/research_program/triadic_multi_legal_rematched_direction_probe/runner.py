"""Run the rematched-policy posthoc directional probe.

The data-path and intervention implementation are the previously audited
directional-probe runner.  Its module-level paths are rebound here before any
prepare/execute call so the new source manifest, four-arm grid and output tree
remain explicit and independently frozen.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

from research_program.triadic_multi_legal_direction_probe import runner as _base
from . import design, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_RUN = ROOT / 'research_program/triadic_multi_legal_rematched_confirmation_study/results/confirm_001'
SOURCE_RESULTS = SOURCE_RUN.parent

CONFIG = dict(
    schema='triadic_multi_legal_rematched_direction_probe_v1', seeds=list(design.SEEDS),
    schedules=list(design.SCHEDULES), conditions=list(design.CONDITIONS), parts=list(design.PARTS),
    checkpoint=design.CHECKPOINT, chunk_size=design.CHUNK_SIZE, sham_rows_per_sender=design.SHAM_ROWS_PER_SENDER,
    overwrite_windows=[True, False], natural_messages='greedy messages from each final rematched/static policy checkpoint',
    donor='same-layout same-owner cyclic-neighbor need; only selected sender W1 outward packet is replaced',
    primary='rematched-by-communication interaction on signed donor-only minus receiver-only legal-plan transfer',
    secondary='signed partner transfer, natural/intervened physical execution and Q, conditional Q, exact sham replay',
    posthoc_relative_to='triadic_multi_legal_rematched_confirmation_study/results/confirm_001',
    new_model_calls=True, training_updates=0, automatic_followon_experiment=False,
)

# Rebind audited base functions to this package.  These assignments affect the
# imported module only; spawned workers import this module and repeat the same
# deterministic rebinding before loading any checkpoint.
_base.HERE = HERE
_base.ROOT = ROOT
_base.SOURCE_RUN = SOURCE_RUN
_base.SEEDS = design.SEEDS
_base.CONDITIONS = design.CONDITIONS
_base.PARTS = design.PARTS
_base.CONFIG = CONFIG
_base.design = design
_base.metrics = metrics


def _source_manifest():
    """Freeze the rematched training package and this probe implementation."""
    paths = [HERE / n for n in ('__init__.py', 'design.py', 'metrics.py', 'runner.py', 'audit.py',
                                'plan.md', 'tests/test_design.py')]
    paths += [
        Path(_base.intervention.__file__), Path(_base.core.__file__), Path(_base.core.base.__file__),
        Path(_base.dataset.__file__), Path(_base.original.__file__), Path(_base.execution_env.__file__),
        SOURCE_RUN / 'plan.json', SOURCE_RUN / 'prepared.json', SOURCE_RUN / 'freeze.json',
        SOURCE_RESULTS / 'analysis_002/results.json', SOURCE_RESULTS / 'audit_001/verification.json',
    ]
    if not all(path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise ValueError('Missing probe source or frozen input: ' + ', '.join(missing))
    return {str(path.resolve().relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def _frozen_inputs():
    source = json.loads((SOURCE_RUN / 'prepared.json').read_text())
    plan = json.loads((SOURCE_RUN / 'plan.json').read_text())
    freeze = json.loads((SOURCE_RUN / 'freeze.json').read_text())
    require = _base.require
    require(hashlib.sha256((SOURCE_RUN / 'plan.json').read_bytes()).hexdigest() == freeze['plan_sha256'], 'Source plan freeze mismatch')
    require(hashlib.sha256((SOURCE_RUN / 'prepared.json').read_bytes()).hexdigest() == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Source prepared freeze mismatch')
    aggregate_path = SOURCE_RESULTS / 'analysis_002/results.json'
    audit_path = SOURCE_RESULTS / 'audit_001/verification.json'
    aggregate = json.loads(aggregate_path.read_text()); audit = json.loads(audit_path.read_text())
    require(str(aggregate.get('status', '')).startswith('completed'), 'Source aggregate incomplete')
    require(audit.get('status') == 'passed', 'Source audit incomplete')
    checkpoints = {}
    for seed in design.SEEDS:
        for condition in design.CONDITIONS:
            directory = SOURCE_RUN / 'execution' / f'seed_{seed}_{condition}'
            result = json.loads((directory / 'result.json').read_text())
            path = directory / 'checkpoint_6000.npz'
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            require(digest == result['final_checkpoint_sha256'], 'Source checkpoint hash mismatch')
            checkpoints[f'{seed}:{condition}'] = dict(path=str(path.resolve()), sha256=digest, seed=seed, condition=condition)
    return dict(plan=plan, prepared=source, freeze=freeze,
                aggregate_sha256=hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
                audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(), checkpoints=checkpoints)


# The copied base runner keeps its audited numerical kernels. Rebind only its
# source and input readers so prepare/verify use the rematched four-arm grid.
_base.source_manifest = _source_manifest
_base.frozen_inputs = _frozen_inputs

# Public aliases used by audit/plot utilities.
core = _base.core
sha = _base.sha
write = _base.write
make_arrays = _base.make_arrays
parameter_hash = _base.parameter_hash
native_bank = _base.native_bank
run_policy = _base.run_policy
frozen_inputs = _base.frozen_inputs
source_manifest = _base.source_manifest
prepared = _base.prepared
prepare = _base.prepare
verify = _base.verify
execute = _base.execute
worker = _base.worker


if __name__ == '__main__':
    import argparse
    import json
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'verify', 'execute')); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
