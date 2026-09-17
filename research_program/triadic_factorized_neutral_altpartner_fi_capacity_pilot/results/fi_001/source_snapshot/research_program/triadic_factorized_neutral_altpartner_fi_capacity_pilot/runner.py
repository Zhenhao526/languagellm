"""Train the independent FI capacity pilot using the audited factorized kernel."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import platform
import shutil
import time
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_factorized_neutral_altpartner_study import runner as source_runner
from research_program.triadic_factorized_neutral_altpartner_study import design as source_design
from research_program.triadic_message_study import runner as core
from . import design

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS, SCHEDULES, CONDITIONS, PARTS = design.SEEDS, design.SCHEDULES, design.CONDITIONS, design.PARTS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def source_manifest():
    paths = [HERE / n for n in ('__init__.py', 'design.py', 'runner.py', 'audit.py',
                               'aggregate.py', 'plot_results.py', 'plan.md', 'README.md',
                               'tests/test_design.py')]
    paths += [Path(source_runner.__file__), Path(source_design.__file__),
              Path(source_runner.kernel.__file__), Path(source_runner.remap.__file__),
              Path(dataset.__file__), Path(source_runner.environment.__file__),
              Path(core.__file__), Path(core.base.__file__)]
    require(all(path.is_file() for path in paths), 'Missing FI pilot source')
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}


def make_arrays(spec):
    arrays = dataset.make_arrays(spec, information=design.INFORMATION)
    arrays['x_PL'] = arrays.pop('x_FI')
    arrays['native_rewards'] = arrays['rewards'].copy()
    require(arrays['x_PL'].shape == (spec['world_count'], 3, 54), 'Invalid FI feature shape')
    require(np.all(arrays['x_PL'][:, :, 53] == 1), 'FI flag is not set for every actor')
    require(np.all((arrays['native_rewards'] == 1).sum(axis=1) == 2),
            'Every FI pilot world must have two full plans')
    return arrays


def _budget(static):
    runs = len(SEEDS) * len(CONDITIONS)
    train_worlds = int(static['partitions']['train']['world_count'])
    new_worlds = int(static['partitions']['new_layouts']['world_count'])
    checkpoints = len(design.UPDATES)
    train_forward = runs * 6000 * 256 * 2 * 9
    eval_forward = (runs * checkpoints * train_worlds + runs * new_worlds) * 9
    return dict(runs=runs, independent_seed_blocks=len(SEEDS), schedules=2, channels=2,
                training_updates=runs * 6000, training_world_samples=runs * 6000 * 256,
                message_trajectories=runs * 6000 * 256 * 2,
                training_forward_module_samples=train_forward,
                checkpoints=runs * checkpoints, compact_checkpoint_worlds=runs * checkpoints * train_worlds,
                final_control_worlds=runs * new_worlds,
                compact_evaluation_forward_module_samples=eval_forward,
                total_forward_module_samples=train_forward + eval_forward,
                information=design.INFORMATION)


def prepared():
    static = design.make_prepared()
    static['source_sha256'] = source_manifest()
    static['source_runner_config'] = source_runner.CONFIG
    static['runtime'] = dict(python=platform.python_version(), numpy=np.__version__)
    static['budget'] = _budget(static)
    return static


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Never overwrite preparation')
    static = prepared(); out.mkdir(parents=True, exist_ok=False)
    for relative in static['source_sha256']:
        target = out / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write(out / 'prepared.json', static)
    write(out / 'plan.json', dict(status='prepared_without_training', created_at=core.base.now(),
                                  config=static, source_sha256=static['source_sha256'],
                                  prepared_sha256=sha(out / 'prepared.json'),
                                  runtime=static['runtime'], no_model_calls=True,
                                  no_training_updates=True, information=design.INFORMATION))
    write(out / 'freeze.json', dict(plan_sha256=sha(out / 'plan.json'),
                                    prepared_sha256=sha(out / 'prepared.json')))
    verify(out)
    return dict(status='prepared_without_training', output=str(out), budget=static['budget'])


def verify(out):
    out = Path(out).resolve()
    static = json.loads((out / 'prepared.json').read_text())
    plan = json.loads((out / 'plan.json').read_text())
    freeze = json.loads((out / 'freeze.json').read_text())
    require(sha(out / 'plan.json') == freeze['plan_sha256'], 'Pilot plan hash mismatch')
    require(sha(out / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'],
            'Pilot prepared hash mismatch')
    require(static == prepared() and plan['source_sha256'] == source_manifest() and
            plan['information'] == design.INFORMATION, 'Current FI source/config differs from freeze')
    for relative, digest in plan['source_sha256'].items():
        require(sha(out / 'source_snapshot' / relative) == digest,
                'FI source snapshot changed: ' + relative)
    return plan, static


def _set_source_seed_universe():
    # source_runner.name() validates the seed against its module global.  The
    # pilot intentionally uses a new seed block while preserving the source
    # training and evaluation code unchanged.
    source_runner.SEEDS = tuple(SEEDS)
    source_runner.SCHEDULES = tuple(SCHEDULES)
    source_runner.CONDITIONS = tuple(CONDITIONS)


def worker(payload):
    seed, static, execution = payload
    _set_source_seed_universe()
    execution = Path(execution)
    arrays = {part: make_arrays(static['partitions'][part]) for part in PARTS}
    keys = ('packed_states', 'native_rewards', 'x_PL')
    hashes = {part: {key: core.base.array_sha(arrays[part][key]) for key in keys} for part in PARTS}
    write(execution / f'seed_{seed}_arrays.json', {'information': design.INFORMATION,
                                                   'array_hashes': hashes})
    runs = []
    for condition in CONDITIONS:
        run = source_runner.train_run(seed, condition, static, arrays, execution)
        runs.append(run)
        print(json.dumps({'completed_run': f'seed_{seed}_{condition}',
                          'information': design.INFORMATION,
                          'elapsed_seconds': run['elapsed_seconds']}), flush=True)
    after = {part: {key: core.base.array_sha(arrays[part][key]) for key in keys} for part in PARTS}
    require(hashes == after, 'FI arrays mutated')
    return runs


def verify_pairing(execution, runs):
    """Use the audited pairing checker after binding its seed universe."""
    _set_source_seed_universe()
    source_runner.verify_pairing(execution, runs)


def measured_budget(runs, static):
    require(len(runs) == len(SEEDS) * len(CONDITIONS), 'Unexpected FI run count')
    trajectories = [row for run in runs for row in run['trajectory']]
    finals = [value for run in runs for value in run['final'].values()]
    require(all([row['update'] for row in run['trajectory']] == list(design.UPDATES)
                for run in runs), 'Incomplete FI trajectory')
    return dict(
        training_forward_module_samples=sum(run['updates'] for run in runs) * 256 * 2 * 9,
        compact_evaluation_forward_module_samples=sum(row['forward_module_samples'] for row in trajectories)
        + sum(9 * row['worlds'] for row in finals),
        compact_evaluation_records=len(trajectories) + len(finals),
        compact_evaluation_worlds=sum(row['compact_worlds'] for row in trajectories)
        + sum(row['worlds'] for row in finals),
        checkpoints=len(trajectories), final_control_records=len(finals),
        information=design.INFORMATION,
    )


def execute(out):
    out = Path(out).resolve(); _, static = verify(out)
    execution = out / 'execution'; require(not execution.exists(), 'Never overwrite FI execution')
    execution.mkdir(parents=False, exist_ok=False); started = time.perf_counter()
    write(execution / 'started.json', dict(started_at=core.base.now(), plan_sha256=sha(out / 'plan.json'),
                                           information=design.INFORMATION))
    try:
        payloads = [(seed, static, str(execution)) for seed in SEEDS]
        with multiprocessing.get_context('spawn').Pool(4) as pool:
            groups = pool.map(worker, payloads)
        runs = [run for group in groups for run in group]
        expected = [(seed, condition) for seed in SEEDS for condition in CONDITIONS]
        require([(run['seed'], run['condition']) for run in runs] == expected,
                'Noncanonical FI run order')
        verify_pairing(execution, runs)
        measured = measured_budget(runs, static)
        result = dict(status='completed', completed_at=core.base.now(),
                      elapsed_seconds=time.perf_counter() - started,
                      plan_sha256=sha(out / 'plan.json'), information=design.INFORMATION,
                      budget=static['budget'], measured_budget=measured, runs=runs,
                      experiment_type='full_information_capacity_control',
                      no_training_updates=False, language_claim_automatically_supported=False)
        write(execution / 'results.json', result)
        write(execution / 'status.json', dict(status='completed', at=core.base.now(),
                                              results_sha256=sha(execution / 'results.json')))
        return dict(status='completed', output=str(execution),
                    elapsed_seconds=result['elapsed_seconds'], measured_budget=measured)
    except BaseException as error:
        write(execution / 'failure.json', dict(status='failed', at=core.base.now(),
                                               error=repr(error),
                                               elapsed_seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'verify', 'execute'))
    parser.add_argument('--out', required=True); args = parser.parse_args()
    answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
