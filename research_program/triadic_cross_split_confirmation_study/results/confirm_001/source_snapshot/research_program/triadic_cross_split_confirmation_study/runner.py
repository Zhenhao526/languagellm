"""Train the bidirectional cross-split confirmation with compact evaluations."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import time
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

from research_program.triadic_factorial_payoff_study import runner as base_runner
from research_program.triadic_message_study import runner as core
from research_program.triadic_reciprocal_execution_study import environment, kernel
from . import design, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS, PAYOFFS, RULES, LIVES, CONDITIONS = design.SEEDS, design.PAYOFFS, design.RULES, design.LIVES, design.CONDITIONS
STEPS = design.UPDATES
sha = core.base.sha
read = core.base.read
write = core.base.write_new
json_bytes = core.base.json_bytes
array_sha = core.base.array_sha
now = core.base.now

CONFIG = dict(
    updates=6000,
    batch_size=256,
    checkpoints=list(STEPS),
    features=54,
    dtype='float64',
    learning_rate=0.001,
    global_gradient_clip=5.0,
    entropy_initial=0.001,
    entropy_zero_after_updates=1000,
    sender_entropy_coefficient=0,
    trajectories_per_state=2,
    sender_windows=2,
    sender_tokens_per_window=4,
    alphabet_size=8,
    actions_per_actor=17,
    worker_count=4,
    multiprocessing_start_method='spawn',
    blas_threads_per_worker=1,
    dimensions={'sender1': [54, 64, 64, 32], 'sender2': [153, 64, 64, 32], 'action': [252, 64, 64, 17]},
    seeds=list(SEEDS),
    payoffs=list(PAYOFFS),
    rules=list(RULES),
    lives=['live', 'silent'],
    conditions=list(CONDITIONS),
    partitions=list(design.PARTS),
    cross_split_directions=['train_to_heldout', 'heldout_to_train'],
    task='bidirectional_cross_split_semantic_generalization_payoff_ecology',
    train_resources=[0, 1, 2, 3, 4, 7],
    heldout_resources=[5, 6],
    training_objective='mean_log_expected_payoff_plus_original_action_entropy; sender_two_trajectory_LOO',
    payoff_intervention='partial: native 0/.5/1; all_or_nothing: training table maps .5 to 0; native settlement unchanged',
    observation='PL_own_need_and_public_layout_only; other_need_blocks_and_FI_flag_zero',
    primary='bidirectional_cross_split_Q_payoff_communication_interaction_centered_time_AUC',
    semantic_holdout='resource predicates 5=wood-long and 6=fiber-short absent from factorial training; marginal values remain present',
    message_panel='six nonempty changed-person×{kind,length} strata in each direction',
    pairing='same initial parameters, world uniforms and message uniforms across all eight arms within each seed',
    selection='fixed16 new seeds ×2 payoff modes ×2 rules ×2 communication channels ×6000 updates; no early stopping',
    evaluation='compact bidirectional cross-split metrics at six checkpoints; checkpoint NPZ retained for audit',
    automatic_followon_experiment=False,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, 'Unknown seed or condition')
    return f'seed_{seed}_{condition}'


def sources():
    paths = [HERE / n for n in ('__init__.py', 'design.py', 'metrics.py', 'runner.py', 'plan.md', 'README.md', 'tests/test_design.py')]
    paths += [Path(base_runner.__file__), Path(base_runner.design.__file__), Path(base_runner.dataset.__file__),
              Path(base_runner.environment.__file__), Path(base_runner.kernel.__file__), Path(base_runner.core.__file__),
              Path(base_runner.cases.__file__), Path(base_runner.factor_cases.__file__)]
    require(all(path.is_file() for path in paths), 'Missing frozen source')
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}


def _budget(static):
    runs = len(SEEDS) * len(CONDITIONS)
    train_worlds = static['partitions']['train']['world_count']
    target_worlds = static['partitions']['heldout_resource']['world_count']
    directions = static['cross_splits']['train_to_heldout']
    source_unique = len({edge['before_index'] for edge in directions['edges']})
    destination_unique = len({edge['after_index'] for edge in directions['edges']})
    compact_worlds_per_checkpoint = (source_unique + destination_unique) * static['cross_splits']['background_count']
    train_forward = runs * CONFIG['updates'] * CONFIG['batch_size'] * 2 * 9
    evaluation_forward = runs * len(STEPS) * compact_worlds_per_checkpoint * 9
    return dict(runs=runs, independent_seed_blocks=len(SEEDS), payoffs=len(PAYOFFS), rules=len(RULES), channels=len(LIVES),
                training_updates=runs * CONFIG['updates'], training_world_samples=runs * CONFIG['updates'] * CONFIG['batch_size'],
                message_trajectories=runs * CONFIG['updates'] * CONFIG['batch_size'] * 2,
                training_forward_module_samples=train_forward, checkpoints=runs * len(STEPS),
                checkpoint_files=runs * len(STEPS), compact_evaluation_worlds=runs * len(STEPS) * compact_worlds_per_checkpoint,
                compact_evaluation_forward_module_samples=evaluation_forward,
                total_forward_module_samples=train_forward + evaluation_forward,
                compact_worlds_per_checkpoint=compact_worlds_per_checkpoint,
                train_world_count=train_worlds, heldout_resource_world_count=target_worlds)


def prepared():
    static = design.make_prepared()
    static['config'] = deepcopy(CONFIG)
    static['source_sha256'] = sources()
    static['runtime'] = dict(python=platform.python_version(), numpy=np.__version__)
    static['budget'] = _budget(static)
    return static


def prepare(out):
    out = Path(out).resolve()
    require(not out.exists(), 'Never overwrite preparation')
    static = prepared()
    out.mkdir(parents=True)
    for relative in static['source_sha256']:
        target = out / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / relative).read_bytes())
    write(out / 'prepared.json', static)
    plan = dict(status='prepared_without_training', created_at=now(), config=CONFIG,
                source_sha256=static['source_sha256'], prepared_sha256=sha(out / 'prepared.json'),
                runtime=static['runtime'], no_model_calls=True, no_training_updates=True)
    write(out / 'plan.json', plan)
    write(out / 'freeze.json', dict(plan_sha256=sha(out / 'plan.json'), prepared_sha256=sha(out / 'prepared.json')))
    verify(out)
    return dict(status='prepared_without_training', output=str(out), plan_sha256=sha(out / 'plan.json'), budget=static['budget'])


def verify(out):
    out = Path(out).resolve()
    plan, static, freeze = [read(out / n) for n in ('plan.json', 'prepared.json', 'freeze.json')]
    require(sha(out / 'plan.json') == freeze['plan_sha256'] and sha(out / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Frozen hash mismatch')
    require(static == prepared() and plan['source_sha256'] == sources() and plan['config'] == CONFIG, 'Current source/config differs from frozen preparation')
    require(plan['runtime'] == dict(python=platform.python_version(), numpy=np.__version__), 'Runtime differs')
    for relative, digest in plan['source_sha256'].items():
        require(sha(out / 'source_snapshot' / relative) == digest, 'Source snapshot changed: ' + relative)
    return plan, static


def make_arrays(spec):
    arrays = base_runner.make_arrays(spec)
    require(arrays['x_PL'].shape == (spec['world_count'], 3, 54) and np.all(arrays['x_PL'][:, :, 53] == 0), 'Bad PL features')
    return arrays


def parameter_hash(networks):
    return core.base.json_hash({f'network_{i}_{key}': array_sha(value) for i, network in enumerate(networks) for key, value in network.items()})


def _selected_indices(static):
    selected = {part: set() for part in design.PARTS}
    for direction in ('train_to_heldout', 'heldout_to_train'):
        split = static['cross_splits'][direction]
        source, destination = split['source'], split['destination']
        for edge in split['edges']:
            for background in range(static['cross_splits']['background_count']):
                selected[source].add(edge['before_index'] * static['cross_splits']['background_count'] + background)
                selected[destination].add(edge['after_index'] * static['cross_splits']['background_count'] + background)
    return {part: np.array(sorted(value), dtype=np.int64) for part, value in selected.items()}


def _evaluate_partition(networks, arrays, ids, rule, live):
    ids = np.asarray(ids, dtype=np.int64)
    states = arrays['packed_states'][ids]
    actual = np.empty(len(ids), dtype=np.int8)
    target = np.empty(len(ids), dtype=np.int8)
    for start in range(0, len(ids), 1024):
        stop = min(start + 1024, len(ids))
        sl = slice(start, stop)
        trace = core.rollout(networks, arrays['x_PL'][ids[sl]], bool(live))
        probs, _ = core.base.policy_distribution(trace['action_logits'])
        actions = probs.argmax(axis=-1)
        physical = environment.settle(states[sl], actions, rule)
        truth = environment.truth_from_rewards(states[sl], arrays['native_rewards'][ids[sl]])
        actual[sl] = physical['actual_pair_index']
        target[sl] = truth['correct_pair_index']
    return actual, target


def _cross_metrics(static, evaluations):
    result = {}
    for direction in ('train_to_heldout', 'heldout_to_train'):
        split = static['cross_splits'][direction]
        source_actual, source_target = evaluations[split['source']]
        destination_actual, destination_target = evaluations[split['destination']]
        result[direction] = metrics.edge_metrics(
            source_actual, source_target, destination_actual, destination_target,
            split['edges'], split['groups'], static['cross_splits']['background_count'])
    return result


def train_run(seed, condition, static, arrays, execution, selected):
    payoff, rule, live = design.parse_condition(condition)
    directory = Path(execution) / name(seed, condition)
    directory.mkdir(exist_ok=False)
    networks = core.make_networks(seed)
    initial = parameter_hash(networks)
    optimizer = core.base.make_adam(networks)
    world_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
    message_rngs = core.make_message_rngs(seed)
    trajectory = []
    started = time.perf_counter()

    def checkpoint(step):
        checkpoint_path = directory / f'checkpoint_{step:04d}.npz'
        checkpoint_sha = core.save_checkpoint(checkpoint_path, networks, optimizer, step, world_rng, message_rngs)
        evaluations = {part: _evaluate_partition(networks, arrays[part], selected[part], rule, live) for part in design.PARTS}
        cross = _cross_metrics(static, evaluations)
        row = dict(update=step, checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
                   cross_trajectory=cross,
                   compact_worlds=sum(len(ids) for ids in selected.values()),
                   forward_module_samples=9 * sum(len(ids) for ids in selected.values()),
                   elapsed_seconds=time.perf_counter() - started)
        trajectory.append(row)
        with (directory / 'cross_trajectory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json_bytes(row).decode())

    checkpoint(0)
    with (directory / 'training.jsonl').open('x', encoding='utf-8') as stream:
        for update in range(1, CONFIG['updates'] + 1):
            world_uniforms = world_rng.random((CONFIG['batch_size'], 3))
            ids = design.sample_indices(static['partitions']['train'], world_uniforms)
            message_uniforms = core.draw_uniforms(message_rngs, CONFIG['batch_size'])
            gradients, row = kernel.training_gradients(
                networks, arrays['train']['x_PL'][ids], arrays['train']['rewards_' + payoff][ids], live,
                message_uniforms, update, rule)
            norm, scale = core.base.adam_step(networks, gradients, optimizer, update)
            row.update(update=update, seed=seed, condition=condition, payoff=payoff, rule=rule,
                       world_uniforms_sha256=array_sha(world_uniforms), batch_indices_sha256=array_sha(ids),
                       batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]),
                       sample_uniforms_sha256=array_sha(message_uniforms), gradient_norm=norm,
                       gradient_clip_scale=scale, elapsed_seconds=time.perf_counter() - started)
            stream.write(json_bytes(row).decode())
            if update in STEPS:
                stream.flush()
                checkpoint(update)
    result = dict(seed=seed, condition=condition, payoff=payoff, rule=rule, live=live, updates=CONFIG['updates'],
                  initial_parameter_sha256=initial, final_parameter_sha256=parameter_hash(networks),
                  final_checkpoint_sha256=sha(directory / 'checkpoint_6000.npz'),
                  training_log_sha256=sha(directory / 'training.jsonl'),
                  cross_trajectory=trajectory, elapsed_seconds=time.perf_counter() - started)
    write(directory / 'result.json', result)
    write(directory / 'status.json', dict(status='completed', seed=seed, condition=condition,
                                          result_sha256=sha(directory / 'result.json'), at=now()))
    return result


def worker(payload):
    seed, static, execution = payload
    arrays = {part: make_arrays(static['partitions'][part]) for part in design.PARTS}
    keys = ('packed_states', 'native_rewards', 'x_PL', 'rewards_partial', 'rewards_all_or_nothing')
    hashes = {part: {key: array_sha(arrays[part][key]) for key in keys} for part in design.PARTS}
    write(Path(execution) / f'seed_{seed}_arrays.json', dict(array_hashes=hashes))
    selected = _selected_indices(static)
    runs = []
    for condition in CONDITIONS:
        runs.append(train_run(seed, condition, static, arrays, execution, selected))
        print(json.dumps(dict(completed_run=name(seed, condition), elapsed_seconds=runs[-1]['elapsed_seconds'])), flush=True)
    require(hashes == {part: {key: array_sha(arrays[part][key]) for key in keys} for part in design.PARTS}, 'Arrays mutated')
    return runs


def verify_pairing(execution, runs):
    execution = Path(execution)
    by = {(r['seed'], r['condition']): r for r in runs}
    expected = {(seed, condition) for seed in SEEDS for condition in CONDITIONS}
    require(set(by) == expected and len(by) == 128, 'Incomplete paired grid')
    for seed in SEEDS:
        selected = [by[seed, condition] for condition in CONDITIONS]
        require(len({r['initial_parameter_sha256'] for r in selected}) == 1, 'Unpaired initialization')
        streams = [(execution / name(seed, condition) / 'training.jsonl').open() for condition in CONDITIONS]
        try:
            count = 0
            for lines in zip_longest(*streams):
                require(all(line is not None for line in lines), 'Unequal paired logs')
                rows = [json.loads(line) for line in lines]
                count += 1
                require(all(row['update'] == count and row['seed'] == seed for row in rows), 'Training identity mismatch')
                for key in ('world_uniforms_sha256', 'sample_uniforms_sha256', 'entropy_coefficient', 'batch_indices_sha256', 'batch_states_sha256'):
                    require(len({row[key] for row in rows}) == 1, 'Paired stream differs: ' + key)
            require(count == CONFIG['updates'], 'Training update count mismatch')
        finally:
            for stream in streams:
                stream.close()


def measured_budget(runs):
    require(len(runs) == 128, 'Expected 128 runs')
    rows = []
    for run in runs:
        require([row['update'] for row in run['cross_trajectory']] == list(STEPS), 'Incomplete cross trajectory')
        rows.extend(run['cross_trajectory'])
    return dict(training_forward_module_samples=sum(run['updates'] for run in runs) * CONFIG['batch_size'] * 2 * 9,
                compact_evaluation_forward_module_samples=sum(row['forward_module_samples'] for row in rows),
                checkpoint_files=len(rows), compact_evaluation_records=len(rows),
                compact_evaluation_worlds=sum(row['compact_worlds'] for row in rows))


def execute(out):
    out = Path(out).resolve()
    plan, static = verify(out)
    execution = out / 'execution'
    require(not execution.exists(), 'Never overwrite execution')
    execution.mkdir()
    started = time.perf_counter()
    write(execution / 'started.json', dict(started_at=now(), pid=os.getpid(), plan_sha256=sha(out / 'plan.json')))
    try:
        with multiprocessing.get_context('spawn').Pool(CONFIG['worker_count']) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in SEEDS])
        runs = [run for group in groups for run in group]
        require([(r['seed'], r['condition']) for r in runs] == [(seed, condition) for seed in SEEDS for condition in CONDITIONS], 'Noncanonical run order')
        verify_pairing(execution, runs)
        measured = measured_budget(runs)
        arrays = [read(execution / f'seed_{seed}_arrays.json')['array_hashes'] for seed in SEEDS]
        require(all(value == arrays[0] for value in arrays), 'Worker arrays differ')
        result = dict(status='completed', completed_at=now(), elapsed_seconds=time.perf_counter() - started,
                      plan_sha256=sha(out / 'plan.json'), budget=static['budget'], measured_budget=measured,
                      runs=runs, array_hashes=arrays[0], primary=metrics.summarize(runs),
                      experiment_type='bidirectional_cross_split_payoff_ecology',
                      language_claim_automatically_supported=False)
        write(execution / 'results.json', result)
        write(execution / 'status.json', dict(status='completed', at=now(), results_sha256=sha(execution / 'results.json')))
        return dict(status='completed', output=str(execution), elapsed_seconds=result['elapsed_seconds'], primary=result['primary'])
    except BaseException as error:
        write(execution / 'failure.json', dict(status='failed', at=now(), error=repr(error), elapsed_seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'verify', 'execute'))
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
