"""Train and evaluate the factorial semantic-generalization payoff study."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

from copy import deepcopy
from itertools import zip_longest
from pathlib import Path
import argparse
import json
import multiprocessing
import platform
import shutil
import time
import numpy as np

from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_factorial_formation_study import factor_cases
from research_program.triadic_need_response_study import cases
from research_program.triadic_reciprocal_execution_study import environment, kernel
from research_program.triadic_message_study import runner as core
from . import design, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS, PAYOFFS, RULES, LIVES = design.SEEDS, design.PAYOFFS, design.RULES, design.LIVES
PARTS, TARGET, STEPS, CONDITIONS = design.PARTS, design.TARGET, design.UPDATES, design.CONDITIONS
sha, read, write, json_bytes, array_sha, now = core.base.sha, core.base.read, core.base.write_new, core.base.json_bytes, core.base.array_sha, core.base.now

CONFIG = dict(
    updates=6000, batch_size=256, checkpoints=list(STEPS), features=54, dtype='float64',
    learning_rate=0.001, global_gradient_clip=5.0, entropy_initial=0.001,
    entropy_zero_after_updates=1000, sender_entropy_coefficient=0,
    trajectories_per_state=2, sender_windows=2, sender_tokens_per_window=4,
    alphabet_size=8, actions_per_actor=17, worker_count=4,
    multiprocessing_start_method='spawn', blas_threads_per_worker=1,
    dimensions={'sender1': [54, 64, 64, 32], 'sender2': [153, 64, 64, 32], 'action': [252, 64, 64, 17]},
    seeds=list(SEEDS), payoffs=list(PAYOFFS), rules=list(RULES), lives=['live', 'silent'],
    conditions=list(CONDITIONS), partitions=list(PARTS), target=TARGET,
    task='factorial_semantic_generalization_payoff_ecology',
    train_resources=[0, 1, 2, 3, 4, 7], heldout_resources=list(design.HELDOUT_RESOURCES),
    training_objective='mean_log_expected_payoff_plus_original_action_entropy; sender_two_trajectory_LOO',
    payoff_intervention='partial: native 0/.5/1; all_or_nothing: training table maps .5 to 0; native settlement unchanged',
    observation='PL_own_need_and_public_layout_only; other_need_blocks_and_FI_flag_zero',
    primary='heldout_changed_actor_Q_payoff_communication_interaction_centered_time_AUC_rule_mean_then16_seeds',
    semantic_holdout='resource predicates 5=wood-long and 6=fiber-short are absent from factorial training; marginal values and opposite singleton corners remain present',
    evaluation='complete heldout_resource trajectory at six checkpoints plus complete train/heldout_resource/heldout_layout/heldout_both final partitions',
    message_panel='target heldout_resource factor response; all nine changed-person×axis strata retained',
    pairing='same initial parameters, world uniforms and message uniforms across all eight arms within each seed',
    selection='fixed16 new seeds ×2 payoff modes ×2 rules ×2 communication channels ×6000 updates; no early stopping',
    automatic_followon_experiment=False,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    return design.parse_condition(condition)


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, 'Unknown seed or condition')
    return f'seed_{seed}_{condition}'


def sources():
    paths = [HERE / n for n in ('__init__.py', 'design.py', 'metrics.py', 'runner.py', 'plan.md', 'README.md', 'tests/test_design.py')]
    paths += [Path(dataset.__file__), Path(environment.__file__), Path(kernel.__file__), Path(core.__file__),
              Path(cases.__file__), Path(factor_cases.__file__)]
    require(all(path.is_file() for path in paths), 'Missing frozen source')
    return {str(path.resolve().relative_to(ROOT)): sha(path) for path in paths}


def _budget(static):
    runs = len(SEEDS) * len(CONDITIONS)
    target_worlds = static['partitions'][TARGET]['world_count']
    eval_worlds = runs * (len(STEPS) * target_worlds + sum(static['partitions'][part]['world_count'] for part in PARTS if part != TARGET))
    train_forward = runs * CONFIG['updates'] * CONFIG['batch_size'] * 2 * 9
    return dict(runs=runs, independent_seed_blocks=len(SEEDS), payoffs=len(PAYOFFS), rules=len(RULES), channels=len(LIVES),
                training_updates=runs * CONFIG['updates'], training_world_samples=runs * CONFIG['updates'] * CONFIG['batch_size'],
                message_trajectories=runs * CONFIG['updates'] * CONFIG['batch_size'] * 2,
                training_forward_module_samples=train_forward, checkpoints=runs * len(STEPS),
                target_trajectory_files=runs * len(STEPS), other_final_files=runs * (len(PARTS) - 1),
                evaluation_files=runs * (len(STEPS) + len(PARTS) - 1), evaluation_worlds=eval_worlds,
                evaluation_forward_module_samples=eval_worlds * 9, total_forward_module_samples=train_forward + eval_worlds * 9,
                target_trajectory_worlds=runs * len(STEPS) * target_worlds)


def prepared():
    static = design.make_prepared()
    static['config'] = deepcopy(CONFIG); static['source_sha256'] = sources()
    static['runtime'] = dict(python=platform.python_version(), numpy=np.__version__)
    static['budget'] = _budget(static)
    return static


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Never overwrite preparation')
    static = prepared(); out.mkdir(parents=True)
    for relative in static['source_sha256']:
        target = out / 'source_snapshot' / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    write(out / 'prepared.json', static)
    plan = dict(status='prepared_without_training', created_at=now(), config=CONFIG,
                source_sha256=static['source_sha256'], prepared_sha256=sha(out / 'prepared.json'),
                runtime=static['runtime'], no_model_calls=True, no_training_updates=True)
    write(out / 'plan.json', plan); write(out / 'freeze.json', dict(plan_sha256=sha(out / 'plan.json'), prepared_sha256=sha(out / 'prepared.json')))
    verify(out)
    return dict(status='prepared_without_training', output=str(out), plan_sha256=sha(out / 'plan.json'), budget=static['budget'])


def verify(out):
    out = Path(out).resolve(); plan, static, freeze = [read(out / n) for n in ('plan.json', 'prepared.json', 'freeze.json')]
    require(sha(out / 'plan.json') == freeze['plan_sha256'] and sha(out / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Frozen hash mismatch')
    require(static == prepared() and plan['source_sha256'] == sources() and plan['config'] == CONFIG,
            'Current source/config differs from frozen preparation')
    require(plan['runtime'] == dict(python=platform.python_version(), numpy=np.__version__), 'Runtime differs')
    for relative, digest in plan['source_sha256'].items():
        require(sha(out / 'source_snapshot' / relative) == digest, 'Source snapshot changed: ' + relative)
    return plan, static


def make_arrays(spec):
    arrays = dataset.make_arrays(spec, information='PL'); arrays.pop('states', None)
    x = arrays['x_PL']; require(x.shape == (spec['world_count'], 3, 54) and np.all(x[:, :, 53] == 0), 'Bad PL features')
    for actor in range(3):
        for other in range(3):
            if actor != other:
                require(np.all(x[:, actor, 7 * other:7 * other + 7] == 0), 'Other private need exposed')
    native = arrays['rewards'].copy(); arrays['native_rewards'] = native
    arrays['rewards_partial'] = design.payoff_rewards(native, 'partial')
    arrays['rewards_all_or_nothing'] = design.payoff_rewards(native, 'all_or_nothing')
    return arrays


def evaluate(networks, arrays, spec, payoff, rule, live, path, case_spec, edge_groups):
    path = Path(path); require(not path.exists(), 'Evaluation path exists')
    n = int(spec['world_count']); ids = np.arange(n, dtype=np.int64); states = arrays['packed_states'][ids]
    cases.validate_partition_arrays(case_spec, states, ids)
    data = dict(states=states, state_indices=ids, messages=np.empty((n, 2, 3, 4), dtype=np.int8),
                action_indices=np.empty((n, 3), dtype=np.int16), action_probabilities=np.empty((n, 3, 17), dtype=np.float64),
                conditional_exact_expected_reward=np.empty(n), conditional_exact_full_success_probability=np.empty(n),
                conditional_exact_execution_probability=np.empty(n), conditional_full_posterior_mass=np.empty(n))
    for start in range(0, n, 1024):
        stop = min(start + 1024, n); sl = slice(start, stop); ix = ids[sl]
        trace = core.rollout(networks, arrays['x_PL'][ix], bool(live)); probs, _ = core.base.policy_distribution(trace['action_logits'])
        data['messages'][sl] = trace['messages']; data['action_probabilities'][sl] = probs; data['action_indices'][sl] = probs.argmax(axis=-1)
        terms = kernel.objective_terms(trace['action_logits'], arrays['rewards_' + payoff][ix], rule)
        data['conditional_exact_expected_reward'][sl] = terms['native_expected_reward']
        data['conditional_exact_full_success_probability'][sl] = terms['full_success_probability']
        data['conditional_exact_execution_probability'][sl] = terms['execution_probability']
        data['conditional_full_posterior_mass'][sl] = terms['full_success_posterior_mass']
    physical = environment.settle(states, data['action_indices'], rule)
    truth = environment.truth_from_rewards(states, arrays['native_rewards'][ids])
    behavior = metrics.behavioral(data | physical, truth, data['action_indices'], rule)
    factor_response = {}
    for group in ('heldout_changed_actor', 'seen_changed_actor'):
        selected = edge_groups[group]
        factor_response[group] = factor_cases.subset_metrics(case_spec, physical['actual_pair_index'], selected) if all(selected) else None
    strict = environment.settle(states, data['action_indices'], 'strict')
    reciprocal = environment.settle(states, data['action_indices'], 'reciprocal')
    data.update(physical); data['target_pair_index'] = truth['correct_pair_index']
    data['native_rewards'] = arrays['native_rewards'][ids]; data['payoff_training_rewards'] = arrays['rewards_' + payoff][ids]
    with path.open('xb') as stream:
        np.savez_compressed(stream, **data)
    return dict(path=str(path), data_sha256=sha(path), worlds=n, partition=spec['partition'], payoff=payoff,
                rule=rule, live=bool(live), forward_module_samples=9 * n, behavioral=behavior,
                native=environment.metrics(physical, truth, data['action_indices'], rule),
                strict=environment.metrics(strict, truth, data['action_indices'], 'strict'),
                common_reciprocal=environment.metrics(reciprocal, truth, data['action_indices'], 'reciprocal'),
                factor_response=factor_response, information='PL', scope='complete_partition')


def train_run(seed, condition, static, arrays, execution):
    payoff, rule, live = parse_condition(condition); directory = Path(execution) / name(seed, condition); directory.mkdir(exist_ok=False)
    networks = core.make_networks(seed); initial = core.parameter_hash(networks); optimizer = core.base.make_adam(networks)
    world_rng, message_rngs = np.random.default_rng(np.random.SeedSequence([seed, 200])), core.make_message_rngs(seed)
    train_spec = static['partitions']['train']; target_spec = static['partitions'][TARGET]
    target_cases = static['need_response_cases'][TARGET]; target_groups = static['factor_edge_groups'][TARGET]
    trajectory = []; previous_messages = None; started = time.perf_counter()

    def checkpoint(step):
        nonlocal previous_messages
        checkpoint_path = directory / f'checkpoint_{step:04d}.npz'; checkpoint_sha = core.save_checkpoint(checkpoint_path, networks, optimizer, step, world_rng, message_rngs)
        evaluation = evaluate(networks, arrays[TARGET], target_spec, payoff, rule, live,
                              directory / f'trajectory_{step:04d}_{TARGET}.npz', target_cases, target_groups)
        with np.load(evaluation['path'], allow_pickle=False) as saved:
            messages = saved['messages']
        transition = None if previous_messages is None else dict(packet_equal_rate=float(np.mean(messages == previous_messages)), token_hamming_fraction=float(np.mean(messages != previous_messages)))
        previous_messages = messages.copy()
        row = dict(update=step, checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha, evaluation=evaluation,
                   message_summary=dict(window0_actor0_entropy_bits=metrics._entropy(messages[:, 0, 0])), message_transition=transition,
                   elapsed_seconds=time.perf_counter() - started)
        trajectory.append(row)
        with (directory / 'trajectory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json_bytes(row).decode())

    checkpoint(0)
    with (directory / 'training.jsonl').open('x', encoding='utf-8') as stream:
        for update in range(1, CONFIG['updates'] + 1):
            world_uniforms = world_rng.random((CONFIG['batch_size'], 3)); ids = design.sample_indices(train_spec, world_uniforms)
            message_uniforms = core.draw_uniforms(message_rngs, CONFIG['batch_size'])
            gradients, row = kernel.training_gradients(networks, arrays['train']['x_PL'][ids], arrays['train']['rewards_' + payoff][ids], live,
                                                       message_uniforms, update, rule)
            norm, scale = core.base.adam_step(networks, gradients, optimizer, update)
            row.update(update=update, seed=seed, condition=condition, payoff=payoff, rule=rule,
                       world_uniforms_sha256=array_sha(world_uniforms), batch_indices_sha256=array_sha(ids),
                       batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]), sample_uniforms_sha256=array_sha(message_uniforms),
                       gradient_norm=norm, gradient_clip_scale=scale, elapsed_seconds=time.perf_counter() - started)
            stream.write(json_bytes(row).decode())
            if update in STEPS:
                stream.flush(); checkpoint(update)
    finals = {}
    for part in PARTS:
        if part == TARGET:
            continue
        finals[part] = evaluate(networks, arrays[part], static['partitions'][part], payoff, rule, live,
                                directory / f'final_{part}.npz', static['need_response_cases'][part], static['factor_edge_groups'][part])
    finals[TARGET] = dict(trajectory[-1]['evaluation'], alias_of='trajectory_update_6000', additional_forward_module_samples=0)
    result = dict(seed=seed, condition=condition, payoff=payoff, rule=rule, live=live, updates=CONFIG['updates'],
                  initial_parameter_sha256=initial, final_parameter_sha256=core.parameter_hash(networks),
                  final_checkpoint_sha256=sha(directory / 'checkpoint_6000.npz'), training_log_sha256=sha(directory / 'training.jsonl'),
                  trajectory=trajectory, final=finals, elapsed_seconds=time.perf_counter() - started)
    write(directory / 'result.json', result); write(directory / 'status.json', dict(status='completed', seed=seed, condition=condition, result_sha256=sha(directory / 'result.json'), at=now()))
    return result


def worker(payload):
    seed, static, execution = payload; execution = Path(execution)
    arrays = {part: make_arrays(static['partitions'][part]) for part in PARTS}
    keys = ('packed_states', 'native_rewards', 'x_PL', 'rewards_partial', 'rewards_all_or_nothing')
    hashes = {part: {key: array_sha(arrays[part][key]) for key in keys} for part in PARTS}
    write(execution / f'seed_{seed}_arrays.json', dict(array_hashes=hashes))
    runs = []
    for condition in CONDITIONS:
        runs.append(train_run(seed, condition, static, arrays, execution))
        print(json.dumps(dict(completed_run=name(seed, condition), elapsed_seconds=runs[-1]['elapsed_seconds'])), flush=True)
    require(hashes == {part: {key: array_sha(arrays[part][key]) for key in keys} for part in PARTS}, 'Arrays mutated')
    return runs


def verify_pairing(execution, runs):
    execution = Path(execution); by = {(r['seed'], r['condition']): r for r in runs}
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
                rows = [json.loads(line) for line in lines]; count += 1
                require(all(row['update'] == count and row['seed'] == seed for row in rows), 'Training identity mismatch')
                for key in ('world_uniforms_sha256', 'sample_uniforms_sha256', 'entropy_coefficient', 'batch_indices_sha256', 'batch_states_sha256'):
                    require(len({row[key] for row in rows}) == 1, 'Paired stream differs: ' + key)
            require(count == CONFIG['updates'], 'Training update count mismatch')
        finally:
            for stream in streams:
                stream.close()


def measured_budget(static, runs):
    require(len(runs) == 128, 'Expected 128 runs'); records = []
    for run in runs:
        require([row['update'] for row in run['trajectory']] == list(STEPS), 'Incomplete trajectory')
        alias = run['final'][TARGET]; endpoint = run['trajectory'][-1]['evaluation']
        require(alias['alias_of'] == 'trajectory_update_6000' and alias['additional_forward_module_samples'] == 0,
                'Target alias mismatch')
        records.extend(row['evaluation'] for row in run['trajectory']); records.extend(run['final'][part] for part in PARTS if part != TARGET)
    require(len({row['path'] for row in records}) == len(records), 'Duplicate evaluation path')
    return dict(training_forward_module_samples=sum(run['updates'] for run in runs) * CONFIG['batch_size'] * 2 * 9,
                evaluation_forward_module_samples=sum(row['forward_module_samples'] for row in records), evaluation_files=len(records),
                evaluation_worlds=sum(row['worlds'] for row in records), checkpoints=len(runs) * len(STEPS), target_aliases=len(runs))


def execute(out):
    out = Path(out).resolve(); plan, static = verify(out); execution = out / 'execution'; require(not execution.exists(), 'Never overwrite execution'); execution.mkdir()
    started = time.perf_counter(); write(execution / 'started.json', dict(started_at=now(), pid=os.getpid(), plan_sha256=sha(out / 'plan.json')))
    try:
        with multiprocessing.get_context('spawn').Pool(CONFIG['worker_count']) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in SEEDS])
        runs = [run for group in groups for run in group]
        require([(r['seed'], r['condition']) for r in runs] == [(seed, condition) for seed in SEEDS for condition in CONDITIONS], 'Noncanonical run order')
        verify_pairing(execution, runs); measured = measured_budget(static, runs)
        arrays = [read(execution / f'seed_{seed}_arrays.json')['array_hashes'] for seed in SEEDS]
        require(all(value == arrays[0] for value in arrays), 'Worker arrays differ')
        result = dict(status='completed', completed_at=now(), elapsed_seconds=time.perf_counter() - started, plan_sha256=sha(out / 'plan.json'),
                      budget=static['budget'], measured_budget=measured, runs=runs, array_hashes=arrays[0], primary=metrics.primary(runs),
                      experiment_type='discrete_symbol_message_co_learning_factorial_payoff_ecology', language_claim_automatically_supported=False)
        write(execution / 'results.json', result); write(execution / 'status.json', dict(status='completed', at=now(), results_sha256=sha(execution / 'results.json')))
        return dict(status='completed', output=str(execution), elapsed_seconds=result['elapsed_seconds'], primary=result['primary'])
    except BaseException as error:
        write(execution / 'failure.json', dict(status='failed', at=now(), error=repr(error), elapsed_seconds=time.perf_counter() - started)); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'verify', 'execute')); parser.add_argument('--out', required=True)
    args = parser.parse_args(); answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
