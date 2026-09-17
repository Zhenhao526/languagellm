"""Train the partner-switching payoff ecology using the audited learner core."""
from __future__ import annotations

import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

from copy import deepcopy
from itertools import product, zip_longest
from pathlib import Path
import argparse
import json
import multiprocessing
import platform
import shutil
import time
import numpy as np

from research_program.triadic_factorial_formation_study import runner as base
from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_reciprocal_execution_study import environment, kernel
from . import design, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS, PAYOFFS, RULES, LIVES = design.SEEDS, design.PAYOFFS, design.RULES, design.LIVES
PARTS, STEPS, CONDITIONS = design.PARTS, design.UPDATES, design.CONDITIONS
core = base.core
sha, read, write, json_bytes, array_sha, now = base.sha, base.read, base.write, base.json_bytes, base.array_sha, base.now

CONFIG = deepcopy(base.CONFIG)
CONFIG.update(
    seeds=list(SEEDS), payoffs=list(PAYOFFS), rules=list(RULES), lives=['live', 'silent'],
    conditions=list(CONDITIONS), partitions=list(PARTS), checkpoints=list(STEPS),
    task='partner_switching_payoff_ecology',
    training_objective='mean_log_expected_utility_plus_original_action_entropy; sender_two_trajectory_LOO',
    payoff_intervention='partial: native 0/.5/1; all_or_nothing: training table maps .5 to 0; native settlement unchanged',
    observation='PL_own_need_and_public_layout_only; other_need_blocks_and_FI_flag_zero',
    primary='heldout_layout_centered_time_AUC_of_(all_or_nothing_live-silent)-(partial_live-silent)_correct_executed_pair_rate',
    pairing='same initial parameters, world uniforms and message uniforms across all eight arms within each seed',
    evaluation='complete heldout_layout at six checkpoints and complete train/heldout_layout final partitions',
    selection='fixed four pilot seeds ×2 payoff modes ×2 rules ×2 channels ×6000 updates; no early stopping',
    automatic_followon_experiment=False,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    payoff, rule, live = design.parse_condition(condition)
    return payoff, rule, live


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, 'Unknown seed or condition')
    return f'seed_{seed}_{condition}'


def sources():
    paths = [HERE / n for n in ('__init__.py', 'design.py', 'metrics.py', 'runner.py', 'plan.md', 'README.md', 'tests/test_design.py')]
    paths += [Path(base.__file__), Path(base.core.__file__), Path(dataset.__file__), Path(environment.__file__), Path(kernel.__file__)]
    require(all(p.is_file() for p in paths), 'Missing frozen source')
    return {str(p.resolve().relative_to(ROOT)): sha(p) for p in paths}


def prepared():
    value = design.make_prepared()
    value['config'] = deepcopy(CONFIG)
    value['source_sha256'] = sources()
    value['runtime'] = dict(python=platform.python_version(), numpy=np.__version__)
    value['budget'] = dict(
        runs=len(SEEDS) * len(CONDITIONS), training_updates=len(SEEDS) * len(CONDITIONS) * CONFIG['updates'],
        training_world_samples=len(SEEDS) * len(CONDITIONS) * CONFIG['updates'] * CONFIG['batch_size'],
        training_forward_module_samples=len(SEEDS) * len(CONDITIONS) * CONFIG['updates'] * CONFIG['batch_size'] * 2 * 9,
        checkpoints=len(SEEDS) * len(CONDITIONS) * len(STEPS),
        target_trajectory_files=len(SEEDS) * len(CONDITIONS) * len(STEPS),
        final_files=len(SEEDS) * len(CONDITIONS) * len(PARTS),
        target_trajectory_worlds=len(SEEDS) * len(CONDITIONS) * len(STEPS) * value['partitions']['heldout_layout']['world_count'],
        final_worlds=len(SEEDS) * len(CONDITIONS) * sum(value['partitions'][p]['world_count'] for p in PARTS),
    )
    return value


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Never overwrite preparation')
    value = prepared(); out.mkdir(parents=True)
    write(out / 'prepared.json', value)
    plan = dict(status='prepared_without_training', created_at=now(), config=CONFIG,
                runtime=value['runtime'], prepared_sha256=sha(out / 'prepared.json'), source_sha256=sources(),
                no_model_calls=True, no_training_updates=True)
    write(out / 'plan.json', plan)
    write(out / 'freeze.json', dict(plan_sha256=sha(out / 'plan.json'), prepared_sha256=sha(out / 'prepared.json')))
    verify(out)
    return dict(status='prepared_without_training', output=str(out), plan_sha256=sha(out / 'plan.json'), budget=value['budget'])


def verify(out):
    out = Path(out).resolve()
    plan, value, freeze = [read(out / n) for n in ('plan.json', 'prepared.json', 'freeze.json')]
    require(sha(out / 'plan.json') == freeze['plan_sha256'], 'Plan changed')
    require(sha(out / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Prepared hash changed')
    require(plan['config'] == CONFIG and plan['source_sha256'] == sources(), 'Frozen source/config changed')
    require(value == prepared(), 'Prepared values changed')
    require(plan['runtime'] == dict(python=platform.python_version(), numpy=np.__version__), 'Runtime changed')
    return plan, value


def make_arrays(spec):
    arrays = base.make_arrays(spec)
    native = arrays['rewards'].copy()
    arrays['native_rewards'] = native
    arrays['rewards_partial'] = design.payoff_rewards(native, 'partial')
    arrays['rewards_all_or_nothing'] = design.payoff_rewards(native, 'all_or_nothing')
    return arrays


def evaluate(networks, arrays, spec, payoff, rule, live, path):
    """Run a complete partition and save the raw action/message arrays."""
    path = Path(path); require(not path.exists(), 'Evaluation path already exists')
    n = int(spec['world_count']); ids = np.arange(n, dtype=np.int64)
    states = arrays['packed_states'][ids]
    data = dict(states=states, state_indices=ids,
                messages=np.empty((n, 2, 3, 4), dtype=np.int8),
                action_indices=np.empty((n, 3), dtype=np.int16),
                action_probabilities=np.empty((n, 3, 17), dtype=np.float64),
                conditional_exact_expected_reward=np.empty(n),
                conditional_exact_full_success_probability=np.empty(n),
                conditional_exact_execution_probability=np.empty(n))
    for start in range(0, n, 1024):
        stop = min(start + 1024, n); sl = slice(start, stop); ix = ids[sl]
        trace = core.rollout(networks, arrays['x_PL'][ix], bool(live))
        probs, _ = core.base.policy_distribution(trace['action_logits'])
        data['messages'][sl] = trace['messages']
        data['action_probabilities'][sl] = probs
        data['action_indices'][sl] = probs.argmax(axis=-1)
        # These are policy-only diagnostics; the payoff arm never changes the
        # physical settlement or the researcher-side truth label.
        reward_table = arrays['rewards_' + payoff][ix]
        terms = kernel.objective_terms(trace['action_logits'], reward_table, rule)
        data['conditional_exact_expected_reward'][sl] = terms['native_expected_reward']
        data['conditional_exact_full_success_probability'][sl] = terms['full_success_probability']
        data['conditional_exact_execution_probability'][sl] = terms['execution_probability']
    data.update(environment.settle(states, data['action_indices'], rule))
    truth = environment.truth_from_rewards(states, arrays['native_rewards'][ids])
    behavior = metrics.behavioral(data, truth, data['action_indices'], rule)
    data['target_pair_index'] = truth['correct_pair_index']
    data['native_rewards'] = arrays['native_rewards'][ids]
    data['payoff_training_rewards'] = arrays['rewards_' + payoff][ids]
    with path.open('xb') as stream:
        np.savez_compressed(stream, **data)
    return dict(path=str(path), data_sha256=sha(path), worlds=n, partition=spec['partition'], payoff=payoff,
                rule=rule, live=bool(live), forward_module_samples=9 * n,
                behavioral=behavior, information='PL', scope='complete_partition')


def train_run(seed, condition, static, arrays, execution):
    payoff, rule, live = parse_condition(condition)
    directory = Path(execution) / name(seed, condition); directory.mkdir(exist_ok=False)
    nets = core.make_networks(seed); initial = core.parameter_hash(nets); optimizer = core.base.make_adam(nets)
    world_rng, message_rngs = base.make_random_streams(seed)
    train_spec = static['partitions']['train']; target_spec = static['partitions']['heldout_layout']
    start = time.perf_counter(); trajectory = []; previous_messages = None

    def checkpoint(step):
        nonlocal previous_messages
        checkpoint_path = directory / f'checkpoint_{step:04d}.npz'
        checkpoint_sha = core.save_checkpoint(checkpoint_path, nets, optimizer, step, world_rng, message_rngs)
        eval_record = evaluate(nets, arrays['heldout_layout'], target_spec, payoff, rule, live,
                               directory / f'trajectory_{step:04d}_heldout_layout.npz')
        with np.load(eval_record['path'], allow_pickle=False) as saved:
            messages = saved['messages']
        snapshot = metrics._entropy(messages[:, 0, 0])
        transition = None if previous_messages is None else dict(
            packet_equal_rate=float(np.mean(messages == previous_messages)),
            token_hamming_fraction=float(np.mean(messages != previous_messages)))
        previous_messages = messages.copy()
        row = dict(update=step, checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
                   evaluation=eval_record, message_summary=dict(window0_actor0_entropy_bits=snapshot),
                   message_transition=transition, elapsed_seconds=time.perf_counter() - start)
        trajectory.append(row)
        with (directory / 'trajectory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json_bytes(row).decode())

    checkpoint(0)
    with (directory / 'training.jsonl').open('x', encoding='utf-8') as stream:
        for update in range(1, CONFIG['updates'] + 1):
            world_uniforms = world_rng.random((CONFIG['batch_size'], 3))
            ids = design.sample_indices(train_spec, world_uniforms)
            message_uniforms = core.draw_uniforms(message_rngs, CONFIG['batch_size'])
            gradients, row = kernel.training_gradients(
                nets, arrays['train']['x_PL'][ids], arrays['train']['rewards_' + payoff][ids], live,
                message_uniforms, update, rule)
            norm, scale = core.base.adam_step(nets, gradients, optimizer, update)
            row.update(update=update, seed=seed, condition=condition, payoff=payoff, rule=rule,
                       world_uniforms_sha256=array_sha(world_uniforms), batch_indices_sha256=array_sha(ids),
                       batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]),
                       sample_uniforms_sha256=array_sha(message_uniforms), gradient_norm=norm,
                       gradient_clip_scale=scale, elapsed_seconds=time.perf_counter() - start)
            stream.write(json_bytes(row).decode())
            if update in STEPS:
                stream.flush(); checkpoint(update)
    finals = {}
    for part in PARTS:
        finals[part] = evaluate(nets, arrays[part], static['partitions'][part], payoff, rule, live,
                                directory / f'final_{part}.npz')
    result = dict(seed=seed, condition=condition, payoff=payoff, rule=rule, live=live, updates=CONFIG['updates'],
                  initial_parameter_sha256=initial, final_parameter_sha256=core.parameter_hash(nets),
                  final_checkpoint_sha256=sha(directory / 'checkpoint_6000.npz'),
                  training_log_sha256=sha(directory / 'training.jsonl'), trajectory=trajectory, final=finals,
                  elapsed_seconds=time.perf_counter() - start)
    write(directory / 'result.json', result)
    write(directory / 'status.json', dict(status='completed', seed=seed, condition=condition,
                                          result_sha256=sha(directory / 'result.json'), at=now()))
    return result


def worker(payload):
    seed, static, execution = payload; execution = Path(execution)
    arrays = {part: make_arrays(static['partitions'][part]) for part in PARTS}
    hashes = {part: {key: array_sha(arrays[part][key]) for key in ('packed_states', 'native_rewards', 'x_PL', 'rewards_partial', 'rewards_all_or_nothing')}
              for part in PARTS}
    write(execution / f'seed_{seed}_arrays.json', dict(array_hashes=hashes))
    runs = []
    for condition in CONDITIONS:
        runs.append(train_run(seed, condition, static, arrays, execution))
        print(json.dumps(dict(completed_run=name(seed, condition), elapsed_seconds=runs[-1]['elapsed_seconds'])), flush=True)
    require(hashes == {part: {key: array_sha(arrays[part][key]) for key in ('packed_states', 'native_rewards', 'x_PL', 'rewards_partial', 'rewards_all_or_nothing')}
                       for part in PARTS}, 'Arrays mutated during worker')
    return runs


def verify_pairing(execution, runs):
    by = {(r['seed'], r['condition']): r for r in runs}
    require(set(by) == {(s, c) for s in SEEDS for c in CONDITIONS}, 'Incomplete paired grid')
    for seed in SEEDS:
        streams = [(Path(execution) / name(seed, c) / 'training.jsonl').open() for c in CONDITIONS]
        try:
            count = 0
            for lines in zip_longest(*streams):
                require(all(line is not None for line in lines), 'Unequal paired logs')
                rows = [json.loads(line) for line in lines]; count += 1
                require(all(row['update'] == count and row['seed'] == seed for row in rows), 'Training row identity mismatch')
                for key in ('world_uniforms_sha256', 'sample_uniforms_sha256', 'entropy_coefficient'):
                    require(len({row[key] for row in rows}) == 1, 'Paired stream differs: ' + key)
                require(len({row['batch_states_sha256'] for row in rows}) == 1, 'World states differ across arms')
            require(count == CONFIG['updates'], 'Training update count mismatch')
        finally:
            for stream in streams:
                stream.close()


def execute(out):
    out = Path(out).resolve(); plan, static = verify(out); execution = out / 'execution'
    require(not execution.exists(), 'Never overwrite execution'); execution.mkdir()
    write(execution / 'started.json', dict(started_at=now(), pid=os.getpid(), plan_sha256=sha(out / 'plan.json'),
                                          runtime=dict(python=platform.python_version(), numpy=np.__version__)))
    start = time.perf_counter()
    try:
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(len(SEEDS)) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in SEEDS])
        runs = [run for group in groups for run in group]
        expected = [(seed, condition) for seed in SEEDS for condition in CONDITIONS]
        require([(r['seed'], r['condition']) for r in runs] == expected, 'Noncanonical run order')
        verify_pairing(execution, runs)
        arrays = [read(execution / f'seed_{seed}_arrays.json')['array_hashes'] for seed in SEEDS]
        require(all(x == arrays[0] for x in arrays), 'Worker array hashes differ')
        result = dict(status='completed', completed_at=now(), elapsed_seconds=time.perf_counter() - start,
                      plan_sha256=sha(out / 'plan.json'), budget=static['budget'], runs=runs,
                      array_hashes=arrays[0], primary=metrics.primary(runs),
                      experiment_type='discrete_symbol_message_co_learning_partner_switch_payoff_ecology',
                      language_claim_automatically_supported=False)
        write(execution / 'results.json', result)
        write(execution / 'status.json', dict(status='completed', at=now(), results_sha256=sha(execution / 'results.json')))
        return dict(status='completed', output=str(execution), elapsed_seconds=result['elapsed_seconds'], primary=result['primary'])
    except BaseException as error:
        write(execution / 'failure.json', dict(status='failed', at=now(), error=repr(error), elapsed_seconds=time.perf_counter() - start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'verify', 'execute')); parser.add_argument('--out', required=True)
    args = parser.parse_args()
    answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
