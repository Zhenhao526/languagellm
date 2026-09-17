"""Train and evaluate the random partner-rematching confirmation study."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import shutil
import time
from copy import deepcopy
from itertools import zip_longest
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

from research_program.triadic_rematched_partner_confirmation_study import design, remap
from research_program.triadic_rematched_partner_confirmation_study import metrics
from research_program.triadic_partner_holdout_confirmation_study import runner as previous
from research_program.triadic_message_study import runner as core

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS, SCHEDULES, CONDITIONS, PARTS = design.SEEDS, design.SCHEDULES, design.CONDITIONS, design.PARTS
STEPS = design.UPDATES

CONFIG = dict(
    updates=6000, batch_size=256, checkpoints=list(STEPS), features=54, dtype='float64',
    learning_rate=0.001, global_gradient_clip=5.0, entropy_initial=0.001,
    entropy_zero_after_updates=1000, sender_entropy_coefficient=0, trajectories_per_state=2,
    sender_windows=2, sender_tokens_per_window=4, alphabet_size=8, actions_per_actor=17,
    worker_count=4, multiprocessing_start_method='spawn', blas_threads_per_worker=1,
    dimensions={'sender1': [54, 64, 64, 32], 'sender2': [153, 64, 64, 32], 'action': [252, 64, 64, 17]},
    seeds=list(SEEDS), schedules=list(SCHEDULES), payoffs=['partial'], rules=['strict'], lives=['live', 'silent'],
    conditions=list(CONDITIONS), partitions=list(PARTS),
    task='random_partner_rematching_social_relation_generalization',
    heldout_pair_rotation='(seed-66301) mod 3 over AB,AC,BC',
    rematching='rematched: independent uniform six-way permutation of need ownership for every training batch world; static: canonical ownership',
    training_objective='mean_log_expected_payoff_plus_original_action_entropy; sender_two_trajectory_LOO',
    payoff_intervention='partial: native 0/.5/1 retained; no payoff factor in this isolation batch',
    observation='PL_own_need_and_public_layout_only; other_need_blocks_and_FI_flag_zero',
    primary='heldout_pair_target_actor_action_rematched_by_communication_interaction_centered_time_AUC',
    pairing='same initial parameters, world uniforms, batch indices and message uniforms across four arms within each seed',
    evaluation='complete heldout pair on train layouts at six checkpoints; new-layout heldout pair and seen-pair controls at final',
    selection='fixed16 new seeds ×2 exposure schedules ×2 communication channels ×6000 updates; no early stopping',
    automatic_followon_experiment=False,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, 'Unknown seed or condition')
    return f'seed_{seed}_{condition}'


def sources():
    paths = [HERE / n for n in ('__init__.py', 'design.py', 'remap.py', 'metrics.py', 'runner.py', 'plan.md', 'README.md', 'tests/test_design.py')]
    paths += [Path(previous.__file__), Path(previous.core.__file__), Path(previous.dataset.__file__),
              Path(previous.environment.__file__), Path(previous.kernel.__file__), Path(previous.base_runner.cases.__file__),
              Path(previous.base_runner.factor_cases.__file__)]
    require(all(path.is_file() for path in paths), 'Missing frozen source')
    return {str(path.resolve().relative_to(ROOT)): core.base.sha(path) for path in paths}


def _budget():
    runs = len(SEEDS) * len(CONDITIONS)
    target_worlds = 1792 * 18 * 6
    compact = runs * len(STEPS) * target_worlds
    final = runs * ((1792 * 6 * 6) + (3584 * 6 * 6))
    train_forward = runs * CONFIG['updates'] * CONFIG['batch_size'] * 2 * 9
    eval_forward = (compact + final) * 9
    return dict(runs=runs, independent_seed_blocks=len(SEEDS), schedules=len(SCHEDULES), channels=2,
                training_updates=runs * CONFIG['updates'], training_world_samples=runs * CONFIG['updates'] * CONFIG['batch_size'],
                message_trajectories=runs * CONFIG['updates'] * CONFIG['batch_size'] * 2,
                training_forward_module_samples=train_forward, checkpoints=runs * len(STEPS), checkpoint_files=runs * len(STEPS),
                compact_checkpoint_worlds=compact, final_control_worlds=final,
                compact_evaluation_forward_module_samples=eval_forward,
                total_forward_module_samples=train_forward + eval_forward,
                heldout_pair_worlds_per_checkpoint=target_worlds)


def prepared():
    static = design.make_prepared(); static['config'] = deepcopy(CONFIG)
    static['source_sha256'] = sources(); static['runtime'] = dict(python=platform.python_version(), numpy=np.__version__)
    static['budget'] = _budget(); return static


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Never overwrite preparation')
    static = prepared(); out.mkdir(parents=True)
    for relative in static['source_sha256']:
        target = out / 'source_snapshot' / relative; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    (out / 'prepared.json').write_text(json.dumps(static, ensure_ascii=False, indent=2) + '\n')
    plan = dict(status='prepared_without_training', created_at=core.base.now(), config=CONFIG,
                source_sha256=static['source_sha256'], prepared_sha256=core.base.sha(out / 'prepared.json'),
                runtime=static['runtime'], no_model_calls=True, no_training_updates=True)
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    (out / 'freeze.json').write_text(json.dumps(dict(plan_sha256=core.base.sha(out / 'plan.json'),
                                                       prepared_sha256=core.base.sha(out / 'prepared.json')), ensure_ascii=False, indent=2) + '\n')
    verify(out); return dict(status='prepared_without_training', output=str(out), plan_sha256=core.base.sha(out / 'plan.json'), budget=static['budget'])


def verify(out):
    out = Path(out).resolve()
    plan = json.loads((out / 'plan.json').read_text()); static = json.loads((out / 'prepared.json').read_text()); freeze = json.loads((out / 'freeze.json').read_text())
    require(core.base.sha(out / 'plan.json') == freeze['plan_sha256'] and core.base.sha(out / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Frozen hash mismatch')
    require(static == prepared() and plan['source_sha256'] == sources() and plan['config'] == CONFIG, 'Current source/config differs from frozen preparation')
    require(plan['runtime'] == dict(python=platform.python_version(), numpy=np.__version__), 'Runtime differs')
    for relative, digest in plan['source_sha256'].items():
        require(core.base.sha(out / 'source_snapshot' / relative) == digest, 'Source snapshot changed: ' + relative)
    return plan, static


def make_arrays(spec):
    return previous.make_arrays(spec)


def parameter_hash(networks):
    return previous.parameter_hash(networks)


def evaluate_compact(networks, arrays, spec, payoff, rule, live):
    return previous.evaluate_compact(networks, arrays, spec, payoff, rule, live)


def train_run(seed, condition, static, arrays, execution):
    schedule, payoff, rule, live = design.parse_condition(condition)
    directory = Path(execution) / name(seed, condition); directory.mkdir(exist_ok=False)
    networks = core.make_networks(seed); initial = parameter_hash(networks); optimizer = core.base.make_adam(networks)
    train_spec = static['seed_partitions'][str(seed)]['train']; target_spec = static['seed_partitions'][str(seed)]['heldout_pair']
    trajectory = []; started = time.perf_counter(); rematch_rng = np.random.default_rng(np.random.SeedSequence([seed, 700]))
    world_rng = np.random.default_rng(np.random.SeedSequence([seed, 200])); message_rngs = core.make_message_rngs(seed)
    rematch_counts = np.zeros(6, dtype=np.int64)

    def checkpoint(step):
        checkpoint_path = directory / f'checkpoint_{step:04d}.npz'
        checkpoint_sha = core.save_checkpoint(checkpoint_path, networks, optimizer, step, world_rng, message_rngs)
        evaluated = evaluate_compact(networks, arrays['heldout_pair'], target_spec, payoff, rule, live); evaluated['update'] = step
        row = dict(update=step, checkpoint_path=str(checkpoint_path), checkpoint_sha256=checkpoint_sha,
                   target_trajectory={key: value for key, value in evaluated.items() if key != 'update'},
                   compact_worlds=target_spec['world_count'], forward_module_samples=9 * target_spec['world_count'],
                   elapsed_seconds=time.perf_counter() - started)
        trajectory.append(row)
        with (directory / 'trajectory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(core.base.json_bytes(row).decode())

    checkpoint(0)
    with (directory / 'training.jsonl').open('x', encoding='utf-8') as stream:
        for update in range(1, CONFIG['updates'] + 1):
            world_uniforms = world_rng.random((CONFIG['batch_size'], 3)); ids = design.sample_indices(train_spec, world_uniforms)
            message_uniforms = core.draw_uniforms(message_rngs, CONFIG['batch_size'])
            x_batch = arrays['train']['x_PL'][ids]; reward_batch = arrays['train']['rewards_partial'][ids]
            packed_batch = arrays['train']['packed_states'][ids]
            if schedule == 'rematched':
                permutation_ids = remap.permutation_indices(rematch_rng, CONFIG['batch_size'])
                x_batch, reward_batch, packed_batch = remap.remap_batch(x_batch, reward_batch, packed_batch, permutation_ids)
                rematch_counts += np.bincount(permutation_ids, minlength=6)
            else:
                permutation_ids = np.zeros(CONFIG['batch_size'], dtype=np.int8)
            gradients, row = previous.kernel.training_gradients(networks, x_batch, reward_batch, live, message_uniforms, update, rule)
            norm, scale = core.base.adam_step(networks, gradients, optimizer, update)
            row.update(update=update, seed=seed, condition=condition, schedule=schedule, payoff=payoff, rule=rule,
                       world_uniforms_sha256=core.base.array_sha(world_uniforms), batch_indices_sha256=core.base.array_sha(ids),
                       batch_states_sha256=core.base.array_sha(packed_batch), sample_uniforms_sha256=core.base.array_sha(message_uniforms),
                       permutation_indices_sha256=core.base.array_sha(permutation_ids), rematched=schedule == 'rematched',
                       gradient_norm=norm, gradient_clip_scale=scale, elapsed_seconds=time.perf_counter() - started)
            stream.write(core.base.json_bytes(row).decode())
            if update in STEPS:
                stream.flush(); checkpoint(update)
    held_new = evaluate_compact(networks, arrays['heldout_pair_new_layout'], static['seed_partitions'][str(seed)]['heldout_pair_new_layout'], payoff, rule, live)
    seen_new = evaluate_compact(networks, arrays['seen_control_new_layout'], static['seed_partitions'][str(seed)]['seen_control_new_layout'], payoff, rule, live)
    result = dict(seed=seed, heldout_pair=design.heldout_pair_for_seed(seed), heldout_pair_name=design.PAIR_NAMES[design.heldout_pair_for_seed(seed)],
                  schedule=schedule, condition=condition, payoff=payoff, rule=rule, live=live, updates=CONFIG['updates'],
                  initial_parameter_sha256=initial, final_parameter_sha256=parameter_hash(networks),
                  final_checkpoint_sha256=core.base.sha(directory / 'checkpoint_6000.npz'), training_log_sha256=core.base.sha(directory / 'training.jsonl'),
                  trajectory=trajectory, final={'heldout_pair_new_layout': held_new, 'seen_control_new_layout': seen_new},
                  rematch_histogram=rematch_counts.tolist(), elapsed_seconds=time.perf_counter() - started)
    (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (directory / 'status.json').write_text(json.dumps(dict(status='completed', seed=seed, condition=condition,
                                                            result_sha256=core.base.sha(directory / 'result.json'), at=core.base.now()), ensure_ascii=False, indent=2) + '\n')
    return result


def worker(payload):
    seed, static, execution = payload; execution = Path(execution); specs = static['seed_partitions'][str(seed)]
    arrays = {part: make_arrays(specs[part]) for part in PARTS}
    keys = ('packed_states', 'native_rewards', 'x_PL', 'rewards_partial')
    hashes_before = {part: {key: core.base.array_sha(arrays[part][key]) for key in keys} for part in PARTS}
    (execution / f'seed_{seed}_arrays.json').write_text(json.dumps({'array_hashes': hashes_before}, ensure_ascii=False, indent=2) + '\n')
    runs = []
    for condition in CONDITIONS:
        runs.append(train_run(seed, condition, static, arrays, execution))
        print(json.dumps(dict(completed_run=name(seed, condition), elapsed_seconds=runs[-1]['elapsed_seconds'])), flush=True)
    hashes_after = {part: {key: core.base.array_sha(arrays[part][key]) for key in keys} for part in PARTS}
    require(hashes_before == hashes_after, 'Arrays mutated')
    return runs


def verify_pairing(execution, runs):
    execution = Path(execution); by = {(r['seed'], r['condition']): r for r in runs}
    expected = {(seed, condition) for seed in SEEDS for condition in CONDITIONS}
    require(set(by) == expected and len(by) == 64, 'Incomplete paired grid')
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
                for key in ('world_uniforms_sha256', 'sample_uniforms_sha256', 'batch_indices_sha256', 'entropy_coefficient'):
                    require(len({row[key] for row in rows}) == 1, 'Paired stream differs: ' + key)
            require(count == CONFIG['updates'], 'Training update count mismatch')
        finally:
            for stream in streams:
                stream.close()


def measured_budget(runs):
    require(len(runs) == 64, 'Expected 64 runs')
    trajectory = [row for run in runs for row in run['trajectory']]; final = [value for run in runs for value in run['final'].values()]
    require(all([row['update'] for row in run['trajectory']] == list(STEPS) for run in runs), 'Incomplete trajectory')
    return dict(training_forward_module_samples=sum(run['updates'] for run in runs) * CONFIG['batch_size'] * 2 * 9,
                compact_evaluation_forward_module_samples=sum(row['forward_module_samples'] for row in trajectory) + sum(9 * row['worlds'] for row in final),
                compact_evaluation_records=len(trajectory) + len(final), compact_evaluation_worlds=sum(row['compact_worlds'] for row in trajectory) + sum(row['worlds'] for row in final),
                checkpoints=len(trajectory), final_control_records=len(final))


def execute(out):
    out = Path(out).resolve(); plan, static = verify(out); execution = out / 'execution'; require(not execution.exists(), 'Never overwrite execution'); execution.mkdir()
    started = time.perf_counter(); (execution / 'started.json').write_text(json.dumps(dict(started_at=core.base.now(), pid=os.getpid(), plan_sha256=core.base.sha(out / 'plan.json')), ensure_ascii=False, indent=2) + '\n')
    try:
        with multiprocessing.get_context('spawn').Pool(CONFIG['worker_count']) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in SEEDS])
        runs = [run for group in groups for run in group]
        require([(r['seed'], r['condition']) for r in runs] == [(seed, condition) for seed in SEEDS for condition in CONDITIONS], 'Noncanonical run order')
        verify_pairing(execution, runs); measured = measured_budget(runs)
        result = dict(status='completed', completed_at=core.base.now(), elapsed_seconds=time.perf_counter() - started,
                      plan_sha256=core.base.sha(out / 'plan.json'), budget=static['budget'], measured_budget=measured,
                      runs=runs, primary=metrics.summarize(runs), experiment_type='random_partner_rematching_confirmation',
                      language_claim_automatically_supported=False)
        (execution / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        (execution / 'status.json').write_text(json.dumps(dict(status='completed', at=core.base.now(), results_sha256=core.base.sha(execution / 'results.json')), ensure_ascii=False, indent=2) + '\n')
        return dict(status='completed', output=str(execution), elapsed_seconds=result['elapsed_seconds'], primary=result['primary'])
    except BaseException as error:
        (execution / 'failure.json').write_text(json.dumps(dict(status='failed', at=core.base.now(), error=repr(error), elapsed_seconds=time.perf_counter() - started), ensure_ascii=False, indent=2) + '\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'verify', 'execute')); parser.add_argument('--out', required=True)
    args = parser.parse_args(); answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
