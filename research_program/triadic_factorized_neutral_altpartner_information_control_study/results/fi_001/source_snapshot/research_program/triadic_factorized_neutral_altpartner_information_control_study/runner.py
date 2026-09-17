"""Train and evaluate factorized neutral/engage policies with partner choice."""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import platform
import shutil
import time
from copy import deepcopy
from itertools import zip_longest, combinations
from pathlib import Path

for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import numpy as np

from research_program.triadic_action_dependency_study import dataset
from research_program.triadic_reciprocal_execution_study import environment
from research_program.triadic_message_study import runner as core
from . import design, kernel, remap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS, SCHEDULES, CONDITIONS, PARTS = design.SEEDS, design.SCHEDULES, design.CONDITIONS, design.PARTS
STEPS = design.UPDATES
MODULES = ('sender1', 'sender2', 'action')
DIMS = {'sender1': (54, 64, 64, 32), 'sender2': (153, 64, 64, 32), 'action': (252, 64, 64, 18)}

CONFIG = dict(
    updates=6000, batch_size=256, checkpoints=list(STEPS), features=54, dtype='float64', learning_rate=0.001,
    global_gradient_clip=5.0, entropy_initial=0.001, entropy_zero_after_updates=1000,
    sender_entropy_coefficient=0, trajectories_per_state=2, sender_windows=2, sender_tokens_per_window=4,
    alphabet_size=8, proposal_count=16, intent_count=2, actions_per_actor=17, action_logits=18,
    worker_count=4, evaluation_batch_size=8192, multiprocessing_start_method='spawn', blas_threads_per_worker=1,
    dimensions={k: list(v) for k, v in DIMS.items()}, seeds=list(SEEDS), schedules=list(SCHEDULES),
    payoffs=['altpair_factorized'], rules=['strict'], information=['FI'], lives=['live', 'silent'], conditions=list(CONDITIONS),
    partitions=list(PARTS), task='alternative_partner_coordination_with_explicit_neutral_action',
    training_objective='mean_log_expected_reward_over_all_full_and_partial_legal_plans; sender_two_trajectory_LOO',
    observation='FI_all_three_needs_and_public_layout; no_cross_agent_message_channel',
    action_factorization='intent softmax [neutral, engage] plus proposal softmax [16 site-destination-partner choices]; proposal ignored when neutral',
    pairing='same initial parameters, world uniforms, batch indices and message uniforms within each seed; rematched FI live/silent share permutation assignments',
    evaluation='two-alternative-partner worlds on train layouts at six checkpoints; new layouts at final',
    primary='FI_rematched_by_communication_interaction_on_target_pair_legal_rate_conditional_on_physical_execution_centered_time_AUC',
    selection='fixed16 new seeds ×2 schedules ×2 channels ×6000 updates; no early stopping',
    automatic_followon_experiment=False,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, 'Unknown seed or condition')
    return f'seed_{seed}_{condition}'


def make_network(seed, dimensions):
    rng = np.random.default_rng(seed)
    result = {}
    for layer, (left, right) in enumerate(zip(dimensions, dimensions[1:]), 1):
        result[f'W{layer}'] = rng.normal(0, math.sqrt(2 / (left + right)), (left, right))
        result[f'b{layer}'] = np.zeros(right, dtype=np.float64)
    return result


def make_networks(seed):
    networks = [make_network(np.random.SeedSequence([seed, a, m, 100]), DIMS[module])
                for a in range(3) for m, module in enumerate(MODULES)]
    for i, j in combinations(range(9), 2):
        require(all(not np.shares_memory(networks[i][k], networks[j][k]) for k in networks[i]),
                'Network parameters are shared')
    return networks


def parameter_hash(networks):
    return core.base.json_hash({f'agent{a}_{module}_{key}': core.base.array_sha(value)
        for a in range(3) for m, module in enumerate(MODULES)
        for key, value in networks[3 * a + m].items()})


def save_checkpoint(path, networks, optimizer, update, world_rng, message_rngs, rematch_rng):
    require(not Path(path).exists(), 'Checkpoint exists')
    payload = {f'agent{a}_{module}_{key}': value
               for a in range(3) for m, module in enumerate(MODULES)
               for key, value in networks[3 * a + m].items()}
    for a in range(3):
        for m, module in enumerate(MODULES):
            for moment in ('m', 'v'):
                payload.update({f'adam_agent{a}_{module}_{moment}_{key}': value
                                for key, value in optimizer[3 * a + m][moment].items()})
    payload['update'] = np.array(update, dtype=np.int64)
    payload['world_rng_json'] = np.array(json.dumps(world_rng.bit_generator.state, sort_keys=True))
    payload['rematch_rng_json'] = np.array(json.dumps(rematch_rng.bit_generator.state, sort_keys=True))
    payload['message_rngs_json'] = np.array(json.dumps({key: rng.bit_generator.state for key, rng in message_rngs.items()}, sort_keys=True))
    np.savez_compressed(path, **payload)
    return core.base.sha(path)


def load_networks(path):
    networks = []
    with np.load(path, allow_pickle=False) as saved:
        for a in range(3):
            for module in MODULES:
                dimensions = DIMS[module]; network = {}
                for layer, (left, right) in enumerate(zip(dimensions, dimensions[1:]), 1):
                    for key, shape in ((f'W{layer}', (left, right)), (f'b{layer}', (right,))):
                        value = saved[f'agent{a}_{module}_{key}'].copy()
                        require(value.shape == shape and value.dtype == np.float64, 'Saved network shape/dtype mismatch')
                        core.base.finite(value, 'Loaded parameter'); network[key] = value
                networks.append(network)
    return networks


def make_arrays(spec):
    arrays = dataset.make_arrays(spec, information='FI')
    arrays['native_rewards'] = arrays['rewards'].copy()
    require(arrays['x_FI'].shape == (spec['world_count'], 3, 54), 'Invalid FI feature shape')
    require(np.all(arrays['x_FI'][:, :, 53] == 1), 'Full-information flag missing from FI')
    require(np.all(arrays['x_FI'][:, :, :21].sum(axis=-1) > 0), 'FI need blocks missing')
    require(np.all((arrays['native_rewards'] == 1).sum(axis=1) == 2), 'Every world must have two full plans')
    return arrays


def _event_pair_mask(full):
    # Full is [B,24]. Each block of eight events belongs to one partner pair.
    return full.reshape(len(full), 3, 8).any(axis=2)


def _greedy_factorized(trace):
    intent_p, intent_lp, proposal_p, proposal_lp = kernel.factorized_distribution(trace['action_logits'])
    intent = intent_p.argmax(axis=-1).astype(np.int16)  # 0 neutral, 1 engage
    proposal = proposal_p.argmax(axis=-1).astype(np.int16)  # 0..15
    actions = np.where(intent == 1, proposal + 1, 0).astype(np.int16)
    return intent_p, proposal_p, intent, proposal, actions


def evaluate_factorized(networks, arrays, spec, information, live):
    n = int(spec['world_count']); ids = np.arange(n, dtype=np.int64)
    engagement_count = neutral_count = physical_count = q_count = pair_legal_count = 0
    proposal_legal_count = proposal_legal_denominator = 0
    third_neutral_count = third_neutral_denominator = 0
    pair_counts = np.zeros(3, dtype=np.int64); plan_counts = np.zeros(24, dtype=np.int64)
    engagement_actor_hits = np.zeros(3, dtype=np.int64); engagement_actor_total = 0
    expected_sum = full_prob_sum = partial_prob_sum = 0.0
    for start in range(0, n, CONFIG['evaluation_batch_size']):
        stop = min(start + CONFIG['evaluation_batch_size'], n); ix = ids[start:stop]
        states = arrays['packed_states'][ix]; full = arrays['native_rewards'][ix] == 1.
        trace = core.rollout(networks, arrays['x_' + information][ix], bool(live))
        terms = kernel.objective_terms(trace['action_logits'], arrays['native_rewards'][ix])
        expected_sum += float(terms['J'].sum()); full_prob_sum += float(terms['full_success_probability'].sum()); partial_prob_sum += float(terms['partial_success_probability'].sum())
        _, _, intent, proposal, actions = _greedy_factorized(trace)
        physical = environment.settle(states, actions, 'strict')
        pair_targets = _event_pair_mask(full)
        engagement_count += int((intent == 1).sum()); neutral_count += int((intent == 0).sum())
        engagement_actor_hits += (intent == 1).sum(axis=0, dtype=np.int64); engagement_actor_total += len(ix)
        legal_actions = np.zeros((len(ix), 3, 17), dtype=bool)
        for event in range(24):
            for actor in range(3):
                legal_actions[:, actor, core.base.JOINT_ACTIONS[event, actor]] |= full[:, event]
        proposal_legal_count += int(legal_actions[np.arange(len(ix))[:, None], np.arange(3)[None, :], actions].astype(bool)[intent == 1].sum())
        proposal_legal_denominator += int((intent == 1).sum())
        executed = physical['executed']; physical_ok = np.any(executed, axis=1)
        actual_pair = physical['actual_pair_index'].astype(np.int64); physical_count += int(physical_ok.sum())
        safe_pair = np.maximum(actual_pair, 0)
        executed_event = safe_pair * 8 + np.maximum(physical['executed_site'], 0).astype(np.int64) * 2 + np.maximum(physical['executed_destination'], 0).astype(np.int64)
        pair_legal = physical_ok & pair_targets[np.arange(len(ix)), safe_pair]
        q_hits = physical_ok & full[np.arange(len(ix)), executed_event]
        pair_legal_count += int(pair_legal.sum()); q_count += int(q_hits.sum())
        if physical_ok.any():
            pair_counts += np.bincount(actual_pair[physical_ok], minlength=3).astype(np.int64)
            plan_counts += np.bincount(executed_event[physical_ok], minlength=24).astype(np.int64)
            third_by_pair = np.asarray([2, 1, 0], dtype=np.int64)
            third_neutral_count += int((intent[np.flatnonzero(physical_ok), third_by_pair[actual_pair[physical_ok]]] == 0).sum())
            third_neutral_denominator += int(physical_ok.sum())
    physical_rate = physical_count / n
    return dict(
        value=float(q_count / n), q_rate=float(q_count / n), conditional_q_rate=float(q_count / physical_count) if physical_count else 0.0,
        target_pair_legal_rate=float(pair_legal_count / physical_count) if physical_count else 0.0,
        proposal_legal_rate=float(proposal_legal_count / proposal_legal_denominator) if proposal_legal_denominator else 0.0,
        physical_execution_rate=physical_rate, engagement_rate=float(engagement_count / (3 * n)), neutral_rate=float(neutral_count / (3 * n)),
        actor_engagement_rates=[float(x / max(engagement_actor_total, 1)) for x in engagement_actor_hits],
        third_agent_neutral_rate=float(third_neutral_count / third_neutral_denominator) if third_neutral_denominator else 0.0,
        legal_plan_selection_counts=plan_counts.tolist(), legal_pair_selection_counts=pair_counts.tolist(),
        worlds=n, physical_worlds=int(physical_count), proposal_legal_denominator=int(proposal_legal_denominator),
        conditional_q_denominator_worlds=int(physical_count), conditional_q_numerator_worlds=int(q_count),
        target_pair_denominator_worlds=int(physical_count), target_pair_numerator_worlds=int(pair_legal_count),
        exact_expected_reward_mean=float(expected_sum / n), exact_full_success_probability_mean=float(full_prob_sum / n),
        exact_partial_success_probability_mean=float(partial_prob_sum / n),
        legal_plan_count=int(full.sum(axis=1).min()), legal_pair_count=2, partition=spec['partition'], live=bool(live),
        action_factorization='intent:neutral/engage; proposal:16-way conditional on engage',
        information=information,
    )


def train_run(seed, condition, static, arrays, execution):
    schedule, payoff, rule, information, live = design.parse_condition(condition)
    directory = Path(execution) / name(seed, condition); directory.mkdir(exist_ok=False)
    networks = make_networks(seed); initial = parameter_hash(networks); optimizer = core.base.make_adam(networks)
    spec = static['partitions']['train']; trajectory = []; started = time.perf_counter()
    world_rng = np.random.default_rng(np.random.SeedSequence([seed, 200]))
    message_rngs = core.make_message_rngs(seed); rematch_rng = np.random.default_rng(np.random.SeedSequence([seed, 700]))
    rematch_counts = np.zeros(6, dtype=np.int64)

    def checkpoint(step):
        checkpoint_path = directory / f'checkpoint_{step:04d}.npz'
        digest = save_checkpoint(checkpoint_path, networks, optimizer, step, world_rng, message_rngs, rematch_rng)
        evaluated = evaluate_factorized(networks, arrays['train'], spec, information, live); evaluated['update'] = step
        row = dict(update=step, checkpoint_path=str(checkpoint_path), checkpoint_sha256=digest,
                   target_trajectory={k: v for k, v in evaluated.items() if k != 'update'},
                   compact_worlds=spec['world_count'], forward_module_samples=9 * spec['world_count'],
                   elapsed_seconds=time.perf_counter() - started)
        trajectory.append(row)
        with (directory / 'trajectory.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(core.base.json_bytes(row).decode())

    checkpoint(0)
    with (directory / 'training.jsonl').open('x', encoding='utf-8') as stream:
        for update in range(1, CONFIG['updates'] + 1):
            world_uniforms = world_rng.random((CONFIG['batch_size'], 3)); ids = design.sample_indices(spec, world_uniforms)
            message_uniforms = core.draw_uniforms(message_rngs, CONFIG['batch_size'])
            canonical_x = arrays['train']['x_' + information][ids]; canonical_rewards = arrays['train']['rewards'][ids]; canonical_states = arrays['train']['packed_states'][ids]
            if schedule == 'rematched':
                permutation_ids = remap.permutation_indices(rematch_rng, CONFIG['batch_size'])
                x_batch, reward_batch, effective_states = remap.remap_batch(canonical_x, canonical_rewards, canonical_states, permutation_ids)
                rematch_counts += np.bincount(permutation_ids, minlength=6)
            else:
                permutation_ids = np.zeros(CONFIG['batch_size'], dtype=np.int8)
                x_batch, reward_batch, effective_states = canonical_x, canonical_rewards, canonical_states
            gradients, row = kernel.training_gradients(networks, x_batch, reward_batch, live, message_uniforms, update)
            norm, scale = core.base.adam_step(networks, gradients, optimizer, update)
            row.update(update=update, seed=seed, condition=condition, schedule=schedule, payoff=payoff, rule=rule,
                       world_uniforms_sha256=core.base.array_sha(world_uniforms), batch_indices_sha256=core.base.array_sha(ids),
                       canonical_batch_states_sha256=core.base.array_sha(canonical_states), effective_batch_states_sha256=core.base.array_sha(effective_states),
                       sample_uniforms_sha256=core.base.array_sha(message_uniforms), permutation_indices_sha256=core.base.array_sha(permutation_ids),
                       rematched=schedule == 'rematched', gradient_norm=norm, gradient_clip_scale=scale,
                       elapsed_seconds=time.perf_counter() - started)
            stream.write(core.base.json_bytes(row).decode())
            if update in STEPS:
                stream.flush(); checkpoint(update)
    final = evaluate_factorized(networks, arrays['new_layouts'], static['partitions']['new_layouts'], information, live)
    result = dict(seed=seed, schedule=schedule, condition=condition, payoff=payoff, rule=rule,
                  information=information, live=live, updates=CONFIG['updates'],
                  initial_parameter_sha256=initial, final_parameter_sha256=parameter_hash(networks),
                  final_checkpoint_sha256=core.base.sha(directory / 'checkpoint_6000.npz'), training_log_sha256=core.base.sha(directory / 'training.jsonl'),
                  trajectory=trajectory, final={'new_layouts': final}, rematch_histogram=rematch_counts.tolist(),
                  rematch_assignments=int(rematch_counts.sum()), elapsed_seconds=time.perf_counter() - started)
    (directory / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (directory / 'status.json').write_text(json.dumps(dict(status='completed', seed=seed, condition=condition,
                                                            result_sha256=core.base.sha(directory / 'result.json'), at=core.base.now()), ensure_ascii=False, indent=2) + '\n')
    return result


def sources():
    paths = [HERE / n for n in ('__init__.py', 'design.py', 'kernel.py', 'runner.py', 'remap.py', 'metrics.py', 'plan.md', 'README.md', 'tests/test_design.py')]
    paths += [Path(dataset.__file__), Path(environment.__file__), Path(core.__file__), Path(core.base.__file__)]
    require(all(path.is_file() for path in paths), 'Missing frozen source')
    return {str(path.resolve().relative_to(ROOT)): core.base.sha(path) for path in paths}


def _budget():
    runs = len(SEEDS) * len(CONDITIONS); target = 1560 * 18 * 6; final = 1560 * 6 * 6
    train_forward = runs * CONFIG['updates'] * CONFIG['batch_size'] * 2 * 9
    eval_forward = (runs * len(STEPS) * target + runs * final) * 9
    return dict(runs=runs, independent_seed_blocks=len(SEEDS), schedules=2, channels=2,
                training_updates=runs * CONFIG['updates'], training_world_samples=runs * CONFIG['updates'] * CONFIG['batch_size'],
                message_trajectories=runs * CONFIG['updates'] * CONFIG['batch_size'] * 2, training_forward_module_samples=train_forward,
                checkpoints=runs * len(STEPS), checkpoint_files=runs * len(STEPS), compact_checkpoint_worlds=runs * len(STEPS) * target,
                final_control_worlds=runs * final, compact_evaluation_forward_module_samples=eval_forward,
                total_forward_module_samples=train_forward + eval_forward, two_alternative_partner_plan_worlds_per_checkpoint=target)


def prepared():
    static = design.make_prepared(); static['config'] = deepcopy(CONFIG); static['source_sha256'] = sources()
    static['runtime'] = dict(python=platform.python_version(), numpy=np.__version__); static['budget'] = _budget(); return static


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Never overwrite preparation')
    static = prepared(); out.mkdir(parents=True)
    for relative in static['source_sha256']:
        target = out / 'source_snapshot' / relative; target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT / relative, target)
    (out / 'prepared.json').write_text(json.dumps(static, ensure_ascii=False, indent=2) + '\n')
    plan = dict(status='prepared_without_training', created_at=core.base.now(), config=CONFIG, source_sha256=static['source_sha256'],
                prepared_sha256=core.base.sha(out / 'prepared.json'), runtime=static['runtime'], no_model_calls=True, no_training_updates=True)
    (out / 'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2) + '\n')
    (out / 'freeze.json').write_text(json.dumps(dict(plan_sha256=core.base.sha(out / 'plan.json'), prepared_sha256=core.base.sha(out / 'prepared.json')), ensure_ascii=False, indent=2) + '\n')
    verify(out); return dict(status='prepared_without_training', output=str(out), plan_sha256=core.base.sha(out / 'plan.json'), budget=static['budget'])


def verify(out):
    out = Path(out).resolve(); plan = json.loads((out / 'plan.json').read_text()); static = json.loads((out / 'prepared.json').read_text()); freeze = json.loads((out / 'freeze.json').read_text())
    require(core.base.sha(out / 'plan.json') == freeze['plan_sha256'] and core.base.sha(out / 'prepared.json') == freeze['prepared_sha256'] == plan['prepared_sha256'], 'Frozen hash mismatch')
    require(static == prepared() and plan['source_sha256'] == sources() and plan['config'] == CONFIG, 'Current source/config differs from frozen preparation')
    require(plan['runtime'] == dict(python=platform.python_version(), numpy=np.__version__), 'Runtime differs')
    for relative, digest in plan['source_sha256'].items(): require(core.base.sha(out / 'source_snapshot' / relative) == digest, 'Source snapshot changed: ' + relative)
    return plan, static


def worker(payload):
    seed, static, execution = payload; execution = Path(execution); arrays = {part: make_arrays(static['partitions'][part]) for part in PARTS}
    keys = ('packed_states', 'native_rewards', 'x_FI')
    before = {part: {key: core.base.array_sha(arrays[part][key]) for key in keys} for part in PARTS}
    (execution / f'seed_{seed}_arrays.json').write_text(json.dumps({'array_hashes': before}, ensure_ascii=False, indent=2) + '\n')
    runs = []
    for condition in CONDITIONS:
        runs.append(train_run(seed, condition, static, arrays, execution)); print(json.dumps(dict(completed_run=name(seed, condition), elapsed_seconds=runs[-1]['elapsed_seconds'])), flush=True)
    after = {part: {key: core.base.array_sha(arrays[part][key]) for key in keys} for part in PARTS}; require(before == after, 'Arrays mutated'); return runs


def verify_pairing(execution, runs):
    execution = Path(execution); by = {(r['seed'], r['condition']): r for r in runs}; expected = {(seed, condition) for seed in SEEDS for condition in CONDITIONS}
    require(set(by) == expected and len(by) == 64, 'Incomplete paired grid')
    for seed in SEEDS:
        selected = [by[seed, condition] for condition in CONDITIONS]; require(len({r['initial_parameter_sha256'] for r in selected}) == 1, 'Unpaired initialization')
        streams = [(execution / name(seed, condition) / 'training.jsonl').open() for condition in CONDITIONS]
        try:
            count = 0
            for lines in zip_longest(*streams):
                require(all(line is not None for line in lines), 'Unequal paired logs'); rows = [json.loads(line) for line in lines]; count += 1
                require(all(row['update'] == count and row['seed'] == seed for row in rows), 'Training identity mismatch')
                for key in ('world_uniforms_sha256', 'sample_uniforms_sha256', 'batch_indices_sha256', 'canonical_batch_states_sha256', 'entropy_coefficient'):
                    require(len({row[key] for row in rows}) == 1, 'Paired stream differs: ' + key)
                for schedule in SCHEDULES:
                    group = [row for row in rows if row['schedule'] == schedule]
                    require(len({row['effective_batch_states_sha256'] for row in group}) == 1 and len({row['permutation_indices_sha256'] for row in group}) == 1, 'Live/silent rematch stream differs')
            require(count == CONFIG['updates'], 'Training update count mismatch')
        finally:
            for stream in streams: stream.close()


def measured_budget(runs):
    require(len(runs) == 64, 'Expected 64 runs'); trajectory = [row for run in runs for row in run['trajectory']]; final = [value for run in runs for value in run['final'].values()]
    require(all([row['update'] for row in run['trajectory']] == list(STEPS) for run in runs), 'Incomplete trajectory')
    return dict(training_forward_module_samples=sum(run['updates'] for run in runs) * CONFIG['batch_size'] * 2 * 9,
                compact_evaluation_forward_module_samples=sum(row['forward_module_samples'] for row in trajectory) + sum(9 * row['worlds'] for row in final),
                compact_evaluation_records=len(trajectory) + len(final), compact_evaluation_worlds=sum(row['compact_worlds'] for row in trajectory) + sum(row['worlds'] for row in final),
                checkpoints=len(trajectory), final_control_records=len(final), rematched_assignments=sum(run['rematch_assignments'] for run in runs))


def execute(out):
    out = Path(out).resolve(); plan, static = verify(out); execution = out / 'execution'; require(not execution.exists(), 'Never overwrite execution'); execution.mkdir(); started = time.perf_counter()
    (execution / 'started.json').write_text(json.dumps(dict(started_at=core.base.now(), pid=os.getpid(), plan_sha256=core.base.sha(out / 'plan.json')), ensure_ascii=False, indent=2) + '\n')
    try:
        with multiprocessing.get_context('spawn').Pool(CONFIG['worker_count']) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in SEEDS])
        runs = [run for group in groups for run in group]
        require([(r['seed'], r['condition']) for r in runs] == [(seed, condition) for seed in SEEDS for condition in CONDITIONS], 'Noncanonical run order')
        verify_pairing(execution, runs); measured = measured_budget(runs)
        result = dict(status='completed', completed_at=core.base.now(), elapsed_seconds=time.perf_counter() - started,
                      plan_sha256=core.base.sha(out / 'plan.json'), budget=static['budget'], measured_budget=measured, runs=runs,
                      primary='see JSON aggregation', experiment_type='factorized_neutral_action_alternative_partner_coordination',
                      language_claim_automatically_supported=False)
        (execution / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        (execution / 'status.json').write_text(json.dumps(dict(status='completed', at=core.base.now(), results_sha256=core.base.sha(execution / 'results.json')), ensure_ascii=False, indent=2) + '\n')
        return dict(status='completed', output=str(execution), elapsed_seconds=result['elapsed_seconds'], measured_budget=measured)
    except BaseException as error:
        (execution / 'failure.json').write_text(json.dumps(dict(status='failed', at=core.base.now(), error=repr(error), elapsed_seconds=time.perf_counter() - started), ensure_ascii=False, indent=2) + '\n'); raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'verify', 'execute')); parser.add_argument('--out', required=True); args = parser.parse_args()
    answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command == 'execute' else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
