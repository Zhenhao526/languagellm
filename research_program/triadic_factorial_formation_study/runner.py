"""Factorial training: unseen object×length conjunctions under communication."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
from copy import deepcopy
from itertools import product, zip_longest
from pathlib import Path
import argparse, json, multiprocessing, platform, shutil, time
import numpy as np

from research_program.triadic_action_dependency_study import runner as old, dataset
from research_program.triadic_reciprocal_execution_study import environment, kernel
from research_program.triadic_need_response_study import cases
from . import factor_cases, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = HERE.parent / 'triadic_action_dependency_study' / 'results' / 'context_001'
PHYSICS = HERE.parent / 'triadic_reciprocal_execution_study' / 'results' / 'rules_001'
SEEDS = metrics.SEEDS
REGIMES = metrics.REGIMES
RULES = metrics.RULES
LIVES = metrics.LIVES
PARTS = metrics.PARTS
STEPS = metrics.UPDATES
TARGET = metrics.TARGET
HELDOUT_RESOURCES = (5, 6)
CONDITIONS = tuple(f'{regime}_{rule}_PL_{"live" if live else "silent"}'
                   for regime in REGIMES for rule in RULES for live in (True, False))

sha = old.sha
read = old.read
write = old.write
json_bytes = old.json_bytes
array_sha = old.array_sha
now = old.now
core = old.core

CONFIG = dict(
    updates=6000, batch_size=256, checkpoints=list(STEPS), features=54, dtype='float64',
    learning_rate=0.001, global_gradient_clip=5.0, entropy_initial=0.001,
    entropy_zero_after_updates=1000, sender_entropy_coefficient=0,
    trajectories_per_state=2, sender_windows=2, sender_tokens_per_window=4,
    alphabet_size=8, actions_per_actor=17, worker_count=4,
    multiprocessing_start_method='spawn', blas_threads_per_worker=1,
    dimensions={'sender1': [54, 64, 64, 32], 'sender2': [153, 64, 64, 32], 'action': [252, 64, 64, 17]},
    seeds=list(SEEDS), regimes=list(REGIMES), rules=list(RULES), lives=['live', 'silent'],
    conditions=list(CONDITIONS), task='factorial_object_length_composition_holdout',
    train_factorial_resources=[0, 1, 2, 3, 4, 7],
    heldout_resources=list(HELDOUT_RESOURCES),
    training_objective='mean_log_exact_expected_native_reward_plus_original_action_entropy; sender_two_trajectory_LOO',
    observation='PL_own_need_and_public_layout_only; other_need_blocks_and_FI_flag_zero',
    primary='heldout_changed_actor_native_Q_regime_interaction_centered_time_AUC_rule_mean_then16_seeds',
    semantic_holdout='resource predicates 5=wood-long and 6=fiber-short are absent from factorial training; broad predicates and the opposite singleton corners remain present',
    evaluation='complete heldout-resource, heldout-layout and double-holdout partitions at six checkpoints plus complete final partitions',
    message_panel='target heldout-resource full trajectories; six actor/window panels; no monitor subsets',
    pairing='same seed initial parameters and world/message streams across all four arms within each regime; same initial parameters and streams across regimes with regime-specific train index mapping',
    selection='fixed16 seeds ×2 regimes ×2 rules ×2 communication channels ×6000 updates; no early stopping or score-dependent changes',
    automatic_followon_experiment=False,
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def parse_condition(condition):
    require(condition in CONDITIONS, 'Unknown condition')
    regime, rule, _, live_word = condition.rsplit('_', 3)
    # rsplit yields regime as the part before rule only when regime has no '_';
    # use the fixed prefix instead for unambiguous metadata.
    for candidate in REGIMES:
        prefix = candidate + '_'
        if condition.startswith(prefix):
            rest = condition[len(prefix):]
            rule, channel = rest.split('_PL_')
            require(rule in RULES and channel in ('live', 'silent'), 'Malformed condition')
            return candidate, rule, channel == 'live'
    raise ValueError('Malformed condition')


def name(seed, condition):
    require(seed in SEEDS and condition in CONDITIONS, 'Unknown seed/condition')
    return f'seed_{seed}_{condition}'


def _original():
    source = read(ORIGINAL / 'prepared.json'); freeze = read(ORIGINAL / 'freeze.json')
    require(sha(ORIGINAL / 'prepared.json') == freeze['prepared_sha256'] ==
            'e555037fa8a9d72a2ef150a0d99d46e5a893139b2efa02314dffa3e2a001a299', 'Original prepared hash mismatch')
    require(sha(ORIGINAL / 'plan.json') == freeze['plan_sha256'] ==
            'df5036145a1bea258a40d6e0a65b66a09c6c188b311b732c954429b6eb38fcba', 'Original plan hash mismatch')
    physics_plan = read(PHYSICS / 'plan.json'); physics_freeze = read(PHYSICS / 'freeze.json')
    require(sha(PHYSICS / 'plan.json') == physics_freeze['plan_sha256'] ==
            '59e9f8d3b2056538b339072867fde4c58b1478274d982e3158d1b4268edd0e53', 'Reciprocal source changed')
    return source


def _spec(partition, needs, layouts, owners, regime, role):
    needs = [list(map(int, row)) for row in sorted(tuple(map(int, row)) for row in needs)]
    layouts = [list(map(int, row)) for row in layouts]
    owners = [list(map(int, row)) for row in owners]
    return dict(partition=partition, regime=regime, role=role, needs=needs, layouts=layouts,
                private_sites=owners, world_count=len(needs) * len(layouts) * len(owners),
                state_order='need-major, then layout, then owner', weighting='uniform needs × layouts × owners')


def _build_specs():
    source = _original()
    train_layouts = source['partitions']['train']['layouts']
    heldout_layouts = source['partitions']['new_layouts']['layouts']
    owners = source['partitions']['train']['private_sites']
    support = [tuple(map(int, row)) for row in old.env.support()]
    factorial_needs = [row for row in support if all(int(row[a]) // 3 not in HELDOUT_RESOURCES for a in range(3))]
    heldout_needs = [row for row in support if any(int(row[a]) // 3 in HELDOUT_RESOURCES for a in range(3))]
    all_needs = support
    require(len(factorial_needs) == 2088 and len(heldout_needs) == 3288 and len(all_needs) == 5376,
            'Unexpected factorial support counts')
    specs = {regime: {} for regime in REGIMES}
    for regime in REGIMES:
        train_needs = factorial_needs if regime == 'factorial_holdout' else all_needs
        layout_train_needs = factorial_needs if regime == 'factorial_holdout' else all_needs
        specs[regime]['train'] = _spec('train', train_needs, train_layouts, owners, regime,
                                       'factorial_train' if regime == 'factorial_holdout' else 'saturated_train')
        specs[regime]['heldout_resource'] = _spec('heldout_resource', heldout_needs, train_layouts, owners, regime, 'unseen_resource')
        specs[regime]['heldout_layout'] = _spec('heldout_layout', layout_train_needs, heldout_layouts, owners, regime, 'unseen_layout')
        specs[regime]['heldout_both'] = _spec('heldout_both', heldout_needs, heldout_layouts, owners, regime, 'unseen_resource_and_layout')
    return specs, dict(train_resources=[0, 1, 2, 3, 4, 7], heldout_resources=list(HELDOUT_RESOURCES),
                       train_need_count=len(factorial_needs), heldout_need_count=len(heldout_needs), all_support_count=len(all_needs),
                       train_layout_count=len(train_layouts), heldout_layout_count=len(heldout_layouts), owner_count=len(owners))


def _budget(specs):
    runs = len(SEEDS) * len(REGIMES) * len(RULES) * len(LIVES)
    updates = runs * CONFIG['updates']
    eval_worlds = 0
    for regime in REGIMES:
        reg = specs[regime]
        per_run = len(STEPS) * reg[TARGET]['world_count'] + sum(reg[p]['world_count'] for p in PARTS if p != TARGET)
        eval_worlds += len(SEEDS) * len(RULES) * len(LIVES) * per_run
    return dict(runs=runs, independent_paired_seed_blocks=len(SEEDS), regimes=len(REGIMES),
                training_updates=updates, training_world_samples=updates * CONFIG['batch_size'],
                message_trajectories=updates * CONFIG['batch_size'] * 2,
                categorical_symbol_samples=updates * CONFIG['batch_size'] * 2 * 2 * 3 * 4,
                training_forward_module_samples=updates * CONFIG['batch_size'] * 2 * 9,
                checkpoints=runs * len(STEPS), training_log_rows=updates,
                target_trajectory_files=runs * len(STEPS), other_final_files=runs * (len(PARTS) - 1),
                evaluation_files=runs * (len(STEPS) + len(PARTS) - 1), target_aliases=runs,
                evaluation_worlds=eval_worlds, evaluation_forward_module_samples=eval_worlds * 9,
                total_forward_module_samples=updates * CONFIG['batch_size'] * 2 * 9 + eval_worlds * 9,
                generated_NPZ_files=runs * (len(STEPS) + len(PARTS) - 1),
                message_snapshots=runs * len(STEPS), message_transitions=runs * (len(STEPS) - 1))


def prepared():
    specs, split = _build_specs()
    response_cases = {regime: {part: cases.build_cases(specs[regime][part]) for part in PARTS} for regime in REGIMES}
    groups = {regime: {part: factor_cases.heldout_edge_indices(response_cases[regime][part], specs[regime][part]['needs'], HELDOUT_RESOURCES)
                       for part in PARTS} for regime in REGIMES}
    require(all(len(response_cases[g][p]['strata']) == 9 and not response_cases[g][p]['empty_strata']
                for g in REGIMES for p in PARTS), 'Every partition must have nine response strata')
    require(all(all(groups[g][p]['heldout_changed_actor']) and all(groups[g][p]['seen_changed_actor'])
                for g in REGIMES for p in ('heldout_resource', 'heldout_both')),
            'Held-out target groups must populate all nine strata')
    budget = _budget(specs)
    return dict(schema='triadic_factorial_formation_v1', regimes=list(REGIMES), partitions=specs,
                need_response_cases=response_cases, factor_edge_groups=groups,
                split=split, source_original_prepared_sha256=sha(ORIGINAL / 'prepared.json'),
                independent_initializations=len(SEEDS), budget=budget,
                scientific_question='Does a learned discrete convention support response to unseen object×length resource predicates, and is that support specifically amplified by live communication?',
                claim_boundary='A live/silent interaction on held-out semantic combinations is evidence about behavioral generalization under this task; it does not alone establish human-like language, compositionality, or autonomous convention formation.')


def sources():
    required = [HERE / n for n in ('__init__.py', 'factor_cases.py', 'metrics.py', 'runner.py', 'plan.md')]
    required += [Path(old.__file__), Path(dataset.__file__), Path(old.env.__file__), Path(core.__file__),
                 Path(kernel.__file__), Path(environment.__file__), Path(cases.__file__)]
    require(all(path.is_file() for path in required), 'Missing frozen source')
    return {str(path.resolve()): sha(path) for path in required}


def prepare(out):
    out = Path(out).resolve(); require(not out.exists(), 'Never overwrite preparation')
    static = prepared(); ss = sources(); out.mkdir(parents=True)
    for path in ss:
        target = out / 'source_snapshot' / Path(path).relative_to(ROOT); target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, target)
    write(out / 'prepared.json', static)
    inputs = [ORIGINAL / f for f in ('plan.json', 'prepared.json', 'freeze.json')]
    inputs += [PHYSICS / f for f in ('plan.json', 'freeze.json')]
    write(out / 'plan.json', dict(status='prepared_without_training', at=now(), config=CONFIG, source_sha256=ss,
                                  inputs_sha256={str(p): sha(p) for p in inputs}, prepared_sha256=sha(out / 'prepared.json'),
                                  runtime=dict(python=platform.python_version(), numpy=np.__version__)))
    write(out / 'freeze.json', dict(plan_sha256=sha(out / 'plan.json'), prepared_sha256=sha(out / 'prepared.json')))
    verify(out)
    return dict(status='prepared_without_training', plan_sha256=sha(out / 'plan.json'), budget=static['budget'])


def verify(out):
    out = Path(out).resolve(); plan = read(out / 'plan.json'); static = read(out / 'prepared.json'); freeze = read(out / 'freeze.json')
    require(sha(out / 'plan.json') == freeze['plan_sha256'] and plan['config'] == CONFIG, 'Plan/config mismatch')
    require(sha(out / 'prepared.json') == plan['prepared_sha256'] == freeze['prepared_sha256'], 'Prepared hash mismatch')
    require(static == prepared() and plan['source_sha256'] == sources(), 'Static source/config mismatch')
    require(plan['runtime'] == dict(python=platform.python_version(), numpy=np.__version__), 'Runtime mismatch')
    for path, digest in plan['inputs_sha256'].items(): require(sha(path) == digest, 'Changed input: ' + path)
    for path, digest in plan['source_sha256'].items():
        require(sha(out / 'source_snapshot' / Path(path).relative_to(ROOT)) == digest, 'Changed source snapshot: ' + path)
    return plan, static


def make_arrays(spec):
    arrays = dataset.make_arrays(spec, information='PL'); arrays.pop('states', None)
    x = arrays['x_PL']; require(x.shape == (spec['world_count'], 3, 54) and np.all(x[:, :, 53] == 0), 'Bad PL feature array')
    for actor in range(3):
        for other in range(3):
            if actor != other: require(np.all(x[:, actor, 7 * other:7 * other + 7] == 0), 'Other private need exposed')
    return arrays


def evaluate(networks, arrays, live, rule, indices, path, case_spec, edge_groups):
    ids = np.asarray(indices, dtype=np.int64); path = Path(path); n = len(ids)
    require(ids.ndim == 1 and len(np.unique(ids)) == n and n == case_spec['world_count'] and
            np.array_equal(ids, np.arange(n, dtype=np.int64)) and not path.exists(), 'Complete canonical evaluation required')
    states = arrays['packed_states'][ids]
    cases.validate_partition_arrays(case_spec, states, ids)
    data = dict(states=states, state_indices=ids, messages=np.empty((n, 2, 3, 4), np.int8),
                action_indices=np.empty((n, 3), np.int16), action_probabilities=np.empty((n, 3, 17)),
                conditional_exact_expected_reward=np.empty(n), conditional_exact_full_success_probability=np.empty(n),
                conditional_exact_execution_probability=np.empty(n), conditional_full_posterior_mass=np.empty(n))
    for start in range(0, n, 1024):
        sl = slice(start, min(start + 1024, n)); ix = ids[sl]
        trace = core.rollout(networks, arrays['x_PL'][ix], bool(live))
        terms = kernel.objective_terms(trace['action_logits'], arrays['rewards'][ix], rule); probs = terms['probabilities']
        data['messages'][sl] = trace['messages']; data['action_probabilities'][sl] = probs; data['action_indices'][sl] = probs.argmax(-1)
        data['conditional_exact_expected_reward'][sl] = terms['native_expected_reward']
        data['conditional_exact_full_success_probability'][sl] = terms['full_success_probability']
        data['conditional_exact_execution_probability'][sl] = terms['execution_probability']
        data['conditional_full_posterior_mass'][sl] = terms['full_success_posterior_mass']
    actions = data['action_indices']; native = environment.settle(states, actions, rule)
    strict = environment.settle(states, actions, 'strict'); common = environment.settle(states, actions, 'reciprocal')
    truth = environment.truth_from_rewards(states, arrays['rewards'][ids])
    factor_response = {}
    for group in ('heldout_changed_actor', 'seen_changed_actor'):
        selected = edge_groups[group]
        factor_response[group] = (factor_cases.subset_metrics(case_spec, native['actual_pair_index'], selected)
                                  if all(selected) else None)
    record = dict(native=environment.metrics(native, truth, actions, rule),
                  strict=environment.metrics(strict, truth, actions, 'strict'),
                  common_reciprocal=environment.metrics(common, truth, actions, 'reciprocal'),
                  need_response=dict(native=cases.metrics(case_spec, native['actual_pair_index']),
                                     common_reciprocal=cases.metrics(case_spec, common['actual_pair_index'])),
                  factor_response=factor_response)
    data.update(native); data.update({f'strict__{k}': v for k, v in strict.items()}); data.update({f'common_reciprocal__{k}': v for k, v in common.items()})
    with path.open('xb') as stream: np.savez_compressed(stream, **data)
    record.update(path=str(path), data_sha256=sha(path), information='PL', live=bool(live), rule=rule,
                  scope='complete_partition', worlds=n, forward_module_samples=9 * n,
                  expected_reward_given_greedy_messages=float(data['conditional_exact_expected_reward'].mean()),
                  full_probability_given_greedy_messages=float(data['conditional_exact_full_success_probability'].mean()),
                  execution_probability_given_greedy_messages=float(data['conditional_exact_execution_probability'].mean()),
                  full_posterior_mass_given_greedy_messages=float(data['conditional_full_posterior_mass'].mean()),
                  state_indices_sha256=array_sha(ids))
    return record, data['messages']


def make_random_streams(seed):
    return np.random.default_rng(np.random.SeedSequence([seed, 200])), core.make_message_rngs(seed)


def train_run(seed, regime, condition, static, arrays, execution):
    directory = Path(execution) / name(seed, condition); directory.mkdir(exist_ok=False)
    _, rule, live = parse_condition(condition); nets = core.make_networks(seed); initial = core.parameter_hash(nets)
    optimizer = core.base.make_adam(nets); world_rng, message_rngs = make_random_streams(seed)
    trajectory = []; start = time.perf_counter(); previous_messages = None
    target_spec = static['partitions'][regime][TARGET]; target_cases = static['need_response_cases'][regime][TARGET]; target_groups = static['factor_edge_groups'][regime][TARGET]

    def checkpoint(step):
        nonlocal previous_messages
        checkpoint_path = directory / f'checkpoint_{step:04d}.npz'; digest = core.save_checkpoint(checkpoint_path, nets, optimizer, step, world_rng, message_rngs)
        evaluation, messages = evaluate(nets, arrays[TARGET], live, rule, np.arange(target_spec['world_count'], dtype=np.int64),
                                        directory / f'trajectory_{step:04d}_{TARGET}.npz', target_cases, target_groups)
        snapshot = metrics.message_snapshot(messages, target_spec)
        transition = None if previous_messages is None else metrics.message_transition(previous_messages, messages, target_spec)
        previous_messages = messages
        row = dict(update=step, checkpoint_path=str(checkpoint_path), checkpoint_sha256=digest, evaluation=evaluation,
                   message_snapshot=snapshot, message_transition=transition, elapsed_seconds=time.perf_counter() - start)
        trajectory.append(row)
        with (directory / 'trajectory.jsonl').open('a') as stream: stream.write(json_bytes(row).decode())

    checkpoint(0)
    with (directory / 'training.jsonl').open('x') as stream:
        for update in range(1, CONFIG['updates'] + 1):
            uniforms_world = world_rng.random((CONFIG['batch_size'], 3)); ids = dataset.sample_indices(static['partitions'][regime]['train'], uniforms_world)
            uniforms_message = core.draw_uniforms(message_rngs, CONFIG['batch_size'])
            gradients, row = kernel.training_gradients(nets, arrays['train']['x_PL'][ids], arrays['train']['rewards'][ids], live, uniforms_message, update, rule)
            norm, scale = core.base.adam_step(nets, gradients, optimizer, update)
            row.update(update=update, seed=seed, regime=regime, rule=rule, condition=condition,
                       world_uniforms_sha256=array_sha(uniforms_world), batch_indices_sha256=array_sha(ids),
                       batch_states_sha256=array_sha(arrays['train']['packed_states'][ids]), sample_uniforms_sha256=array_sha(uniforms_message),
                       gradient_norm=norm, gradient_clip_scale=scale)
            stream.write(json_bytes(row).decode())
            if update in STEPS:
                stream.flush(); checkpoint(update)
    finals = {}
    for part in PARTS:
        if part == TARGET: continue
        spec = static['partitions'][regime][part]
        finals[part], _ = evaluate(nets, arrays[part], live, rule, np.arange(spec['world_count'], dtype=np.int64),
                                   directory / f'final_{part}.npz', static['need_response_cases'][regime][part], static['factor_edge_groups'][regime][part])
    finals[TARGET] = dict(trajectory[-1]['evaluation'], alias_of='trajectory_update_6000', additional_forward_module_samples=0)
    result = dict(seed=seed, regime=regime, condition=condition, rule=rule, live=live, updates=CONFIG['updates'],
                  initial_parameter_sha256=initial, final_parameter_sha256=core.parameter_hash(nets),
                  final_checkpoint_sha256=sha(directory / 'checkpoint_6000.npz'), training_log_sha256=sha(directory / 'training.jsonl'),
                  trajectory=trajectory, final=finals, elapsed_seconds=time.perf_counter() - start)
    write(directory / 'result.json', result)
    write(directory / 'status.json', dict(status='completed', seed=seed, regime=regime, condition=condition,
                                          result_sha256=sha(directory / 'result.json'), at=now()))
    return result


def worker(payload):
    seed, static, execution = payload; execution = Path(execution); start = time.perf_counter()
    arrays = {regime: {part: make_arrays(static['partitions'][regime][part]) for part in PARTS} for regime in REGIMES}
    hashes = {regime: {part: {k: array_sha(arrays[regime][part][k]) for k in ('packed_states', 'rewards', 'x_PL')} for part in PARTS} for regime in REGIMES}
    write(execution / f'seed_{seed}_arrays.json', dict(array_hashes=hashes, build_seconds=time.perf_counter() - start))
    runs = []
    for condition in CONDITIONS:
        regime, _, _ = parse_condition(condition)
        runs.append(train_run(seed, regime, condition, static, arrays[regime], execution))
    require(hashes == {g: {p: {k: array_sha(arrays[g][p][k]) for k in ('packed_states', 'rewards', 'x_PL')} for p in PARTS} for g in REGIMES}, 'Training mutated arrays')
    return runs


def verify_pairing(execution, runs):
    execution = Path(execution); by = {(r['seed'], r['regime'], r['condition']): r for r in runs}
    require(set(by) == {(s, g, c) for s in SEEDS for g in REGIMES for c in CONDITIONS if parse_condition(c)[0] == g}, 'Incomplete paired grid')
    for seed in SEEDS:
        initial_hashes = []
        for regime in REGIMES:
            regime_conditions = [c for c in CONDITIONS if parse_condition(c)[0] == regime]
            selected = [by[seed, regime, c] for c in regime_conditions]; initial_hashes.extend(r['initial_parameter_sha256'] for r in selected)
            require(len(set(initial_hashes[-4:])) == 1, 'Unpaired initialization within regime')
            streams = [(execution / name(seed, c) / 'training.jsonl').open() for c in regime_conditions]
            try:
                count = 0
                for lines in zip_longest(*streams):
                    require(all(line is not None for line in lines), 'Unequal paired logs')
                    rows = [json.loads(line) for line in lines]; count += 1
                    require(all(r['update'] == count and r['seed'] == seed and r['regime'] == regime and
                                r['condition'] == c and r['rule'] == parse_condition(c)[1] for r, c in zip(rows, regime_conditions)), 'Training identity mismatch')
                    for key in ('world_uniforms_sha256', 'sample_uniforms_sha256', 'entropy_coefficient'):
                        require(len({r[key] for r in rows}) == 1, 'Paired stream differs: ' + key)
                    for key in ('batch_indices_sha256', 'batch_states_sha256'):
                        require(len({r[key] for r in rows}) == 1, 'Paired batch differs: ' + key)
                require(count == CONFIG['updates'], 'Training update count mismatch')
            finally:
                for stream in streams: stream.close()
        require(len(set(initial_hashes)) == 1, 'Cross-regime initial parameters differ')


def measured_budget(static, runs):
    require(len(runs) == 128, 'Incomplete run grid'); actual = []
    for run in runs:
        require(run['updates'] == CONFIG['updates'] and [row['update'] for row in run['trajectory']] == list(STEPS), 'Incomplete trajectory')
        endpoint = run['trajectory'][-1]['evaluation']; alias = run['final'][TARGET]
        require(alias['alias_of'] == 'trajectory_update_6000' and alias['additional_forward_module_samples'] == 0 and
                {k: v for k, v in alias.items() if k not in ('alias_of', 'additional_forward_module_samples')} == endpoint, 'Target alias mismatch')
        actual.extend(row['evaluation'] for row in run['trajectory']); actual.extend(run['final'][part] for part in PARTS if part != TARGET)
    require(len({row['path'] for row in actual}) == len(actual), 'Duplicate evaluation path')
    return dict(training_forward_module_samples=sum(r['updates'] for r in runs) * CONFIG['batch_size'] * 2 * 9,
                evaluation_forward_module_samples=sum(r['forward_module_samples'] for r in actual), checkpoints=len(runs) * len(STEPS),
                evaluation_files=len(actual), evaluation_worlds=sum(r['worlds'] for r in actual), target_aliases=len(runs))


def execute(out):
    out = Path(out).resolve(); plan, static = verify(out); execution = out / 'execution'; require(not execution.exists(), 'Never overwrite execution'); execution.mkdir()
    start = time.perf_counter(); write(execution / 'started.json', dict(at=now(), pid=os.getpid(), plan_sha256=sha(out / 'plan.json')))
    try:
        with multiprocessing.get_context('spawn').Pool(CONFIG['worker_count']) as pool:
            groups = pool.map(worker, [(seed, static, str(execution)) for seed in SEEDS])
        runs = [run for group in groups for run in group]
        expected = [(seed, regime, condition) for seed in SEEDS for regime in REGIMES for condition in CONDITIONS if parse_condition(condition)[0] == regime]
        require([(r['seed'], r['regime'], r['condition']) for r in runs] == expected, 'Noncanonical run order')
        verify_pairing(execution, runs)
        array_hashes = [read(execution / f'seed_{seed}_arrays.json')['array_hashes'] for seed in SEEDS]; require(all(x == array_hashes[0] for x in array_hashes), 'Worker arrays differ')
        measured = measured_budget(static, runs); expected_measured = dict(training_forward_module_samples=static['budget']['training_forward_module_samples'],
            evaluation_forward_module_samples=measured['evaluation_forward_module_samples'], checkpoints=measured['checkpoints'],
            evaluation_files=measured['evaluation_files'], evaluation_worlds=measured['evaluation_worlds'], target_aliases=measured['target_aliases'])
        require(measured == expected_measured, 'Internal measured budget mismatch')
        result = dict(status='completed', completed_at=now(), elapsed_seconds=time.perf_counter() - start,
                      plan_sha256=sha(out / 'plan.json'), budget=static['budget'], measured_budget=measured, runs=runs,
                      array_hashes=array_hashes[0], primary=metrics.primary(runs), pairing='same initial parameters and streams within each regime',
                      experiment_type='discrete_symbol_message_co_learning_factorial_generalization', language_claim_automatically_supported=False)
        write(execution / 'results.json', result); write(execution / 'status.json', dict(status='completed', at=now(), results_sha256=sha(execution / 'results.json')))
        return dict(status='completed', elapsed_seconds=result['elapsed_seconds'], primary=result['primary'])
    except BaseException as error:
        write(execution / 'failure.json', dict(status='failed', at=now(), error=repr(error), elapsed_seconds=time.perf_counter() - start)); raise


run = execute

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('command', choices=('prepare', 'verify', 'execute', 'run')); parser.add_argument('--out', required=True); args = parser.parse_args()
    answer = prepare(args.out) if args.command == 'prepare' else execute(args.out) if args.command in ('execute', 'run') else verify(args.out)[0]
    print(json.dumps(answer, ensure_ascii=False))
