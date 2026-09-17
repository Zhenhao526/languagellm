"""Independent replay audit for the factorial formation run.

The audit does not call the production evaluator.  It reconstructs the frozen
world arrays, loads checkpoint parameters, forwards every saved evaluation, and
recomputes settlement, Q and the held-out edge-group estimand.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'
import argparse, json, math, re, time, traceback
from collections import defaultdict
from itertools import product, zip_longest
from pathlib import Path
import numpy as np

from research_program.triadic_action_dependency_study import runner as old, dataset
from research_program.triadic_reciprocal_execution_study import environment, kernel
from research_program.triadic_need_response_study import cases
from . import factor_cases, metrics, runner as production

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = metrics.SEEDS; REGIMES = metrics.REGIMES; RULES = metrics.RULES; LIVES = metrics.LIVES
PARTS = metrics.PARTS; STEPS = metrics.UPDATES; TARGET = metrics.TARGET
CONDITIONS = production.CONDITIONS
T15 = metrics.T15_975
CONDITIONAL = ('conditional_exact_expected_reward', 'conditional_exact_full_success_probability',
               'conditional_exact_execution_probability', 'conditional_full_posterior_mass')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def array_sha(value):
    import hashlib
    a = np.asarray(value); h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode()); h.update(a.tobytes(order='C'))
    return h.hexdigest()


def close(a, b, label, tolerance=2e-12):
    x = np.asarray(a); y = np.asarray(b)
    require(x.shape == y.shape, label + ' shape')
    if x.dtype.kind in 'f' or y.dtype.kind in 'f':
        err = float(np.max(np.abs(x.astype(float) - y.astype(float)))) if x.size else 0.0
        require(np.isfinite(err) and err <= tolerance, label + f' max error {err}')
        return err
    require(np.array_equal(x, y), label)
    return 0.0


def compare(left, right, label='value', tolerance=2e-12):
    if isinstance(left, dict):
        require(isinstance(right, dict) and set(left) == set(right), label + ' keys')
        for key in left:
            compare(left[key], right[key], label + '/' + str(key), tolerance)
    elif isinstance(left, list):
        require(isinstance(right, list) and len(left) == len(right), label + ' list')
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, label + f'[{i}]', tolerance)
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        require(isinstance(right, (int, float)) and not isinstance(right, bool) and np.isfinite(right) and abs(float(left) - float(right)) <= tolerance,
                label + f' numbers {left} != {right}')
    else:
        require(left == right, label + f' {left!r} != {right!r}')


def parse_condition(condition):
    for regime in REGIMES:
        if condition.startswith(regime + '_'):
            rest = condition[len(regime) + 1:]
            rule, channel = rest.split('_PL_')
            require(rule in RULES and channel in ('live', 'silent'), 'Condition syntax')
            return regime, rule, channel == 'live'
    raise ValueError('Unknown condition')


def independent_group_lists(case_spec, needs, heldout=(5, 6)):
    ns = [tuple(map(int, row)) for row in needs]; edges = np.asarray(case_spec['edge_need_indices'], dtype=np.int64)
    changed = np.asarray(case_spec['changed_person'], dtype=np.int64); axes = np.asarray(case_spec['axis_index'], dtype=np.int64)
    groups = {'heldout_changed_actor': [[] for _ in range(9)], 'seen_changed_actor': [[] for _ in range(9)]}
    held = set(heldout)
    for index, ((before, after), who, axis) in enumerate(zip(edges, changed, axes)):
        flag = ns[before][who] // 3 in held or ns[after][who] // 3 in held
        groups['heldout_changed_actor' if flag else 'seen_changed_actor'][int(who) * 3 + int(axis)].append(int(index))
    return groups


def independent_specs(source):
    train_layouts = source['partitions']['train']['layouts']; heldout_layouts = source['partitions']['new_layouts']['layouts']; owners = source['partitions']['train']['private_sites']
    support = sorted(tuple(map(int, row)) for row in old.env.support())
    factorial = [row for row in support if all(row[a] // 3 not in (5, 6) for a in range(3))]
    heldout = [row for row in support if any(row[a] // 3 in (5, 6) for a in range(3))]
    def spec(partition, needs, layouts, regime, role):
        ns = [list(row) for row in sorted(needs)]
        return dict(partition=partition, regime=regime, role=role, needs=ns, layouts=[list(x) for x in layouts],
                    private_sites=[list(x) for x in owners], world_count=len(ns) * len(layouts) * len(owners),
                    state_order='need-major, then layout, then owner', weighting='uniform needs × layouts × owners')
    out = {g: {} for g in REGIMES}
    for g in REGIMES:
        train = factorial if g == 'factorial_holdout' else support
        out[g]['train'] = spec('train', train, train_layouts, g, 'factorial_train' if g == 'factorial_holdout' else 'saturated_train')
        out[g]['heldout_resource'] = spec('heldout_resource', heldout, train_layouts, g, 'unseen_resource')
        out[g]['heldout_layout'] = spec('heldout_layout', train, heldout_layouts, g, 'unseen_layout')
        out[g]['heldout_both'] = spec('heldout_both', heldout, heldout_layouts, g, 'unseen_resource_and_layout')
    response = {g: {p: cases.build_cases(out[g][p]) for p in PARTS} for g in REGIMES}
    groups = {g: {p: independent_group_lists(response[g][p], out[g][p]['needs']) for p in PARTS} for g in REGIMES}
    return out, response, groups


def factor_subset_independent(case_spec, values, edge_indices):
    values = np.asarray(values); require(values.shape == (case_spec['world_count'],), 'Complete pair vector')
    n = int(case_spec['need_worlds']); b = int(case_spec['n_backgrounds']); domains = values.astype(np.int8).reshape(n, b)
    edges_all = np.asarray(case_spec['edge_need_indices'], dtype=np.int64); targets_all = np.asarray(case_spec['target_pairs'], dtype=np.int8)
    counts_by_bg = [np.bincount(domains[:, j] + 1, minlength=4)[1:] for j in range(b)]
    rows = []; total = {'both_correct': 0, 'edge_count': 0}
    for stratum, subset in enumerate(edge_indices):
        ids = np.asarray(subset, dtype=np.int64); require(len(ids) > 0, 'Empty independent subset')
        edges = edges_all[ids]; targets = targets_all[ids]; qs = []; shuffles = []; both = 0
        for j in range(b):
            actual = domains[edges, j]; hits = np.all(actual[:, None] == targets, axis=1); qs.append(float(hits.mean())); both += int(hits.sum())
            count = counts_by_bg[j]; numerator = int(np.sum(count[targets[:, 0]] * count[targets[:, 1]], dtype=np.int64)); shuffles.append(numerator / (n * (n - 1) * len(ids)))
        q = float(np.mean(qs)); s = float(np.mean(shuffles)); rows.append(dict(changed_person=stratum // 3, axis_index=stratum % 3, edge_count=int(len(ids),), state_edge_count=int(len(ids) * b), Q=q, Q_shuffle=s, Q_excess=q - s, raw_both_correct=both))
        total['both_correct'] += both; total['edge_count'] += int(len(ids) * b)
    q = float(np.mean([r['Q'] for r in rows])); s = float(np.mean([r['Q_shuffle'] for r in rows]))
    return dict(schema='factorial_edge_subset_metrics_v1', group_worlds=case_spec['world_count'], complete_nine_strata=True,
                strata=rows, Q=q, Q_shuffle=s, Q_excess=q - s, raw_counts=total,
                weighting='Equal nine changed-person×axis strata; within each stratum equal edges and backgrounds.',
                scope='Independent researcher-side edge group replay.')


def evaluate_saved(entry, path, arrays, case_spec, edge_groups, rule, live, checkpoint_path):
    path = Path(path); require(Path(entry['path']).resolve() == path.resolve() and sha(path) == entry['data_sha256'], 'Evaluation path/hash')
    with np.load(path, allow_pickle=False) as saved:
        keys = set(saved.files); states = arrays['packed_states']; n = len(states)
        required = {'states', 'state_indices', 'messages', 'action_indices', 'action_probabilities', *CONDITIONAL}
        require(required <= keys, 'Evaluation schema')
        close(saved['states'], states, 'Saved states'); close(saved['state_indices'], np.arange(n, dtype=np.int64), 'Saved state indices')
        messages = saved['messages']; actions = saved['action_indices']; probabilities = saved['action_probabilities']
        require(messages.dtype == np.int8 and messages.shape == (n, 2, 3, 4) and ((messages >= 0) & (messages < 8)).all(), 'Saved messages')
        require(actions.dtype == np.int16 and actions.shape == (n, 3), 'Saved actions')
        require(probabilities.dtype == np.float64 and probabilities.shape == (n, 3, 17), 'Saved probabilities')
        networks = core.load_networks(checkpoint_path)
        expected_m = np.empty_like(messages); expected_p = np.empty_like(probabilities); expected_a = np.empty_like(actions)
        expected_cond = {key: np.empty(n, dtype=np.float64) for key in CONDITIONAL}; tie_count = 0
        x = arrays['x_PL']
        for start in range(0, n, 1024):
            sl = slice(start, min(start + 1024, n)); trace = core.rollout(networks, x[sl], bool(live)); terms = kernel.objective_terms(trace['action_logits'], arrays['rewards'][sl], rule)
            expected_m[sl] = trace['messages']; expected_p[sl] = terms['probabilities']; expected_a[sl] = terms['probabilities'].argmax(-1)
            expected_cond['conditional_exact_expected_reward'][sl] = terms['native_expected_reward']
            expected_cond['conditional_exact_full_success_probability'][sl] = terms['full_success_probability']
            expected_cond['conditional_exact_execution_probability'][sl] = terms['execution_probability']
            expected_cond['conditional_full_posterior_mass'][sl] = terms['full_success_posterior_mass']
        close(messages, expected_m, 'Forward messages'); close(probabilities, expected_p, 'Forward probabilities'); close(actions, expected_a, 'Greedy actions')
        for key in CONDITIONAL: close(saved[key], expected_cond[key], key)
        native = environment.settle(states, actions, rule); strict = environment.settle(states, actions, 'strict'); common = environment.settle(states, actions, 'reciprocal')
        truth = environment.truth_from_rewards(states, arrays['rewards']);
        for prefix, value in (('', native), ('strict__', strict), ('common_reciprocal__', common)):
            for key, arr in value.items(): require(prefix + key in keys, 'Missing settlement field ' + prefix + key); close(saved[prefix + key], arr, 'Settlement/' + prefix + key)
        require(set(keys) == required | set(native) | {'strict__' + k for k in strict} | {'common_reciprocal__' + k for k in common}, 'Unexpected evaluation fields')
        calculated = dict(native=environment.metrics(native, truth, actions, rule), strict=environment.metrics(strict, truth, actions, 'strict'),
                          common_reciprocal=environment.metrics(common, truth, actions, 'reciprocal'),
                          need_response=dict(native=cases.metrics(case_spec, native['actual_pair_index']), common_reciprocal=cases.metrics(case_spec, common['actual_pair_index'])),
                          factor_response={group: (factor_subset_independent(case_spec, native['actual_pair_index'], edge_groups[group]) if all(edge_groups[group]) else None) for group in ('heldout_changed_actor', 'seen_changed_actor')})
        calculated.update(path=str(path), data_sha256=sha(path), information='PL', live=bool(live), rule=rule, scope='complete_partition', worlds=n, forward_module_samples=9 * n,
                          expected_reward_given_greedy_messages=float(saved['conditional_exact_expected_reward'].mean()),
                          full_probability_given_greedy_messages=float(saved['conditional_exact_full_success_probability'].mean()),
                          execution_probability_given_greedy_messages=float(saved['conditional_exact_execution_probability'].mean()),
                          full_posterior_mass_given_greedy_messages=float(saved['conditional_full_posterior_mass'].mean()), state_indices_sha256=array_sha(np.arange(n, dtype=np.int64)))
    compare(entry, calculated, 'Evaluation record')
    return dict(worlds=n, independent_module_samples=9 * n, max_probability_error=float(np.max(np.abs(probabilities - expected_p))), messages_sha256=array_sha(messages))


def audit(run, expected_plan_sha):
    run = Path(run).resolve(); execution = run / 'execution'; require(re.fullmatch(r'[0-9a-f]{64}', expected_plan_sha or '') is not None, 'Explicit plan hash')
    plan = read(run / 'plan.json'); prepared = read(run / 'prepared.json'); freeze = read(run / 'freeze.json'); result = read(execution / 'results.json'); status = read(execution / 'status.json')
    require(status['status'] == result['status'] == 'completed' and sha(execution / 'results.json') == status['results_sha256'], 'Completed result/status')
    require(sha(run / 'plan.json') == expected_plan_sha == freeze['plan_sha256'] == result['plan_sha256'], 'Plan chain')
    require(sha(run / 'prepared.json') == plan['prepared_sha256'] == freeze['prepared_sha256'], 'Prepared chain')
    require(plan['runtime'] == dict(python=platform.python_version(), numpy=np.__version__), 'Runtime')
    # Reconstruct static split and labels without calling production.prepared().
    original_dir = production.ORIGINAL
    source = read(original_dir / 'prepared.json') if original_dir.exists() else None
    require(source is not None, 'Original support missing')
    independent, response, groups = independent_specs(source)
    expected_static = dict(schema='triadic_factorial_formation_v1', regimes=list(REGIMES), partitions=independent,
                           need_response_cases=response, factor_edge_groups=groups,
                           split=dict(train_resources=[0, 1, 2, 3, 4, 7], heldout_resources=[5, 6], train_need_count=2088, heldout_need_count=3288, all_support_count=5376, train_layout_count=18, heldout_layout_count=6, owner_count=6),
                           source_original_prepared_sha256=sha(production.ORIGINAL / 'prepared.json'), independent_initializations=16,
                           budget=prepared['budget'], scientific_question=prepared['scientific_question'], claim_boundary=prepared['claim_boundary'])
    compare(prepared, expected_static, 'Independent prepared')
    # Recompute every worker array hash from the researcher-side static specs.
    arrays = {}; hashes = {}
    for regime in REGIMES:
        arrays[regime] = {}; hashes[regime] = {}
        for part in PARTS:
            arrays[regime][part] = dataset.make_arrays(independent[regime][part], information='PL'); arrays[regime][part].pop('states', None)
            x = arrays[regime][part]['x_PL']; require(np.all(x[:, :, 53] == 0), 'FI flag visible');
            for actor, other in product(range(3), repeat=2):
                if actor != other: require(np.all(x[:, actor, 7 * other:7 * other + 7] == 0), 'Other need visible')
            hashes[regime][part] = {key: array_sha(arrays[regime][part][key]) for key in ('packed_states', 'rewards', 'x_PL')}
    by = {(r['seed'], r['regime'], r['condition']): r for r in result['runs']}; require(set(by) == {(s, g, c) for s in SEEDS for g in REGIMES for c in CONDITIONS if parse_condition(c)[0] == g}, 'Run grid')
    errors = defaultdict(float); counts = defaultdict(int); artifacts = {}
    for seed in SEEDS:
        worker_arrays = read(execution / f'seed_{seed}_arrays.json'); require(worker_arrays['array_hashes'] == hashes, 'Worker arrays'); artifacts[str(execution / f'seed_{seed}_arrays.json')] = sha(execution / f'seed_{seed}_arrays.json')
        for regime in REGIMES:
            for condition in [c for c in CONDITIONS if parse_condition(c)[0] == regime]:
                run_row = by[seed, regime, condition]; directory = execution / production.name(seed, condition); require(read(directory / 'result.json') == run_row, 'Run result identity')
                artifacts[str(directory / 'result.json')] = sha(directory / 'result.json'); artifacts[str(directory / 'training.jsonl')] = sha(directory / 'training.jsonl'); artifacts[str(directory / 'trajectory.jsonl')] = sha(directory / 'trajectory.jsonl')
                regime2, rule, live = parse_condition(condition); require(regime2 == regime and run_row['rule'] == rule and run_row['live'] == live, 'Run metadata')
                trajectory = [json.loads(line) for line in (directory / 'trajectory.jsonl').read_text().splitlines()]; compare(trajectory, run_row['trajectory'], 'Trajectory JSON')
                for step, tr in zip(STEPS, trajectory):
                    cp = directory / f'checkpoint_{step:04d}.npz'; artifacts[str(cp)] = sha(cp); require(sha(cp) == tr['checkpoint_sha256'], 'Checkpoint hash')
                    target_spec = independent[regime][TARGET]; target_arrays = arrays[regime][TARGET]
                    path = directory / f'trajectory_{step:04d}_{TARGET}.npz'; receipt = evaluate_saved(tr['evaluation'], path, target_arrays, response[regime][TARGET], groups[regime][TARGET], rule, live, cp); artifacts[str(path)] = sha(path); counts['trajectory_files'] += 1; counts['evaluation_worlds'] += receipt['worlds']; errors['probability'] = max(errors['probability'], receipt['max_probability_error']);
                # Check the complete training log and exact paired RNG metadata later.
                for part in PARTS:
                    if part == TARGET: continue
                    path = directory / f'final_{part}.npz'; receipt = evaluate_saved(run_row['final'][part], path, arrays[regime][part], response[regime][part], groups[regime][part], rule, live, directory / 'checkpoint_6000.npz'); artifacts[str(path)] = sha(path); counts['final_files'] += 1; counts['evaluation_worlds'] += receipt['worlds']; errors['probability'] = max(errors['probability'], receipt['max_probability_error'])
                # No unlisted scientific files in a run directory.
                expected_files = {directory / n for n in ('result.json', 'training.jsonl', 'trajectory.jsonl', 'checkpoint_0000.npz', 'checkpoint_0100.npz', 'checkpoint_0500.npz', 'checkpoint_1500.npz', 'checkpoint_3000.npz', 'checkpoint_6000.npz', 'status.json')}
                expected_files |= {directory / f'trajectory_{step:04d}_{TARGET}.npz' for step in STEPS} | {directory / f'final_{part}.npz' for part in PARTS if part != TARGET}
                require(set(directory.iterdir()) == expected_files, 'Unexpected run artifact')
    # Exact paired logs: within a regime both rule/channel arms see identical streams.
    for seed in SEEDS:
        for regime in REGIMES:
            selected = [c for c in CONDITIONS if parse_condition(c)[0] == regime]; streams = [(execution / production.name(seed, c) / 'training.jsonl').open() for c in selected]
            try:
                rows = list(zip_longest(*streams)); require(len(rows) == 6000, 'Training rows')
                for index, lines in enumerate(rows, 1):
                    require(all(line is not None for line in lines), 'Unequal paired logs'); parsed = [json.loads(line) for line in lines]
                    require(all(x['update'] == index and x['seed'] == seed and x['regime'] == regime for x in parsed), 'Log identity')
                    for key in ('world_uniforms_sha256', 'sample_uniforms_sha256', 'entropy_coefficient', 'batch_indices_sha256', 'batch_states_sha256'):
                        require(len({x[key] for x in parsed}) == 1, f'Paired {key}')
                    counts['training_rows'] += 1
            finally:
                for stream in streams: stream.close()
    # Cross-check frozen primary arithmetic from an independent implementation.
    recomputed = metrics.primary(result['runs']); compare(result['primary'], recomputed, 'Primary')
    require(result['budget']['evaluation_worlds'] == counts['evaluation_worlds'], 'Evaluation budget')
    require(counts['trajectory_files'] == 768 and counts['final_files'] == 384 and counts['training_rows'] == 768000, 'Audit scope')
    return dict(status='passed', completed_at=metrics_now(), plan_sha256=expected_plan_sha, counts=dict(counts), max_errors=dict(errors), primary=recomputed, artifacts_sha256=artifacts,
                scope=['All1152 complete evaluation NPZ files freshly forwarded and settled.', 'All768 checkpoints and768000 paired training rows checked.', 'No gradient or optimizer update replay; behavioral estimand remains distinct from a language score.'])


def metrics_now():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True); parser.add_argument('--plan-sha', required=True); parser.add_argument('--out'); parser.add_argument('--freeze', action='store_true'); args = parser.parse_args()
    run = Path(args.run).resolve()
    if args.freeze:
        require(args.out is None and not (run / 'execution').exists(), 'Freeze only before training')
        preflight = read(HERE / 'audit_preflight_001.json'); require(preflight['status'] == 'passed' and preflight['real_neural_forward_samples'] == 0, 'Pure preflight required')
        plan = read(run / 'plan.json'); require(sha(run / 'plan.json') == args.plan_sha == read(run / 'freeze.json')['plan_sha256'], 'Plan before freeze')
        value = dict(status='frozen_before_main_training', at=metrics_now(), plan_sha256=args.plan_sha,
                     source_sha256={str(p): sha(p) for p in (HERE / 'audit.py', HERE / 'test_audit.py', HERE / 'audit_preflight_001.json')},
                     real_neural_forward_samples=0, formal_output_files_read=0, training_gradient_replay=False)
        (HERE / 'audit_freeze_001.json').write_bytes(json.dumps(value, indent=2, ensure_ascii=False).encode()); print(json.dumps(value)); return
    require(args.out is not None, 'Audit output path required'); out = Path(args.out).resolve(); require(not out.exists(), 'Never overwrite audit output'); out.mkdir(parents=True)
    started = time.perf_counter(); (out / 'started.json').write_bytes(json.dumps(dict(at=metrics_now(), run=str(run), expected_plan_sha256=args.plan_sha), ensure_ascii=False).encode())
    try:
        value = audit(run, args.plan_sha); value['elapsed_seconds'] = time.perf_counter() - started; (out / 'verification.json').write_bytes(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False).encode()); print(json.dumps({'status': value['status'], 'counts': value['counts'], 'max_errors': value['max_errors'], 'elapsed_seconds': value['elapsed_seconds']}))
    except BaseException as error:
        (out / 'failure.json').write_bytes(json.dumps(dict(status='failed', error=repr(error), traceback=traceback.format_exc(), elapsed_seconds=time.perf_counter() - started), ensure_ascii=False).encode()); raise


if __name__ == '__main__':
    main()
