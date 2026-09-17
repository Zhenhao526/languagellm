"""Pure completed-run descriptions; no networks, checkpoints, or training calls.

Every natural/closed endpoint and monitor record is read from its actual NPZ.
Silent closed records are verified aliases. Truth comes from the unique native
plan in each actual material layout, never from an observed dominant role.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import time

import numpy as np

from research_program.triadic_action_dependency_study import metrics as native

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEEDS = (53101, 53102, 53103, 53104)
PAYOFFS = (('a50', .5), ('a10', .1))
CONDITIONS = ('FI_silent', 'PL_silent', 'PL_live')
PARTS = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')
STEPS = (0, 100, 500, 1500, 3000, 6000)
PAIR_NAMES = ('AB', 'AC', 'BC', 'invalid_role_configuration')
ROLE_OPTIONS = np.asarray(list(product(*[[-1] + [b for b in range(3) if b != a]
                                        for a in range(3)])), dtype=np.int8)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def role_codes(roles):
    r = np.asarray(roles)
    require(r.ndim == 2 and r.shape[1] == 3 and len(r) > 0
            and r.dtype.kind in 'iu', 'Roles require nonempty integer N×3')
    require(np.isin(r, (-1, 0, 1, 2)).all()
            and all(not np.any(r[:, a] == a) for a in range(3)), 'Illegal role/self partner')
    return ((r[:, 0].astype(np.int16) + 1) * 16
            + (r[:, 1].astype(np.int16) + 1) * 4 + r[:, 2] + 1).astype(np.uint8)


LEGAL_CODES = role_codes(ROLE_OPTIONS)
CODE_ROLES = np.full((64, 3), -9, dtype=np.int8)
CODE_ROLES[LEGAL_CODES] = ROLE_OPTIONS
PAIR_ROLES = np.asarray(((1, 0, -1), (2, -1, 0), (-1, 2, 1)), dtype=np.int8)
PAIR_CODES = role_codes(PAIR_ROLES)
PAIR_LOOKUP = np.full(64, 3, dtype=np.int8)
PAIR_LOOKUP[PAIR_CODES] = np.arange(3)


def _codes(values):
    values = np.asarray(values)
    require(values.ndim == 1 and len(values) and values.dtype.kind in 'iu'
            and np.isin(values, LEGAL_CODES).all(), 'Invalid saved role codes')
    return values.astype(np.uint8, copy=False)


def role_description(codes):
    codes = _codes(codes)
    n = len(codes)
    counts = np.bincount(codes, minlength=64)
    pairs = PAIR_LOOKUP[codes]
    pair_counts = np.bincount(pairs, minlength=4)
    roles = CODE_ROLES[codes]
    observed_pair_ids = np.flatnonzero(pair_counts[:3])
    fixed_pair = PAIR_NAMES[observed_pair_ids[0]] if len(observed_pair_ids) == 1 and pair_counts[3] == 0 else None
    return dict(
        all_27_joint_role_counts=[dict(partner_indices=r.tolist(), worlds=int(counts[c]))
                                 for r, c in zip(ROLE_OPTIONS, LEGAL_CODES)],
        observed_joint_role_count=int(np.count_nonzero(counts)),
        reciprocal_pair_counts={key: int(pair_counts[i]) for i, key in enumerate(PAIR_NAMES)},
        reciprocal_pair_rates={key: float(pair_counts[i] / n) for i, key in enumerate(PAIR_NAMES)},
        observed_reciprocal_pairs=[PAIR_NAMES[i] for i in observed_pair_ids],
        constant_joint_role_across_all_worlds=bool(np.count_nonzero(counts) == 1),
        all_worlds_same_reciprocal_pair=fixed_pair,
        only_one_reciprocal_pair_when_roles_form_pair=(PAIR_NAMES[observed_pair_ids[0]] if len(observed_pair_ids) == 1 else None),
        maximum_reciprocal_pair_fraction=float(pair_counts[:3].max() / n),
        any_world_uses_a_different_joint_role=bool(np.count_nonzero(counts) > 1),
        any_actor_uses_both_possible_partners_across_worlds=bool(any(len(set(roles[:, a]) - {-1}) == 2 for a in range(3))),
        actors={native.env.AGENTS[a]: dict(
            wait_worlds=int(np.count_nonzero(roles[:, a] == -1)),
            partner_worlds={native.env.AGENTS[b]: int(np.count_nonzero(roles[:, a] == b)) for b in range(3) if b != a},
            distinct_partner_count_excluding_wait=len(set(map(int, roles[:, a])) - {-1})) for a in range(3)},
        role_pair_scope='Reciprocal partner proposals with the third agent waiting; sites and destinations may still disagree. Physical execution is reported separately.')


def world_summary(states, actions, alpha):
    require(alpha in (.5, .1) and not isinstance(alpha, (bool, np.bool_)), 'Unsupported partial utility')
    packed = native._states(states)
    actions, _ = native._policy(actions, None, len(packed))
    truth = native.truth_actions(packed)
    settled = native.settle_arrays(packed, actions)
    truth_roles = native.PARTNER[np.arange(3), truth]
    roles = settled['partners']
    actual_code = role_codes(roles)
    truth_code = role_codes(truth_roles)
    truth_pair = PAIR_LOOKUP[truth_code]
    require(np.all(truth_pair < 3), 'Truth must be one reciprocal active pair with the third waiting')
    role_ok = actual_code == truth_code
    full = np.all(actions == truth, axis=1)
    reward = settled['reward']
    require(np.array_equal(full, reward == 1), 'Unique native full plan disagrees with settlement')
    utility = np.where(reward == .5, alpha, reward)
    physical = settled['executed'].any(1)
    n = len(packed)

    def aggregate(mask):
        count = int(np.count_nonzero(mask))
        if not count:
            return dict(worlds=0, role_success_rate=None, full_success_rate=None,
                        reward_mean=None, utility_mean=None, physical_execution_rate=None,
                        role_success_worlds=0, full_success_worlds=0)
        return dict(worlds=count, role_success_rate=float(role_ok[mask].mean()),
                    full_success_rate=float(full[mask].mean()), reward_mean=float(reward[mask].mean()),
                    utility_mean=float(utility[mask].mean()), physical_execution_rate=float(physical[mask].mean()),
                    role_success_worlds=int(role_ok[mask].sum()), full_success_worlds=int(full[mask].sum()))

    summary = aggregate(np.ones(n, dtype=bool))
    summary.update(native_outcome_counts={str(v): int(np.count_nonzero(reward == v)) for v in (0., .5, 1.)},
                   partial_utility=float(alpha),
                   truth_pair_strata={PAIR_NAMES[p]: {**aggregate(truth_pair == p),
                        'actual_reciprocal_pair_counts': {PAIR_NAMES[q]: int(np.count_nonzero((truth_pair == p) & (PAIR_LOOKUP[actual_code] == q))) for q in range(4)}} for p in range(3)},
                   role_description=role_description(actual_code),
                   all_actual_joint_action_counts=[dict(action_indices=a.tolist(), worlds=int(c)) for a, c in zip(*np.unique(actions, axis=0, return_counts=True))])
    derived = dict(role_codes=actual_code, truth_role_codes=truth_code,
                   reward=reward, utility=utility, executed=settled['executed'], satisfied=settled['satisfied'])
    return summary, derived


def role_transition(before, after, truth):
    before, after, truth = map(_codes, (before, after, truth))
    require(before.shape == after.shape == truth.shape, 'Transition worlds must align exactly')
    n = len(before)
    first, second = CODE_ROLES[before], CODE_ROLES[after]
    before_desc, after_desc = role_description(before), role_description(after)
    before_pair, after_pair = PAIR_LOOKUP[before], PAIR_LOOKUP[after]
    require((PAIR_LOOKUP[truth] < 3).all(), 'Invalid true role in transition')
    joint = before.astype(np.int64) * 64 + after
    values, counts = np.unique(joint, return_counts=True)
    matrix = np.bincount(before_pair.astype(np.int64) * 4 + after_pair, minlength=16).reshape(4, 4)
    fixed_before = before_desc['all_worlds_same_reciprocal_pair']
    fixed_after = after_desc['all_worlds_same_reciprocal_pair']
    changed = before != after
    correct_before, correct_after = before == truth, after == truth
    return dict(worlds=n, joint_role_changed_worlds=int(changed.sum()), joint_role_changed_rate=float(changed.mean()),
        any_world_changes_joint_role=bool(changed.any()),
        only_changed_from_one_all_worlds_fixed_pair_to_another=bool(fixed_before and fixed_after and fixed_before != fixed_after),
        before_all_worlds_fixed_pair=fixed_before, after_all_worlds_fixed_pair=fixed_after,
        newly_role_correct_worlds=int((~correct_before & correct_after).sum()),
        newly_role_incorrect_worlds=int((correct_before & ~correct_after).sum()),
        role_success_rate_difference=float(correct_after.mean() - correct_before.mean()),
        pair_order=list(PAIR_NAMES), reciprocal_pair_transition_counts=matrix.tolist(),
        all_observed_joint_role_transitions=[dict(before_partner_indices=CODE_ROLES[v // 64].tolist(),
                                                after_partner_indices=CODE_ROLES[v % 64].tolist(), worlds=int(c)) for v, c in zip(values, counts)],
        actors={native.env.AGENTS[a]: dict(role_changed_including_wait_worlds=int((first[:, a] != second[:, a]).sum()),
            changed_partner_while_active_in_both_worlds=int(((first[:, a] >= 0) & (second[:, a] >= 0) & (first[:, a] != second[:, a])).sum())) for a in range(3)},
        truth_pair_strata={PAIR_NAMES[p]: dict(worlds=int((PAIR_LOOKUP[truth] == p).sum()),
            joint_role_changed_worlds=int((changed & (PAIR_LOOKUP[truth] == p)).sum()),
            newly_role_correct_worlds=int((~correct_before & correct_after & (PAIR_LOOKUP[truth] == p)).sum()),
            newly_role_incorrect_worlds=int((correct_before & ~correct_after & (PAIR_LOOKUP[truth] == p)).sum())) for p in range(3)})


def expected_states(spec, ids):
    ids = np.asarray(ids)
    require(ids.ndim == 1 and ids.dtype.kind in 'iu' and len(ids)
            and ((ids >= 0) & (ids < spec['world_count'])).all(), 'Invalid state indices')
    require(np.all(np.diff(ids.astype(np.int64)) > 0), 'Indices must be unique and ascending')
    nl, no = len(spec['layouts']), len(spec['private_sites'])
    require(spec['world_count'] == len(spec['needs']) * nl * no, 'Static world count disagrees')
    return np.concatenate((np.asarray(spec['needs'])[ids // (nl * no)],
                           np.asarray(spec['layouts'])[(ids // no) % nl],
                           np.asarray(spec['private_sites'])[ids % no]), axis=1)


def array_sha(value):
    a = np.ascontiguousarray(value)
    metadata = (json.dumps({'shape': list(a.shape), 'dtype': a.dtype.str}, ensure_ascii=False,
                           sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
    return sha256(metadata + a.tobytes()).hexdigest()


MEAN_FIELDS = {
    'conditional_exact_expected_reward': 'expected_reward_given_greedy_messages',
    'conditional_exact_expected_utility': 'expected_utility_given_greedy_messages',
    'conditional_exact_full_success_probability': 'full_probability_given_greedy_messages',
    'conditional_full_posterior_mass': 'full_posterior_mass_given_greedy_messages',
    'conditional_exact_execution_probability': 'execution_probability_given_greedy_messages',
}


def summarize_record(record, spec, expected_ids, alpha, expected_path, bindings, *, semantic=False, pairs=None):
    path = Path(record['path']).resolve()
    require(path == Path(expected_path).resolve(), 'Record path does not match its own run/partition/mode')
    require(sha(path) == record['data_sha256'], 'Changed NPZ bytes: ' + str(path))
    bindings[str(path)] = record['data_sha256']
    with np.load(path, allow_pickle=False) as archive:
        needed = ('states', 'state_indices', 'action_indices', 'greedy_reward', 'greedy_utility', 'executed', 'satisfied') + tuple(MEAN_FIELDS)
        require(set(needed) <= set(archive.files), 'NPZ missing required saved arrays')
        data = {key: archive[key] for key in needed}
        if semantic:
            data['action_probabilities'] = archive['action_probabilities']
    ids = data['state_indices']; n = len(ids)
    require(np.array_equal(ids, expected_ids), 'Saved world indices do not match complete endpoint/monitor domain')
    require(record['worlds'] == n and array_sha(ids) == record['state_indices_sha256'], 'World index identity disagrees')
    require(np.array_equal(data['states'], expected_states(spec, ids)), 'Packed worlds disagree with static IDs')
    summary, derived = world_summary(data['states'], data['action_indices'], alpha)
    for saved, computed in (('greedy_reward', 'reward'), ('greedy_utility', 'utility'), ('executed', 'executed'), ('satisfied', 'satisfied')):
        require(np.array_equal(data[saved], derived[computed]), 'Native/utility settlement differs: ' + saved)
    for key in ('reward_mean', 'utility_mean', 'full_success_rate', 'role_success_rate', 'physical_execution_rate'):
        require(abs(summary[key] - record[key]) <= 2e-12, 'Saved aggregate differs: ' + key)
    actual = summary['all_actual_joint_action_counts']
    require(actual == record['raw_joint_action_counts'], 'Saved joint action frequencies differ')
    observed_roles = [r for r in summary['role_description']['all_27_joint_role_counts'] if r['worlds']]
    require(observed_roles == record['raw_joint_role_counts'], 'Saved joint role frequencies differ')
    expectations = {}
    for key, label in MEAN_FIELDS.items():
        value = data[key]
        require(value.shape == (n,) and np.isfinite(value).all()
                and ((value >= -2e-12) & (value <= 1 + 2e-12)).all(), 'Invalid saved probability/expectation: ' + key)
        expectations[label] = float(value.mean())
        if label in record:
            require(abs(expectations[label] - record[label]) <= 2e-12, 'Saved expectation mean differs: ' + key)
    full = data['conditional_exact_full_success_probability']
    partial = 2 * (data['conditional_exact_expected_reward'] - full)
    require((partial >= -2e-12).all() and (partial <= 1 + 2e-12).all(), 'Partial probability reconstructed outside support')
    u = data['conditional_exact_expected_utility']
    require(np.allclose(u, full + alpha * partial, rtol=0, atol=2e-12), 'Expected utility fails F + alpha P')
    stable = u > 1e-280
    require(np.allclose(data['conditional_full_posterior_mass'][stable], full[stable] / u[stable], rtol=2e-10, atol=2e-12), 'Full posterior differs from per-world F/U')
    expectations['partial_probability_given_greedy_messages'] = float(partial.mean())
    expectations['full_posterior_direct_ratio_checked_worlds'] = int(stable.sum())
    summary.update(path=str(path), data_sha256=record['data_sha256'], state_indices_sha256=record['state_indices_sha256'],
                   saved_expectation_means=expectations,
                   expected_value_scope='Saved independent action sampling conditional on greedy messages; arithmetic checked here, no probability network or sender sampling replay.',
                   reused_natural=False)
    if semantic:
        summary['semantic_metrics'] = native.semantic_metrics(spec=spec, states=data['states'], state_indices=ids,
            action_indices=data['action_indices'], action_probabilities=data['action_probabilities'], pairs=pairs, include_rows=False)
        require(summary['semantic_metrics']['coverage'] == 'full_partition', 'Endpoint semantic coverage is incomplete')
        summary['semantic_interpretation'] = ('Frozen content/role case families; three axes equal, then six ordered sender/listener strata equal. '
            'The role family changes requirements and associated unique actions; it is not a pure-partner intervention. '
            'Natural two-endpoint appropriateness is not causal message use or compositionality.')
        if record['information'] == 'PL' and not record['live']:
            for family in ('content', 'role'):
                require(summary['semantic_metrics'][family]['macro']['both_endpoints_apt'] == 0,
                        'PL silent/closed same-observation structural zero failed')
    return summary, derived['role_codes'], derived['truth_role_codes']


def primary_from_points(points):
    rows = []
    for seed in SEEDS:
        cells = {f'{p}_{c}': points[(seed, p, c, 'final', 'new_needs_and_layouts')]['natural'] for p, _ in PAYOFFS for c in CONDITIONS}
        contrasts = {metric: (cells['a10_PL_live'][metric] - cells['a10_PL_silent'][metric])
                     - (cells['a50_PL_live'][metric] - cells['a50_PL_silent'][metric])
                     for metric in ('role_success_rate', 'full_success_rate', 'reward_mean')}
        rows.append(dict(seed=seed, contrasts=contrasts, cells={key: {m: row[m] for m in ('role_success_rate', 'full_success_rate', 'reward_mean', 'utility_mean')} for key, row in cells.items()}))
    return dict(metric='role_success_rate', partition='new_needs_and_layouts', paired_seeds=rows,
                mean_difference=float(np.mean([r['contrasts']['role_success_rate'] for r in rows])))


def summarize(run, out):
    run, out = Path(run).resolve(), Path(out).resolve()
    require(not out.exists(), 'Summary output already exists')
    # Completion gate must precede output creation and any actual NPZ read.
    require(not (run / 'execution/failure.json').exists(), 'Failed execution is not a completed study')
    require(read(run / 'execution/status.json')['status'] == 'completed', 'Main execution is not completed')
    results = read(run / 'execution/results.json')
    require(results['status'] == 'completed', 'Main result is not completed')
    plan, static, freeze = read(run / 'plan.json'), read(run / 'prepared.json'), read(run / 'freeze.json')
    plan_sha = sha(run / 'plan.json')
    require(plan_sha == freeze['plan_sha256'] == results['plan_sha256'], 'Main plan identities differ')
    require(sha(run / 'prepared.json') == plan['prepared_sha256'], 'Static data identity changed')
    require(tuple(plan['config']['seeds']) == SEEDS and tuple(plan['config']['conditions']) == CONDITIONS, 'Unexpected cohort')
    require(plan['config']['primary'] == '6000_double_holdout_world_role_success_(a10_PL_live-a10_PL_silent)-(a50_PL_live-a50_PL_silent)', 'Frozen primary differs')
    expected_runs = [(s, p, c) for s in SEEDS for p, _ in PAYOFFS for c in CONDITIONS]
    require([(r['seed'], r['payoff'], r['condition']) for r in results['runs']] == expected_runs, 'Incomplete/reordered 24 runs')
    bindings = {str(run / f): sha(run / f) for f in ('plan.json', 'prepared.json', 'freeze.json', 'execution/results.json', 'execution/status.json')}
    for module in (native, native.env, native.dataset):
        source = Path(module.__file__).resolve()
        require(str(source) in plan['source_sha256'], 'Native measurement source not bound by main plan')
        digest = plan['source_sha256'][str(source)]
        snapshot = run / 'source_snapshot' / source.relative_to(ROOT)
        require(sha(source) == digest and sha(snapshot) == digest, 'Changed native measurement source/snapshot')
        bindings[str(source)] = digest; bindings[str(snapshot)] = digest
    out.mkdir(parents=True)
    start = time.perf_counter()
    write(out / 'started.json', dict(main_plan_sha256=plan_sha, main_result_sha256=bindings[str(run / 'execution/results.json')]))
    points = {}; summaries = []; transitions = []; files_read = 0; aliases = 0; rows_read = 0
    try:
        pairs_by_part = {part: native.dataset.content_pairs(static['partitions'][part]) for part in PARTS}
        for part, pairs in pairs_by_part.items():
            write(out / f'fixed_semantic_cases_{part}.json', pairs)
        by = {(r['seed'], r['payoff'], r['condition']): r for r in results['runs']}
        for seed in SEEDS:
            cache = {}; truth_by_domain = {}
            for payoff, alpha in PAYOFFS:
                for condition in CONDITIONS:
                    r = by[seed, payoff, condition]
                    directory = run / 'execution' / f'seed_{seed}_{payoff}_{condition}'
                    require(read(directory / 'result.json') == r, 'Embedded result differs from per-run result')
                    bindings[str(directory / 'result.json')] = sha(directory / 'result.json')
                    require(r['updates'] == 6000 and r['partial_utility'] == alpha, 'Wrong update count or payoff')
                    require([v['update'] for v in r['monitor']] == list(STEPS), 'Missing/reordered checkpoints')
                    phases = [(f'monitor_{m["update"]:04d}', m['monitor']) for m in r['monitor']] + [('final', r['final'])]
                    for phase, evaluations in phases:
                        require(set(evaluations) == set(PARTS), 'Missing partition')
                        for part in PARTS:
                            spec = static['partitions'][part]
                            ids = np.arange(spec['world_count'], dtype=np.int64) if phase == 'final' else np.asarray(spec['monitor_indices'], dtype=np.int64)
                            modes = evaluations[part]
                            require(set(modes) == {'natural', 'closed'}, 'Missing evaluation mode')
                            require(not modes['natural']['reused_natural'], 'Natural output is an alias')
                            point = dict(seed=seed, payoff=payoff, partial_utility=alpha, condition=condition, phase=phase,
                                         update=6000 if phase == 'final' else int(phase.split('_')[1]), partition=part,
                                         distribution='complete_uniform_partition' if phase == 'final' else 'fixed_monitor_worlds_only')
                            for mode in ('natural', 'closed'):
                                rec = modes[mode]
                                require(rec['information'] == condition.split('_')[0] and rec['partial_utility'] == alpha, 'Record cell metadata mismatch')
                                if mode == 'closed' and condition.endswith('silent'):
                                    alias = deepcopy(modes['natural']); alias['reused_natural'] = True
                                    require(rec == alias, 'Silent closed is not an exact natural alias')
                                    point[mode] = deepcopy(point['natural']); point[mode]['reused_natural'] = True
                                    cache[payoff, condition, phase, part, mode] = cache[payoff, condition, phase, part, 'natural']
                                    aliases += 1
                                    continue
                                expected_live = condition.endswith('live') and mode == 'natural'
                                require(rec['live'] is expected_live and not rec['reused_natural'], 'Actual record visibility/alias differs')
                                record_path = directory / f'{phase}_{part}_{mode}.npz'
                                summary, codes, truth = summarize_record(rec, spec, ids, alpha, record_path, bindings,
                                    semantic=phase == 'final', pairs=pairs_by_part[part])
                                point[mode] = summary; files_read += 1; rows_read += len(codes)
                                cache[payoff, condition, phase, part, mode] = codes
                                domain = (phase, part)
                                if domain in truth_by_domain:
                                    require(np.array_equal(truth_by_domain[domain], truth), 'Truth roles changed between experimental cells')
                                else:
                                    truth_by_domain[domain] = truth
                            point['natural_minus_closed'] = {key: point['natural'][key] - point['closed'][key] for key in ('role_success_rate', 'full_success_rate', 'reward_mean', 'utility_mean', 'physical_execution_rate')}
                            if phase == 'final':
                                point['semantic_natural_minus_closed'] = {family: {
                                    key: point['natural']['semantic_metrics'][family]['macro'][key] - point['closed']['semantic_metrics'][family]['macro'][key]
                                    for key in native.PAIR_METRICS} for family in ('content', 'role')}
                            key = (seed, payoff, condition, phase, part); points[key] = point
                            path = out / f'{seed}_{payoff}_{condition}_{phase}_{part}.json'
                            write(path, point); summaries.append(dict(path=str(path), sha256=sha(path)))
            for part in PARTS:
                for condition in CONDITIONS:
                    for phase in [f'monitor_{s:04d}' for s in STEPS] + ['final']:
                        for mode in ('natural', 'closed'):
                            a = cache['a50', condition, phase, part, mode]
                            b = cache['a10', condition, phase, part, mode]
                            transitions.append(dict(kind='payoff_a50_to_a10', seed=seed, condition=condition, partition=part, phase=phase, mode=mode,
                                                    **role_transition(a, b, truth_by_domain[phase, part])))
                    for payoff, _ in PAYOFFS:
                        if condition == 'PL_live':
                            for phase in [f'monitor_{s:04d}' for s in STEPS] + ['final']:
                                transitions.append(dict(kind='natural_to_closed', seed=seed, payoff=payoff, condition=condition,
                                    partition=part, phase=phase,
                                    **role_transition(cache[payoff, condition, phase, part, 'natural'],
                                                      cache[payoff, condition, phase, part, 'closed'], truth_by_domain[phase, part])))
                        for first, second in zip(STEPS, STEPS[1:]):
                            ph1, ph2 = f'monitor_{first:04d}', f'monitor_{second:04d}'
                            require(np.array_equal(truth_by_domain[ph1, part], truth_by_domain[ph2, part]), 'Monitor truth ordering differs between times')
                            for mode in ('natural', 'closed'):
                                transitions.append(dict(kind='adjacent_monitor_time', seed=seed, payoff=payoff, condition=condition, partition=part,
                                                        before_update=first, after_update=second, mode=mode,
                                                        **role_transition(cache[payoff, condition, ph1, part, mode], cache[payoff, condition, ph2, part, mode], truth_by_domain[ph1, part])))
            del cache, truth_by_domain
        require(files_read == 896 and aliases == 448 and len(summaries) == 672, 'Incomplete actual/alias/point coverage')
        primary = primary_from_points(points)
        require(primary == results['primary'], 'Raw-data frozen primary differs from main result')
        write(out / 'role_transitions.json', transitions)
        endpoint_rows = []
        for s, p, c in expected_runs:
            for part in PARTS:
                v = points[s, p, c, 'final', part]
                endpoint_rows.append({k: v[k] for k in ('seed', 'payoff', 'partial_utility', 'condition', 'partition', 'distribution', 'natural', 'closed', 'natural_minus_closed', 'semantic_natural_minus_closed')})
        write(out / 'all_endpoint_cells.json', endpoint_rows)
        for path, digest in bindings.items():
            require(sha(path) == digest, 'Bound input changed during postprocessing: ' + path)
        report = dict(status='completed', schema='partial_payoff_description_v1', main_plan_sha256=plan_sha,
            main_result_sha256=bindings[str(run / 'execution/results.json')], primary=primary,
            coverage=dict(policies=24, paired_initializations=4, partitions=4, monitor_checkpoints=6, full_endpoints_per_policy=4,
                          actual_npz_read=files_read, actual_saved_world_rows_read=rows_read, silent_closed_alias_records=aliases,
                          point_summaries=len(summaries), role_transition_descriptions=len(transitions),
                          semantic_actual_endpoint_analyses=128, semantic_silent_closed_aliases=64),
            no_new_computation=dict(network_forward_calls=0, parameter_loads=0, training_updates=0),
            semantic_scope='All four full endpoint partitions use existing fixed content/role cases with three-axis/six-stratum macro and all axis/S-L rows. Monitors have native and role descriptions only. Role cases are not pure-partner interventions. True pair strata use each world unique full native plan. Role variation alone does not prove task adaptation or partner expression.',
            probability_verification_scope='Saved scalar arrays and their arithmetic/means checked; original neural probabilities and optimization are not replayed by this descriptive script.',
            point_files=summaries,
            artifact_files={f: dict(path=str(out / f), sha256=sha(out / f)) for f in ('role_transitions.json', 'all_endpoint_cells.json', *[f'fixed_semantic_cases_{part}.json' for part in PARTS])},
            sources_sha256={str(p): sha(p) for p in (Path(__file__), HERE / 'tests/test_summarize_results.py', Path(native.__file__), Path(native.env.__file__))},
            input_bindings_sha256=bindings, elapsed_seconds=time.perf_counter() - start)
        write(out / 'summary.json', report)
        write(out / 'status.json', dict(status='completed'))
        return {k: report[k] for k in ('status', 'primary', 'coverage', 'elapsed_seconds')}
    except BaseException as error:
        write(out / 'failure.json', dict(status='failed', error=repr(error), elapsed_seconds=time.perf_counter() - start))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.run, args.out), ensure_ascii=False))
