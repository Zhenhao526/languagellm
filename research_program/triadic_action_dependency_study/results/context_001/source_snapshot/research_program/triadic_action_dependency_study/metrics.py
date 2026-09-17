"""Pure measurements for the frozen action-dependency task, without model calls.

The native world distribution and the equal-axis content-pair distribution are
separate estimands. A pair's two endpoints are not independent replicates.
Canonical plans identify a MATERIAL; actual layouts determine action SITE.
No truth labels from this module are supplied to actors or training.
"""
from functools import lru_cache
from itertools import combinations, product
import numpy as np
from . import environment as env
from . import dataset

AXES = ('kind', 'length', 'destination')
PARTITIONS = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')
CONDITIONS = ('FI_silent', 'FI_live', 'PL_silent', 'PL_live', 'LL_silent', 'LL_live')
SEEDS = (51101, 51102, 51103, 51104)
PROB_ATOL = 1e-10
ACTIONS = tuple(tuple(env.all_actions(a)) for a in env.AGENTS)
PARTNER = np.full((3, 17), -1, dtype=np.int8)
SITE = np.full((3, 17), -1, dtype=np.int8)
DESTINATION = np.full((3, 17), -1, dtype=np.int8)
for _a in range(3):
    for _k, _action in enumerate(ACTIONS[_a]):
        if _k:
            PARTNER[_a, _k] = env.AGENTS.index(_action['partner'])
            SITE[_a, _k] = env.SITES.index(_action['site'])
            DESTINATION[_a, _k] = env.DESTINATIONS.index(_action['destination'])
ACCEPTANCE = np.asarray([[[env.accepts(n, m, d) for d in range(2)]
                          for m in range(4)] for n in range(24)], dtype=bool)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value, shape, low, high, name):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'iu', name + ' shape/dtype')
    require(np.all((a >= low) & (a <= high)), name + ' range')
    return a


def _states(value):
    a = np.asarray(value)
    require(a.ndim == 2 and a.shape[1] == 10 and len(a) > 0 and a.dtype.kind in 'iu',
            'states must be nonempty integer N×10')
    require(np.all((a[:, :3] >= 0) & (a[:, :3] < 24)), 'invalid needs')
    require(np.all(np.sort(a[:, 3:7], axis=1) == np.arange(4)), 'invalid actual layouts')
    require(np.all(np.sort(a[:, 7:10], axis=1) == np.arange(1, 4)), 'invalid private-site owners')
    return a


def _policy(action_indices, probabilities, n):
    actions = _integer(action_indices, (n, 3), 0, 16, 'action_indices')
    if probabilities is None:
        return actions, None
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.shape == (n, 3, 17) and np.isfinite(p).all()
            and np.all((p >= 0) & (p <= 1)), 'invalid action_probabilities')
    require(np.allclose(p.sum(-1), 1, atol=PROB_ATOL, rtol=0), 'probabilities not normalized')
    require(np.array_equal(actions, np.argmax(p, axis=-1)),
            'actions must be first argmax of saved probabilities, not raw logits')
    return actions, p


def _plan_actions(i, j, site, destination):
    result = [0, 0, 0]
    for who, partner in ((i, j), (j, i)):
        action = dict(kind='transport', site=env.SITES[site],
                      destination=env.DESTINATIONS[destination], partner=env.AGENTS[partner])
        result[who] = ACTIONS[who].index(action)
    return result


STRUCTURAL_PLANS = tuple((i, j, s, d) for i, j in combinations(range(3), 2)
                         for s, d in product(range(4), range(2)))
STRUCTURAL_ACTIONS = np.asarray([_plan_actions(*p) for p in STRUCTURAL_PLANS], dtype=np.int16)


@lru_cache(maxsize=13824)
def _need_truth(needs):
    plans = env.full_success_plans(needs, (0, 1, 2, 3))
    require(len(plans) == 1, 'measurement requires exactly one native full-success plan')
    i, j, canonical_material, destination = plans[0]
    return i, j, canonical_material, destination


def truth_actions(states):
    """Unique native full-success action, using current material→site mapping.

    env.full_success_plans is cached by need triple. Its canonical site is a
    material ID only. It is NEVER compared directly with an actual site action.
    """
    packed = _states(states)
    needs, inverse = np.unique(packed[:, :3], axis=0, return_inverse=True)
    plans = np.asarray([_need_truth(tuple(map(int, n))) for n in needs], dtype=np.int16)[inverse]
    sites = np.argmax(packed[:, 3:7] == plans[:, 2, None], axis=1)
    result = np.zeros((len(packed), 3), dtype=np.int16)
    for who in range(3):
        active = (plans[:, 0] == who) | (plans[:, 1] == who)
        other = np.where(plans[:, 0] == who, plans[:, 1], plans[:, 0])
        partner_slot = np.asarray([a for a in range(3) if a != who])
        slot = np.argmax(other[:, None] == partner_slot, axis=1)
        result[active, who] = 1 + 4 * sites[active] + 2 * plans[active, 3] + slot[active]
    return result


def settle_arrays(states, action_indices):
    """All 17 actions remain possible; invalid joint coordination receives R=0."""
    packed = _states(states)
    actions, _ = _policy(action_indices, None, len(packed))
    active = actions != 0
    partners = PARTNER[np.arange(3), actions]
    sites = SITE[np.arange(3), actions]
    dest = DESTINATION[np.arange(3), actions]
    executed = np.zeros_like(active)
    for who in range(3):
        for other in range(3):
            if who == other:
                continue
            executed[:, who] |= ((active.sum(1) == 2) & active[:, who] & active[:, other]
                & (partners[:, who] == other) & (partners[:, other] == who)
                & (sites[:, who] == sites[:, other]) & (dest[:, who] == dest[:, other]))
    material = packed[np.arange(len(packed))[:, None], 3 + np.maximum(sites, 0)]
    satisfied = executed & ACCEPTANCE[packed[:, :3], material, np.maximum(dest, 0)]
    return dict(reward=satisfied.sum(1) / 2, executed=executed, satisfied=satisfied,
                active=active, partners=partners, sites=sites, destinations=dest)


def _fraction(value):
    return float(np.asarray(value, dtype=np.float64).mean())


def world_metrics(*, states, action_indices, action_probabilities=None, greedy_reward=None):
    """Uniform saved world set; full domain and two-background monitor must be labeled separately."""
    packed = _states(states); n = len(packed)
    actions, probs = _policy(action_indices, action_probabilities, n)
    truth = truth_actions(packed)
    settled = settle_arrays(packed, actions)
    reward = settled['reward']
    if greedy_reward is not None:
        require(np.array_equal(np.asarray(greedy_reward), reward), 'saved reward differs from native settlement')
    target_partner = PARTNER[np.arange(3), truth]
    target_active = truth != 0
    role = settled['partners'] == target_partner
    site = settled['active'] & (settled['sites'] == SITE[np.arange(3), truth])
    dest = settled['active'] & (settled['destinations'] == DESTINATION[np.arange(3), truth])
    site_joint = np.all(site | ~target_active, axis=1)
    dest_joint = np.all(dest | ~target_active, axis=1)
    role_joint = role.all(1)
    complete = np.all(actions == truth, axis=1)
    require(np.array_equal(complete, reward == 1), 'unique action and native full-success disagree')
    require(np.array_equal(complete, role_joint & site_joint & dest_joint), 'component reconstruction differs')
    joint, counts = np.unique(actions, axis=0, return_counts=True)
    result = dict(schema='action_dependency_world_metrics_v1', worlds=n,
        reward_mean=_fraction(reward), full_success_rate=_fraction(complete),
        full_success_worlds=int(complete.sum()), failure_worlds=int((~complete).sum()),
        reward_counts={str(r): int(np.count_nonzero(reward == r)) for r in (0., .5, 1.)},
        role_success_rate=_fraction(role_joint), site_success_rate=_fraction(site_joint),
        destination_success_rate=_fraction(dest_joint),
        role_site_success_rate=_fraction(role_joint & site_joint),
        role_destination_success_rate=_fraction(role_joint & dest_joint),
        site_destination_success_rate=_fraction(site_joint & dest_joint),
        role_site_destination_success_rate=_fraction(complete),
        individual_role_accuracy=_fraction(role),
        required_active_site_accuracy=_fraction(site[target_active]),
        required_active_destination_accuracy=_fraction(dest[target_active]),
        physical_execution_rate=_fraction(settled['executed'].any(1)),
        actual_active_agent_count={str(k): int(np.count_nonzero(settled['active'].sum(1) == k)) for k in range(4)},
        joint_action_count=len(joint),
        joint_action_distribution=[dict(action_indices=a.tolist(), worlds=int(c)) for a, c in zip(joint, counts)],
        agent_proposals={agent: dict(wait_worlds=int(np.count_nonzero(actions[:, who] == 0)),
            partner_worlds={other: int(np.count_nonzero(settled['partners'][:, who] == j))
                for j, other in enumerate(env.AGENTS) if j != who},
            distinct_action_indices=np.unique(actions[:, who]).tolist()) for who, agent in enumerate(env.AGENTS)},
        component_scope='Role includes wait versus active and the stated partner for all three agents. Site/destination require both true active actors to propose transport at the correct value, without conditioning on role correctness; the third actor is assessed by role. Individual site/destination denominators are 2N, role denominator 3N.',
        distribution='Uniform over supplied saved worlds; not semantic-pair weights.')
    if probs is not None:
        e = np.zeros(n); physical = np.zeros(n)
        for plan, joint in zip(STRUCTURAL_PLANS, STRUCTURAL_ACTIONS):
            i, j, s, d = plan
            q = np.prod(probs[:, np.arange(3), joint], axis=1)
            mat = packed[:, 3 + s]
            r = (ACCEPTANCE[packed[:, i], mat, d].astype(float)
                 + ACCEPTANCE[packed[:, j], mat, d]) / 2
            e += q * r; physical += q
        full = np.prod(probs[np.arange(n)[:, None], np.arange(3), truth], axis=1)
        result.update(expected_reward_given_saved_messages=_fraction(e),
                      full_probability_given_saved_messages=_fraction(full),
                      physical_probability_given_saved_messages=_fraction(physical),
                      probability_scope='Independent receiver action sampling conditional on these saved greedy messages; no sender sampling or new rollout.')
    return result


@lru_cache(maxsize=8)
def _expected_cases(needs):
    spec = dict(needs=needs, layouts=[[0, 1, 2, 3]], private_sites=[[1, 2, 3]])
    return dataset.content_pairs(spec)['rows']


def _saved_grid(spec, states, state_indices):
    packed = _states(states)
    ids = _integer(state_indices, (len(packed),), 0, spec['world_count'] - 1, 'state_indices')
    require(np.all(np.diff(ids.astype(np.int64)) > 0), 'state_indices must be unique and ascending')
    nn, nl, no = len(spec['needs']), len(spec['layouts']), len(spec['private_sites'])
    require(spec['world_count'] == nn * nl * no, 'spec world count disagrees')
    nb = nl * no
    backgrounds = np.unique(ids % nb)
    expected_ids = (np.arange(nn, dtype=np.int64)[:, None] * nb + backgrounds).ravel()
    require(np.array_equal(ids, expected_ids), 'Every need must have all the same saved backgrounds; no missing pair endpoint')
    expected = np.concatenate((np.asarray(spec['needs'])[ids // nb],
        np.asarray(spec['layouts'])[(ids // no) % nl], np.asarray(spec['private_sites'])[ids % no]), axis=1)
    require(np.array_equal(packed, expected), 'Saved states do not match actual spec/index ordering')
    return packed, ids, backgrounds


PAIR_METRICS = ('both_endpoints_apt', 'mean_endpoint_apt', 'endpoint0_apt', 'endpoint1_apt',
    'mean_endpoint_correct_probability', 'both_endpoint_correct_probability_product',
    'listener_action_changed', 'both_team_full_success', 'mean_team_full_success', 'mean_native_reward')


def _case_aggregate(case_means, rows, classification):
    selected = np.asarray([r['classification'] == classification for r in rows])
    group = dict(case_count=int(selected.sum()), by_axis={}, macro=None,
                 weighting='Three axes equal; six ordered sender/listener strata equal within axis; all predeclared cases equal within stratum; all saved backgrounds equal within case.')
    if not selected.any():
        group['availability'] = 'no_cases'; return group
    all_axes = []
    for axis, name in enumerate(AXES):
        strata = {}; stratum_means = []
        for sender in range(3):
            for listener in range(3):
                if sender == listener:
                    continue
                mask = selected & np.asarray([r['axis_index'] == axis and r['sender'] == sender
                                              and r['listener'] == listener for r in rows])
                count = int(mask.sum())
                for r in (rows[k] for k in np.flatnonzero(mask)):
                    require(r['within_axis_case_weight_denominator'] == 6 * count,
                            'Declared case weights do not equal six-strata uniform weights')
                means = {k: _fraction(v[mask]) for k, v in case_means.items()} if count else None
                strata[env.AGENTS[sender] + '>' + env.AGENTS[listener]] = dict(case_count=count, metrics=means)
                if means is not None:
                    stratum_means.append(means)
        complete = len(stratum_means) == 6
        means = {k: float(np.mean([m[k] for m in stratum_means])) for k in case_means} if complete else None
        group['by_axis'][name] = dict(complete_six_strata=complete, strata=strata, metrics=means)
        if means is not None:
            all_axes.append(means)
    if len(all_axes) == 3:
        group['macro'] = {k: float(np.mean([m[k] for m in all_axes])) for k in case_means}
        group['availability'] = 'all_three_axes_and_six_strata'
    else:
        group['availability'] = 'missing_strata_no_macro_no_silent_renormalization'
    return group


def semantic_metrics(*, spec, states, state_indices, action_indices,
                     action_probabilities, pairs=None, include_rows=False):
    """Natural two-endpoint content probe; no policy-dependent case selection.

    All need triples × the same backgrounds are required. Thus the formal full
    domain and every-need × two-background monitors share an exact API.
    include_rows adds compact per-case means and pair labels, not N×2 policy copies.
    Original actions/probabilities/messages remain in the runner's raw NPZ.
    """
    packed, ids, backgrounds = _saved_grid(spec, states, state_indices)
    actions, probs = _policy(action_indices, action_probabilities, len(packed))
    require(probs is not None, 'Saved probabilities required')
    expected_rows = _expected_cases(tuple(tuple(map(int, n)) for n in spec['needs']))
    rows = expected_rows if pairs is None else (pairs['rows'] if isinstance(pairs, dict) else pairs)
    require(rows == expected_rows, 'Case list differs from all predetermined dataset cases')
    require(len(rows) > 0, 'No semantic cases')
    truth = truth_actions(packed)
    settled = settle_arrays(packed, actions)
    g = len(backgrounds); c = len(rows)
    means = {k: np.empty(c, dtype=np.float64) for k in PAIR_METRICS}
    integer_sums = {k: 0 for k in ('both_endpoints_apt', 'both_team_full_success')}
    for start in range(0, c, 256):
        stop = min(start + 256, c); batch = rows[start:stop]
        need_idx = np.asarray([r['endpoint_need_indices'] for r in batch], dtype=np.int64)
        listeners = np.asarray([r['listener'] for r in batch], dtype=np.int64)
        # Need-major, common background ordering; shape case × background × endpoint.
        ix = need_idx[:, None, :] * g + np.arange(g)[None, :, None]
        who = listeners[:, None, None]
        chosen = actions[ix, who]
        correct = truth[ix, who]
        apt = chosen == correct
        probability = probs[ix, who, correct]
        reward = settled['reward'][ix]
        full = reward == 1
        values = dict(both_endpoints_apt=apt.all(-1), mean_endpoint_apt=apt.mean(-1),
            endpoint0_apt=apt[..., 0], endpoint1_apt=apt[..., 1],
            mean_endpoint_correct_probability=probability.mean(-1),
            both_endpoint_correct_probability_product=probability.prod(-1),
            listener_action_changed=chosen[..., 0] != chosen[..., 1],
            both_team_full_success=full.all(-1), mean_team_full_success=full.mean(-1),
            mean_native_reward=reward.mean(-1))
        for k, v in values.items():
            means[k][start:stop] = v.mean(1)
        for k in integer_sums:
            integer_sums[k] += int(values[k].sum())
    content = _case_aggregate(means, rows, 'content')
    require(content['macro'] is not None, 'Primary content estimand missing one of three axes/six strata')
    result = dict(schema='action_dependency_semantic_metrics_v1',
        saved_worlds=len(packed), full_partition_worlds=spec['world_count'],
        saved_background_count=g, full_background_count=len(spec['layouts']) * len(spec['private_sites']),
        saved_background_indices=backgrounds.tolist(),
        coverage='full_partition' if len(packed) == spec['world_count'] else 'all_needs_fixed_background_subset',
        case_count=c, case_background_pairs=c*g, endpoint_directions=2*c*g,
        content=content, role=_case_aggregate(means, rows, 'role'),
        all_classes_raw=dict(case_count=c, case_background_pairs=c*g,
                            means={k: _fraction(v) for k, v in means.items()}, integer_sums=integer_sums),
        class_case_counts={label: sum(r['classification'] == label for r in rows)
                           for label in ('content', 'role', 'descriptive')},
        truth='Unique env full-success plan cached by needs as material identity, remapped to actual site for every saved world; canonical action indices never used as actual answers.',
        scope='Natural conditional content appropriateness, not a causal message intervention or compositionality claim. Failures remain in every denominator. Pair endpoints/backgrounds are not independent training seeds.',
        probability_scope='Product is two independently sampled listener actions conditional on the two saved natural greedy-message contexts; it is not a joint-team success probability or sender-stochastic experiment.')
    if include_rows:
        result['case_rows'] = [{k: r[k] for k in ('case_id', 'classification', 'axis_index', 'sender', 'listener', 'endpoint_need_indices')} for r in rows]
        result['case_mean_values'] = means
    return result


COMPARISON_METRICS = ('content_both_endpoints_apt', 'full_success_rate', 'reward_mean',
    'role_success_rate', 'site_success_rate', 'destination_success_rate',
    'role_site_destination_success_rate', 'physical_execution_rate')
CONTRASTS = {
    'PL_live_minus_LL_live': {'PL_live': 1, 'LL_live': -1},
    'PL_communication_effect': {'PL_live': 1, 'PL_silent': -1},
    'LL_communication_effect': {'LL_live': 1, 'LL_silent': -1},
    'FI_communication_effect': {'FI_live': 1, 'FI_silent': -1},
    'communication_difference_in_differences': {'PL_live': 1, 'PL_silent': -1, 'LL_live': -1, 'LL_silent': 1}}


def _cell(value):
    semantic, world = value['semantic_metrics'], value['world_metrics']
    result = dict(content_both_endpoints_apt=semantic['content']['macro']['both_endpoints_apt'])
    result.update({k: world[k] for k in COMPARISON_METRICS if k != 'content_both_endpoints_apt'})
    require(all(np.isfinite(v) and 0 <= v <= 1 for v in result.values()), 'Invalid comparison metric')
    return result


def _paired_summary(seed_values):
    require(len(seed_values) == 4, 'Four paired independent initializations required')
    return dict(seed_values=[dict(seed=s, difference=float(v)) for s, v in zip(SEEDS, seed_values)],
        equal_seed_mean=float(np.mean(seed_values)), minimum=float(min(seed_values)), maximum=float(max(seed_values)))


def primary_comparison(records):
    """Enriched copies only: records24[*].final[part][natural/closed].*_metrics.

    Caller must establish actual completion/source hashes; this function validates
    the complete fixed numerical grid and its paired estimands, not file provenance.
    """
    require(len(records) == 24, 'Require all 24 completed runs, never select successful runs')
    by = {(r['seed'], r['condition']): r for r in records}
    require(len(by) == 24 and set(by) == set(product(SEEDS, CONDITIONS)), 'Wrong or duplicate seed/condition grid')
    require(all(r.get('updates') == 6000 for r in records), 'Formal comparisons require update 6000')
    partitions = {}
    for part in PARTITIONS:
        seed_rows = []
        for seed in SEEDS:
            cells = {cond: {mode: _cell(by[seed, cond]['final'][part][mode])
                for mode in ('natural', 'closed')} for cond in CONDITIONS}
            for cond in ('PL_silent', 'LL_silent'):
                require(cells[cond]['natural']['content_both_endpoints_apt'] == 0,
                    'PL/LL silent disjoint-answers same-observation structural zero violated')
            for cond in CONDITIONS:
                for mode in ('natural', 'closed'):
                    data = by[seed, cond]['final'][part][mode]['semantic_metrics']
                    require(data['coverage'] == 'full_partition', 'Two-background monitor cannot substitute for full endpoint')
                if cond.endswith('_silent'):
                    require(cells[cond]['natural'] == cells[cond]['closed'], 'Silent closed must reuse natural result')
            seed_rows.append(dict(seed=seed, cells=cells))
        contrast_results = {name: {metric: _paired_summary([
            sum(coef * row['cells'][cond]['natural'][metric] for cond, coef in coefficients.items())
            for row in seed_rows]) for metric in COMPARISON_METRICS}
            for name, coefficients in CONTRASTS.items()}
        closed = {info: {metric: _paired_summary([
            row['cells'][info+'_live']['natural'][metric] - row['cells'][info+'_live']['closed'][metric]
            for row in seed_rows]) for metric in COMPARISON_METRICS} for info in ('FI', 'PL', 'LL')}
        means = {cond: {mode: {metric: float(np.mean([row['cells'][cond][mode][metric] for row in seed_rows]))
            for metric in COMPARISON_METRICS} for mode in ('natural', 'closed')} for cond in CONDITIONS}
        partitions[part] = dict(seed_rows=seed_rows, equal_seed_cell_means=means,
                                paired_contrasts=contrast_results, natural_minus_closed=closed)
    primary = partitions['new_needs_and_layouts']['paired_contrasts']['PL_live_minus_LL_live']['content_both_endpoints_apt']
    secondary = partitions['new_needs_and_layouts']['paired_contrasts']['communication_difference_in_differences']['full_success_rate']
    return dict(schema='action_dependency_paired_comparison_v1', updates=6000,
        primary=dict(partition='new_needs_and_layouts', metric='content_both_endpoints_apt',
                     contrast='PL_live_minus_LL_live', **primary),
        secondary_task_difference_in_differences=dict(partition='new_needs_and_layouts',
            metric='full_success_rate', contrast='(PL_live-PL_silent)-(LL_live-LL_silent)', **secondary),
        partitions=partitions, independent_team_initializations=4,
        limitations='No significance test. All four paired seeds retained. Background worlds, pair endpoints and conditions are not independent training repeats. Natural appropriateness and closed-channel effects do not establish reusable message components.')
