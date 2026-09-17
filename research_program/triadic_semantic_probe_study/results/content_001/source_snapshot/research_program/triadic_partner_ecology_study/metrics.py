"""Weighted behavioral measures. Pure arrays; no policy, runner or model import.

Role correctness ignores location/destination agreement. Physical matching and
native reward remain separate. Every population quantity uses supplied weights.
"""
from itertools import combinations, product
import math
import numpy as np

AGENTS = ('A', 'B', 'C')
PAIRS = ((0, 1), (0, 2), (1, 2))
PAIR_NAMES = ('AB', 'AC', 'BC')
SEEDS = (49101, 49102, 49103, 49104)
ECOLOGIES = ('unique', 'multiple')
CONDITIONS = ('FI_silent', 'FI_live', 'PI_silent', 'PI_live')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def compatibility(needs):
    """Independent bit-set implementation of the original acceptance predicate."""
    needs = np.asarray(needs)
    require(needs.ndim == 2 and needs.shape[1] == 3 and needs.dtype.kind in 'iu'
        and ((needs >= 0) & (needs < 12)).all(), 'Invalid need ids')
    resource = np.array((0b0011, 0b1100, 0b0101, 0b1010))[needs//3]
    destination = np.array((0b01, 0b10, 0b11))[needs % 3]
    return np.stack([((resource[:, i] & resource[:, j]) != 0)
        & ((destination[:, i] & destination[:, j]) != 0) for i, j in PAIRS], axis=1)


def decode_actions(actions):
    actions = np.asarray(actions)
    require(actions.ndim == 2 and actions.shape[1] == 3 and actions.dtype.kind in 'iu'
        and ((actions >= 0) & (actions < 17)).all(), 'Actions must be integer indices0..16')
    active = actions != 0
    packed = np.maximum(actions.astype(np.int64)-1, 0)
    site, destination = packed//4, (packed % 4)//2
    partner = np.empty_like(packed)
    for a in range(3):
        partner[:, a] = np.array([b for b in range(3) if b != a])[packed[:, a] % 2]
    return dict(active=active, site=np.where(active, site, -1),
        destination=np.where(active, destination, -1), partner=np.where(active, partner, -1))


def summarize(states, actions, reward, weights):
    states, actions = np.asarray(states), np.asarray(actions)
    reward, weights = np.asarray(reward, dtype=np.float64), np.asarray(weights, dtype=np.float64)
    n = len(states)
    require(n > 0 and states.shape == (n, 10) and states.dtype.kind in 'iu', 'Invalid packed states')
    require(((states[:, :3] >= 0) & (states[:, :3] < 12)).all()
        and np.array_equal(np.sort(states[:, 3:7], axis=1), np.tile(np.arange(4), (n, 1)))
        and np.array_equal(np.sort(states[:, 7:], axis=1), np.tile(np.arange(1, 4), (n, 1))), 'Invalid semantic world')
    require(actions.shape == (n, 3) and reward.shape == weights.shape == (n,), 'Array lengths differ')
    require(np.isfinite(weights).all() and (weights >= 0).all()
        and abs(float(weights.sum())-1) <= 1e-12, 'Weights must sum to1 within1e-12; never silently normalize')
    require(np.isin(reward, (0, .5, 1)).all(), 'Invalid native reward')
    dec = decode_actions(actions)
    active, site, destination, partner = [dec[k] for k in ('active', 'site', 'destination', 'partner')]
    active_count = active.sum(1)
    edge = compatibility(states[:, :3])
    require((edge.sum(1) >= 1).all(), 'The candidate only includes solvable demands')
    proposed, topology, physical = [], [], []
    for i, j in PAIRS:
        proposed.append((active_count == 2) & active[:, i] & active[:, j])
        topology.append(proposed[-1] & (partner[:, i] == j) & (partner[:, j] == i))
        physical.append(topology[-1] & (site[:, i] == site[:, j]) & (destination[:, i] == destination[:, j]))
    proposed, topology, physical = map(lambda x: np.stack(x, axis=1), (proposed, topology, physical))
    role = topology & edge
    executed = np.zeros((n, 3), dtype=bool)
    for k, (i, j) in enumerate(PAIRS):
        executed[:, i] |= physical[:, k]
        executed[:, j] |= physical[:, k]
    material = states[np.arange(n)[:, None], 3+np.maximum(site, 0)]
    resource_masks = np.array((3, 12, 5, 10))[states[:, :3]//3]
    destination_masks = np.array((1, 2, 3))[states[:, :3] % 3]
    satisfies = executed & ((resource_masks & (1 << material)) != 0)
    satisfies &= (destination_masks & (1 << np.maximum(destination, 0))) != 0
    require(np.array_equal(satisfies.sum(1)/2, reward), 'Supplied native rewards do not match physical actions')
    full = reward == 1
    role_any, topo_any, physical_any = role.any(1), topology.any(1), physical.any(1)
    require((~full | role_any).all() and (~full | physical_any).all(), 'Full success lacks a compatible physical pair')
    mass = lambda values: float(np.dot(weights, np.asarray(values, dtype=np.float64)))
    conditional = lambda values, mask: (mass(np.asarray(values) & mask)/mass(mask) if mass(mask) > 0 else None)
    # Aggregate by needs first: layout/owner duplication never changes an oracle weight.
    unique_needs, inverse = np.unique(states[:, :3], axis=0, return_inverse=True)
    need_weights = np.bincount(inverse, weights=weights, minlength=len(unique_needs))
    gamma = np.dot(need_weights, compatibility(unique_needs).astype(float))
    best_gamma = float(gamma.max())
    pairs = {name: dict(active_pair_weight=mass(proposed[:, k]),
        proposal_weight=mass(topology[:, k]), compatible_proposal_weight=mass(role[:, k]),
        executed_weight=mass(physical[:, k]), full_success_weight=mass(full & physical[:, k]))
        for k, name in enumerate(PAIR_NAMES)}
    oracle = {name: dict(gamma=float(gamma[k]), full_success_oracle=float(gamma[k]),
        mean_reward_oracle=float((1+gamma[k])/2), mean_log_J_oracle=float(-(1-gamma[k])*math.log(2)))
        for k, name in enumerate(PAIR_NAMES)}
    unique_subgroups = {}
    for k, name in enumerate(PAIR_NAMES):
        group = (edge.sum(1) == 1) & edge[:, k]
        i, j = PAIRS[k]; outsider = next(a for a in range(3) if a not in (i, j))
        unique_subgroups[name] = dict(population_weight=mass(group), raw_worlds=int(group.sum()),
            compatible_role_rate=conditional(role_any, group), topology_consistency_rate=conditional(topo_any, group),
            physical_match_rate=conditional(physical_any, group), full_success_rate=conditional(full, group),
            mean_reward=(mass(reward*group)/mass(group) if mass(group) else None),
            isolated_wait_rate=conditional(~active[:, outsider], group),
            required_agents_active_rate=conditional(active[:, i] & active[:, j], group),
            topology_confusion_conditional={target: conditional(topology[:, t], group)
                for t, target in enumerate(PAIR_NAMES)},
            no_mutual_pair_rate=conditional(~topo_any, group))
    unique_actions, action_inverse, counts = np.unique(actions, axis=0, return_inverse=True, return_counts=True)
    action_mass = np.bincount(action_inverse, weights=weights, minlength=len(counts))
    distribution = [dict(action_indices=a.tolist(), raw_worlds=int(c), probability_weight=float(w))
        for a, c, w in zip(unique_actions, counts, action_mass)]
    weighted = dict(compatible_role_rate=mass(role_any), topology_consistency_rate=mass(topo_any),
        physical_match_rate=mass(physical_any), full_success_rate=mass(full), reward_mean=mass(reward),
        compatible_role_but_not_physical_rate=mass(role_any & ~physical_any),
        topology_but_incompatible_rate=mass(topo_any & ~role_any),
        active_count_weights={str(k): mass(active_count == k) for k in range(4)},
        native_reward_weights={str(v): mass(reward == v) for v in (0., .5, 1.)},
        two_active_not_mutual_rate=mass((active_count == 2) & ~topo_any),
        site_mismatch_given_topology=conditional(np.any([topology[:, k] & (site[:, i] != site[:, j]) for k, (i, j) in enumerate(PAIRS)], axis=0), topo_any),
        destination_mismatch_given_topology=conditional(np.any([topology[:, k] & (destination[:, i] != destination[:, j]) for k, (i, j) in enumerate(PAIRS)], axis=0), topo_any),
        pair_weights=pairs,
        agent_rates={a: dict(wait_rate=mass(~active[:, i]), participant_rate=mass(active[:, i]),
            executed_participant_rate=mass(executed[:, i]), own_need_satisfied_rate=mass(satisfies[:, i]))
            for i, a in enumerate(AGENTS)},
        fixed_pair_oracles=oracle, best_fixed_pair_gamma=best_gamma,
        compatible_role_excess_best_fixed_pair=mass(role_any)-best_gamma,
        full_success_excess_best_fixed_pair=mass(full)-best_gamma,
        best_fixed_pair_reward_oracle=(1+best_gamma)/2,
        reward_excess_best_fixed_pair=mass(reward)-(1+best_gamma)/2,
        best_fixed_pair_mean_log_J_oracle=-(1-best_gamma)*math.log(2),
        full_information_unrestricted_oracle=dict(compatible_role_rate=1., full_success_rate=1., reward_mean=1., mean_log_J=0.),
        unique_edge_subgroups=unique_subgroups)
    return dict(weighted=weighted, raw=dict(worlds=n, weights_sum=float(weights.sum()),
        unique_need_tables=len(unique_needs), joint_action_domain_size=17**3,
        observed_joint_actions=len(counts), joint_action_distribution=distribution,
        omitted_joint_actions_have_zero_count_and_weight=True,
        compatible_role_worlds=int(role_any.sum()), topology_consistent_worlds=int(topo_any.sum()),
        physical_matched_worlds=int(physical_any.sum()), full_success_worlds=int(full.sum())),
        definitions=dict(role='exactly two active, mutually name each other, selected pair has intersecting resource and destination needs; location/destination agreement not required',
            topology='exactly two active and mutually name each other',
            physical='topology plus same site and destination',
            proposal_weight='mutual proposals; active_pair_weight also retains two-active nonmutual cases',
            participant_rate='proposed active transport, distinct from actually executed participation',
            oracles='fixed partner identities, optimally state-adaptive site/destination with complete information; mean logJ is deterministic-policy supremum, not mean log realizedR'))


def primary_comparison(results):
    index = {(r['seed'], r['ecology'], r['condition']): r for r in results}
    expected = set(product(SEEDS, ECOLOGIES, CONDITIONS))
    require(len(results) == len(index) == 32 and set(index) == expected, 'Exactly4seeds×2ecologies×4conditions required')
    rows = []
    for seed in SEEDS:
        cells = {e: {c: index[seed, e, c]['final']['heldout_layouts']['natural']['weighted']
            for c in CONDITIONS} for e in ECOLOGIES}
        rates = {e: {c: {key: float(cells[e][c][key]) for key in
            ('compatible_role_rate', 'topology_consistency_rate', 'physical_match_rate', 'full_success_rate', 'reward_mean',
             'best_fixed_pair_gamma', 'compatible_role_excess_best_fixed_pair')}
            for c in CONDITIONS} for e in ECOLOGIES}
        require(all(math.isfinite(v) for e in rates.values() for c in e.values() for v in c.values()), 'Nonfinite primary cell')
        pi = {e: rates[e]['PI_live']['compatible_role_rate']-rates[e]['PI_silent']['compatible_role_rate'] for e in ECOLOGIES}
        fi = {e: rates[e]['FI_live']['compatible_role_rate']-rates[e]['FI_silent']['compatible_role_rate'] for e in ECOLOGIES}
        full = {e: rates[e]['PI_live']['full_success_rate']-rates[e]['PI_silent']['full_success_rate'] for e in ECOLOGIES}
        rows.append(dict(seed=seed, eight_cells=rates,
            PI_live_minus_silent_by_ecology=pi, FI_live_minus_silent_by_ecology=fi,
            primary_DiD_unique_minus_multiple=pi['unique']-pi['multiple'],
            FI_DiD_unique_minus_multiple=fi['unique']-fi['multiple'],
            full_success_PI_live_minus_silent_by_ecology=full,
            full_success_DiD_unique_minus_multiple=full['unique']-full['multiple']))
    return dict(partition='heldout_layouts', primary_metric='compatible_role_rate',
        primary_estimand='(PI_live−PI_silent)_unique − (PI_live−PI_silent)_multiple',
        independent_paired_seeds=4, seed_pairs=rows,
        equal_seed_means={key: sum(r[key] for r in rows)/4 for key in
            ('primary_DiD_unique_minus_multiple', 'FI_DiD_unique_minus_multiple', 'full_success_DiD_unique_minus_multiple')},
        significance_test=None,
        interpretation='Weighted role compatibility; not physical completion, compositional language, or an isolated change to partner count with all other joint task properties equal.')
