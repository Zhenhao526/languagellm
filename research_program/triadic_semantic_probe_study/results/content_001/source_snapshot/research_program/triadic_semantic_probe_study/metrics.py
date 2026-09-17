"""Pure array metrics for the fixed semantic probe; no files or models read.

One row is an UNORDERED demand pair × listener × recipient layout × owner.
The second axis b=0,1 is an endpoint/direction, never an independent replicate.
In an intervention direction b the current mask is masks[b], the target mask
is masks[1-b], BOTH in the recipient layout. Native team rewards must be
settled externally in that current recipient world, not the donor world.

Public reports contain JSON-compatible summaries and `row_values` NumPy
arrays. Intervention reports also hold `_labels` arrays for paired contrasts.
Use summary_only for JSON; save all row_values (e.g. NPZ), retaining failures. Content axis
weights each sum to 1; the primary macro averages THREE axis means. Role
weights independently sum to 1 for each of TWO axes. Raw counts/means are
descriptive and are not the original ecological world distribution.
"""
from __future__ import annotations

from hashlib import sha256
import numpy as np

AXES = ('kind_wood_fiber', 'length_short_long', 'destination_L_R')
AGENTS = ('A', 'B', 'C')
CLASS_NAMES = ('descriptive', 'content', 'role')
WEIGHT_ATOL = 1e-10
PROB_ATOL = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, shape, low, high, name):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'iu', name+' shape/dtype')
    require(np.all((a >= low) & (a <= high)), name+' range')
    return a


def _policy(action_indices, action_probs, n):
    actions = _integer(action_indices, (n, 2, 3), 0, 16, 'action_indices')
    probabilities = np.asarray(action_probs, dtype=np.float64)
    require(probabilities.shape == (n, 2, 3, 17), 'action_probs shape')
    require(np.isfinite(probabilities).all() and np.all(probabilities >= 0)
            and np.all(probabilities <= 1), 'action_probs finite probabilities')
    require(np.allclose(probabilities.sum(axis=-1), 1, atol=PROB_ATOL, rtol=0),
            'action_probs must sum to 1 (not silently renormalized)')
    require(np.array_equal(actions, np.argmax(probabilities, axis=-1)),
            'Frozen greedy action_indices differ from first-argmax probabilities')
    return actions, probabilities


def _native_rewards(value, n):
    rewards = np.asarray(value, dtype=np.float64)
    require(rewards.shape == (n, 2) and np.isfinite(rewards).all()
            and np.all(np.isin(rewards, (0., .5, 1.))), 'Invalid recipient native reward')
    return rewards


def summary_only(report):
    """JSON-compatible summary; callers must retain row_values separately."""
    return {k: v for k, v in report.items() if k not in ('row_values', '_labels')}


def _labels(*, n, success_action_masks, listener, axis, sender,
            listener_partners, classification, content_within_axis_weight,
            role_within_axis_weight, content_only=False):
    d = dict(masks=_integer(success_action_masks, (n, 2), 1, (1 << 17)-1, 'success_action_masks'),
             listener=_integer(listener, (n,), 0, 2, 'listener'),
             sender=_integer(sender, (n,), 0, 2, 'sender'),
             axis=_integer(axis, (n,), 0, 2, 'axis'),
             partners=_integer(listener_partners, (n, 2), -1, 2, 'listener_partners'),
             classification=_integer(classification, (n,), 0, 2, 'classification'))
    require(n > 0 and np.all(d['sender'] != d['listener']), 'Empty rows or sender=listener')
    require(np.all((d['partners'] == -1) | (d['partners'] != d['listener'][:, None])),
            'A listener cannot partner with itself')
    for name, values, code, axes in (
            ('content', content_within_axis_weight, 1, (0, 1, 2)),
            ('role', role_within_axis_weight, 2, (0, 1))):
        w = np.asarray(values, dtype=np.float64)
        require(w.shape == (n,) and np.isfinite(w).all() and np.all(w >= 0), name+' weights')
        selected = d['classification'] == code
        require(np.array_equal(w > 0, selected), name+' positive weights must cover every selected row')
        require(np.all(np.isin(d['axis'][selected], axes)), name+' unexpected axis')
        if selected.any():
            for a in axes:
                require(abs(float(w[d['axis'] == a].sum())-1) <= WEIGHT_ATOL,
                        name+' missing axis or incorrect within-axis normalization')
            require(np.all((d['masks'][selected, 0] & d['masks'][selected, 1]) == 0),
                    name+' success masks must be disjoint')
        d[name+'_weights'] = w
    require(np.any(d['classification'] == 1), 'All three content axes are required')
    content = d['classification'] == 1
    require(np.all(d['partners'][content] == d['sender'][content, None]),
            'Content cases must keep sender as listener partner in both endpoints')
    if content_only:
        require(np.all(content), 'Interventions accept only the predetermined content rows')
    return d


def _listener_values(actions, probabilities, masks, listeners):
    n = len(actions)
    chosen = actions[np.arange(n)[:, None], np.arange(2)[None, :], listeners[:, None]]
    policy = probabilities[np.arange(n)[:, None], np.arange(2)[None, :], listeners[:, None]]
    accepted = (masks[:, :, None].astype(np.uint32) >> np.arange(17, dtype=np.uint32)) & 1
    apt = ((masks.astype(np.uint32) >> chosen.astype(np.uint32)) & 1).astype(bool)
    mass = np.sum(policy * accepted, axis=-1)
    return chosen, apt, mass


def _json_number(value):
    a = np.asarray(value)
    return a.item() if a.ndim == 0 else a.tolist()


def _summary(values, selected, weights=None):
    count = int(np.count_nonzero(selected))
    result = dict(n_case_worlds=count, n_endpoint_directions=2*count, metrics={})
    if count == 0:
        result['weight_sum'] = 0.0 if weights is not None else None
        return result
    w = None if weights is None else weights[selected]
    total = None if w is None else float(w.sum())
    require(w is None or total > 0, 'Selected summary has zero weight')
    result['weight_sum'] = total
    for key, value in values.items():
        v = np.asarray(value)[selected]
        raw_sum = v.sum(axis=0)
        raw_mean = v.mean(axis=0)
        mean = raw_mean if w is None else np.tensordot(w/total, v, axes=(0, 0))
        result['metrics'][key] = dict(mean=_json_number(mean), raw_sum=_json_number(raw_sum),
                                      raw_mean=_json_number(raw_mean))
    return result


def _axis_report(values, selected, weights, axis, allowed_axes):
    by_axis = {AXES[a]: _summary(values, selected & (axis == a), weights) for a in allowed_axes}
    present = [a for a in allowed_axes if by_axis[AXES[a]]['n_case_worlds']]
    macro = {key: _json_number(np.mean([by_axis[AXES[a]]['metrics'][key]['mean']
                for a in present], axis=0)) for key in values} if present else {}
    return dict(by_axis=by_axis, macro=macro,
                macro_axes=[AXES[a] for a in present],
                macro_rule='Equal mean of listed axis-specific weighted means; directions remain within rows.',
                raw_unweighted=_summary(values, selected))


def _strata(d):
    # Retain endpoint partner transitions, including waiting encoded -1.
    partner_name = lambda p: 'wait_or_no_unique_partner' if p == -1 else AGENTS[int(p)]
    sets = {
        'sender': {a: d['sender'] == i for i, a in enumerate(AGENTS)},
        'listener': {a: d['listener'] == i for i, a in enumerate(AGENTS)},
        'ordered_sender_listener': {s+'>'+l: (d['sender'] == i) & (d['listener'] == j)
            for i, s in enumerate(AGENTS) for j, l in enumerate(AGENTS) if i != j},
        'listener_partner_transition': {partner_name(p)+'>'+partner_name(q):
            (d['partners'][:, 0] == p) & (d['partners'][:, 1] == q)
            for p, q in sorted(set(map(tuple, d['partners'].tolist())))},
        'sender_listener_pair': {AGENTS[i]+AGENTS[j]:
            ((d['sender'] == i) & (d['listener'] == j)) | ((d['sender'] == j) & (d['listener'] == i))
            for i, j in ((0, 1), (0, 2), (1, 2))}}
    return sets


def _aggregate(values, d, group):
    code, axes = (1, (0, 1, 2)) if group == 'content' else (2, (0, 1))
    selected, weights = d['classification'] == code, d[group+'_weights']
    report = _axis_report(values, selected, weights, d['axis'], axes)
    report['strata'] = {name: {label: _axis_report(values, selected & mask, weights, d['axis'], axes)
        for label, mask in groups.items()} for name, groups in _strata(d).items()}
    return report


def _signature(d, *arrays):
    h = sha256()
    for key, a in sorted(d.items()):
        a = np.ascontiguousarray(a)
        h.update((key+'|'+str(a.shape)+'|'+a.dtype.str+'\n').encode()); h.update(a.tobytes())
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update((str(a.shape)+'|'+a.dtype.str+'\n').encode()); h.update(a.tobytes())
    return h.hexdigest()


def natural_metrics(*, action_indices, action_probs, success_action_masks,
                    listener, axis, sender, listener_partners, classification,
                    content_within_axis_weight, role_within_axis_weight,
                    native_rewards=None):
    """All unique rows; only classes 1/2 receive probe weights.

    Optional native_rewards contains saved actual whole-team R at the two
    original endpoints. This function validates its values, not provenance.
    """
    n = len(action_indices)
    d = _labels(n=n, success_action_masks=success_action_masks, listener=listener,
                axis=axis, sender=sender, listener_partners=listener_partners,
                classification=classification, content_within_axis_weight=content_within_axis_weight,
                role_within_axis_weight=role_within_axis_weight)
    actions, probs = _policy(action_indices, action_probs, n)
    chosen, apt, mass = _listener_values(actions, probs, d['masks'], d['listener'])
    values = dict(both_endpoints_apt=np.all(apt, axis=1), endpoint_apt=apt,
                  mean_endpoint_apt=apt.mean(axis=1), endpoint_probability_mass=mass,
                  mean_endpoint_probability_mass=mass.mean(axis=1),
                  listener_action_changed=chosen[:, 0] != chosen[:, 1],
                  any_team_action_changed=np.any(actions[:, 0] != actions[:, 1], axis=1))
    if native_rewards is not None:
        rewards = _native_rewards(native_rewards, n)
        full = rewards == 1
        values.update(native_reward=rewards, mean_native_reward=rewards.mean(axis=1),
            endpoint_team_full_success=full, mean_endpoint_team_full_success=full.mean(axis=1),
            both_team_full_success=np.all(full, axis=1))
    return dict(schema='semantic_natural_metrics_v1', row_values=values,
        content=_aggregate(values, d, 'content'), role=_aggregate(values, d, 'role'),
        all_rows_unweighted=_summary(values, np.ones(n, dtype=bool)),
        by_class_unweighted={name: _summary(values, d['classification'] == i)
                             for i, name in enumerate(CLASS_NAMES)},
        design_signature=_signature(d),
        interpretation='Probability mass is measured from saved policies; not stochastic replay or joint team success.')


def intervention_metrics(*, action_indices, action_probs, natural_action_indices,
                         natural_action_probs, native_rewards, window,
                         success_action_masks, listener, axis, sender,
                         listener_partners, classification,
                         content_within_axis_weight, role_within_axis_weight):
    """CONTENT only; b current endpoint, 1-b counterfactual recipient target.

    native_rewards[N,2] must contain the complete team's real R in each
    recipient world, supplied by independent native settlement. This function
    checks the allowed reward values but cannot certify its source.
    """
    require(window in ('w1', 'w2', 'both'), 'Unknown fixed window')
    n = len(action_indices)
    d = _labels(n=n, success_action_masks=success_action_masks, listener=listener,
                axis=axis, sender=sender, listener_partners=listener_partners,
                classification=classification, content_within_axis_weight=content_within_axis_weight,
                role_within_axis_weight=role_within_axis_weight, content_only=True)
    actions, probs = _policy(action_indices, action_probs, n)
    na, np_ = _policy(natural_action_indices, natural_action_probs, n)
    rewards = _native_rewards(native_rewards, n)
    chosen, current, current_mass = _listener_values(actions, probs, d['masks'], d['listener'])
    _, target, target_mass = _listener_values(actions, probs, d['masks'][:, ::-1], d['listener'])
    nc, natural_apt, natural_mass = _listener_values(na, np_, d['masks'], d['listener'])
    values = dict(current_apt=current, counterfactual_apt=target,
        current_probability_mass=current_mass, counterfactual_probability_mass=target_mass,
        listener_action_changed_from_natural=chosen != nc,
        any_team_action_changed_from_natural=np.any(actions != na, axis=-1),
        native_reward=rewards, native_full_success=rewards == 1,
        natural_current_apt=natural_apt, natural_current_probability_mass=natural_mass,
        current_apt_minus_natural=current.astype(float)-natural_apt,
        current_probability_mass_minus_natural=current_mass-natural_mass)
    # Every directional outcome has an explicit per-unordered-row mean.
    values.update({'direction_mean_'+k: v.mean(axis=1) for k, v in list(values.items())})
    return dict(schema='semantic_intervention_metrics_v1', window=window,
        window_status='predetermined_secondary' if window == 'both' else 'prespecified_window_diagnostic',
        row_values=values, content=_aggregate(values, d, 'content'),
        pairing_signature=_signature(d, na, np_),
        _labels=d,
        interpretation='Target mask uses opposite need in recipient layout. Native R evaluates current recipient truth.')


def remote_contrast(same, opposite):
    """Paired same/opposite donor contrast, each produced by intervention_metrics.

    These reports must share the exact row order, labels, weights, masks,
    natural policy, and window. Their donor backgrounds must be matched by
    the caller; this pure function has no donor state access to certify that.
    """
    require(same['schema'] == opposite['schema'] == 'semantic_intervention_metrics_v1', 'Wrong report schema')
    require(same['window'] == opposite['window'] and same['pairing_signature'] == opposite['pairing_signature'],
            'Remote same/opposite pairing differs')
    s, o = same['row_values'], opposite['row_values']
    values = dict(
        target_apt_opposite_minus_same=o['counterfactual_apt'].astype(float)-s['counterfactual_apt'],
        target_probability_mass_opposite_minus_same=o['counterfactual_probability_mass']-s['counterfactual_probability_mass'],
        current_apt_same_minus_natural=s['current_apt'].astype(float)-s['natural_current_apt'],
        current_apt_loss_natural_minus_same=s['natural_current_apt'].astype(float)-s['current_apt'],
        current_probability_mass_same_minus_natural=s['current_probability_mass']-s['natural_current_probability_mass'],
        current_probability_mass_loss_natural_minus_same=s['natural_current_probability_mass']-s['current_probability_mass'],
        native_reward_opposite_minus_same=o['native_reward']-s['native_reward'])
    values.update({'direction_mean_'+k: v.mean(axis=1) for k, v in list(values.items())})
    return dict(schema='semantic_remote_contrast_v1', window=same['window'],
        window_status=same['window_status'], row_values=values,
        content=_aggregate(values, same['_labels'], 'content'),
        pairing_signature=same['pairing_signature'],
        interpretation='Paired whole-message functional transfer; not composition. A positive target contrast need not improve true recipient reward.')
