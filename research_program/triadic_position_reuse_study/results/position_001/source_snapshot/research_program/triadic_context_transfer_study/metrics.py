"""Pure fixed-denominator measurements for whole-packet context transfer.

No networks, optimization, or outcome-selected cases are used. Each axis gives
six ordered sender/listener strata equal weight, then cases and recipient
backgrounds equal weight. Eligible donors are averaged WITHIN each background;
the conservative gate never removes a row or changes a denominator.
"""
import numpy as np
from research_program.triadic_action_dependency_study import metrics as native

AXES = ('kind', 'length', 'destination')
PARTITIONS = ('train', 'new_needs', 'new_layouts', 'new_needs_and_layouts')
SEEDS = (51101, 51102, 51103, 51104)
CONDITIONS = ('PL_silent', 'PL_live', 'LL_silent', 'LL_live')
REMOTE_MODES = ('remote_same_both', 'remote_opposite_both')
CONTROL_MODES = ('sham_both', 'local_opposite_both')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def build_pool_context(pool, information):
    """Cache native truth once per saved endpoint; only arrays are read.

    Correct canonical MATERIAL is remapped through each actual layout by the
    frozen native truth function. No canonical site is used as an actual action.
    Pool arrays are referenced, never modified. No actor parameters are needed.
    """
    require(information in ('PL', 'LL'), 'Only PL and LL policies are included')
    states = native._states(pool['states'])
    actions, probabilities = native._policy(pool['action_indices'], pool['action_probabilities'], len(states))
    messages = np.asarray(pool['messages'])
    require(messages.shape == (len(states), 2, 3, 4) and messages.dtype.kind in 'iu'
            and np.all((messages >= 0) & (messages < 8)), 'Invalid saved messages')
    truth = native.truth_actions(states)
    physics = native.settle_arrays(states, actions)
    for key, expected in (('greedy_reward', physics['reward']), ('executed', physics['executed']),
                          ('satisfied', physics['satisfied'])):
        require(np.array_equal(pool[key], expected), 'Saved natural physics mismatch: ' + key)
    return dict(pool=pool, information=information, truth_actions=truth)


def _selection(spec, rows):
    n = len(spec['endpoint_indices'])
    if rows is None:
        selected = np.arange(n)
    elif isinstance(rows, slice):
        # Avoid allocating all N indices for every small streaming batch.
        selected = np.arange(*rows.indices(n))
    else:
        selected = np.asarray(rows)
        require(selected.dtype.kind in 'iu' and np.all((selected >= 0) & (selected < n)), 'Invalid row indices')
    require(selected.ndim == 1 and len(selected) > 0, 'Rows must be a nonempty one-dimensional selection')
    return selected


def _donor_pair(spec, shift, mode, rows):
    if mode in REMOTE_MODES:
        require(isinstance(shift, (int, np.integer)) and 0 <= shift < len(spec['donor_endpoint_indices']), 'Invalid donor shift')
        return spec['donor_endpoint_indices'][shift, rows]
    require(mode in CONTROL_MODES and shift == -1, 'Unknown intervention or invalid control shift')
    return spec['endpoint_indices'][rows]


def _same_observation(states, recipient_ids, donor_ids, listener, information):
    # Within this dataset only sender need differs, and ownership is unchanged.
    # Exact own need/assignment checks precede the visible-material comparison.
    r = states[recipient_ids]; d = states[donor_ids]; ar = np.arange(len(r))
    require(np.array_equal(r[ar, listener], d[ar, listener]), 'Listener own need changed')
    require(np.array_equal(r[:, 7:10], d[:, 7:10]), 'Owner assignment changed')
    if information == 'PL':
        return np.all(r[:, 3:7] == d[:, 3:7], axis=1)
    own_site = r[ar, 7 + listener]
    return (r[:, 3] == d[:, 3]) & (r[ar, 3 + own_site] == d[ar, 3 + own_site])


def cell_values(spec, context, data, *, mode, shift, direction, live, rows=None):
    """Return one float/bool vector per measurement for selected saved rows.

    `data` is the runner's complete intervention NPZ mapping, or None for a
    silent alias. `rows` selects the SAME N-axis in spec and data. Fixed donor
    endpoint 0/1 markers are converted using direction, never assumed to mean
    same/opposite. Native and counterfactual team scores are separately replayed.
    """
    require(direction in (0, 1), 'Invalid endpoint direction')
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    ii = _selection(spec, rows); ar = np.arange(len(ii))
    pool = context['pool']; states = pool['states']; truth = context['truth_actions']
    listener = np.asarray(spec['listener'])[ii]
    sender = np.asarray(spec['sender'])[ii]
    require(listener.shape == sender.shape and np.all(listener != sender)
            and np.all((listener >= 0) & (listener < 3)) and np.all((sender >= 0) & (sender < 3)), 'Invalid sender/listener')
    endpoint = spec['endpoint_indices'][ii]
    ids, cfids = endpoint[:, direction], endpoint[:, 1-direction]
    donor_pair = _donor_pair(spec, shift, mode, ii)
    same_ids, opposite_ids = donor_pair[:, direction], donor_pair[:, 1-direction]
    selected_donor = same_ids if mode in ('sham_both', 'remote_same_both') else opposite_ids
    require(np.array_equal(truth[endpoint], spec['correct_actions'][ii]), 'Recipient truth/layout mismatch')
    donor_truth = spec['donor_correct_actions'][shift, ii] if shift >= 0 else spec['correct_actions'][ii]
    require(np.array_equal(truth[donor_pair], donor_truth), 'Donor truth/layout mismatch')
    # The two endpoint worlds differ only in the designated sender's need.
    difference = states[ids] != states[cfids]
    expected_difference = np.zeros_like(difference); expected_difference[ar, sender] = True
    require(np.array_equal(difference, expected_difference), 'Pair changes more than sender need')
    current, target = truth[ids, listener], truth[cfids, listener]
    require(np.all(current != target) and np.all(current != 0) and np.all(target != 0),
            'Content listeners need distinct active unique answers')
    if live:
        require(data is not None, 'Live intervention data required')
        for key, expected in (('dataset_rows', ii), ('recipient_indices', ids),
                ('counterfactual_recipient_indices', cfids), ('donor_indices', selected_donor)):
            require(np.array_equal(np.asarray(data[key])[ii], expected), 'Saved row identity mismatch: ' + key)
        actions, probabilities = native._policy(np.asarray(data['action_indices'])[ii],
                                               np.asarray(data['action_probabilities'])[ii], len(ii))
    else:
        require(data is None, 'Silent output must be a saved-natural alias')
        actions, probabilities = native._policy(pool['action_indices'][ids], pool['action_probabilities'][ids], len(ii))
    original_messages = pool['messages'][ids]
    generated_messages = np.asarray(data['messages'])[ii] if live else original_messages
    require(generated_messages.shape == (len(ii), 2, 3, 4) and generated_messages.dtype.kind in 'iu'
            and np.all((generated_messages >= 0) & (generated_messages < 8)), 'Invalid generated messages')
    actual = actions[ar, listener]
    natural_actual = pool['action_indices'][ids, listener]
    donor_same_actual = pool['action_indices'][same_ids, listener]
    donor_opposite_actual = pool['action_indices'][opposite_ids, listener]
    same_view = _same_observation(states, ids, same_ids, listener, context['information'])
    opposite_view = _same_observation(states, ids, opposite_ids, listener, context['information'])
    require(np.array_equal(same_view, opposite_view), 'Donor backgrounds or listener facts differ across arms')
    if shift >= 0 and context['information'] == 'LL':
        require(np.array_equal(same_view, spec['view_equal_LL'][shift, ii]), 'Static listener view marker mismatch')
    if live:
        equal = np.asarray(data['action_input_equals_donor'])[ii]
        require(equal.shape == (len(ii), 3, 2) and equal.dtype.kind == 'b', 'Invalid full-input identity markers')
        input_same = equal[ar, listener, direction]
        input_opposite = equal[ar, listener, 1-direction]
    else:
        # Silent inputs are own observation + own two messages + invariant zero
        # foreign blocks/bits. This exact equality needs no neural forward.
        own_messages = pool['messages'][ids, :, listener, :]
        input_same = same_view & np.all(own_messages == pool['messages'][same_ids, :, listener, :], axis=(1, 2))
        input_opposite = opposite_view & np.all(own_messages == pool['messages'][opposite_ids, :, listener, :], axis=(1, 2))
    require(np.all(~input_same | (actual == donor_same_actual)), 'Equal input did not reproduce donor same action')
    require(np.all(~input_opposite | (actual == donor_opposite_actual)), 'Equal input did not reproduce donor opposite action')
    # With a unique target, this shared gate is equivalent to excluding literal
    # copies of BOTH donors' actual listener actions, even when those were wrong.
    gate = (target != donor_same_actual) & (target != donor_opposite_actual)
    apt = actual == target
    physics = native.settle_arrays(states[ids], actions)
    counterfactual = native.settle_arrays(states[cfids], actions)
    if live:
        for key, expected in (('greedy_reward', physics['reward']), ('executed', physics['executed']),
                ('satisfied', physics['satisfied']), ('counterfactual_reward', counterfactual['reward']),
                ('counterfactual_satisfied', counterfactual['satisfied'])):
            require(np.array_equal(np.asarray(data[key])[ii], expected), 'Saved intervention physics mismatch: ' + key)
    require(np.array_equal(physics['executed'], counterfactual['executed']), 'Demand change altered action execution')
    require(np.array_equal(physics['reward'] == 1, np.all(actions == truth[ids], axis=1)), 'Native full-plan mismatch')
    require(np.array_equal(counterfactual['reward'] == 1, np.all(actions == truth[cfids], axis=1)), 'Counterfactual full-plan mismatch')
    return dict(current_apt=actual == current, target_apt=apt,
        current_probability=probabilities[ar, listener, current], target_probability=probabilities[ar, listener, target],
        natural_current_apt=natural_actual == current,
        current_minus_natural_apt=(actual == current).astype(float)-(natural_actual == current),
        conservative_gate=gate, conservative_target_apt=gate & apt,
        native_reward=physics['reward'], native_team_full=physics['reward'] == 1,
        counterfactual_reward=counterfactual['reward'], counterfactual_team_full=counterfactual['reward'] == 1,
        physical_pair_executed=np.any(physics['executed'], axis=1),
        listener_action_changed=actual != natural_actual,
        any_generated_message_changed=np.any(generated_messages != original_messages, axis=(1, 2, 3)),
        listener_own_second_message_changed=np.any(generated_messages[ar, 1, listener] != original_messages[ar, 1, listener], axis=1),
        donor_same_correct=donor_same_actual == truth[same_ids, listener],
        donor_opposite_correct=donor_opposite_actual == truth[opposite_ids, listener],
        copy_donor_same_current_apt=donor_same_actual == current,
        copy_donor_same_target_apt=donor_same_actual == target,
        copy_donor_opposite_current_apt=donor_opposite_actual == current,
        copy_donor_opposite_target_apt=donor_opposite_actual == target,
        action_equals_donor_same=actual == donor_same_actual,
        action_equals_donor_opposite=actual == donor_opposite_actual,
        observation_equal_donor=same_view,
        input_equal_donor_same=input_same, input_equal_donor_opposite=input_opposite)


def contrast_values(same, opposite):
    """Paired differences, without selecting successful or gate-passing rows."""
    require(set(same) == set(opposite) and len(same) > 0, 'Arm measurement schemas differ')
    invariants = ('conservative_gate', 'natural_current_apt', 'donor_same_correct', 'donor_opposite_correct',
                  'copy_donor_same_current_apt', 'copy_donor_same_target_apt',
                  'copy_donor_opposite_current_apt', 'copy_donor_opposite_target_apt', 'observation_equal_donor')
    for key in invariants:
        require(np.array_equal(same[key], opposite[key]), 'Paired contrast changed shared quantity: ' + key)
    out = {}
    for key in ('target_apt', 'target_probability', 'conservative_target_apt', 'current_apt',
                'native_reward', 'native_team_full', 'counterfactual_reward', 'counterfactual_team_full'):
        a, b = np.asarray(same[key], dtype=float), np.asarray(opposite[key], dtype=float)
        require(a.ndim == 1 and a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Invalid paired values')
        out[key + '_gain'] = b-a
    out['conservative_gate'] = np.asarray(same['conservative_gate'], dtype=float)
    return out


def support_weights(spec, shift, layer, rows=None):
    """Each direction gets half weight; caller must include both directions.

    Every layer has total weight one within each axis after ALL donor shifts and
    both directions. A gate/view marker is never an argument to this function.
    """
    ii = _selection(spec, rows)
    base = np.asarray(spec['base_within_axis_weight'], dtype=float)[ii]
    require(np.isfinite(base).all() and np.all(base > 0), 'Invalid base weights')
    if layer == 'control':
        require(shift == -1, 'Controls do not have a remote donor shift')
        return base / 2
    k = len(spec['donor_endpoint_indices'])
    require(0 <= shift < k and layer in ('eligible', 'all_other'), 'Invalid remote support')
    if layer == 'all_other':
        return base / (2*k)
    count = np.asarray(spec['eligible_donor_count'])[ii]
    require(np.all(count > 0), 'Cannot drop a background lacking eligible donors')
    require(np.array_equal(count, np.asarray(spec['eligible'])[:, ii].sum(axis=0)), 'Eligible counts inconsistent')
    return base * np.asarray(spec['eligible'])[shift, ii] / count / 2


class WeightedAccumulator:
    """Small streaming accumulator; all rows count even when their weight is zero."""
    def __init__(self):
        self.keys = None
        self.sums = None
        self.weights = np.zeros(3, dtype=float)
        self.counts = np.zeros(3, dtype=np.int64)

    def update(self, values, axis, weights):
        keys = tuple(sorted(values))
        require(bool(keys), 'No measurements')
        axis = np.asarray(axis); w = np.asarray(weights, dtype=float)
        require(axis.ndim == 1 and axis.dtype.kind in 'iu' and w.shape == axis.shape
                and np.all((axis >= 0) & (axis < 3)) and np.isfinite(w).all() and np.all(w >= 0), 'Invalid weights/axes')
        if self.keys is None:
            self.keys = keys; self.sums = np.zeros((3, len(keys)), dtype=float)
        require(keys == self.keys, 'Measurement schema changed between batches')
        a = np.stack([np.asarray(values[k], dtype=float) for k in keys], axis=1)
        require(a.shape == (len(w), len(keys)) and np.isfinite(a).all(), 'Invalid measurement vectors')
        for j in range(3):
            take = axis == j
            self.weights[j] += w[take].sum()
            self.counts[j] += int(take.sum())
            self.sums[j] += w[take] @ a[take]

    def finish(self):
        require(self.keys is not None, 'No accumulated rows')
        require(np.allclose(self.weights, 1, atol=2e-11, rtol=0), 'Incomplete or duplicate support/directions: each axis weight must be one')
        # Fixed weights are checked, not silently renormalized after filtering.
        by_axis = {name: dict(weight_sum=float(self.weights[j]), row_count=int(self.counts[j]),
            values={k: float(self.sums[j, i]) for i, k in enumerate(self.keys)}) for j, name in enumerate(AXES)}
        return dict(by_axis=by_axis, macro={k: float(self.sums[:, i].mean()) for i, k in enumerate(self.keys)},
                    weighting='fixed axis/S-L/case/recipient/donor/direction; no gate or outcome renormalization')


def primary_comparison(records):
    """Require all 16 policies and four partitions, retaining every seed.

    Input record: seed, condition, partitions[part][eligible/all_other]['contrast']
    = WeightedAccumulator.finish() for paired remote same/opposite differences.
    This function never reads an individual best seed or a selected checkpoint.
    """
    require(len(records) == 16, 'Exactly all 16 frozen policies are required')
    lookup = {(int(r['seed']), r['condition']): r for r in records}
    require(set(lookup) == {(s, c) for s in SEEDS for c in CONDITIONS}, 'Missing/duplicate policy or unexpected seed')
    comparisons = {}
    for part in PARTITIONS:
        comparisons[part] = {}
        for support in ('eligible', 'all_other'):
            seed_pairs = []
            for seed in SEEDS:
                values = {}
                for condition in CONDITIONS:
                    record = lookup[seed, condition]
                    require(set(record['partitions']) == set(PARTITIONS), 'Four complete partitions required')
                    summary = record['partitions'][part][support]['contrast']
                    require(set(summary['by_axis']) == set(AXES), 'Three complete axes required')
                    for name in AXES:
                        require(abs(summary['by_axis'][name]['weight_sum'] - 1) <= 2e-11, 'Incomplete support')
                    values[condition] = {k: float(summary['macro'][k]) for k in
                        ('target_apt_gain', 'conservative_target_apt_gain')}
                    require(all(np.isfinite(x) for x in values[condition].values()), 'Nonfinite effect')
                    if condition.endswith('_silent'):
                        require(all(abs(x) <= 2e-12 for x in values[condition].values()), 'Silent target effects must be structural zero')
                seed_pairs.append(dict(seed=seed, conditions=values,
                    PL_minus_LL=values['PL_live']['target_apt_gain']-values['LL_live']['target_apt_gain'],
                    conservative_PL_minus_LL=values['PL_live']['conservative_target_apt_gain']-values['LL_live']['conservative_target_apt_gain']))
            comparisons[part][support] = dict(seed_pairs=seed_pairs,
                equal_seed_mean_PL_minus_LL=float(np.mean([r['PL_minus_LL'] for r in seed_pairs])),
                equal_seed_mean_conservative_PL_minus_LL=float(np.mean([r['conservative_PL_minus_LL'] for r in seed_pairs])),
                equal_seed_condition_means={c: {metric: float(np.mean([r['conditions'][c][metric] for r in seed_pairs]))
                    for metric in ('target_apt_gain', 'conservative_target_apt_gain')} for c in CONDITIONS})
    return dict(primary=dict(partition='new_needs_and_layouts', support='eligible',
        measurement='remote_opposite target_apt minus remote_same target_apt; then PL_live minus LL_live',
        **comparisons['new_needs_and_layouts']['eligible']), all_partitions=comparisons,
        statistical_unit='four paired saved-policy training seeds; backgrounds/cases/directions are repeated measurements',
        interpretation='whole-packet functional transfer; no component reuse, language-origin, or new-training claim')
