"""Row measures and preregistered selection of the training-chosen positions."""
import numpy as np
from research_program.triadic_action_dependency_study import environment as env
from research_program.triadic_context_transfer_study import channel_intervention as old_channel
from .discovery import packet_codes

AXES = ('kind', 'length', 'destination')
ATTRIBUTES = ('kind', 'length', 'destination', 'partner')
MATERIAL_KINDS = np.asarray([('wood', 'fiber').index(m[0]) for m in env.MATERIALS], dtype=np.int8)
MATERIAL_LENGTHS = np.asarray([('short', 'long').index(m[1]) for m in env.MATERIALS], dtype=np.int8)
ACTION_SITES = np.full((3, 17), -1, np.int8)
ACTION_DESTINATIONS = ACTION_SITES.copy()
ACTION_PARTNERS = ACTION_SITES.copy()
for actor in range(3):
    for index, action in enumerate(env.all_actions(env.AGENTS[actor])):
        if action['kind'] == 'transport':
            ACTION_SITES[actor, index] = env.SITES.index(action['site'])
            ACTION_DESTINATIONS[actor, index] = env.DESTINATIONS.index(action['destination'])
            ACTION_PARTNERS[actor, index] = env.AGENTS.index(action['partner'])


def require(condition, message):
    if not condition:
        raise ValueError(message)


def decoded_attributes(states, actions, listeners):
    states, actions, listeners = np.asarray(states), np.asarray(actions), np.asarray(listeners)
    n = len(states)
    require(states.shape == (n, 10) and actions.shape == (n, 3) and listeners.shape == (n,), 'Bad decode shapes')
    require(states.dtype.kind in 'iu' and actions.dtype.kind in 'iu' and listeners.dtype.kind in 'iu', 'Decode requires integer states/actions/listeners')
    require(np.all((actions >= 0) & (actions < 17)) and np.all((listeners >= 0) & (listeners < 3)), 'Decode action/listener out of range')
    require(np.all(np.sort(states[:, 3:7], axis=1) == np.arange(4))
            and np.all(np.sort(states[:, 7:10], axis=1) == np.arange(1, 4)), 'Decode requires valid layout/owner permutations')
    require(np.all((states[:, :3] >= 0) & (states[:, :3] < 24)), 'Decode needs out of range')
    chosen = actions[np.arange(n), listeners]
    active = chosen > 0
    site = ACTION_SITES[listeners, chosen]
    material = states[np.arange(n), 3 + np.maximum(site, 0)]
    attrs = np.stack((MATERIAL_KINDS[material], MATERIAL_LENGTHS[material],
                      ACTION_DESTINATIONS[listeners, chosen], ACTION_PARTNERS[listeners, chosen]), axis=-1)
    attrs[~active] = -1
    return attrs


def row_values(spec, direction, recipient_states, natural, data, code_sets, *, counterfactual_states):
    """Keep every row; explicit source counterpart states supply only needs.

    counterfactual_states is pool['states'][endpoint_indices[:,1-direction]].
    Only its first three needs are copied onto recipient physical state; neither
    donor layout nor an absent spec field is ever used for counterfactual scoring.
    """
    require(direction in (0, 1), 'Direction must be0 or1')
    s, l = np.asarray(spec['sender']), np.asarray(spec['listener'])
    n = len(s); rows = np.arange(n)
    require(s.shape == l.shape == (n,) and s.dtype.kind in 'iu' and l.dtype.kind in 'iu'
            and np.all((s >= 0) & (s < 3)) and np.all((l >= 0) & (l < 3)) and np.all(s != l), 'Invalid sender/listener')
    recipient_states, counterfactual_states = np.asarray(recipient_states), np.asarray(counterfactual_states)
    require(recipient_states.shape == counterfactual_states.shape == (n, 10)
            and recipient_states.dtype.kind in 'iu' and counterfactual_states.dtype.kind in 'iu', 'Explicit recipient/counterfactual states must be integer N×10')
    require(np.array_equal(recipient_states[:, 3:], counterfactual_states[:, 3:]), 'Counterfactual must retain recipient physical layout and owner')
    expected_changed = np.zeros((n, 3), dtype=bool); expected_changed[rows, s] = True
    require(np.array_equal(recipient_states[:, :3] != counterfactual_states[:, :3], expected_changed), 'Counterfactual must change only sender need')
    actions = np.asarray(data['action_indices']); p = np.asarray(data['action_probabilities'])
    require(actions.shape == (n, 3) and p.shape == (n, 3, 17), 'Bad action dimensions')
    require(actions.dtype.kind in 'iu' and np.all((actions >= 0) & (actions < 17)) and np.isfinite(p).all()
            and np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(-1), 1, atol=2e-12, rtol=0), 'Bad action distribution')
    require(np.array_equal(actions, np.argmax(p, -1)), 'Saved action does not follow greedy probabilities')
    current = spec['correct_actions'][:, direction]
    target = spec['correct_actions'][:, 1-direction]
    a = actions[rows, l]; a_current = current[rows, l]; a_target = target[rows, l]
    require(np.all(a_current != a_target) and np.all(a_current > 0) and np.all(a_target > 0), 'Content truth must change active action')
    attrs = decoded_attributes(recipient_states, actions, l)
    current_attrs = decoded_attributes(recipient_states, current, l)
    target_attrs = decoded_attributes(recipient_states, target, l)
    equal_truth = current_attrs == target_attrs
    equality = attrs == target_attrs
    native = old_channel.settle(recipient_states, actions)
    cf_states = recipient_states.copy()
    cf_states[:, :3] = counterfactual_states[:, :3]
    cf = old_channel.settle(cf_states, actions)
    require(np.array_equal(native['executed'], cf['executed']), 'Physical result changed with demand-only counterfactual')
    require(np.array_equal(cf['greedy_reward'] == 1, np.all(actions == target, axis=1)), 'Unique counterfactual solution mismatch')
    for name in ('greedy_reward', 'executed', 'satisfied'):
        require(np.array_equal(data[name], native[name]), 'Saved native settlement mismatch: '+name)
    require(np.array_equal(data['counterfactual_reward'], cf['greedy_reward']), 'Saved counterfactual settlement mismatch')
    packets = data['patched_outward_packets']
    codes = packet_codes(packets)
    require(codes.shape == (n,) and len(code_sets) == 3, 'Wrong patched packet/code-set dimensions')
    for name, source in (('natural', natural), ('intervention', data)):
        messages = np.asarray(source['messages'])
        require(messages.shape == (n, 2, 3, 4) and messages.dtype.kind in 'iu'
                and np.all((messages >= 0) & (messages < 8)), 'Invalid ' + name + ' messages')
    require(np.shape(natural['action_indices']) == (n, 3), 'Invalid natural action dimensions')
    seen = np.zeros(n, bool)
    for actor in range(3):
        mask = s == actor
        seen[mask] = np.isin(codes[mask], code_sets[actor])
    output = dict(target_apt=a == a_target, current_apt=a == a_current,
        target_probability=p[rows, l, a_target], current_probability=p[rows, l, a_current],
        natural_current_apt=natural['action_indices'][rows, l] == a_current,
        listener_action_changed=a != natural['action_indices'][rows, l],
        native_reward=native['greedy_reward'], native_team_full=native['greedy_reward'] == 1,
        counterfactual_reward=cf['greedy_reward'], counterfactual_team_full=cf['greedy_reward'] == 1,
        physical_pair_executed=native['executed'].sum(-1) == 2,
        unchanged_requirements_apt=np.all(equality | ~equal_truth, axis=1),
        packet_in_train_endpoint_greedy_set=seen,
        any_self_generated_message_changed=np.any(data['messages'] != natural['messages'], axis=(1, 2, 3)),
        listener_second_message_changed=np.any(data['messages'][:, 1][rows, l] != natural['messages'][:, 1][rows, l], axis=-1))
    for column, attribute in enumerate(ATTRIBUTES):
        output[attribute + '_apt'] = equality[:, column]
    return output


def contrast(same, opposite):
    require(set(same) == set(opposite), 'Different arm metrics')
    require(bool(same), 'Empty arm metrics')
    shape = np.shape(next(iter(same.values())))
    require((len(shape) == 1 or (len(shape) == 2 and shape[0] == 2))
            and all(np.shape(v) == shape and np.isfinite(v).all()
                    for v in (*same.values(), *opposite.values())), 'Invalid same/opposite row vectors')
    result = {k: np.asarray(opposite[k], float)-np.asarray(same[k], float) for k in same}
    require(np.array_equal(same['natural_current_apt'], opposite['natural_current_apt']), 'Natural control differs across arms')
    outside_both = ~np.asarray(same['packet_in_train_endpoint_greedy_set'], bool) & ~np.asarray(opposite['packet_in_train_endpoint_greedy_set'], bool)
    result['both_packets_outside_train_greedy_set'] = outside_both
    result['outside_both_target_apt_gain'] = outside_both * result['target_apt']
    return result


def aggregate_rows(spec, values):
    """Input vectors have shape (2,N), directions then static validation rows."""
    n = len(spec['axis'])
    require(n == 144 and bool(values) and all(np.shape(v) == (2, n) and np.isfinite(v).all() for v in values.values()), 'Keep both directions and every validation row')
    require(np.asarray(spec['axis']).dtype.kind in 'iu' and np.all((spec['axis'] >= 0) & (spec['axis'] < 3)), 'Invalid validation axes')
    require(np.shape(spec['sender']) == np.shape(spec['listener']) == (n,)
            and np.all(spec['sender'] != spec['listener']), 'Invalid validation sender/listener')
    require(np.shape(spec['base_within_axis_weight']) == (n,)
            and np.allclose(spec['base_within_axis_weight'], 1 / 48, rtol=0, atol=1e-15), 'Validation weights differ from fixed balanced sample')
    axes = {}; by_sender_listener = {}; by_direction = {}
    for axis, label in enumerate(AXES):
        mask = spec['axis'] == axis
        require(mask.sum() == 48, 'Each axis must contain 48 undirected validation rows')
        axes[label] = {k: float(np.mean(v[:, mask])) for k, v in values.items()}
        by_sender_listener[label] = {}
        for sender in range(3):
            for listener in range(3):
                if sender == listener:
                    continue
                m = mask & (spec['sender'] == sender) & (spec['listener'] == listener)
                require(m.sum() == 8, 'Every ordered pair keeps four cases and two backgrounds')
                by_sender_listener[label][f'{sender}_{listener}'] = {k: float(np.mean(v[:, m])) for k,v in values.items()}
        by_direction[label] = {str(d): {k: float(np.mean(v[d, mask])) for k,v in values.items()} for d in (0,1)}
    return dict(by_axis=axes, macro={k:float(np.mean([axes[a][k] for a in AXES])) for k in values},
                by_sender_listener=by_sender_listener, by_direction=by_direction)


def selection_summary(spec, selection, position_reports):
    """Positions are selected solely from training, before these row measures exist.

    position_reports[u] contains same/opposite/contrast metric arrays (2,N).
    Keep all eight per-position reports as output; never select by held-out scores.
    """
    chosen = np.asarray(selection['selected_positions'])
    require(chosen.shape == (3, 3) and chosen.dtype.kind in 'iu' and np.all((chosen >= 0) & (chosen < 8)), 'Bad frozen selected positions')
    require(set(position_reports) == set(range(8)), 'All eight positions required')
    matrix = np.zeros((3,3)); selected = {}; n=len(spec['sender'])
    schemas = {arm: set(position_reports[0][arm]) for arm in ('same', 'opposite', 'contrast')}
    require(schemas['same'] == schemas['opposite'], 'Arm measure schemas differ')
    natural_reference = np.asarray(position_reports[0]['same']['natural_current_apt'])
    for unit in range(8):
        require(set(position_reports[unit]) == set(schemas), 'Wrong position arms')
        for arm in schemas:
            report = position_reports[unit][arm]
            require(set(report) == schemas[arm] and all(np.shape(value) == (2, n) and np.isfinite(value).all()
                    for value in report.values()), 'All positions must retain the same complete row schema')
        same, opposite, difference = (position_reports[unit][arm] for arm in ('same', 'opposite', 'contrast'))
        require(np.array_equal(same['natural_current_apt'], natural_reference)
                and np.array_equal(opposite['natural_current_apt'], natural_reference), 'Position evaluation changed natural sample/control')
        require('target_apt' in difference and np.array_equal(difference['target_apt'],
                np.asarray(opposite['target_apt'], float) - np.asarray(same['target_apt'], float)), 'Target contrast does not pair the same two arm rows')
    for q, label in enumerate(AXES):
        units = chosen[spec['sender'], q]
        selected[label] = {}
        for arm in ('same','opposite','contrast'):
            keys = position_reports[0][arm]
            values = {k:np.zeros((2,n),float) for k in keys}
            for unit in range(8):
                mask = units == unit
                for key in keys:
                    values[key][:,mask] = position_reports[unit][arm][key][:,mask]
            selected[label][arm] = aggregate_rows(spec,values)
        for r, axis in enumerate(AXES):
            matrix[q,r] = selected[label]['contrast']['by_axis'][axis]['target_apt']
    uniform = {}
    for arm in ('same','opposite','contrast'):
        values={k:sum(position_reports[u][arm][k].astype(float) for u in range(8))/8 for k in position_reports[0][arm]}
        uniform[arm]=aggregate_rows(spec,values)
    diagonal = float(np.trace(matrix)/3)
    offdiagonal = float((matrix.sum()-np.trace(matrix))/6)
    baseline = uniform['contrast']['macro']['target_apt']
    # Equivalent balanced H contrast, written with within-column differences so
    # literally shared q rows yield exact zero without a tolerance/sign clamp.
    specificity = float(sum((matrix[i, i] - matrix[j, i]) - (matrix[i, j] - matrix[j, j])
                            for i in range(3) for j in range(i + 1, 3)) / 6)
    return dict(matrix_rows_discovered_axis_columns_test_axis=matrix.tolist(), selected=selected, uniform_position=uniform,
        diagonal_effect=diagonal, offdiagonal_effect=offdiagonal, selectivity=specificity,
        uniform_position_effect=baseline, diagonal_minus_uniform_position=diagonal-baseline,
        interpretation='Directional one-listener action appropriateness. Positive selectivity alone is insufficient if diagonal effect or advantage over the average position is absent.')
