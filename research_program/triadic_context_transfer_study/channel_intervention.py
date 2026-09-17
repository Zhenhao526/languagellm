"""Whole-packet interventions on one sender's outward edges in the new task.

Only two-window packets are replaced. All W2 descendants and all action heads
are recomputed; sender self channels and recipient observations stay local.
"""
import numpy as np
from research_program.triadic_message_study import runner as core
from research_program.triadic_action_dependency_study import dataset as native_dataset
from research_program.triadic_action_dependency_study import environment as env
from research_program.triadic_action_dependency_study import metrics as native_metrics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _tokens(value, shape, name):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in 'iu'
            and np.all((a >= 0) & (a < 8)), name + ' must contain integer symbols 0..7')
    return a


def _input(observations, natural_messages, senders, donor_packets):
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and len(x) > 0
            and np.isfinite(x).all(), 'observations must be finite B×3×54')
    n = len(x)
    m = _tokens(natural_messages, (n, 2, 3, 4), 'natural_messages')
    donor = _tokens(donor_packets, (n, 2, 4), 'donor_packets')
    senders = np.asarray(senders)
    require(senders.shape == (n,) and senders.dtype.kind in 'iu'
            and np.all((senders >= 0) & (senders < 3)), 'Invalid sender indices')
    return x, m, senders, donor


def replace_outward(tokens, senders, replacements, live):
    tokens = np.asarray(tokens); n = len(tokens)
    _tokens(tokens, (n, 3, 4), 'tokens')
    replacements = _tokens(replacements, (n, 4), 'replacements')
    senders = np.asarray(senders)
    require(senders.shape == (n,) and senders.dtype.kind in 'iu'
            and np.all((senders >= 0) & (senders < 3)), 'Invalid sender indices')
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    route = core.routed_window(tokens, bool(live))
    if live:
        encoded = np.eye(8, dtype=np.float64)[replacements].reshape(n, 32)
        for viewer in range(3):
            rows = np.flatnonzero(senders != viewer)
            slots = 32 * senders[rows, None] + np.arange(32)
            route[rows[:, None], viewer, slots] = encoded[rows]
    return route


def natural_action_inputs(observations, natural_messages, live):
    x = np.asarray(observations, dtype=np.float64); n = len(x)
    require(x.shape == (n, 3, 54) and np.isfinite(x).all(), 'Invalid observations')
    m = _tokens(natural_messages, (n, 2, 3, 4), 'natural_messages')
    return np.concatenate((x, core.routed_window(m[:, 0], live),
                           core.routed_window(m[:, 1], live)), axis=-1)


def silent_routes(observations, natural_messages, senders, donor_packets):
    """No forward: an invisible replacement leaves the saved natural inputs intact."""
    x, m, senders, donor = _input(observations, natural_messages, senders, donor_packets)
    r1 = replace_outward(m[:, 0], senders, donor[:, 0], False)
    r2 = replace_outward(m[:, 1], senders, donor[:, 1], False)
    inputs = np.concatenate((x, r1, r2), axis=-1)
    require(np.array_equal(inputs, natural_action_inputs(x, m, False)), 'Silent routing changed input')
    return dict(first_routes=r1, second_routes=r2, action_inputs=inputs,
                reused_natural=True, neural_forward_samples=0)


def intervene(networks, observations, natural_messages, senders, donor_packets, live):
    """B rows, six module forwards each. Caller uses silent_routes for aliases.

    Return `messages` contains each agent's actual self-generated W1/W2, not the
    patched outgoing packet. Returned routes record what recipients received.
    Greedy symbols/actions are selected AFTER the original stable softmax.
    """
    x, m, senders, donor = _input(observations, natural_messages, senders, donor_packets)
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    route1 = replace_outward(m[:, 0], senders, donor[:, 0], bool(live))
    second_input = np.concatenate((x, route1), axis=-1)
    logits2 = np.stack([core.base.actor_forward(networks[3*a+1], second_input[:, a])[0]
                       .reshape(-1, 4, 8) for a in range(3)], axis=1)
    p2, _ = core.base.policy_distribution(logits2)
    second = core.categorical_tokens(p2)
    route2 = replace_outward(second, senders, donor[:, 1], bool(live))
    action_input = np.concatenate((x, route1, route2), axis=-1)
    logits = np.stack([core.base.actor_forward(networks[3*a+2], action_input[:, a])[0]
                       for a in range(3)], axis=1)
    probabilities, _ = core.base.policy_distribution(logits)
    return dict(messages=np.stack((m[:, 0], second), axis=1),
        action_inputs=action_input, first_routes=route1, second_routes=route2,
        action_probabilities=probabilities,
        action_indices=np.argmax(probabilities, axis=-1).astype(np.int16),
        reused_natural=False, neural_forward_samples=6*len(x))


def observations(states, information):
    """Official 24-demand acceptance-set encoder; old 12-demand helper is incompatible."""
    require(information in ('PL', 'LL'), 'Only frozen PL/LL policies are included')
    packed = np.asarray(states)
    require(packed.ndim == 2 and packed.shape[1] == 10 and packed.dtype.kind in 'iu', 'Invalid packed states')
    worlds = [env.State(tuple(map(int, row[:3])), tuple(map(int, row[3:7])),
                        tuple(map(int, row[7:10]))) for row in packed]
    return native_dataset.encode_observations([
        {a: env.observe(s, a, information=information) for a in env.AGENTS} for s in worlds])


def settle(states, actions):
    r = native_metrics.settle_arrays(states, actions)
    return dict(greedy_reward=r['reward'], executed=r['executed'], satisfied=r['satisfied'])
