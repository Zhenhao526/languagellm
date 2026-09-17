"""First-window message replacement and causal downstream recomputation."""
from __future__ import annotations

import numpy as np

from research_program.triadic_message_study import runner as core


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _tokens(value, shape, name):
    a = np.asarray(value)
    require(a.shape == shape and a.dtype.kind in "iu" and np.all((a >= 0) & (a < 8)), name + " must be integer symbols 0..7")
    return a


def route_window(tokens, live):
    tokens = np.asarray(tokens)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4) and tokens.dtype.kind in "iu" and np.all((tokens >= 0) & (tokens < 8)), "Invalid message tokens")
    visibility = np.ones((3, 3), dtype=np.float64) if bool(live) else np.eye(3, dtype=np.float64)
    onehot = np.eye(8, dtype=np.float64)[tokens]
    visible = onehot[:, None, :, :, :] * visibility[None, :, :, None, None]
    return np.concatenate((visible.reshape(len(tokens), 3, 96), np.broadcast_to(visibility[None], (len(tokens), 3, 3))), axis=-1)


def replace_outward(first_messages, senders, donor_packets, live=True):
    first = _tokens(first_messages, (len(first_messages), 3, 4), "first_messages")
    donor = _tokens(donor_packets, (len(first_messages), 4), "donor_packets")
    s = np.asarray(senders)
    require(s.shape == (len(first),) and s.dtype.kind in "iu" and np.all((s >= 0) & (s < 3)), "Invalid senders")
    route = route_window(first, live)
    if live:
        encoded = np.eye(8, dtype=np.float64)[donor].reshape(len(first), 32)
        for viewer in range(3):
            rows = np.flatnonzero(s != viewer)
            slots = 32 * s[rows, None] + np.arange(32)
            route[rows[:, None], viewer, slots] = encoded[rows]
    return route


def first_window_intervene(networks, observations, natural_messages, senders, donor_packets):
    """Replace only the sender's outward W1 packet, then greedily recompute W2/action."""
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and np.isfinite(x).all(), "Invalid PL observations")
    n = len(x)
    m = _tokens(natural_messages, (n, 2, 3, 4), "natural_messages")
    s = np.asarray(senders)
    d = _tokens(donor_packets, (n, 4), "donor_packets")
    route1 = replace_outward(m[:, 0], s, d, True)
    second_input = np.concatenate((x, route1), axis=-1)
    logits2 = np.stack([core.base.actor_forward(networks[3 * a + 1], second_input[:, a])[0].reshape(n, 4, 8)
                         for a in range(3)], axis=1)
    second_probabilities, _ = core.base.policy_distribution(logits2)
    second = np.argmax(second_probabilities, axis=-1).astype(np.int8)
    route2 = route_window(second, True)
    action_input = np.concatenate((x, route1, route2), axis=-1)
    logits = np.stack([core.base.actor_forward(networks[3 * a + 2], action_input[:, a])[0]
                       for a in range(3)], axis=1)
    probabilities, _ = core.base.policy_distribution(logits)
    actions = np.argmax(probabilities, axis=-1).astype(np.int16)
    return dict(messages=np.stack((m[:, 0], second), axis=1), first_routes=route1,
                second_routes=route2, action_inputs=action_input,
                action_probabilities=probabilities, action_indices=actions,
                neural_forward_samples=6 * n)


def observations(states):
    """Official PL encoder, kept as a small execution helper."""
    from research_program.triadic_action_dependency_study import dataset as task_dataset
    from research_program.triadic_action_dependency_study import environment as env
    packed = np.asarray(states)
    require(packed.ndim == 2 and packed.shape[1] == 10 and packed.dtype.kind in "iu", "Invalid packed states")
    worlds = [env.State(tuple(map(int, row[:3])), tuple(map(int, row[3:7])), tuple(map(int, row[7:10]))) for row in packed]
    return task_dataset.encode_observations([{a: env.observe(s, a, information="PL") for a in env.AGENTS} for s in worlds])
