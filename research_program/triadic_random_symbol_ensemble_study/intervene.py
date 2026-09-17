"""First-window live intervention for a random symbol permutation."""
from __future__ import annotations

import numpy as np

from research_program.triadic_message_study import runner as core


def require(ok, message):
    if not ok:
        raise ValueError(message)


def route(tokens):
    t = np.asarray(tokens)
    require(t.ndim == 3 and t.shape[1:] == (3, 4) and t.dtype.kind in "iu" and np.all((t >= 0) & (t < 8)), "Invalid message tokens")
    visibility = np.ones((3, 3), dtype=np.float64)
    one_hot = np.eye(8, dtype=np.float64)[t]
    return np.concatenate(((one_hot[:, None] * visibility[None, :, :, None, None]).reshape(len(t), 3, 96),
                           np.broadcast_to(visibility[None], (len(t), 3, 3))), axis=-1)


def intervene_first(networks, observations, natural_messages, senders, donor_packets):
    """Replace the selected sender's W1 packet, then greedily replay W2/actions."""
    x = np.asarray(observations, dtype=np.float64); n = len(x)
    natural = np.asarray(natural_messages); senders = np.asarray(senders); donor = np.asarray(donor_packets)
    require(x.shape == (n, 3, 54) and np.isfinite(x).all(), "Invalid observations")
    require(natural.shape == (n, 2, 3, 4) and natural.dtype.kind in "iu" and np.all((natural >= 0) & (natural < 8)), "Invalid natural messages")
    require(senders.shape == (n,) and senders.dtype.kind in "iu" and np.all((senders >= 0) & (senders < 3)), "Invalid senders")
    require(donor.shape == (n, 4) and donor.dtype.kind in "iu" and np.all((donor >= 0) & (donor < 8)), "Invalid donor packet")
    first = natural[:, 0]
    routed_first = route(first)
    encoded = np.eye(8, dtype=np.float64)[donor].reshape(n, 32)
    for viewer in range(3):
        rows = np.flatnonzero(senders != viewer)
        slots = 32 * senders[rows, None] + np.arange(32)
        routed_first[rows[:, None], viewer, slots] = encoded[rows]
    second_input = np.concatenate((x, routed_first), axis=-1)
    second_logits = np.stack([
        core.base.actor_forward(networks[3 * actor + 1], second_input[:, actor])[0].reshape(n, 4, 8)
        for actor in range(3)], axis=1)
    second_probabilities, _ = core.base.policy_distribution(second_logits)
    second = np.argmax(second_probabilities, axis=-1).astype(np.int8)
    routed_second = route(second)
    action_input = np.concatenate((x, routed_first, routed_second), axis=-1)
    action_logits = np.stack([
        core.base.actor_forward(networks[3 * actor + 2], action_input[:, actor])[0]
        for actor in range(3)], axis=1)
    probabilities, _ = core.base.policy_distribution(action_logits)
    return dict(messages=np.stack((first, second), axis=1), action_probabilities=probabilities,
                action_indices=np.argmax(probabilities, axis=-1).astype(np.int16),
                first_routes=routed_first, second_routes=routed_second,
                neural_forward_samples=6 * n)


def observations(states):
    from research_program.triadic_action_dependency_study import dataset as task_dataset
    from research_program.triadic_action_dependency_study import environment as env
    packed = np.asarray(states)
    require(packed.ndim == 2 and packed.shape[1] == 10 and packed.dtype.kind in "iu", "Invalid packed states")
    worlds = [env.State(tuple(map(int, row[:3])), tuple(map(int, row[3:7])), tuple(map(int, row[7:10]))) for row in packed]
    return task_dataset.encode_observations([
        {actor: env.observe(world, actor, information="PL") for actor in env.AGENTS}
        for world in worlds])
