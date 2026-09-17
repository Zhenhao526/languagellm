"""First-window intervention used by the symbol recoding runner."""
from __future__ import annotations

import numpy as np

from research_program.triadic_message_study import runner as core


def require(ok, message):
    if not ok:
        raise ValueError(message)


def route(tokens):
    t = np.asarray(tokens)
    require(t.ndim == 3 and t.shape[1:] == (3, 4) and t.dtype.kind in "iu" and np.all((t >= 0) & (t < 8)), "Invalid message tokens")
    vis = np.ones((3, 3), dtype=np.float64); one = np.eye(8, dtype=np.float64)[t]
    return np.concatenate(((one[:, None] * vis[None, :, :, None, None]).reshape(len(t), 3, 96), np.broadcast_to(vis[None], (len(t), 3, 3))), axis=-1)


def intervene_first(networks, observations, natural_messages, senders, donor_packets):
    x = np.asarray(observations, dtype=np.float64); n = len(x); m = np.asarray(natural_messages)
    s = np.asarray(senders); d = np.asarray(donor_packets)
    require(x.shape == (n, 3, 54) and np.isfinite(x).all(), "Invalid observations")
    require(m.shape == (n, 2, 3, 4) and m.dtype.kind in "iu" and np.all((m >= 0) & (m < 8)), "Invalid natural messages")
    require(s.shape == (n,) and s.dtype.kind in "iu" and np.all((s >= 0) & (s < 3)), "Invalid senders")
    require(d.shape == (n, 4) and d.dtype.kind in "iu" and np.all((d >= 0) & (d < 8)), "Invalid donor packet")
    r1 = route(m[:, 0]); enc = np.eye(8, dtype=np.float64)[d].reshape(n, 32)
    for viewer in range(3):
        rows = np.flatnonzero(s != viewer); slots = 32 * s[rows, None] + np.arange(32)
        r1[rows[:, None], viewer, slots] = enc[rows]
    z2 = np.concatenate((x, r1), axis=-1)
    l2 = np.stack([core.base.actor_forward(networks[3 * a + 1], z2[:, a])[0].reshape(n, 4, 8) for a in range(3)], axis=1)
    p2, _ = core.base.policy_distribution(l2); second = np.argmax(p2, axis=-1).astype(np.int8); r2 = route(second)
    ai = np.concatenate((x, r1, r2), axis=-1)
    logits = np.stack([core.base.actor_forward(networks[3 * a + 2], ai[:, a])[0] for a in range(3)], axis=1)
    p, _ = core.base.policy_distribution(logits)
    return dict(messages=np.stack((m[:, 0], second), axis=1), action_probabilities=p,
                action_indices=np.argmax(p, axis=-1).astype(np.int16), first_routes=r1,
                second_routes=r2, neural_forward_samples=6 * n)


def observations(states):
    from research_program.triadic_action_dependency_study import dataset as task_dataset
    from research_program.triadic_action_dependency_study import environment as env
    packed = np.asarray(states)
    require(packed.ndim == 2 and packed.shape[1] == 10 and packed.dtype.kind in "iu", "Invalid packed states")
    worlds = [env.State(tuple(map(int, row[:3])), tuple(map(int, row[3:7])), tuple(map(int, row[7:10]))) for row in packed]
    return task_dataset.encode_observations([{a: env.observe(world, a, information="PL") for a in env.AGENTS} for world in worlds])


def observations(states):
    from research_program.triadic_action_dependency_study import dataset as task_dataset
    from research_program.triadic_action_dependency_study import environment as env
    packed = np.asarray(states)
    require(packed.ndim == 2 and packed.shape[1] == 10 and packed.dtype.kind in "iu", "Invalid packed states")
    worlds = [env.State(tuple(map(int, row[:3])), tuple(map(int, row[3:7])), tuple(map(int, row[7:10]))) for row in packed]
    return task_dataset.encode_observations([{a: env.observe(world, a, information="PL") for a in env.AGENTS} for world in worlds])
