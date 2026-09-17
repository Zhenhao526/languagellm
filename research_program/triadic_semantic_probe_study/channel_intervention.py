"""Causal interventions on one sender's outward packets, never its own memory."""
import numpy as np
from research_program.triadic_message_study import runner as core
from research_program.triadic_partner_ecology_study.metrics import decode_actions


def replace_outward(tokens, senders, replacements, live):
    tokens, senders, replacements = map(np.asarray, (tokens, senders, replacements))
    n = len(tokens)
    assert tokens.shape == (n, 3, 4) and senders.shape == (n,) and replacements.shape == (n, 4)
    assert senders.dtype.kind in 'iu' and np.all((senders >= 0) & (senders < 3))
    assert replacements.dtype.kind in 'iu' and np.all((replacements >= 0) & (replacements < 8))
    route = core.routed_window(tokens, live)
    if live:
        encoded = np.eye(8)[replacements].reshape(n, 32)
        for viewer in range(3):
            rows = np.flatnonzero(senders != viewer)
            slots = 32 * senders[rows, None] + np.arange(32)
            route[rows[:, None], viewer, slots] = encoded[rows]
    return route


def intervene(networks, observations, natural_messages, senders, donor_messages, windows, live):
    """Reuse W1 (upstream), recompute every W2 and every action (six modules).

    donor_messages is [B,2,4], from the designated sender alone. Replacement
    affects only the two other agents' views; all visibility bits are retained.
    """
    x, m, donor = map(np.asarray, (observations, natural_messages, donor_messages))
    assert x.shape == (len(x), 3, 54) and m.shape == (len(x), 2, 3, 4)
    assert donor.shape == (len(x), 2, 4) and tuple(windows) in ((0,), (1,), (0, 1))
    route1 = replace_outward(m[:, 0], senders, donor[:, 0], live) if 0 in windows else core.routed_window(m[:, 0], live)
    second_input = np.concatenate((x, route1), axis=-1)
    second_logits = np.stack([core.base.actor_forward(networks[3*a+1], second_input[:, a])[0]
                              .reshape(-1, 4, 8) for a in range(3)], axis=1)
    second_probabilities, _ = core.base.policy_distribution(second_logits)
    second = core.categorical_tokens(second_probabilities)
    route2 = replace_outward(second, senders, donor[:, 1], live) if 1 in windows else core.routed_window(second, live)
    action_input = np.concatenate((x, route1, route2), axis=-1)
    logits = np.stack([core.base.actor_forward(networks[3*a+2], action_input[:, a])[0] for a in range(3)], axis=1)
    probabilities, _ = core.base.policy_distribution(logits)
    return dict(messages=np.stack((m[:, 0], second), axis=1),
                action_inputs=action_input, first_routes=route1, second_routes=route2,
                action_probabilities=probabilities, action_indices=np.argmax(probabilities, axis=-1).astype(np.int16))


def observations(states, full):
    """Official observation encoder, called once per complete partition."""
    return core.base.encode_observations([
        {a: core.base.env.observe(core.base.env.State(tuple(map(int, row[:3])), tuple(map(int, row[3:7])),
                                                    tuple(map(int, row[7:]))), a,
                                 shared_needs=full, full_information=full) for a in core.base.AGENTS}
        for row in states])


def settle(states, actions):
    """Vectorized original native score. No policy-dependent success labels."""
    states, actions = np.asarray(states), np.asarray(actions)
    assert states.shape == (len(actions), 10)
    decoded = decode_actions(actions)
    active, sites, dests, partners = (decoded[k] for k in ('active', 'site', 'destination', 'partner'))
    reward = np.zeros(len(actions), dtype=np.float64)
    executed = np.zeros((len(actions), 3), dtype=bool)
    satisfied = np.zeros_like(executed)
    resource_bits = np.array([3, 12, 5, 10]); dest_bits = np.array([1, 2, 3])
    for i, j, k in ((0, 1, 2), (0, 2, 1), (1, 2, 0)):
        match = active[:, i] & active[:, j] & ~active[:, k]
        match &= (partners[:, i] == j) & (partners[:, j] == i)
        match &= (sites[:, i] == sites[:, j]) & (dests[:, i] == dests[:, j])
        rows = np.flatnonzero(match)
        material = states[rows, 3 + sites[rows, i]]
        destination = dests[rows, i]
        for a in (i, j):
            need = states[rows, a]
            good = ((resource_bits[need // 3] & (1 << material)) != 0) & ((dest_bits[need % 3] & (1 << destination)) != 0)
            reward[rows] += good / 2
            executed[rows, a] = True; satisfied[rows, a] = good
    return dict(greedy_reward=reward, executed=executed, satisfied=satisfied)
