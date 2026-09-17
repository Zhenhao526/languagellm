"""Differentiable-learning wrapper with sender-slot permutations.

The internal token emitted by each sender is first passed through a fixed wire
map.  In a masked live condition, the three wire packets are then placed into
random sender slots independently for each world and message window.  The
permutation is exogenous and shared across the two paired trajectories.
"""
from __future__ import annotations

import numpy as np

from research_program.triadic_message_study import runner as core


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_maps(maps):
    maps = tuple(np.asarray(row, dtype=np.int8) for row in maps)
    require(len(maps) == 3, "Three sender maps required")
    for row in maps:
        require(row.shape == (8,) and np.array_equal(np.sort(row), np.arange(8)), "Invalid wire bijection")
    return maps


def validate_permutations(permutations, batch_size):
    if permutations is None:
        return np.broadcast_to(np.arange(3, dtype=np.int8), (batch_size, 3)).copy()
    permutations = np.asarray(permutations, dtype=np.int8)
    require(permutations.shape == (batch_size, 3), "Invalid sender-slot permutation shape")
    require(np.all(np.sort(permutations, axis=1) == np.arange(3)), "Each sender-slot row must be a permutation")
    return permutations


def recode_tokens(tokens, maps):
    tokens = np.asarray(tokens, dtype=np.int8)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4), "Invalid token tensor")
    maps = validate_maps(maps)
    output = np.empty_like(tokens)
    for sender, mapping in enumerate(maps):
        output[:, sender] = mapping[tokens[:, sender]]
    return output


def routed_window(tokens, live, permutations=None):
    """Return per-viewer routed one-hot packets and visibility bits.

    `permutations` maps sender index to visible slot.  Identity is used for
    stable routes and for silent self-only channels.  Live masked routes expose
    all three packets to every viewer but do not expose the permutation itself.
    """
    tokens = np.asarray(tokens, dtype=np.int8)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4), "Invalid routed token tensor")
    require(((tokens >= 0) & (tokens < 8)).all(), "Token outside alphabet")
    batch_size = len(tokens)
    permutations = validate_permutations(permutations, batch_size)
    slots = np.empty_like(tokens)
    for sender in range(3):
        slots[np.arange(batch_size), permutations[:, sender]] = tokens[:, sender]
    onehot = np.eye(8, dtype=np.float64)[slots]
    if live:
        visibility = np.ones((3, 3), dtype=np.float64)
    else:
        visibility = np.eye(3, dtype=np.float64)
    visible = onehot[:, None, :, :, :] * visibility[None, :, :, None, None]
    bits = np.broadcast_to(visibility[None], (batch_size, 3, 3))
    return np.concatenate((visible.reshape(batch_size, 3, 96), bits), axis=-1)


def rollout(networks, observations, live, maps, route_permutations=None, uniforms=None):
    """Causal two-window rollout with internal and on-wire token traces."""
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and len(networks) == 9, "Invalid rollout input")
    require(np.isfinite(x).all(), "Non-finite observations")
    batch_size = len(x)
    maps = validate_maps(maps)
    if uniforms is not None:
        uniforms = np.asarray(uniforms, dtype=np.float64)
        require(uniforms.shape == (batch_size, 2, 3, 4), "Invalid message uniforms")
    if route_permutations is not None:
        route_permutations = np.asarray(route_permutations, dtype=np.int8)
        require(route_permutations.shape == (batch_size, 2, 3), "Invalid route permutation tensor")
    caches = [None] * 9
    sender_logits, sender_probabilities, sender_log_probabilities = [], [], []
    internal_messages, wire_messages = [], []
    inputs = x
    for window in range(2):
        logits = []
        for actor in range(3):
            z, caches[3 * actor + window] = core.base.actor_forward(networks[3 * actor + window], inputs[:, actor])
            logits.append(z.reshape(batch_size, 4, 8))
        z = np.stack(logits, axis=1)
        probabilities, log_probabilities = core.base.policy_distribution(z)
        internal = core.categorical_tokens(probabilities, None if uniforms is None else uniforms[:, window])
        on_wire = recode_tokens(internal, maps)
        sender_logits.append(z)
        sender_probabilities.append(probabilities)
        sender_log_probabilities.append(log_probabilities)
        internal_messages.append(internal)
        wire_messages.append(on_wire)
        if window == 0:
            inputs = np.concatenate((x, routed_window(on_wire, live, None if not live else route_permutations[:, window])), axis=-1)
    action_inputs = np.concatenate((
        x,
        routed_window(wire_messages[0], live, None if not live else route_permutations[:, 0]),
        routed_window(wire_messages[1], live, None if not live else route_permutations[:, 1]),
    ), axis=-1)
    action_logits = []
    for actor in range(3):
        z, caches[3 * actor + 2] = core.base.actor_forward(networks[3 * actor + 2], action_inputs[:, actor])
        action_logits.append(z)
    return dict(
        messages=np.stack(internal_messages, axis=1),
        wire_messages=np.stack(wire_messages, axis=1),
        sender_logits=np.stack(sender_logits, axis=1),
        sender_probabilities=np.stack(sender_probabilities, axis=1),
        sender_log_probabilities=np.stack(sender_log_probabilities, axis=1),
        action_logits=np.stack(action_logits, axis=1),
        action_inputs=action_inputs,
        caches=caches,
    )


def training_gradients(networks, observations, rewards, live, uniforms, route_permutations,
                       update, maps, rule="reciprocal"):
    """Exact receiver gradients plus paired score-function sender gradients."""
    x = np.asarray(observations, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    route_permutations = np.asarray(route_permutations, dtype=np.int8)
    require(x.ndim == 3 and x.shape[1:] == (3, 54), "Invalid training observations")
    require(rewards.shape == (len(x), 24), "Invalid training rewards")
    require(uniforms.shape == (2, len(x), 2, 3, 4), "Invalid paired uniforms")
    require(route_permutations.shape == (2, len(x), 2, 3), "Invalid paired route permutations")
    trace = rollout(networks, np.concatenate((x, x)), live, maps,
                    route_permutations.reshape(2 * len(x), 2, 3),
                    uniforms.reshape(2 * len(x), 2, 3, 4))
    require(rule == "reciprocal", "This package freezes the audited reciprocal objective")
    terms, receiver, f_value = core.trajectory_objective(
        trace["action_logits"], np.concatenate((rewards, rewards)), update)
    sender = core.paired_sender_derivative(
        trace["sender_probabilities"].reshape(2, len(x), 2, 3, 4, 8),
        trace["sender_log_probabilities"].reshape(2, len(x), 2, 3, 4, 8),
        trace["messages"].reshape(2, len(x), 2, 3, 4),
        f_value.reshape(2, len(x)),
    )
    sender_derivative = sender["derivative"].reshape(2 * len(x), 2, 3, 4, 8)
    gradients = []
    for actor in range(3):
        for module in range(3):
            derivative = receiver["derivative"][:, actor] if module == 2 else sender_derivative[:, module, actor].reshape(2 * len(x), 32)
            gradients.append(core.base.actor_backward(networks[3 * actor + module], trace["caches"][3 * actor + module], derivative))
    row = dict(
        mean_expected_reward=float(terms["J"].mean()),
        mean_log_expected_reward=float(terms["log_J"].mean()),
        mean_f=float(f_value.mean()),
        receiver_loss=receiver["loss"],
        mean_actor_entropy=receiver["mean_actor_entropy"],
        entropy_coefficient=receiver["entropy_coefficient"],
        sender_surrogate_loss=sender["surrogate_loss"],
        sender_advantage_mean=float(sender["advantage"].mean()),
        sender_advantage_abs_mean=float(np.abs(sender["advantage"]).mean()),
        sampled_internal_messages_sha256=core.array_sha(trace["messages"]),
        sampled_wire_messages_sha256=core.array_sha(trace["wire_messages"]),
        route_permutations_sha256=core.array_sha(route_permutations),
        wire_maps=[list(map(int, row)) for row in maps],
        rule=rule,
    )
    return gradients, row
