"""Differentiable-learning wrapper with an explicit discrete wire map.

The map is applied to sampled/greedy token indices before routed one-hot
features are built.  Sender score gradients still use the internal sampled
tokens; the wire transformation is discrete and therefore has no derivative.
"""
from __future__ import annotations

import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_reciprocal_execution_study import kernel


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_maps(maps):
    maps = tuple(np.asarray(row, dtype=np.int8) for row in maps)
    require(len(maps) == 3, "Three sender maps required")
    for row in maps:
        require(row.shape == (8,) and np.array_equal(np.sort(row), np.arange(8)), "Invalid wire bijection")
    return maps


def recode_tokens(tokens, maps):
    tokens = np.asarray(tokens, dtype=np.int8)
    require(tokens.ndim == 3 and tokens.shape[1:] == (3, 4), "Invalid token tensor")
    maps = validate_maps(maps)
    output = np.empty_like(tokens)
    for sender, mapping in enumerate(maps):
        output[:, sender] = mapping[tokens[:, sender]]
    return output


def rollout(networks, observations, live, maps, uniforms=None):
    """Two-window rollout with internal and wire token traces."""
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and len(networks) == 9, "Invalid rollout input")
    require(np.isfinite(x).all(), "Non-finite observations")
    maps = validate_maps(maps)
    if uniforms is not None:
        uniforms = np.asarray(uniforms, dtype=np.float64)
        require(uniforms.shape == (len(x), 2, 3, 4), "Invalid message uniforms")
    caches = [None] * 9
    sender_logits = []
    sender_probabilities = []
    sender_log_probabilities = []
    messages = []
    wire_messages = []
    inputs = x
    for window in range(2):
        logits = []
        for actor in range(3):
            z, caches[3 * actor + window] = core.base.actor_forward(networks[3 * actor + window], inputs[:, actor])
            logits.append(z.reshape(len(x), 4, 8))
        z = np.stack(logits, axis=1)
        probabilities, log_probabilities = core.base.policy_distribution(z)
        internal = core.categorical_tokens(
            probabilities, None if uniforms is None else uniforms[:, window]
        )
        on_wire = recode_tokens(internal, maps)
        sender_logits.append(z)
        sender_probabilities.append(probabilities)
        sender_log_probabilities.append(log_probabilities)
        messages.append(internal)
        wire_messages.append(on_wire)
        if window == 0:
            inputs = np.concatenate((x, core.routed_window(on_wire, bool(live))), axis=-1)
    action_inputs = np.concatenate(
        (x, core.routed_window(wire_messages[0], bool(live)), core.routed_window(wire_messages[1], bool(live))), axis=-1
    )
    action_logits = []
    for actor in range(3):
        z, caches[3 * actor + 2] = core.base.actor_forward(networks[3 * actor + 2], action_inputs[:, actor])
        action_logits.append(z)
    return dict(
        messages=np.stack(messages, axis=1),
        wire_messages=np.stack(wire_messages, axis=1),
        sender_logits=np.stack(sender_logits, axis=1),
        sender_probabilities=np.stack(sender_probabilities, axis=1),
        sender_log_probabilities=np.stack(sender_log_probabilities, axis=1),
        action_logits=np.stack(action_logits, axis=1),
        action_inputs=action_inputs,
        caches=caches,
    )


def training_gradients(networks, observations, rewards, live, uniforms, update, rule, maps):
    """Exact receiver gradients plus paired score-function sender gradients."""
    x = np.asarray(observations, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    uniforms = np.asarray(uniforms, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54), "Invalid training observations")
    require(rewards.ndim == 2 and rewards.shape[0] == len(x), "Invalid training rewards")
    require(uniforms.shape == (2, len(x), 2, 3, 4), "Invalid paired uniforms")
    trace = rollout(networks, np.concatenate((x, x)), live, maps, uniforms.reshape(2 * len(x), 2, 3, 4))
    terms = kernel.objective_terms(trace["action_logits"], np.concatenate((rewards, rewards)), rule)
    receiver = core.coordination.loss_and_derivative(terms, "mean_log_J", update)
    entropy, _ = core.base.entropy_and_logit_gradient(terms["probabilities"], terms["log_probabilities"])
    f_value = terms["log_J"] + receiver["entropy_coefficient"] * entropy
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
        wire_maps=[list(map(int, row)) for row in maps],
        rule=rule,
    )
    return gradients, row

