"""Greedy formation measurements; no model loading, training, or checkpoint I/O.

The caller determines the complete train/validation domains and checkpoint
identities. Each function accepts a batch of official 54-dimensional observations.
Sender-only generation costs six module samples/world; natural rollout costs
nine. Interventions reuse the frozen earlier implementations, without editing
them. A module sample is one world passed through one of the nine networks.
"""
import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_position_reuse_study import channel_intervention as position
from research_program.triadic_context_transfer_study import channel_intervention as whole

ALPHABET = tuple(core.ALPHABET)
env = whole.env


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _input(networks, observations, live):
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and len(x) > 0
            and np.isfinite(x).all(), 'Expected finite nonempty B×3×54 observations')
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    require(networks is not None and len(networks) == 9, 'Nine independent modules required')
    return x


def _routing_hashes(out):
    return {key: core.array_sha(out[key])
            for key in ('first_routes', 'second_routes', 'action_inputs') if key in out}


def generate_messages(networks, observations, live, *, include_routes=False):
    """Greedy W1 then W2, with a complete synchronous barrier before routing.

    This never calls action heads (indices2/5/8). For a complete train pass the
    caller must concatenate every batch, including states outside discovery
    pairs. No random draws are made. Greedy means argmax after stable softmax.
    """
    x = _input(networks, observations, live)
    require(isinstance(include_routes, (bool, np.bool_)), 'include_routes must be boolean')
    n = len(x)
    logits1 = np.stack([core.base.actor_forward(networks[3*a], x[:, a])[0]
                       .reshape(n, 4, 8) for a in range(3)], axis=1)
    p1, _ = core.base.policy_distribution(logits1)
    first = core.categorical_tokens(p1)
    route1 = core.routed_window(first, bool(live))
    second_input = np.concatenate((x, route1), axis=-1)
    logits2 = np.stack([core.base.actor_forward(networks[3*a+1], second_input[:, a])[0]
                       .reshape(n, 4, 8) for a in range(3)], axis=1)
    p2, _ = core.base.policy_distribution(logits2)
    second = core.categorical_tokens(p2)
    out = dict(messages=np.stack((first, second), axis=1),
               neural_forward_samples=6*n, neural_forward_calls=6,
               first_window_forward_samples=3*n, second_window_forward_samples=3*n,
               action_forward_samples=0, reused_natural=False)
    if include_routes:
        out.update(first_routes=route1, second_routes=core.routed_window(second, bool(live)))
        out['routing_sha256'] = _routing_hashes(out)
    return out


def natural_rollout(networks, observations, live):
    """Complete receiver-checkpoint natural rollout: 3W1 + 3W2 + 3actions."""
    x = _input(networks, observations, live)
    out = generate_messages(networks, x, live, include_routes=True)
    inputs = np.concatenate((x, out['first_routes'], out['second_routes']), axis=-1)
    logits = np.stack([core.base.actor_forward(networks[3*a+2], inputs[:, a])[0]
                       for a in range(3)], axis=1)
    probabilities, _ = core.base.policy_distribution(logits)
    out.update(action_inputs=inputs, action_probabilities=probabilities,
               action_indices=np.argmax(probabilities, axis=-1).astype(np.int16),
               neural_forward_samples=9*len(x), neural_forward_calls=9,
               action_forward_samples=3*len(x))
    out['routing_sha256'] = _routing_hashes(out)
    return out


def position_intervention(networks, observations, natural_messages, senders,
                          donor_packets, windows, positions, live=True):
    """Use the unchanged single-symbol splice. Unseen mixed packets stay legal.

    W1 rows cost6 module samples, W2 rows cost3; silent aliases cost0.
    The saved natural history must belong to the supplied receiver checkpoint.
    """
    return position.intervene(networks, observations, natural_messages, senders,
                              donor_packets, windows, positions, live)


def whole_intervention(receiver_networks, observations, receiver_natural_messages,
                       senders, donor_packets, live=True):
    """Replace only focal sender outgoing W1/W2 to the other two agents.

    Receiver observations, natural W1, and all recomputed modules are from t_r.
    Donor packets may come from t_s; donor actions/observations never enter this
    API. Recompute all three receiver W2 and all three actions (6B samples).
    The focal sender's own W2 is exactly its natural t_r W2 because its W1 input
    is unchanged. Its final action need not stay natural: the other agents' W2
    messages can change. Silent routes are verified aliases with zero forwards.
    """
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and len(x) > 0
            and np.isfinite(x).all(), 'Expected finite nonempty B×3×54 observations')
    n = len(x)
    s = position._indices(senders, n, 3, 'senders')
    m = position._tokens(receiver_natural_messages, (n, 2, 3, 4), 'receiver_natural_messages')
    donor = position._tokens(donor_packets, (n, 2, 4), 'donor_packets')
    if live:
        _input(receiver_networks, x, live)
        out = whole.intervene(receiver_networks, x, m, s, donor, True)
        require(np.array_equal(out['messages'][np.arange(n), 1, s], m[np.arange(n), 1, s]),
                'Focal sender W2 differs from its receiver-checkpoint natural history')
        out.update(neural_forward_calls=6, first_window_forward_samples=0,
                   second_window_forward_samples=3*n, action_forward_samples=3*n)
    else:
        out = whole.silent_routes(x, m, s, donor)
        out.update(messages=m.copy(), neural_forward_calls=0,
                   first_window_forward_samples=0, second_window_forward_samples=0,
                   action_forward_samples=0)
    out.update(patched_outward_packets=donor.copy(), senders=s.copy(),
               outward_patch_visible=bool(live), own_sender_second_equals_receiver_natural=True)
    out['routing_sha256'] = _routing_hashes(out)
    return out


def cross_time_whole(receiver_networks, observations, receiver_natural_messages,
                     senders, donor_packets, live=True):
    """Named cross-time entry; same routing/cost as whole_intervention.

    For a6×6 checkpoint grid, costs are6×36×B=216B module samples for one set of
    B recipient rows, before accounting for additional arms/directions/policies.
    Saved donor/natural generation costs are separate and must not be counted as
    zero if newly generated. No checkpoint time is inserted into agent inputs.
    """
    return whole_intervention(receiver_networks, observations, receiver_natural_messages,
                              senders, donor_packets, live)


def observations(states, information):
    """Unchanged official 24-need PL/LL54 encoder; no network evaluation."""
    return whole.observations(states, information)


def settle(states, actions):
    """Unchanged recipient-world native R and physical settlement."""
    return whole.settle(states, actions)
