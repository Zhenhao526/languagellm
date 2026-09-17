"""Single-position researcher splices on a frozen sender's outward edges.

The splice is alphabet/length legal even if its mixed packet never occurred
naturally. W1 changes can affect subsequent W2 messages; W2 splices cannot.
No policy is loaded or evaluated on import. Silent interventions are aliases.
"""
import numpy as np

from research_program.triadic_message_study import runner as core
from research_program.triadic_context_transfer_study import channel_intervention as previous

ALPHABET = tuple(core.ALPHABET)
env = previous.env


def require(ok, message):
    if not ok:
        raise ValueError(message)


def _tokens(value, shape, name):
    value = np.asarray(value)
    require(value.shape == shape and value.dtype.kind in 'iu'
            and np.all((value >= 0) & (value < 8)),
            name + ' must have the requested shape and integer symbols 0..7')
    return value


def _indices(value, n, upper, name):
    value = np.asarray(value)
    require(value.dtype.kind in 'iu' and value.shape in ((), (n,)),
            name + ' must be an integer scalar or length-B vector')
    require(np.all((value >= 0) & (value < upper)), name + ' out of range')
    return (np.full(n, int(value), dtype=np.int64) if value.ndim == 0
            else value.astype(np.int64, copy=False))


def _input(observations, natural_messages, senders, donor_packets, windows, positions):
    x = np.asarray(observations, dtype=np.float64)
    require(x.ndim == 3 and x.shape[1:] == (3, 54) and len(x) > 0
            and np.isfinite(x).all(), 'observations must be finite B×3×54')
    n = len(x)
    m = _tokens(natural_messages, (n, 2, 3, 4), 'natural_messages')
    donor = _tokens(donor_packets, (n, 2, 4), 'donor_packets')
    s = _indices(senders, n, 3, 'senders')
    w = _indices(windows, n, 2, 'windows')
    p = _indices(positions, n, 4, 'positions')
    symbols = donor[np.arange(n), w, p]
    return x, m, s, donor, w, p, symbols


def replace_outward(tokens, senders, positions, replacement_symbols, live):
    """Route one window and splice exactly one symbol to each non-sender.

    tokens[B,3,4]; scalar or B-vector senders/positions/replacement_symbols.
    Sender self-route, all other symbols, and the three visibility bits are kept.
    """
    tokens = np.asarray(tokens)
    require(tokens.ndim == 3 and len(tokens) > 0, 'Invalid window tokens')
    n = len(tokens)
    _tokens(tokens, (n, 3, 4), 'tokens')
    s = _indices(senders, n, 3, 'senders')
    p = _indices(positions, n, 4, 'positions')
    symbols = _indices(replacement_symbols, n, 8, 'replacement_symbols')
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    routes = core.routed_window(tokens, bool(live))
    if live:
        encoded = np.eye(8, dtype=np.float64)[symbols]
        for viewer in range(3):
            rows = np.flatnonzero(s != viewer)
            slots = 32*s[rows, None] + 8*p[rows, None] + np.arange(8)
            routes[rows[:, None], viewer, slots] = encoded[rows]
    return routes


def natural_action_inputs(observations, natural_messages, live):
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    return previous.natural_action_inputs(observations, natural_messages, bool(live))


def _result(x, messages, senders, windows, positions, symbols, first, second, *, live):
    n = len(x)
    patched = messages[np.arange(n), :, senders, :].copy()
    patched[np.arange(n), windows, positions] = symbols
    action_inputs = np.concatenate((x, first, second), axis=-1)
    return dict(messages=messages, patched_outward_packets=patched,
                replacement_symbols=symbols.copy(),
                senders=senders.copy(), windows=windows.copy(), positions=positions.copy(),
                first_routes=first, second_routes=second, action_inputs=action_inputs,
                routing_sha256=dict(first_routes=core.array_sha(first),
                                    second_routes=core.array_sha(second),
                                    action_inputs=core.array_sha(action_inputs)),
                outward_patch_visible=bool(live))


def silent_routes(observations, natural_messages, senders, donor_packets, windows, positions):
    """Construct actual silent routes and check exact input equality; zero NN.

    patched_outward_packets records the proposed splice, which is NOT delivered
    under silent routing. Saved natural actions/probabilities must be gathered by
    the caller; this alias return deliberately contains no invented action output.
    """
    x, m, s, donor, w, p, symbols = _input(
        observations, natural_messages, senders, donor_packets, windows, positions)
    first = replace_outward(m[:, 0], s, p, symbols, False)
    second = replace_outward(m[:, 1], s, p, symbols, False)
    out = _result(x, m.copy(), s, w, p, symbols, first, second, live=False)
    require(np.array_equal(out['action_inputs'], natural_action_inputs(x, m, False)),
            'Silent splice changed an action input')
    out.update(reused_natural=True, neural_forward_samples=0, neural_forward_calls=0,
               second_window_forward_samples=0, action_forward_samples=0,
               window0_rows=int(np.sum(w == 0)), window1_rows=int(np.sum(w == 1)))
    return out


def intervene(networks, observations, natural_messages, senders, donor_packets,
              windows, positions, live=True):
    """Greedy original policy under one outward symbol splice per world.

    Controls accept integer scalars or B-vectors. The donor supplies B×2×4 focal
    sender tokens; only donor[row, selected_window, selected_position] is used.
    W1 rows: 3 recomputed sender2 modules + 3 action modules. W2 rows: saved
    W1/W2 and only 3 action modules. A mixed batch uses 6*nW1 + 3*nW2 samples.
    Symbols/actions are argmax AFTER the original stable softmax, including ties.
    live=False routes to the zero-forward alias even when networks is None.
    """
    require(isinstance(live, (bool, np.bool_)), 'live must be boolean')
    if not live:
        return silent_routes(observations, natural_messages, senders, donor_packets, windows, positions)
    x, m, s, donor, w, p, symbols = _input(
        observations, natural_messages, senders, donor_packets, windows, positions)
    require(networks is not None and len(networks) == 9, 'Nine independent modules required')
    n = len(x)
    early, late = np.flatnonzero(w == 0), np.flatnonzero(w == 1)
    first = core.routed_window(m[:, 0], True)
    if len(early):
        first[early] = replace_outward(m[early, 0], s[early], p[early], symbols[early], True)
    messages = m.copy()
    if len(early):
        second_inputs = np.concatenate((x[early], first[early]), axis=-1)
        logits = np.stack([
            core.base.actor_forward(networks[3*a+1], second_inputs[:, a])[0].reshape(-1, 4, 8)
            for a in range(3)], axis=1)
        probabilities, _ = core.base.policy_distribution(logits)
        messages[early, 1] = core.categorical_tokens(probabilities)
    second = core.routed_window(messages[:, 1], True)
    if len(late):
        second[late] = replace_outward(messages[late, 1], s[late], p[late], symbols[late], True)
    out = _result(x, messages, s, w, p, symbols, first, second, live=True)
    logits = np.stack([
        core.base.actor_forward(networks[3*a+2], out['action_inputs'][:, a])[0]
        for a in range(3)], axis=1)
    probabilities, _ = core.base.policy_distribution(logits)
    out.update(action_probabilities=probabilities,
               action_indices=np.argmax(probabilities, axis=-1).astype(np.int16),
               reused_natural=False,
               neural_forward_samples=3*len(early)+3*n,
               second_window_forward_samples=3*len(early), action_forward_samples=3*n,
               neural_forward_calls=3+3*bool(len(early)),
               window0_rows=len(early), window1_rows=len(late))
    return out


def observations(states, information):
    """Official frozen 24-demand PL/LL encoder (54 dimensions), without model work."""
    return previous.observations(states, information)


def settle(states, actions):
    """Original task reward/physical execution, never donor-layout scoring."""
    return previous.settle(states, actions)
