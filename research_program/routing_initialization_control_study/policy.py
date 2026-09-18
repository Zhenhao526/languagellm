"""Holistic, fixed-factor and learnably routed tabular policies.

The routed arm keeps the atomic factor tables but does not tell the policy
which slot carries which attribute. The ``routed`` arm learns independent
sender and receiver routing matrices; ``tied_routed`` shares one matrix
between the two roles.
"""
from __future__ import annotations

import hashlib
import numpy as np
from . import design


def softmax(logits, axis=-1):
    z = np.asarray(logits, dtype=np.float64)
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sample(prob, uniform):
    return (np.asarray(uniform)[..., None] >= np.cumsum(prob, axis=-1)).sum(axis=-1).astype(np.int64)


def _normal(rng, shape):
    return rng.normal(0.0, 0.02, size=shape).astype(np.float64)


def make_policy(seed, architecture, *, visible=False):
    design.require(architecture in design.ARCHITECTURES, "bad architecture")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), design.POLICY_SALT,
                                int(architecture == "factorized"), int(architecture in ("routed", "sync_routed")),
                                int(architecture == "tied_routed")])
    )
    if architecture == "holistic":
        sh = _normal(rng, (design.MEANINGS, design.MESSAGE_LENGTH, design.ALPHABET_SIZE))
        rh = _normal(rng, (design.MESSAGE_STATE_COUNT, design.MEANINGS))
        if visible:
            sv = np.repeat(sh[None, ...], design.PARTNERS, axis=0).copy()
            rv = np.repeat(rh[None, ...], design.PARTNERS, axis=0).copy()
        else:
            sv = np.empty((0, design.MEANINGS, design.MESSAGE_LENGTH, design.ALPHABET_SIZE))
            rv = np.empty((0, design.MESSAGE_STATE_COUNT, design.MEANINGS))
        return {"architecture": architecture, "sender_hidden": sh, "sender_visible": sv,
                "receiver_hidden": rh, "receiver_visible": rv}
    if architecture == "factorized":
        sh = _normal(rng, (design.MESSAGE_LENGTH, design.VALUES, design.ALPHABET_SIZE))
        rh = _normal(rng, (design.MESSAGE_LENGTH, design.ALPHABET_SIZE, design.VALUES))
        if visible:
            sv = np.repeat(sh[None, ...], design.PARTNERS, axis=0).copy()
            rv = np.repeat(rh[None, ...], design.PARTNERS, axis=0).copy()
        else:
            sv = np.empty((0, design.MESSAGE_LENGTH, design.VALUES, design.ALPHABET_SIZE))
            rv = np.empty((0, design.MESSAGE_LENGTH, design.ALPHABET_SIZE, design.VALUES))
        return {"architecture": architecture, "sender_hidden": sh, "sender_visible": sv,
                "receiver_hidden": rh, "receiver_visible": rv}
    # Routed sender factors are [attribute, value, token]. Each slot mixes
    # attribute-specific logits with a learned row-softmax.
    sf = _normal(rng, (design.ATTRIBUTES, design.VALUES, design.ALPHABET_SIZE))
    sr = rng.normal(0.0, design.ROUTE_INIT_SCALE,
                    size=(design.MESSAGE_LENGTH, design.ATTRIBUTES)).astype(np.float64)
    rf = _normal(rng, (design.MESSAGE_LENGTH, design.ALPHABET_SIZE, design.ATTRIBUTES, design.VALUES))
    rr = rng.normal(0.0, design.ROUTE_INIT_SCALE,
                    size=(design.MESSAGE_LENGTH, design.ATTRIBUTES)).astype(np.float64)
    if architecture == "sync_routed":
        rr = sr.copy()
    if visible:
        sv = np.repeat(sf[None, ...], design.PARTNERS, axis=0).copy()
        rv = np.repeat(rf[None, ...], design.PARTNERS, axis=0).copy()
        if architecture == "tied_routed":
            routev = np.repeat(sr[None, ...], design.PARTNERS, axis=0).copy()
        else:
            svr = np.repeat(sr[None, ...], design.PARTNERS, axis=0).copy()
            rvr = np.repeat(rr[None, ...], design.PARTNERS, axis=0).copy()
    else:
        sv = np.empty((0, design.ATTRIBUTES, design.VALUES, design.ALPHABET_SIZE))
        rv = np.empty((0, design.MESSAGE_LENGTH, design.ALPHABET_SIZE, design.ATTRIBUTES, design.VALUES))
        if architecture == "tied_routed":
            routev = np.empty((0, design.MESSAGE_LENGTH, design.ATTRIBUTES))
        else:
            svr = np.empty((0, design.MESSAGE_LENGTH, design.ATTRIBUTES))
            rvr = np.empty((0, design.MESSAGE_LENGTH, design.ATTRIBUTES))
    if architecture == "tied_routed":
        return {"architecture": architecture, "sender_hidden": sf, "route_hidden": sr,
                "sender_visible": sv, "route_visible": routev, "receiver_hidden": rf,
                "receiver_visible": rv}
    return {"architecture": architecture, "sender_hidden": sf, "sender_route_hidden": sr,
            "sender_visible": sv, "sender_route_visible": svr, "receiver_hidden": rf,
            "receiver_route_hidden": rr, "receiver_visible": rv, "receiver_route_visible": rvr}


def clone(params):
    return {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in params.items()}


def parameter_hash(params):
    h = hashlib.sha256()
    h.update(str(params["architecture"]).encode("utf8"))
    for key in sorted(key for key, value in params.items() if isinstance(value, np.ndarray)):
        h.update(key.encode("utf8"))
        h.update(np.asarray(params[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def empty_grad(params):
    return {key: np.zeros_like(value) for key, value in params.items() if isinstance(value, np.ndarray)}


def _routed_sender_logits(factors, routes, attrs):
    route = softmax(design.ROUTE_TEMPERATURE * routes, axis=-1)
    return np.stack(
        [sum(route[slot, attr] * factors[attr, attrs[:, attr], :] for attr in range(design.ATTRIBUTES))
         for slot in range(design.MESSAGE_LENGTH)], axis=1
    )


def sender_arrays(params, meanings, partner_id, visibility):
    architecture = params["architecture"]
    meanings = np.asarray(meanings, dtype=np.int64)
    partner_id = np.asarray(partner_id, dtype=np.int64)
    attrs = design.attrs_from_meaning(meanings)
    probs = np.zeros((len(meanings), design.MESSAGE_LENGTH, design.ALPHABET_SIZE), dtype=np.float64)
    context = meanings.copy()
    for partner in range(design.PARTNERS):
        idx = np.flatnonzero(partner_id == partner)
        if not len(idx):
            continue
        if architecture == "holistic":
            if visibility == "hidden":
                logits = params["sender_hidden"][meanings[idx]]
            else:
                context[idx] = partner * design.MEANINGS + meanings[idx]
                logits = params["sender_visible"][partner, meanings[idx]]
        elif architecture == "factorized":
            if visibility == "hidden":
                logits = np.stack([params["sender_hidden"][slot, attrs[idx, slot], :]
                                   for slot in range(design.MESSAGE_LENGTH)], axis=1)
            else:
                context[idx] = partner * design.MEANINGS + meanings[idx]
                logits = np.stack([params["sender_visible"][partner, slot, attrs[idx, slot], :]
                                   for slot in range(design.MESSAGE_LENGTH)], axis=1)
        else:
            if visibility == "hidden":
                factors = params["sender_hidden"]
                routes = params["route_hidden"] if architecture == "tied_routed" else params["sender_route_hidden"]
            else:
                context[idx] = partner * design.MEANINGS + meanings[idx]
                factors = params["sender_visible"][partner]
                routes = (params["route_visible"][partner] if architecture == "tied_routed"
                          else params["sender_route_visible"][partner])
            logits = _routed_sender_logits(factors, routes, attrs[idx])
        probs[idx] = softmax(logits, axis=-1)
    return probs, context


def receiver_arrays(params, message_state, scene_meanings, partner_id, visibility):
    architecture = params["architecture"]
    state = np.asarray(message_state, dtype=np.int64)
    scene = np.asarray(scene_meanings, dtype=np.int64)
    partner_id = np.asarray(partner_id, dtype=np.int64)
    scene_attrs = design.attrs_from_meaning(scene)
    tokens = design.decode_message_state(state)
    scores = np.zeros((len(state), design.SCENE_SIZE), dtype=np.float64)
    context = state.copy()
    for partner in range(design.PARTNERS):
        idx = np.flatnonzero(partner_id == partner)
        if not len(idx):
            continue
        if architecture == "holistic":
            if visibility == "hidden":
                logits = params["receiver_hidden"][state[idx]]
            else:
                context[idx] = partner * design.MESSAGE_STATE_COUNT + state[idx]
                logits = params["receiver_visible"][partner, state[idx]]
            scores[idx] = logits[np.arange(len(idx))[:, None], scene[idx]]
            continue
        if architecture == "factorized":
            if visibility == "hidden":
                table = params["receiver_hidden"]
            else:
                context[idx] = partner * design.MESSAGE_STATE_COUNT + state[idx]
                table = params["receiver_visible"][partner]
            for row in idx:
                for slot in range(design.MESSAGE_LENGTH):
                    token = tokens[row, slot]
                    if token >= 0:
                        scores[row] += table[slot, token, scene_attrs[row, :, slot]]
            continue
        if visibility == "hidden":
            table = params["receiver_hidden"]
            route_raw = params["route_hidden"] if architecture == "tied_routed" else params["receiver_route_hidden"]
            routes = softmax(design.ROUTE_TEMPERATURE * route_raw, axis=-1)
        else:
            context[idx] = partner * design.MESSAGE_STATE_COUNT + state[idx]
            table = params["receiver_visible"][partner]
            route_raw = (params["route_visible"][partner] if architecture == "tied_routed"
                         else params["receiver_route_visible"][partner])
            routes = softmax(design.ROUTE_TEMPERATURE * route_raw, axis=-1)
        for slot in range(design.MESSAGE_LENGTH):
            token = tokens[idx, slot]
            valid = token >= 0
            token_safe = np.maximum(token, 0)
            for attr in range(design.ATTRIBUTES):
                values = scene_attrs[idx, :, attr]
                contribution = table[slot, token_safe[:, None], attr, values]
                contribution[~valid, :] = 0.0
                scores[idx] += routes[slot, attr] * contribution
    return softmax(scores, axis=-1), context


def sender_sequence(params, meaning, *, partner_id=0, visibility="hidden", external=False, permutation=None):
    probs, _ = sender_arrays(params, np.asarray([meaning]), np.asarray([partner_id]), visibility)
    sequence = probs[0].argmax(axis=-1).astype(np.int64)
    if external and permutation is not None:
        sequence = np.asarray(permutation, dtype=np.int64)[sequence]
    return sequence


def routing_probabilities(params, *, side="sender", partner_id=0, visibility="hidden"):
    if params["architecture"] not in ("routed", "sync_routed", "tied_routed"):
        return None
    if side == "sender":
        key = ("route_hidden" if params["architecture"] == "tied_routed" else "sender_route_hidden") if visibility == "hidden" else ("route_visible" if params["architecture"] == "tied_routed" else "sender_route_visible")
    elif side == "receiver":
        key = ("route_hidden" if params["architecture"] == "tied_routed" else "receiver_route_hidden") if visibility == "hidden" else ("route_visible" if params["architecture"] == "tied_routed" else "receiver_route_visible")
    else:
        raise ValueError(side)
    raw = params[key] if visibility == "hidden" else params[key][int(partner_id)]
    return softmax(design.ROUTE_TEMPERATURE * raw, axis=-1)
