"""Holistic and factor-shared policies for the structured referential test."""
from __future__ import annotations

import hashlib
import numpy as np

from . import design


def softmax(logits, axis=-1):
    z = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def sample(prob, uniform):
    return (np.asarray(uniform)[..., None] >= np.cumsum(prob, axis=-1)).sum(axis=-1).astype(np.int64)


def make_policy(seed, architecture, *, visible=False):
    design.require(architecture in design.ARCHITECTURES, "bad architecture")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), design.POLICY_SALT, int(architecture == "factorized")]))
    if architecture == "holistic":
        sh = rng.normal(0.0, 0.02, size=(design.MEANINGS, design.MESSAGE_LENGTH, design.ALPHABET_SIZE))
        rh = rng.normal(0.0, 0.02, size=(design.MESSAGE_STATE_COUNT, design.MEANINGS))
        if visible:
            sv = np.repeat(sh[None, ...], design.PARTNERS, axis=0).copy()
            rv = np.repeat(rh[None, ...], design.PARTNERS, axis=0).copy()
        else:
            sv = np.empty((0, design.MEANINGS, design.MESSAGE_LENGTH, design.ALPHABET_SIZE), dtype=np.float64)
            rv = np.empty((0, design.MESSAGE_STATE_COUNT, design.MEANINGS), dtype=np.float64)
        return {"architecture": architecture, "sender_hidden": sh, "sender_visible": sv, "receiver_hidden": rh, "receiver_visible": rv}
    # The structural arm shares parameters across complete meanings. Slot j
    # only receives the j-th attribute value; the receiver sums slot-wise
    # evidence for the two attributes. Token identities remain unlabelled.
    sh = rng.normal(0.0, 0.02, size=(design.MESSAGE_LENGTH, design.VALUES, design.ALPHABET_SIZE))
    rh = rng.normal(0.0, 0.02, size=(design.MESSAGE_LENGTH, design.ALPHABET_SIZE, design.VALUES))
    if visible:
        sv = np.repeat(sh[None, ...], design.PARTNERS, axis=0).copy()
        rv = np.repeat(rh[None, ...], design.PARTNERS, axis=0).copy()
    else:
        sv = np.empty((0, design.MESSAGE_LENGTH, design.VALUES, design.ALPHABET_SIZE), dtype=np.float64)
        rv = np.empty((0, design.MESSAGE_LENGTH, design.ALPHABET_SIZE, design.VALUES), dtype=np.float64)
    return {"architecture": architecture, "sender_hidden": sh, "sender_visible": sv, "receiver_hidden": rh, "receiver_visible": rv}


def clone(params):
    return {key: (value.copy() if isinstance(value, np.ndarray) else value) for key, value in params.items()}


def parameter_hash(params):
    h = hashlib.sha256()
    h.update(str(params["architecture"]).encode("utf8"))
    for key in ("sender_hidden", "sender_visible", "receiver_hidden", "receiver_visible"):
        h.update(key.encode("utf8")); h.update(np.asarray(params[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def empty_grad(params):
    return {key: np.zeros_like(value) for key, value in params.items() if isinstance(value, np.ndarray)}


def sender_arrays(params, meanings, partner_id, visibility):
    architecture = params["architecture"]
    meanings = np.asarray(meanings, dtype=np.int64)
    partner_id = np.asarray(partner_id, dtype=np.int64)
    n = len(meanings)
    probs = np.zeros((n, design.MESSAGE_LENGTH, design.ALPHABET_SIZE), dtype=np.float64)
    context = meanings.copy()
    attrs = design.attrs_from_meaning(meanings)
    for partner in range(design.PARTNERS):
        idx = np.flatnonzero(partner_id == partner)
        if not len(idx): continue
        if visibility == "hidden":
            if architecture == "holistic":
                logits = params["sender_hidden"][meanings[idx]]
            else:
                logits = np.stack([params["sender_hidden"][slot, attrs[idx, slot], :] for slot in range(design.MESSAGE_LENGTH)], axis=1)
        else:
            context[idx] = partner * design.MEANINGS + meanings[idx]
            if architecture == "holistic":
                logits = params["sender_visible"][partner, meanings[idx]]
            else:
                logits = np.stack([params["sender_visible"][partner, slot, attrs[idx, slot], :] for slot in range(design.MESSAGE_LENGTH)], axis=1)
        probs[idx] = softmax(logits, axis=-1)
    return probs, context


def receiver_arrays(params, message_state, scene_meanings, partner_id, visibility):
    architecture = params["architecture"]
    state = np.asarray(message_state, dtype=np.int64)
    scene = np.asarray(scene_meanings, dtype=np.int64)
    partner_id = np.asarray(partner_id, dtype=np.int64)
    n = len(state); scores = np.zeros((n, design.SCENE_SIZE), dtype=np.float64); context = state.copy()
    scene_attrs = design.attrs_from_meaning(scene)
    tokens = design.decode_message_state(state)
    for partner in range(design.PARTNERS):
        idx = np.flatnonzero(partner_id == partner)
        if not len(idx): continue
        if architecture == "holistic":
            if visibility == "hidden": logits = params["receiver_hidden"][state[idx]]
            else:
                context[idx] = partner * design.MESSAGE_STATE_COUNT + state[idx]
                logits = params["receiver_visible"][partner, state[idx]]
            scores[idx] = logits[np.arange(len(idx))[:, None], scene[idx]]
        else:
            if visibility == "hidden": table = params["receiver_hidden"]
            else:
                context[idx] = partner * design.MESSAGE_STATE_COUNT + state[idx]
                table = params["receiver_visible"][partner]
            # The explicit loop avoids advanced-index ambiguity and is
            # still cheap at the 512-example batch size.
            scores[idx] = 0.0
            for j, row_idx in enumerate(idx):
                if tokens[row_idx, 0] >= 0:
                    scores[row_idx] += table[0, tokens[row_idx, 0], scene_attrs[row_idx, :, 0]]
                if tokens[row_idx, 1] >= 0:
                    scores[row_idx] += table[1, tokens[row_idx, 1], scene_attrs[row_idx, :, 1]]
    return softmax(scores, axis=-1), context


def sender_sequence(params, meaning, *, partner_id=0, visibility="hidden", external=False, permutation=None):
    probs, _ = sender_arrays(params, np.asarray([meaning]), np.asarray([partner_id]), visibility)
    seq = probs[0].argmax(axis=-1).astype(np.int64)
    if external and permutation is not None: seq = np.asarray(permutation, dtype=np.int64)[seq]
    return seq
