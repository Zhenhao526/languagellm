"""Small tabular policies for the role-symmetric referential game."""
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


def make_policy(seed, *, visible=False):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), design.POLICY_SALT]))
    hidden_sender = rng.normal(0.0, 0.02, size=(design.MEANINGS, design.MESSAGE_LENGTH, design.ALPHABET_SIZE))
    hidden_receiver = rng.normal(0.0, 0.02, size=(design.MESSAGE_STATE_COUNT, design.MEANINGS))
    if visible:
        visible_sender = np.repeat(hidden_sender[None, ...], design.PARTNERS, axis=0).copy()
        visible_receiver = np.repeat(hidden_receiver[None, ...], design.PARTNERS, axis=0).copy()
    else:
        visible_sender = np.empty((0, design.MEANINGS, design.MESSAGE_LENGTH, design.ALPHABET_SIZE), dtype=np.float64)
        visible_receiver = np.empty((0, design.MESSAGE_STATE_COUNT, design.MEANINGS), dtype=np.float64)
    return {
        "sender_hidden": hidden_sender,
        "sender_visible": visible_sender,
        "receiver_hidden": hidden_receiver,
        "receiver_visible": visible_receiver,
    }


def clone(params):
    return {key: np.asarray(value).copy() for key, value in params.items()}


def parameter_hash(params):
    h = hashlib.sha256()
    for key in ("sender_hidden", "sender_visible", "receiver_hidden", "receiver_visible"):
        h.update(key.encode("utf8"))
        h.update(np.asarray(params[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def empty_grad(params):
    return {key: np.zeros_like(value) for key, value in params.items()}


def sender_sequence(params, meaning, *, partner_id=0, visibility="hidden", external=False, permutation=None):
    meaning = int(meaning)
    if visibility == "hidden":
        logits = params["sender_hidden"][meaning]
    else:
        logits = params["sender_visible"][int(partner_id), meaning]
    seq = np.asarray([softmax(logits[slot]).argmax() for slot in range(design.MESSAGE_LENGTH)], dtype=np.int64)
    if external and permutation is not None:
        seq = np.asarray(permutation, dtype=np.int64)[seq]
    return seq


def receiver_scores(params, message, *, partner_id=0, visibility="hidden"):
    state = int(design.message_index(message))
    if visibility == "hidden":
        return np.asarray(params["receiver_hidden"][state], dtype=np.float64)
    return np.asarray(params["receiver_visible"][int(partner_id), state], dtype=np.float64)
