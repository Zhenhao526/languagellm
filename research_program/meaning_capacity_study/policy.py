"""Tabular sender and recurrent worker policies for higher-entropy signaling."""
from __future__ import annotations
import hashlib
import numpy as np
from . import design


def softmax(logits):
    z = np.asarray(logits, dtype=np.float64) - np.max(logits, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def sample(prob, uniform):
    return (np.asarray(uniform)[..., None] >= np.cumsum(prob, axis=-1)).sum(axis=-1).astype(np.int64)


def make_policy(seed, form):
    k = design.alphabet_size(form); length = design.message_length(form)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 712, k, length, design.GOAL_COUNT]))
    hidden = rng.normal(0.0, 0.02, size=(design.GOAL_COUNT, length, k))
    return {
        "sender_logits_hidden": hidden.copy(),
        "sender_logits_visible": np.repeat(hidden, design.WORKERS, axis=0).copy(),
        "worker_logits": rng.normal(0.0, 0.02, size=(design.WORKERS, design.state_count(form), design.HORIZON, 2, 2 * design.CAPACITY + 1, design.ACTION_COUNT)),
    }


def clone(params):
    return {key: np.asarray(value).copy() for key, value in params.items()}


def combined_parameter_hash(params):
    h = hashlib.sha256()
    for key in ("sender_logits_hidden", "sender_logits_visible", "worker_logits"):
        h.update(key.encode()); h.update(np.asarray(params[key], dtype=np.float64).tobytes())
    return h.hexdigest()
