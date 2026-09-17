"""Tabular sender and recurrent worker policies for compositional signaling."""
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


def state_count(form):
    return design.alphabet_size(form) ** design.message_length(form) + 1


def make_policy(seed, form):
    k = design.alphabet_size(form); length = design.message_length(form)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 521, k, length]))
    hidden = rng.normal(0.0, 0.02, size=(4, length, k))
    return {
        "sender_logits_hidden": hidden.copy(),
        "sender_logits_visible": np.repeat(hidden, design.WORKERS, axis=0).copy(),
        "worker_logits": rng.normal(0.0, 0.02, size=(
            design.WORKERS, state_count(form), design.HORIZON, 2,
            2 * design.CAPACITY + 1, design.ACTION_COUNT)),
    }


def clone(p):
    return {k: np.asarray(v).copy() for k, v in p.items()}


def combined_parameter_hash(p):
    h = hashlib.sha256()
    for key in ("sender_logits_hidden", "sender_logits_visible", "worker_logits"):
        h.update(key.encode()); h.update(np.asarray(p[key], dtype=np.float64).tobytes())
    return h.hexdigest()
