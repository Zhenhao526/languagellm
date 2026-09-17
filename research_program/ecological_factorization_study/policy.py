"""Tabular sender and receiver policies for ecological factorization."""
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


def make_policy(seed, form, representation="joint_history", *, sender_logits=None):
    design.require(form in design.FORMS and representation in design.REPRESENTATIONS, "bad policy factors")
    k = design.alphabet_size(form); length = design.message_length(form)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 881, k, length, int(representation == "slot_local")]))
    sender = rng.normal(0.0, 0.02, size=(design.OBJECT_TYPES ** 2, length, k)) if sender_logits is None else np.asarray(sender_logits, dtype=np.float64).copy()
    workers = rng.normal(0.0, 0.02, size=(design.WORKERS, design.state_count(form, representation), design.HORIZON, design.SCENE_COUNT, design.OBJECT_TYPES * design.CAPACITY + 1, design.ACTION_COUNT))
    return {"sender_logits": sender, "worker_logits": workers}


def clone(params):
    return {key: np.asarray(value).copy() for key, value in params.items()}


def parameter_hash(params):
    h = hashlib.sha256()
    for key in ("sender_logits", "worker_logits"):
        h.update(key.encode("utf8")); h.update(np.asarray(params[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def sender_context(goal):
    return design.goal_index(goal).astype(np.int64)
