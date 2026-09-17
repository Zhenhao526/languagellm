"""Tabular sender/worker policies for multi-partner alignment."""
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


def make_policy(seed):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), design.POLICY_SALT]))
    sender_hidden = rng.normal(0.0, 0.02, size=(len(design.GOAL_PAIRS), design.MESSAGE_LENGTH, design.ALPHABET_SIZE))
    sender_visible = rng.normal(0.0, 0.02, size=(design.WORKERS, len(design.GOAL_PAIRS), design.MESSAGE_LENGTH, design.ALPHABET_SIZE))
    workers = rng.normal(
        0.0,
        0.02,
        size=(
            design.WORKERS,
            design.ALPHABET_SIZE ** design.MESSAGE_LENGTH + 1,
            design.HORIZON,
            design.SCENE_COUNT,
            design.OBJECT_TYPES * design.CAPACITY + 1,
            design.ACTION_COUNT,
        ),
    )
    return {"sender_logits_hidden": sender_hidden, "sender_logits_visible": sender_visible, "worker_logits": workers}


def clone(params):
    return {key: np.asarray(value).copy() for key, value in params.items()}


def parameter_hash(params):
    h = hashlib.sha256()
    for key in ("sender_logits_hidden", "sender_logits_visible", "worker_logits"):
        h.update(key.encode("utf8"))
        h.update(np.asarray(params[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def sender_context(goal, partner_id=None, visibility="hidden"):
    g = design.goal_index(goal).astype(np.int64)
    if visibility == "hidden":
        return g
    design.require(visibility == "visible" and partner_id is not None, "visible sender needs partner id")
    return g, np.asarray(partner_id, dtype=np.int64)


def deterministic_sequence(params, goal, visibility="hidden", partner_id=0):
    g = int(design.goal_index(np.asarray(goal, dtype=np.int8)))
    if visibility == "hidden":
        logits = params["sender_logits_hidden"][g]
    else:
        logits = params["sender_logits_visible"][int(partner_id), g]
    return np.asarray([softmax(logits[slot]).argmax() for slot in range(design.MESSAGE_LENGTH)], dtype=np.int64)
