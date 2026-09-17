"""Small tabular sender and recurrent worker policies."""
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


def make_policy(seed, role="parent"):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 417, 0 if role == "parent" else 7001]))
    return {
        "sender_logits": rng.normal(0.0, 0.02, size=(2, design.ALPHABET_SIZE)),
        "worker_logits": rng.normal(0.0, 0.02, size=(
            design.WORKERS, design.ALPHABET_SIZE + 1, design.HORIZON,
            2, design.HORIZON + 1, design.ACTION_COUNT)),
    }


def make_replacement(seed, role):
    if role not in design.ROLES:
        raise ValueError(role)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 9001, 0 if role == "worker" else 1]))
    if role == "worker":
        return rng.normal(0.0, 0.02, size=(design.ALPHABET_SIZE + 1, design.HORIZON, 2, design.HORIZON + 1, design.ACTION_COUNT))
    return rng.normal(0.0, 0.02, size=(2, design.ALPHABET_SIZE))


def clone(p):
    return {k: np.asarray(v, dtype=np.float64).copy() for k, v in p.items()}


def parameter_hash(p):
    h = hashlib.sha256()
    for key in ("sender_logits", "worker_logits"):
        h.update(key.encode()); h.update(np.asarray(p[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def sender_hash(p):
    return hashlib.sha256(np.asarray(p["sender_logits"], dtype=np.float64).tobytes()).hexdigest()


def worker_hash(p, worker):
    return hashlib.sha256(np.asarray(p["worker_logits"][int(worker)], dtype=np.float64).tobytes()).hexdigest()
