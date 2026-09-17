"""Small tabular policies for the population signaling study."""
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
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 417]))
    return {
        "sender_logits": rng.normal(0.0, 0.02, size=(2, design.ALPHABET_SIZE)),
        "worker_logits": rng.normal(0.0, 0.02, size=(design.WORKERS, design.ALPHABET_SIZE + 1,
                                                       design.HORIZON, 2, design.HORIZON + 1, 3)),
    }


def clone(p):
    return {"sender_logits": p["sender_logits"].copy(), "worker_logits": p["worker_logits"].copy()}


def parameter_hash(p):
    h = hashlib.sha256()
    for key in ("sender_logits", "worker_logits"):
        h.update(key.encode())
        h.update(np.asarray(p[key], dtype=np.float64).tobytes())
    return h.hexdigest()


def zero_grads(p):
    return {"sender_logits": np.zeros_like(p["sender_logits"]),
            "worker_logits": np.zeros_like(p["worker_logits"])}


def step(p, g, learning_rate):
    norm = float(np.sqrt(sum(float((x * x).sum()) for x in g.values())))
    scale = min(1.0, 5.0 / max(norm, 1e-12))
    for key in ("sender_logits", "worker_logits"):
        p[key] -= learning_rate * scale * g[key]
    return norm, scale
