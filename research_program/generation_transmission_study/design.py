"""Frozen design for a tabular cultural-transmission study.

A parent population first learns a hidden-partner signaling protocol.  One
member is then replaced by a fresh agent while the other side of the protocol
is frozen.  The role and channel controls test whether a learned convention is
recoverable by a new listener or a new sender.
"""
from __future__ import annotations
import hashlib
import numpy as np

HORIZON = 5
ACTION_START = 1
ALPHABET_SIZE = 6
NULL_MESSAGE = ALPHABET_SIZE
ACTION_COUNT = 3
WORKERS = 4
CAPACITY = 4
SEEDS = (76101, 76102, 76103, 76104, 76105, 76106, 76107, 76108)
UPDATES = 3000
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
EVAL_SEED_OFFSET = 920000
PHASE_OFFSETS = {"parent": 100000, "worker": 200000, "sender": 300000}
ROLES = ("worker", "sender")
CHANNELS = ("live", "silent", "scrambled")
CONDITIONS = tuple(f"{role}_{channel}" for role in ROLES for channel in CHANNELS)
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
TARGET_WORKER = 0


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    parts = condition.split("_")
    require(len(parts) == 2, f"bad condition {condition}")
    role, channel = parts
    require(role in ROLES and channel in CHANNELS, f"unknown condition {condition}")
    return role, channel


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def episode_stream(seed, phase, count, *, evaluation=False, update=0):
    require(phase in PHASE_OFFSETS, "unknown phase")
    require(count > 0, "count must be positive")
    # Channel arms never enter this seed sequence: all channel interventions
    # receive exactly the same world, target, partner and action/message draws.
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), PHASE_OFFSETS[phase], EVAL_SEED_OFFSET if evaluation else 0, int(update)
    ]))
    site0 = rng.integers(0, 2, size=count, dtype=np.int8)
    site_type = np.stack([site0, 1 - site0], axis=1).astype(np.int8)
    goal = rng.integers(0, 2, size=count, dtype=np.int8)
    partner_uniform = rng.random(count)
    partner_id = np.floor(partner_uniform * WORKERS).astype(np.int8)
    return {
        "site_type": site_type,
        "goal": goal,
        "partner_uniform": partner_uniform,
        "partner_id": partner_id,
        "message_uniforms": rng.random((count, HORIZON)),
        "action_uniforms": rng.random((count, HORIZON)),
        "scramble_uniforms": rng.random(count),
        "capacity": np.full(count, CAPACITY, dtype=np.int8),
        "seed": int(seed), "phase": phase, "evaluation": bool(evaluation), "update": int(update),
    }


def prepare():
    return {
        "schema": "generation_transmission_v1",
        "horizon": HORIZON, "action_start": ACTION_START,
        "alphabet_size": ALPHABET_SIZE, "null_message": NULL_MESSAGE,
        "actions": ["wait", "take_site0", "take_site1"], "workers": WORKERS,
        "capacity": CAPACITY, "target_worker": TARGET_WORKER,
        "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "updates": UPDATES, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "entropy_initial": ENTROPY_INITIAL,
        "entropy_zero_after": ENTROPY_ZERO_AFTER, "checkpoints": list(CHECKPOINTS),
        "runtime_policy": "float64 NumPy only",
        "parent_phase": "rotating hidden-partner population trained live from scratch",
        "replacement": "replace worker 0 or the sender after parent checkpoint; freeze the complementary side",
        "channels": {
            "live": "natural selected token delivered",
            "silent": "NULL delivered while outcome rewards remain",
            "scrambled": "a fresh random cyclic symbol offset is applied per episode"
        },
        "pairing": "parent and all child channel arms share phase-specific worlds, targets, partners, message/action uniforms and scramble draws",
        "no_teacher_or_language_prior": True,
        "primary_readouts": ["new-agent natural return", "live-minus-silent", "live-minus-scrambled", "learning-curve AUC", "parent-code fidelity"],
        "interpretation_boundary": "successful recovery is protocol transmission, not natural-language semantics or syntax",
    }
