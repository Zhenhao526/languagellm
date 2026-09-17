"""Frozen design for a fixed-versus-rotating partner signaling study."""
from __future__ import annotations

import hashlib
import numpy as np

HORIZON = 5
ACTION_START = 1
ALPHABET_SIZE = 6
NULL_MESSAGE = ALPHABET_SIZE
ACTION_COUNT = 3                         # wait, take site 0, take site 1
WORKERS = 4
MEMORY = "recurrent"                     # held fixed in this population study
SCARCITIES = ("abundant", "scarce")
PARTNER_MODES = ("fixed", "rotating")
PARTNER_VISIBILITY = ("hidden", "visible")
CHANNELS = ("silent", "live")
TASK = "persistent"
SEEDS = (74101, 74102, 74103, 74104, 74105, 74106, 74107, 74108)
UPDATES = 3000
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
EVAL_SEED_OFFSET = 900000
CONDITIONS = tuple(
    f"{pm}_{pv}_{ch}_{sc}"
    for pm in PARTNER_MODES
    for pv in PARTNER_VISIBILITY
    for ch in CHANNELS
    for sc in SCARCITIES
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    parts = condition.split("_")
    require(len(parts) == 4, f"bad condition {condition}")
    pm, pv, ch, sc = parts
    require(pm in PARTNER_MODES and pv in PARTNER_VISIBILITY and ch in CHANNELS and sc in SCARCITIES,
            f"unknown condition {condition}")
    return pm, pv, ch, sc


def capacity(scarcity):
    require(scarcity in SCARCITIES, "unknown scarcity")
    return 4 if scarcity == "abundant" else 2


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def episode_stream(seed, partner_mode, partner_visibility, scarcity, count, *, evaluation=False, update=0):
    require(count > 0, "count must be positive")
    require(partner_mode in PARTNER_MODES and partner_visibility in PARTNER_VISIBILITY and scarcity in SCARCITIES,
            "bad episode factors")
    # World, target and score uniforms are invariant across all factorial
    # arms.  Only the partner assignment is interpreted differently by the
    # fixed/rotating condition.  This makes partner, visibility and capacity
    # contrasts paired rather than merely equal-sized.
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update),
    ]))
    # A worker observes site 0's local type. The second site is the opposite
    # type, so this one-bit observation identifies the whole two-site map while
    # retaining a genuine private spatial permutation.
    site0 = rng.integers(0, 2, size=count, dtype=np.int8)
    site_type = np.stack([site0, 1 - site0], axis=1).astype(np.int8)
    goal = rng.integers(0, 2, size=count, dtype=np.int8)
    partner_uniform = rng.random(count)
    if partner_mode == "fixed":
        partner_id = np.zeros(count, dtype=np.int8)
    else:
        partner_id = np.floor(partner_uniform * WORKERS).astype(np.int8)
    capacity_arr = np.full(count, capacity(scarcity), dtype=np.int8)
    return {
        "site_type": site_type,
        "goal": goal,
        "partner_uniform": partner_uniform,
        "partner_id": partner_id,
        "message_uniforms": rng.random((count, HORIZON)),
        "action_uniforms": rng.random((count, HORIZON)),
        "capacity": capacity_arr,
        "seed": int(seed), "partner_mode": partner_mode,
        "partner_visibility": partner_visibility, "scarcity": scarcity,
        "evaluation": bool(evaluation), "update": int(update),
    }


def prepare():
    return {
        "schema": "population_signaling_v1", "horizon": HORIZON, "action_start": ACTION_START,
        "alphabet_size": ALPHABET_SIZE, "null_message": NULL_MESSAGE,
        "actions": ["wait", "take_site0", "take_site1"], "workers": WORKERS,
        "seeds": list(SEEDS), "conditions": list(CONDITIONS), "updates": UPDATES,
        "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "entropy_initial": ENTROPY_INITIAL, "entropy_zero_after": ENTROPY_ZERO_AFTER,
        "capacity": {"scarce": 2, "abundant": 4},
        "task": "one scout privately observes a persistent resource type and broadcasts one token to a fixed or rotating worker",
        "worker_observation": "worker sees site 0's local type, which determines the opposite type at site 1, plus inventory; goal remains hidden",
        "partner_visibility": "hidden keeps worker identity out of the sender state; visible gives sender the partner id",
        "pairing": "worlds, target bits, partner draws and action/message uniforms are shared across partner, visibility, channel and capacity arms; only the partner assignment interpretation changes",
        "controls": ["silent channel", "closed evaluation", "within-worker token permutation"],
        "no_teacher_or_language_prior": True,
        "population_test": "a single sender must use one token convention across independent workers when partner is rotating",
        "automatic_followon_experiment": False,
    }
