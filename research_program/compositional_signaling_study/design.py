"""Frozen design for a two-factor compositional signaling study."""
from __future__ import annotations
import hashlib
import numpy as np

HORIZON = 6
MESSAGE_START = 0
MESSAGE_SLOTS = 2
ACTION_START = 2
SUBTASKS = 2
STEPS_PER_SUBTASK = 2
ALPHABET_SIZES = {"mono4": 4, "dual2": 2}
MESSAGE_LENGTHS = {"mono4": 1, "dual2": 2}
FORMS = ("mono4", "dual2")
TASKS = ("factorized", "entangled")
PARTNER_MODES = ("fixed", "rotating")
PARTNER_VISIBILITY = ("hidden", "visible")
CHANNELS = ("silent", "live")
CAPACITY = 2
WORKERS = 4
NULL_MESSAGE = -1
ACTION_COUNT = 3
SEEDS = (75101, 75102, 75103, 75104, 75105, 75106, 75107, 75108)
UPDATES = 3000
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
EVAL_SEED_OFFSET = 910000
CONDITIONS = (
    "mono4_rotating_hidden_live_factorized",
    "mono4_rotating_hidden_silent_factorized",
    "dual2_rotating_hidden_live_factorized",
    "dual2_rotating_hidden_silent_factorized",
    "dual2_rotating_hidden_live_entangled",
    "dual2_rotating_hidden_silent_entangled",
    "dual2_rotating_visible_live_factorized",
    "dual2_rotating_visible_silent_factorized",
    "dual2_fixed_hidden_live_factorized",
    "dual2_fixed_hidden_silent_factorized",
)
CHECKPOINTS = (0, 500, 1000, 2000, 3000)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    parts = condition.split("_")
    require(len(parts) == 5, f"bad condition {condition}")
    form, pm, pv, ch, task = parts
    require(form in FORMS and pm in PARTNER_MODES and pv in PARTNER_VISIBILITY and ch in CHANNELS and task in TASKS, f"unknown condition {condition}")
    return form, pm, pv, ch, task


def alphabet_size(form):
    require(form in FORMS, "unknown form"); return ALPHABET_SIZES[form]


def message_length(form):
    require(form in FORMS, "unknown form"); return MESSAGE_LENGTHS[form]


def goal_index(goal):
    g = np.asarray(goal, dtype=np.int64); return g[..., 0] * 2 + g[..., 1]


def target_bits(goal, task):
    g = np.asarray(goal, dtype=np.int8)
    if task == "factorized": return g.copy()
    require(task == "entangled", "unknown task")
    return np.stack([g[..., 0], np.bitwise_xor(g[..., 0], g[..., 1])], axis=-1).astype(np.int8)


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x): return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def episode_stream(seed, partner_mode, partner_visibility, task, count, *, evaluation=False, update=0):
    require(count > 0, "count must be positive")
    require(partner_mode in PARTNER_MODES and partner_visibility in PARTNER_VISIBILITY and task in TASKS, "bad episode factors")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update)]))
    local0 = rng.integers(0, 2, size=(count, SUBTASKS), dtype=np.int8)
    site_type = np.stack([local0, 1 - local0], axis=-1).astype(np.int8)
    goal = rng.integers(0, 2, size=(count, 2), dtype=np.int8)
    partner_uniform = rng.random(count)
    partner_id = np.zeros(count, dtype=np.int8) if partner_mode == "fixed" else np.floor(partner_uniform * WORKERS).astype(np.int8)
    return {"site_type": site_type, "goal": goal, "target_bits": target_bits(goal, task),
            "partner_uniform": partner_uniform, "partner_id": partner_id,
            "message_uniforms": rng.random((count, MESSAGE_SLOTS)), "action_uniforms": rng.random((count, HORIZON)),
            "capacity": np.full((count, SUBTASKS, 2), CAPACITY, dtype=np.int8), "seed": int(seed),
            "partner_mode": partner_mode, "partner_visibility": partner_visibility, "task": task,
            "evaluation": bool(evaluation), "update": int(update)}


def prepare():
    return {"schema": "compositional_signaling_v1", "horizon": HORIZON, "message_slots": MESSAGE_SLOTS,
            "action_start": ACTION_START, "subtasks": SUBTASKS, "steps_per_subtask": STEPS_PER_SUBTASK,
            "forms": {k: {"alphabet_size": ALPHABET_SIZES[k], "length": MESSAGE_LENGTHS[k]} for k in FORMS},
            "null_message": NULL_MESSAGE, "actions": ["wait", "take_site0", "take_site1"], "workers": WORKERS,
            "seeds": list(SEEDS), "conditions": list(CONDITIONS), "updates": UPDATES, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "entropy_initial": ENTROPY_INITIAL, "entropy_zero_after": ENTROPY_ZERO_AFTER,
            "capacity": CAPACITY, "task": "two binary private factors drive two resource sub-tasks",
            "worker_observation": "worker sees each sub-task local site-0 type; site 1 is its complement; goal is hidden",
            "partner_visibility": "hidden removes partner id; visible permits partner-specific sequences",
            "pairing": "worlds, goals, partner draws and action/message uniforms are shared across form, task, partner, visibility and channel arms",
            "controls": ["from-scratch silent", "closed evaluation", "whole-sequence permutation", "slot recombination readout"],
            "no_teacher_or_language_prior": True,
            "compositionality_test": "dual2 token slots are recombined from factor-matched sender contexts on held-out worlds"}
