"""Frozen design for ternary compositional signaling."""
from __future__ import annotations

import hashlib
import numpy as np

HORIZON = 6
ACTION_START = 2
SUBTASKS = 2
STEPS_PER_SUBTASK = 2
FORMS = ("mono9", "tri3")
ALPHABET_SIZES = {"mono9": 9, "tri3": 3}
MESSAGE_LENGTHS = {"mono9": 1, "tri3": 2}
MESSAGE_SLOTS = 2
TASKS = ("factorized", "entangled")
PROTOCOLS = ("simultaneous", "staged")
CHANNELS = ("silent", "live")
CAPACITY = 2
WORKERS = 4
OBJECT_TYPES = 3
ACTION_COUNT = 4  # wait plus three sites
NULL_MESSAGE = -1
SEEDS = tuple(range(77101, 77110))
UPDATES = 3000
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
EVAL_SEED_OFFSET = 991000
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
CONDITIONS = tuple(f"{form}_{task}_{protocol}_{channel}" for form in FORMS for task in TASKS for protocol in PROTOCOLS for channel in CHANNELS)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    parts = condition.split("_")
    require(len(parts) == 4, f"bad condition {condition}")
    form, task, protocol, channel = parts
    require(form in FORMS and task in TASKS and protocol in PROTOCOLS and channel in CHANNELS, f"unknown condition {condition}")
    return form, task, protocol, channel


def alphabet_size(form):
    require(form in FORMS, "unknown form")
    return ALPHABET_SIZES[form]


def message_length(form):
    require(form in FORMS, "unknown form")
    return MESSAGE_LENGTHS[form]


def state_count(form):
    return alphabet_size(form) ** message_length(form) + 1


def goal_index(goal):
    g = np.asarray(goal, dtype=np.int64)
    return g[..., 0] * OBJECT_TYPES + g[..., 1]


def target_bits(goal, task):
    g = np.asarray(goal, dtype=np.int8)
    if task == "factorized":
        return g.copy()
    require(task == "entangled", "unknown task")
    return np.stack([g[..., 0], (g[..., 0] + g[..., 1]) % OBJECT_TYPES], axis=-1).astype(np.int8)


def message_arrival_times(protocol, form):
    length = message_length(form)
    require(protocol in PROTOCOLS, "unknown protocol")
    if protocol == "simultaneous":
        return tuple(range(length))
    return (1,) if length == 1 else (1, 3)


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def episode_stream(seed, count, form, task, *, evaluation=False, update=0):
    require(count > 0, "count must be positive")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update)]))
    random_sites = rng.random((count, SUBTASKS, OBJECT_TYPES))
    order = np.argsort(random_sites, axis=-1)
    base_values = rng.integers(0, OBJECT_TYPES, size=(count, SUBTASKS), dtype=np.int8)
    # Every subtask contains each of the three object types exactly once; the
    # random permutation prevents a fixed site-position shortcut.
    site_type = (order + base_values[..., None]) % OBJECT_TYPES
    goal = rng.integers(0, OBJECT_TYPES, size=(count, 2), dtype=np.int8)
    partner_uniform = rng.random(count)
    partner_id = np.floor(partner_uniform * WORKERS).astype(np.int8)
    length = message_length(form)
    return {"site_type": site_type.astype(np.int8), "goal": goal, "target_bits": target_bits(goal, task), "partner_uniform": partner_uniform, "partner_id": partner_id, "message_uniforms": rng.random((count, length)), "action_uniforms": rng.random((count, HORIZON)), "capacity": np.full((count, SUBTASKS, OBJECT_TYPES), CAPACITY, dtype=np.int8), "seed": int(seed), "form": form, "task": task, "protocol": "", "evaluation": bool(evaluation), "update": int(update)}


def prepare():
    return {"schema": "ternary_composition_study_v1", "horizon": HORIZON, "action_start": ACTION_START, "subtasks": SUBTASKS, "steps_per_subtask": STEPS_PER_SUBTASK, "forms": {form: {"alphabet_size": alphabet_size(form), "length": message_length(form), "state_count": state_count(form)} for form in FORMS}, "tasks": list(TASKS), "protocols": {protocol: {form: list(message_arrival_times(protocol, form)) for form in FORMS} for protocol in PROTOCOLS}, "channels": list(CHANNELS), "workers": WORKERS, "object_types": OBJECT_TYPES, "actions": ["wait", "take_site0", "take_site1", "take_site2"], "capacity": CAPACITY, "seeds": list(SEEDS), "conditions": list(CONDITIONS), "updates": UPDATES, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "entropy_initial": ENTROPY_INITIAL, "entropy_zero_after": ENTROPY_ZERO_AFTER, "pairing": "forms, tasks, protocols and channel arms share worlds, goals, partners, message uniforms and action uniforms within each seed/update", "compositionality": "tri3 slot recombination selects donors by target values; mono9 has no slot recombination readout", "no_teacher_or_language_prior": True}
