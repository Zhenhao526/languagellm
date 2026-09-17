"""Frozen design for capacity-matched redundancy and protocol repair."""
from __future__ import annotations

import hashlib
import numpy as np

HORIZON = 6
ACTION_START = 2
SUBTASKS = 2
STEPS_PER_SUBTASK = 2
TASK = "factorized"
PARTNER_MODE = "rotating"
PARTNER_VISIBILITY = "hidden"
FORMS = ("dual2", "triple2", "atomic8")
ALPHABET_SIZES = {"dual2": 2, "triple2": 2, "atomic8": 8}
MESSAGE_LENGTHS = {"dual2": 2, "triple2": 3, "atomic8": 1}
# All message forms are delivered simultaneously immediately before the first
# action.  This removes timing as a confound in the redundancy comparison.
ARRIVAL_TIMES = {"dual2": (2, 2), "triple2": (2, 2, 2), "atomic8": (2,)}
CAPACITY = 2
WORKERS = 4
NULL_MESSAGE = -1
ACTION_COUNT = 3
SEEDS = tuple(range(78101, 78110))
NOISE_LEVELS = (0.0, 0.10, 0.25)
NOISE_KEYS = {"p00": 0.0, "p10": 0.10, "p25": 0.25}
ADAPTATIONS = ("scratch", "worker_only", "coadapt")
UPDATES = 3000
PARENT_UPDATES = 3000
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
EVAL_SEED_OFFSET = 992000
NOISE_STREAM_SALT = 993001
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
CONDITIONS = tuple(
    f"{form}_{adaptation}_{key}"
    for form in FORMS
    for adaptation in ADAPTATIONS
    for key in NOISE_KEYS
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    parts = condition.split("_")
    if len(parts) == 4 and parts[1:3] == ["worker", "only"]:
        form, adaptation, key = parts[0], "worker_only", parts[3]
    else:
        require(len(parts) == 3, f"bad condition {condition}")
        form, adaptation, key = parts
    require(form in FORMS and adaptation in ADAPTATIONS and key in NOISE_KEYS, f"unknown condition {condition}")
    return form, adaptation, NOISE_KEYS[key], key


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
    return g[..., 0] * 2 + g[..., 1]


def target_bits(goal):
    return np.asarray(goal, dtype=np.int8).copy()


def arrival_times(form):
    require(form in FORMS, "unknown form")
    return ARRIVAL_TIMES[form]


def atomic_corruption_probability(bit_flip_p):
    # Match the probability that at least one of three binary coordinates is
    # corrupted.  This makes atomic8 a raw-capacity control for triple2.
    p = float(bit_flip_p)
    return 1.0 - (1.0 - p) ** 3


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def episode_stream(seed, count, form, *, evaluation=False, update=0):
    require(count > 0, "count must be positive")
    require(form in FORMS, "unknown form")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update)]))
    local0 = rng.integers(0, 2, size=(count, SUBTASKS), dtype=np.int8)
    site_type = np.stack([local0, 1 - local0], axis=-1).astype(np.int8)
    goal = rng.integers(0, 2, size=(count, 2), dtype=np.int8)
    partner_uniform = rng.random(count)
    partner_id = np.floor(partner_uniform * WORKERS).astype(np.int8)
    length = message_length(form)
    noise_rng = np.random.default_rng(np.random.SeedSequence([int(seed), NOISE_STREAM_SALT if evaluation else NOISE_STREAM_SALT - 1, int(update)]))
    return {
        "site_type": site_type,
        "goal": goal,
        "target_bits": target_bits(goal),
        "partner_uniform": partner_uniform,
        "partner_id": partner_id,
        "message_uniforms": rng.random((count, length)),
        "action_uniforms": rng.random((count, HORIZON)),
        "noise_uniforms": noise_rng.random((count, length, 2)),
        "capacity": np.full((count, SUBTASKS, 2), CAPACITY, dtype=np.int8),
        "seed": int(seed),
        "form": form,
        "evaluation": bool(evaluation),
        "update": int(update),
    }


def prepare():
    return {
        "schema": "error_correcting_signaling_study_v1",
        "horizon": HORIZON,
        "action_start": ACTION_START,
        "subtasks": SUBTASKS,
        "steps_per_subtask": STEPS_PER_SUBTASK,
        "task": TASK,
        "partner_mode": PARTNER_MODE,
        "partner_visibility": PARTNER_VISIBILITY,
        "forms": {form: {"alphabet_size": alphabet_size(form), "length": message_length(form), "state_count": state_count(form), "arrival_times": list(arrival_times(form))} for form in FORMS},
        "adaptations": list(ADAPTATIONS),
        "noise_levels": list(NOISE_LEVELS),
        "noise_keys": NOISE_KEYS,
        "atomic_corruption_probability": {key: atomic_corruption_probability(value) for key, value in NOISE_KEYS.items()},
        "workers": WORKERS,
        "capacity": CAPACITY,
        "actions": ["wait", "take_site0", "take_site1"],
        "seeds": list(SEEDS),
        "conditions": list(CONDITIONS),
        "updates": UPDATES,
        "parent_updates": PARENT_UPDATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "checkpoints": list(CHECKPOINTS),
        "pairing": "within each seed/form/adaptation, noise arms share worlds, goals, partners, message uniforms, action uniforms and noise uniforms",
        "capacity_control": "triple2 and atomic8 have eight raw message states; dual2 has four",
        "no_teacher_or_language_prior": True,
    }
