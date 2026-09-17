"""Frozen design for a higher-entropy repetition-pressure curve."""
from __future__ import annotations
import hashlib
import numpy as np

HORIZON = 10
ACTION_START = 2
SUBTASKS = 8
STEPS_PER_SUBTASK = 1
GOAL_BITS = 3
GOAL_COUNT = 2 ** GOAL_BITS
TASKS = ("unique8", "repeat4", "shared8")
PARTNER_MODE = "rotating"
PARTNER_VISIBILITY = "hidden"
FORMS = ("triple2", "quad2", "atomic16")
ALPHABET_SIZES = {"triple2": 2, "quad2": 2, "atomic16": 16}
MESSAGE_LENGTHS = {"triple2": 3, "quad2": 4, "atomic16": 1}
ARRIVAL_TIMES = {form: tuple([ACTION_START] * MESSAGE_LENGTHS[form]) for form in FORMS}
CAPACITY = 2
WORKERS = 4
NULL_MESSAGE = -1
ACTION_COUNT = 3
SEEDS = tuple(range(82101, 82110))
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
EVAL_SEED_OFFSET = 994000
NOISE_STREAM_SALT = 995001
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
MAX_MESSAGE_LENGTH = max(MESSAGE_LENGTHS.values())
CONDITIONS = tuple(
    f"{task}_{form}_{adaptation}_{key}"
    for task in TASKS
    for form in FORMS
    for adaptation in ADAPTATIONS
    for key in NOISE_KEYS
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    parts = condition.split("_")
    if len(parts) == 5 and parts[2:4] == ["worker", "only"]:
        task, form, adaptation, key = parts[0], parts[1], "worker_only", parts[4]
    else:
        require(len(parts) == 4, f"bad condition {condition}")
        task, form, adaptation, key = parts
    require(task in TASKS and form in FORMS and adaptation in ADAPTATIONS and key in NOISE_KEYS, f"unknown condition {condition}")
    return task, form, adaptation, NOISE_KEYS[key], key


def alphabet_size(form):
    require(form in FORMS, "unknown form")
    return ALPHABET_SIZES[form]


def message_length(form):
    require(form in FORMS, "unknown form")
    return MESSAGE_LENGTHS[form]


def state_count(form):
    return alphabet_size(form) ** message_length(form) + 1


def goal_table():
    return np.asarray([[a, b, c] for a in (0, 1) for b in (0, 1) for c in (0, 1)], dtype=np.int8)


def goal_index(goal):
    g = np.asarray(goal, dtype=np.int64)
    return g[..., 0] * 4 + g[..., 1] * 2 + g[..., 2]


def target_bits(goal, task):
    g = np.asarray(goal, dtype=np.int8)
    require(task in TASKS, "unknown task")
    a, b, c = g[..., 0], g[..., 1], g[..., 2]
    ab = np.bitwise_xor(a, b)
    ac = np.bitwise_xor(a, c)
    bc = np.bitwise_xor(b, c)
    parity = np.bitwise_xor(ab, c)
    if task == "unique8":
        # Eight distinct balanced Boolean functions, one per stage.
        return np.stack([a, b, c, ab, ac, bc, parity, 1 - parity], axis=-1).astype(np.int8)
    if task == "repeat4":
        # Four meanings recur twice each; every meaning remains balanced.
        return np.stack([a, a, b, b, c, c, ab, ab], axis=-1).astype(np.int8)
    return np.repeat(parity[..., None], SUBTASKS, axis=-1).astype(np.int8)


def meaning_classes(task):
    require(task in TASKS, "unknown task")
    if task != "shared8":
        return [[i] for i in range(GOAL_COUNT)]
    goals = goal_table()
    parity = np.bitwise_xor(np.bitwise_xor(goals[:, 0], goals[:, 1]), goals[:, 2])
    return [np.flatnonzero(parity == value).tolist() for value in (0, 1)]


def arrival_times(form):
    require(form in FORMS, "unknown form")
    return ARRIVAL_TIMES[form]


def atomic_corruption_probability(bit_flip_p):
    p = float(bit_flip_p)
    return 1.0 - (1.0 - p) ** 4


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def episode_stream(seed, count, form, task, *, evaluation=False, update=0):
    require(count > 0, "count must be positive")
    require(form in FORMS and task in TASKS, "unknown form or task")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update)]))
    local0 = rng.integers(0, 2, size=(count, SUBTASKS), dtype=np.int8)
    site_type = np.stack([local0, 1 - local0], axis=-1).astype(np.int8)
    goal = rng.integers(0, 2, size=(count, GOAL_BITS), dtype=np.int8)
    partner_uniform = rng.random(count)
    partner_id = np.floor(partner_uniform * WORKERS).astype(np.int8)
    # Draw maximum-length streams before slicing so form arms share worlds,
    # goals, partners and action uniforms exactly.
    message_uniforms_all = rng.random((count, MAX_MESSAGE_LENGTH))
    action_uniforms = rng.random((count, HORIZON))
    noise_rng = np.random.default_rng(np.random.SeedSequence([int(seed), NOISE_STREAM_SALT if evaluation else NOISE_STREAM_SALT - 1, int(update)]))
    noise_uniforms_all = noise_rng.random((count, MAX_MESSAGE_LENGTH, 2))
    length = message_length(form)
    return {
        "site_type": site_type,
        "goal": goal,
        "target_bits": target_bits(goal, task),
        "partner_uniform": partner_uniform,
        "partner_id": partner_id,
        "message_uniforms": message_uniforms_all[:, :length],
        "action_uniforms": action_uniforms,
        "noise_uniforms": noise_uniforms_all[:, :length],
        "capacity": np.full((count, SUBTASKS, 2), CAPACITY, dtype=np.int8),
        "seed": int(seed),
        "form": form,
        "task": task,
        "evaluation": bool(evaluation),
        "update": int(update),
    }


def prepare():
    return {
        "schema": "meaning_capacity_study_v1",
        "horizon": HORIZON,
        "action_start": ACTION_START,
        "subtasks": SUBTASKS,
        "steps_per_subtask": STEPS_PER_SUBTASK,
        "goal_bits": GOAL_BITS,
        "goal_count": GOAL_COUNT,
        "tasks": list(TASKS),
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
        "pairing": "within each seed/task/adaptation, form and noise arms share worlds, goals, partners and action uniforms; message/noise streams are sliced from common maximum-length draws",
        "meaning": {
            "unique8": "eight distinct balanced Boolean functions, one per stage",
            "repeat4": "four independent balanced Boolean meanings each recur in two stages",
            "shared8": "the same hidden three-bit parity recurs in all eight stages",
        },
        "capacity_control": "quad2 and atomic16 have sixteen raw message states; triple2 has eight",
        "candidate_rule": "natural return >= 0.60, cross-class/minimum Hamming distance >= 2; shared8 also requires identical codewords within each parity class",
        "no_teacher_or_language_prior": True,
    }
