"""Frozen design for a ternary ecological factorization study."""
from __future__ import annotations

import hashlib
import numpy as np

HORIZON = 6
ACTION_START = 2
SUBTASKS = 2
STEPS_PER_SUBTASK = 2
OBJECT_TYPES = 3
SCENE_COUNT = OBJECT_TYPES ** OBJECT_TYPES
FORMS = ("mono9", "tri3")
ALPHABET_SIZES = {"mono9": 9, "tri3": 3}
MESSAGE_LENGTHS = {"mono9": 1, "tri3": 2}
ARRIVAL_TIMES = {"mono9": (1,), "tri3": (1, 3)}
TASKS = ("factorized", "holistic")
REPRESENTATIONS = ("joint_history", "slot_local")
SUPPORTS = ("full", "leave_one_out")
CHANNELS = ("live", "silent")
PARTNER_MODE = "rotating"
PARTNER_VISIBILITY = "hidden"
WORKERS = 4
CAPACITY = 2
NULL_MESSAGE = -1
ACTION_COUNT = 4
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
SEEDS = tuple(range(78101, 78110))
UPDATES = 3000
PARENT_UPDATES = UPDATES
CHILD_UPDATES = UPDATES
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
EVAL_SEED_OFFSET = 981000
HOLISTIC_SALT = 981001
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
TARGET_WORKER = 0

GOAL_PAIRS = tuple((a, b) for a in range(OBJECT_TYPES) for b in range(OBJECT_TYPES))
# A fixed bijection preserves the uniform marginal target distribution while
# destroying the alignment between private factor labels and stage targets.
HOLISTIC_TARGET_MAP = (4, 0, 8, 2, 7, 1, 6, 3, 5)

PARENT_CONDITIONS = tuple(f"{form}_{task}" for form in FORMS for task in TASKS)
CHILD_CONDITIONS = tuple(
    f"{representation}_{form}_{task}_{support}_{channel}"
    for representation in REPRESENTATIONS
    for form in FORMS
    for task in TASKS
    for support in SUPPORTS
    for channel in CHANNELS
    if not (representation == "slot_local" and form == "mono9")
)
CONDITIONS = CHILD_CONDITIONS


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_parent_condition(condition):
    parts = condition.split("_")
    require(len(parts) == 2 and parts[0] in FORMS and parts[1] in TASKS, f"bad parent condition {condition}")
    return parts[0], parts[1]


def parse_child_condition(condition):
    representation = next((value for value in REPRESENTATIONS if condition.startswith(value + "_")), None)
    require(representation is not None, f"bad child condition {condition}")
    parts = condition[len(representation) + 1 :].split("_")
    if len(parts) == 4:
        form, task, support, channel = parts
    elif len(parts) == 6 and parts[2:5] == ["leave", "one", "out"]:
        form, task, leave, one, out, channel = parts
        support = leave + "_" + one + "_" + out
    else:
        raise ValueError(f"bad child condition {condition}")
    require(representation in REPRESENTATIONS and form in FORMS and task in TASKS and support in SUPPORTS and channel in CHANNELS, f"bad child condition {condition}")
    require(not (representation == "slot_local" and form == "mono9"), "slot_local mono9 is undefined")
    return representation, form, task, support, channel


def alphabet_size(form):
    require(form in FORMS, "unknown form")
    return ALPHABET_SIZES[form]


def message_length(form):
    require(form in FORMS, "unknown form")
    return MESSAGE_LENGTHS[form]


def state_count(form, representation="joint_history"):
    require(form in FORMS and representation in REPRESENTATIONS, "unknown form or representation")
    if representation == "slot_local":
        require(form == "tri3", "slot-local representation is only defined for tri3")
        return alphabet_size(form) + 1
    return alphabet_size(form) ** message_length(form) + 1


def goal_index(goal):
    g = np.asarray(goal, dtype=np.int64)
    return g[..., 0] * OBJECT_TYPES + g[..., 1]


def target_bits(goal, task):
    g = np.asarray(goal, dtype=np.int8)
    require(task in TASKS, "unknown task")
    if task == "factorized":
        return g.copy()
    idx = goal_index(g)
    mapped = np.asarray(HOLISTIC_TARGET_MAP, dtype=np.int8)[idx]
    return np.asarray(GOAL_PAIRS, dtype=np.int8)[mapped]


def heldout_goal(seed):
    require(int(seed) in SEEDS, "unknown seed")
    return int((int(seed) - SEEDS[0]) % len(GOAL_PAIRS))


def message_arrival_times(form):
    require(form in FORMS, "unknown form")
    return ARRIVAL_TIMES[form]


def entropy_coefficient(update, *, child=False):
    require(1 <= int(update) <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (int(update) - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def _masked_goals(raw, heldout):
    """Map the held-out index to a seen index without changing array shape."""
    idx = goal_index(raw).astype(np.int64)
    replacement = (int(heldout) + 1) % len(GOAL_PAIRS)
    idx = np.where(idx == int(heldout), replacement, idx)
    return np.asarray(GOAL_PAIRS, dtype=np.int8)[idx]


def episode_stream(seed, count, *, task, support="full", heldout=None, evaluation=False, update=0):
    require(count > 0 and task in TASKS and support in SUPPORTS, "bad episode factors")
    require(int(seed) in SEEDS or evaluation, "unknown seed")
    if heldout is None:
        heldout = heldout_goal(seed) if int(seed) in SEEDS else 0
    require(0 <= int(heldout) < len(GOAL_PAIRS), "bad heldout goal")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update), HOLISTIC_SALT]))
    # Each stage contains one copy of each object type in a random order. The
    # random permutation blocks a fixed site-position shortcut.
    site_type = np.empty((count, SUBTASKS, OBJECT_TYPES), dtype=np.int8)
    for i in range(count):
        for stage in range(SUBTASKS):
            site_type[i, stage] = rng.permutation(OBJECT_TYPES)
    raw_goal = rng.integers(0, OBJECT_TYPES, size=(count, 2), dtype=np.int8)
    goal = _masked_goals(raw_goal, heldout) if support == "leave_one_out" else raw_goal
    partner_uniform = rng.random(count)
    partner_id = np.floor(partner_uniform * WORKERS).astype(np.int8)
    length = message_length("tri3")
    # Draw the maximum length for paired mono9/tri3 arms. Each form slices its
    # own stream deterministically in the runner.
    message_uniforms = rng.random((count, length))
    action_uniforms = rng.random((count, HORIZON))
    return {
        "site_type": site_type,
        "goal": goal,
        "target_bits": target_bits(goal, task),
        "partner_uniform": partner_uniform,
        "partner_id": partner_id,
        "message_uniforms": message_uniforms,
        "action_uniforms": action_uniforms,
        "capacity": np.full((count, SUBTASKS, OBJECT_TYPES), CAPACITY, dtype=np.int8),
        "seed": int(seed), "task": task, "support": support,
        "heldout_goal": int(heldout), "evaluation": bool(evaluation), "update": int(update),
    }


def balanced_eval_stream(seed, count, goal_kind, task, heldout):
    require(goal_kind in ("all", "seen", "heldout"), "bad goal kind")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET + 17, int(heldout), HOLISTIC_SALT]))
    site_type = np.empty((count, SUBTASKS, OBJECT_TYPES), dtype=np.int8)
    for i in range(count):
        for stage in range(SUBTASKS):
            site_type[i, stage] = rng.permutation(OBJECT_TYPES)
    if goal_kind == "heldout":
        idx = np.full(count, int(heldout), dtype=np.int64)
    elif goal_kind == "seen":
        choices = np.asarray([i for i in range(len(GOAL_PAIRS)) if i != int(heldout)], dtype=np.int64)
        idx = choices[np.arange(count) % len(choices)]
    else:
        idx = np.arange(count) % len(GOAL_PAIRS)
    goal = np.asarray(GOAL_PAIRS, dtype=np.int8)[idx]
    # Balance partner identities so every target worker has the same number of
    # active evaluation episodes.  The sequence is intentionally nonconstant:
    # the permutation control must be able to exchange messages within a
    # worker's stream.
    partner_id = (np.arange(count, dtype=np.int64) % WORKERS).astype(np.int8)
    partner_uniform = (partner_id.astype(np.float64) + 0.5) / WORKERS
    return {
        "site_type": site_type, "goal": goal, "target_bits": target_bits(goal, task),
        "partner_uniform": partner_uniform, "partner_id": partner_id,
        "message_uniforms": rng.random((count, 2)), "action_uniforms": rng.random((count, HORIZON)),
        "capacity": np.full((count, SUBTASKS, OBJECT_TYPES), CAPACITY, dtype=np.int8),
        "seed": int(seed), "task": task, "support": "evaluation", "heldout_goal": int(heldout),
        "evaluation": True, "update": 0,
    }


def prepare():
    return {
        "schema": "ecological_factorization_study_v1",
        "horizon": HORIZON, "action_start": ACTION_START, "subtasks": SUBTASKS,
        "steps_per_subtask": STEPS_PER_SUBTASK, "object_types": OBJECT_TYPES,
        "forms": {f: {"alphabet_size": alphabet_size(f), "length": message_length(f), "state_count_joint_history": state_count(f), "state_count_slot_local": state_count(f, "slot_local") if f == "tri3" else None, "arrival_times": list(message_arrival_times(f))} for f in FORMS},
        "tasks": list(TASKS), "representations": list(REPRESENTATIONS), "supports": list(SUPPORTS), "channels": list(CHANNELS),
        "partner_mode": PARTNER_MODE, "partner_visibility": PARTNER_VISIBILITY, "workers": WORKERS,
        "capacity": CAPACITY, "actions": ["wait", "take_site0", "take_site1", "take_site2"],
        "seeds": list(SEEDS), "parent_conditions": list(PARENT_CONDITIONS), "child_conditions": list(CHILD_CONDITIONS),
        "updates": UPDATES, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "entropy_initial": ENTROPY_INITIAL, "entropy_zero_after": ENTROPY_ZERO_AFTER, "checkpoints": list(CHECKPOINTS),
        "goal_pairs": [list(x) for x in GOAL_PAIRS], "holistic_target_map": list(HOLISTIC_TARGET_MAP),
        "heldout_assignment": {str(seed): heldout_goal(seed) for seed in SEEDS},
        "pairing": "forms, tasks, representation, support and live/silent arms share site permutations, goals, partners, message uniforms and action uniforms wherever the factor is held fixed",
        "zero_shot_rule": "leave-one-out live heldout natural team return >= 0.60",
        "no_teacher_or_language_prior": True,
    }
