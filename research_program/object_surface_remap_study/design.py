"""Frozen design for the binary object-surface remapping study."""
from __future__ import annotations

import hashlib
import numpy as np

HORIZON = 6
ACTION_START = 2
SUBTASKS = 2
STEPS_PER_SUBTASK = 2
OBJECT_TYPES = 2
SCENE_COUNT = OBJECT_TYPES ** OBJECT_TYPES
FORMS = ("mono4", "dual2")
ALPHABET_SIZES = {"mono4": 4, "dual2": 2}
MESSAGE_LENGTHS = {"mono4": 1, "dual2": 2}
ARRIVAL_TIMES = {"mono4": (1,), "dual2": (1, 3)}
TASKS = ("factorized",)
MAPPINGS = ("identity", "swap")
REPRESENTATIONS = ("joint_history", "slot_local")
SUPPORTS = ("full", "leave_one_out")
CHANNELS = ("live", "silent")
PARTNER_MODE = "rotating"
PARTNER_VISIBILITY = "hidden"
WORKERS = 4
CAPACITY = 2
NULL_MESSAGE = -1
ACTION_COUNT = 3
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
SEEDS = tuple(range(78201, 78210))
UPDATES = 3000
PARENT_UPDATES = UPDATES
CHILD_UPDATES = UPDATES
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
EVAL_SEED_OFFSET = 982000
MAPPING_SALT = 982001
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
TARGET_WORKER = 0
GOAL_PAIRS = tuple((a, b) for a in range(OBJECT_TYPES) for b in range(OBJECT_TYPES))
PARENT_CONDITIONS = tuple(f"{form}_{task}" for form in FORMS for task in TASKS)
CHILD_CONDITIONS = tuple(
    f"{representation}_{form}_{task}_{mapping}_{support}_{channel}"
    for representation in REPRESENTATIONS
    for form in FORMS
    for task in TASKS
    for mapping in MAPPINGS
    for support in SUPPORTS
    for channel in CHANNELS
    if not (representation == "slot_local" and form == "mono4")
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
    representation = next((v for v in REPRESENTATIONS if condition.startswith(v + "_")), None)
    require(representation is not None, f"bad child condition {condition}")
    parts = condition[len(representation) + 1 :].split("_")
    if len(parts) == 5:
        form, task, mapping, support, channel = parts
    elif len(parts) == 7 and parts[3:6] == ["leave", "one", "out"]:
        form, task, mapping, _, _, _, channel = parts
        support = "leave_one_out"
    else:
        raise ValueError(f"bad child condition {condition}")
    require(representation in REPRESENTATIONS and form in FORMS and task in TASKS and mapping in MAPPINGS and support in SUPPORTS and channel in CHANNELS, f"bad child condition {condition}")
    require(not (representation == "slot_local" and form == "mono4"), "slot-local representation is only defined for dual2")
    return representation, form, task, mapping, support, channel


def alphabet_size(form):
    require(form in FORMS, "unknown form")
    return ALPHABET_SIZES[form]


def message_length(form):
    require(form in FORMS, "unknown form")
    return MESSAGE_LENGTHS[form]


def state_count(form, representation="joint_history"):
    require(form in FORMS and representation in REPRESENTATIONS, "unknown form or representation")
    if representation == "slot_local":
        require(form == "dual2", "slot-local representation is only defined for dual2")
        return alphabet_size(form) + 1
    return alphabet_size(form) ** message_length(form) + 1


def goal_index(goal):
    g = np.asarray(goal, dtype=np.int64)
    return g[..., 0] * OBJECT_TYPES + g[..., 1]


def target_bits(goal, task):
    require(task in TASKS, "unknown task")
    return np.asarray(goal, dtype=np.int8).copy()


def heldout_goal(seed):
    require(int(seed) in SEEDS, "unknown seed")
    return int((int(seed) - SEEDS[0]) % len(GOAL_PAIRS))


def surface_permutation(mapping):
    require(mapping in MAPPINGS, "unknown surface mapping")
    return np.asarray((0, 1) if mapping == "identity" else (1, 0), dtype=np.int8)


def message_arrival_times(form):
    require(form in FORMS, "unknown form")
    return ARRIVAL_TIMES[form]


def entropy_coefficient(update, *, child=False):
    require(1 <= int(update) <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (int(update) - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def _masked_goals(raw, heldout):
    idx = goal_index(raw).astype(np.int64)
    replacement = (int(heldout) + 1) % len(GOAL_PAIRS)
    idx = np.where(idx == int(heldout), replacement, idx)
    return np.asarray(GOAL_PAIRS, dtype=np.int8)[idx]


def _world_stream(rng, count):
    semantic = np.empty((count, SUBTASKS, OBJECT_TYPES), dtype=np.int8)
    for i in range(count):
        for stage in range(SUBTASKS):
            semantic[i, stage] = rng.permutation(OBJECT_TYPES)
    return semantic


def episode_stream(seed, count, *, task, support="full", heldout=None, mapping="identity", evaluation=False, update=0):
    require(count > 0 and task in TASKS and support in SUPPORTS and mapping in MAPPINGS, "bad episode factors")
    require(int(seed) in SEEDS or evaluation, "unknown seed")
    if heldout is None:
        heldout = heldout_goal(seed) if int(seed) in SEEDS else 0
    require(0 <= int(heldout) < len(GOAL_PAIRS), "bad heldout goal")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update), MAPPING_SALT]))
    site_semantic = _world_stream(rng, count)
    site_surface = surface_permutation(mapping)[site_semantic]
    raw_goal = rng.integers(0, OBJECT_TYPES, size=(count, 2), dtype=np.int8)
    goal = _masked_goals(raw_goal, heldout) if support == "leave_one_out" else raw_goal
    partner_uniform = rng.random(count)
    partner_id = np.floor(partner_uniform * WORKERS).astype(np.int8)
    message_uniforms = rng.random((count, max(MESSAGE_LENGTHS.values())))
    action_uniforms = rng.random((count, HORIZON))
    return {"site_semantic": site_semantic, "site_surface": site_surface, "goal": goal, "target_bits": target_bits(goal, task), "partner_uniform": partner_uniform, "partner_id": partner_id, "message_uniforms": message_uniforms, "action_uniforms": action_uniforms, "capacity": np.full((count, SUBTASKS, OBJECT_TYPES), CAPACITY, dtype=np.int8), "mapping": mapping, "seed": int(seed), "task": task, "support": support, "heldout_goal": int(heldout), "evaluation": bool(evaluation), "update": int(update)}


def balanced_eval_stream(seed, count, goal_kind, task, heldout, mapping="identity"):
    require(goal_kind in ("all", "seen", "heldout"), "bad goal kind")
    require(task in TASKS and mapping in MAPPINGS, "bad evaluation factors")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET + 17, int(heldout), MAPPING_SALT]))
    site_semantic = _world_stream(rng, count)
    site_surface = surface_permutation(mapping)[site_semantic]
    if goal_kind == "heldout":
        idx = np.full(count, int(heldout), dtype=np.int64)
    elif goal_kind == "seen":
        choices = np.asarray([i for i in range(len(GOAL_PAIRS)) if i != int(heldout)], dtype=np.int64)
        idx = choices[np.arange(count) % len(choices)]
    else:
        idx = np.arange(count) % len(GOAL_PAIRS)
    goal = np.asarray(GOAL_PAIRS, dtype=np.int8)[idx]
    partner_id = (np.arange(count, dtype=np.int64) % WORKERS).astype(np.int8)
    partner_uniform = (partner_id.astype(np.float64) + 0.5) / WORKERS
    return {"site_semantic": site_semantic, "site_surface": site_surface, "goal": goal, "target_bits": target_bits(goal, task), "partner_uniform": partner_uniform, "partner_id": partner_id, "message_uniforms": rng.random((count, max(MESSAGE_LENGTHS.values()))), "action_uniforms": rng.random((count, HORIZON)), "capacity": np.full((count, SUBTASKS, OBJECT_TYPES), CAPACITY, dtype=np.int8), "mapping": mapping, "seed": int(seed), "task": task, "support": "evaluation", "heldout_goal": int(heldout), "evaluation": True, "update": 0}


def prepare():
    return {"schema": "object_surface_remap_study_v1", "horizon": HORIZON, "action_start": ACTION_START, "subtasks": SUBTASKS, "steps_per_subtask": STEPS_PER_SUBTASK, "object_types": OBJECT_TYPES, "forms": {f: {"alphabet_size": alphabet_size(f), "length": message_length(f), "state_count_joint_history": state_count(f), "state_count_slot_local": state_count(f, "slot_local") if f == "dual2" else None, "arrival_times": list(message_arrival_times(f))} for f in FORMS}, "tasks": list(TASKS), "mappings": list(MAPPINGS), "representations": list(REPRESENTATIONS), "supports": list(SUPPORTS), "channels": list(CHANNELS), "partner_mode": PARTNER_MODE, "partner_visibility": PARTNER_VISIBILITY, "workers": WORKERS, "capacity": CAPACITY, "actions": ["wait", "take_site0", "take_site1"], "seeds": list(SEEDS), "parent_conditions": list(PARENT_CONDITIONS), "child_conditions": list(CHILD_CONDITIONS), "updates": UPDATES, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE, "entropy_initial": ENTROPY_INITIAL, "entropy_zero_after": ENTROPY_ZERO_AFTER, "checkpoints": list(CHECKPOINTS), "goal_pairs": [list(x) for x in GOAL_PAIRS], "heldout_assignment": {str(seed): heldout_goal(seed) for seed in SEEDS}, "pairing": "identity/swap arms share semantic scenes, goals, partners, message uniforms and action uniforms; only surface labels differ", "zero_shot_rule": "leave-one-out live heldout natural team return >= 0.60", "no_teacher_or_language_prior": True, "parent_runs": len(SEEDS) * len(PARENT_CONDITIONS), "child_runs": len(SEEDS) * len(CHILD_CONDITIONS), "runs": len(SEEDS) * (len(PARENT_CONDITIONS) + len(CHILD_CONDITIONS)), "runtime_policy": "float64 NumPy only"}
