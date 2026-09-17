"""Frozen design for multi-partner sender alignment under perceptual heterogeneity."""
from __future__ import annotations

import hashlib
import numpy as np

HORIZON = 6
ACTION_START = 2
SUBTASKS = 2
STEPS_PER_SUBTASK = 2
OBJECT_TYPES = 2
SCENE_COUNT = OBJECT_TYPES ** OBJECT_TYPES
FORM = "dual2"
ALPHABET_SIZE = 2
MESSAGE_LENGTH = 2
ARRIVAL_TIMES = (1, 3)
TASK = "factorized"
MAPPINGS = ("identity", "swap")
POPULATION_MODES = ("homogeneous", "heterogeneous")
VISIBILITIES = ("hidden", "visible")
ADAPTATIONS = ("sender_only", "coadapt")
SUPPORTS = ("full", "leave_one_out")
CHANNELS = ("live", "silent")
PARTNER_MODE = "rotating"
WORKERS = 4
CAPACITY = 2
NULL_MESSAGE = -1
ACTION_COUNT = 3
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
SEEDS = tuple(range(78401, 78410))
UPDATES = 3000
PARENT_UPDATES = UPDATES
CHILD_UPDATES = UPDATES
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
EVAL_SEED_OFFSET = 984000
MAPPING_SALT = 984001
POLICY_SALT = 984002
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
GOAL_PAIRS = tuple((a, b) for a in range(OBJECT_TYPES) for b in range(OBJECT_TYPES))
PARENT_CONDITIONS = tuple(POPULATION_MODES)
CHILD_CONDITIONS = tuple(
    f"{mode}_{visibility}_{adaptation}_{support}"
    for mode in POPULATION_MODES
    for visibility in VISIBILITIES
    for adaptation in ADAPTATIONS
    for support in SUPPORTS
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_parent_condition(condition):
    require(condition in PARENT_CONDITIONS, f"bad parent condition {condition}")
    return condition


def parse_child_condition(condition):
    parts = condition.split("_")
    require(len(parts) >= 4, f"bad child condition {condition}")
    mode, visibility = parts[:2]
    if parts[2:4] == ["sender", "only"]:
        adaptation = "sender_only"
        rest = parts[4:]
    else:
        adaptation = parts[2]
        rest = parts[3:]
    if rest == ["full"]:
        support = "full"
    elif rest == ["leave", "one", "out"]:
        support = "leave_one_out"
    else:
        raise ValueError(f"bad child condition {condition}")
    require(mode in POPULATION_MODES and visibility in VISIBILITIES and adaptation in ADAPTATIONS and support in SUPPORTS, f"bad child condition {condition}")
    return mode, visibility, adaptation, support


def goal_index(goal):
    g = np.asarray(goal, dtype=np.int64)
    return g[..., 0] * OBJECT_TYPES + g[..., 1]


def target_bits(goal, task=TASK):
    require(task == TASK, "unknown task")
    return np.asarray(goal, dtype=np.int8).copy()


def heldout_goal(seed):
    require(int(seed) in SEEDS, "unknown seed")
    return int((int(seed) - SEEDS[0]) % len(GOAL_PAIRS))


def surface_permutation(mapping):
    require(mapping in MAPPINGS, "unknown surface mapping")
    return np.asarray((0, 1) if mapping == "identity" else (1, 0), dtype=np.int8)


def entropy_coefficient(update):
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


def _surface_stream(site_semantic, partner_id, population_mode):
    out = np.empty_like(site_semantic)
    for worker in range(WORKERS):
        idx = np.flatnonzero(np.asarray(partner_id) == worker)
        if not len(idx):
            continue
        if population_mode == "heterogeneous" and worker % 2 == 1:
            mapping = "swap"
        else:
            mapping = "identity"
        out[idx] = surface_permutation(mapping)[site_semantic[idx]]
    return out


def episode_stream(seed, count, *, support="full", heldout=None, population_mode="homogeneous", evaluation=False, update=0):
    require(count > 0 and support in SUPPORTS and population_mode in POPULATION_MODES, "bad episode factors")
    require(int(seed) in SEEDS or evaluation, "unknown seed")
    if heldout is None:
        heldout = heldout_goal(seed) if int(seed) in SEEDS else 0
    require(0 <= int(heldout) < len(GOAL_PAIRS), "bad heldout goal")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update), MAPPING_SALT]))
    site_semantic = _world_stream(rng, count)
    partner_uniform = rng.random(count)
    partner_id = np.floor(partner_uniform * WORKERS).astype(np.int8)
    site_surface = _surface_stream(site_semantic, partner_id, population_mode)
    raw_goal = rng.integers(0, OBJECT_TYPES, size=(count, 2), dtype=np.int8)
    goal = _masked_goals(raw_goal, heldout) if support == "leave_one_out" else raw_goal
    return {
        "site_semantic": site_semantic,
        "site_surface": site_surface,
        "goal": goal,
        "target_bits": target_bits(goal),
        "partner_uniform": partner_uniform,
        "partner_id": partner_id,
        "message_uniforms": rng.random((count, MESSAGE_LENGTH)),
        "action_uniforms": rng.random((count, HORIZON)),
        "capacity": np.full((count, SUBTASKS, OBJECT_TYPES), CAPACITY, dtype=np.int8),
        "population_mode": population_mode,
        "support": support,
        "heldout_goal": int(heldout),
        "evaluation": bool(evaluation),
        "update": int(update),
    }


def balanced_eval_stream(seed, count, goal_kind, heldout, population_mode="homogeneous"):
    require(goal_kind in ("all", "seen", "heldout"), "bad goal kind")
    require(population_mode in POPULATION_MODES, "bad population mode")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET + 17, int(heldout), MAPPING_SALT]))
    site_semantic = _world_stream(rng, count)
    if goal_kind == "heldout":
        idx = np.full(count, int(heldout), dtype=np.int64)
    elif goal_kind == "seen":
        choices = np.asarray([i for i in range(len(GOAL_PAIRS)) if i != int(heldout)], dtype=np.int64)
        idx = choices[np.arange(count) % len(choices)]
    else:
        idx = np.arange(count) % len(GOAL_PAIRS)
    goal = np.asarray(GOAL_PAIRS, dtype=np.int8)[idx]
    partner_id = ((np.arange(count, dtype=np.int64) // len(GOAL_PAIRS)) % WORKERS).astype(np.int8)
    partner_uniform = (partner_id.astype(np.float64) + 0.5) / WORKERS
    return {
        "site_semantic": site_semantic,
        "site_surface": _surface_stream(site_semantic, partner_id, population_mode),
        "goal": goal,
        "target_bits": target_bits(goal),
        "partner_uniform": partner_uniform,
        "partner_id": partner_id,
        "message_uniforms": rng.random((count, MESSAGE_LENGTH)),
        "action_uniforms": rng.random((count, HORIZON)),
        "capacity": np.full((count, SUBTASKS, OBJECT_TYPES), CAPACITY, dtype=np.int8),
        "population_mode": population_mode,
        "support": "evaluation",
        "heldout_goal": int(heldout),
        "evaluation": True,
        "update": 0,
    }


def prepare():
    return {
        "schema": "partner_exposure_study_v1",
        "horizon": HORIZON,
        "action_start": ACTION_START,
        "subtasks": SUBTASKS,
        "steps_per_subtask": STEPS_PER_SUBTASK,
        "object_types": OBJECT_TYPES,
        "form": {"name": FORM, "alphabet_size": ALPHABET_SIZE, "length": MESSAGE_LENGTH, "arrival_times": list(ARRIVAL_TIMES), "joint_state_count": ALPHABET_SIZE ** MESSAGE_LENGTH + 1},
        "task": TASK,
        "mappings": list(MAPPINGS),
        "population_modes": list(POPULATION_MODES),
        "visibilities": list(VISIBILITIES),
        "adaptations": list(ADAPTATIONS),
        "supports": list(SUPPORTS),
        "evaluation_channels": list(CHANNELS),
        "partner_mode": PARTNER_MODE,
        "workers": WORKERS,
        "heterogeneous_rule": "worker ids 0 and 2 use identity; ids 1 and 3 use swap",
        "partner_exposure": "every child batch contains all four partner ids; sender may or may not observe partner id",
        "capacity": CAPACITY,
        "actions": ["wait", "take_site0", "take_site1"],
        "seeds": list(SEEDS),
        "parent_conditions": list(PARENT_CONDITIONS),
        "child_conditions": list(CHILD_CONDITIONS),
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "entropy_initial": ENTROPY_INITIAL,
        "entropy_zero_after": ENTROPY_ZERO_AFTER,
        "checkpoints": list(CHECKPOINTS),
        "goal_pairs": [list(x) for x in GOAL_PAIRS],
        "heldout_assignment": {str(seed): heldout_goal(seed) for seed in SEEDS},
        "pairing": "population modes share semantic worlds, goals, partners, messages and actions; heterogeneity changes only partner-specific surface labels",
        "primary_readout": "fresh sender leave-one-out live held-out return with hidden versus visible partner identity",
        "functional_rule": "held-out natural team return >= 0.60",
        "no_teacher_or_language_prior": True,
        "parent_runs": len(SEEDS) * len(PARENT_CONDITIONS),
        "child_runs": len(SEEDS) * len(CHILD_CONDITIONS),
        "runs": len(SEEDS) * (len(PARENT_CONDITIONS) + len(CHILD_CONDITIONS)),
        "runtime_policy": "float64 NumPy only",
    }
