"""Frozen design for leave-one-goal-combination-out recovery."""
from __future__ import annotations

import hashlib
import numpy as np

from research_program.action_dependent_signaling_study import design as base

HORIZON = base.HORIZON
ACTION_START = base.ACTION_START
SUBTASKS = base.SUBTASKS
STEPS_PER_SUBTASK = base.STEPS_PER_SUBTASK
FORM = "dual2"
PARTNER_MODE = "rotating"
VISIBILITY = "hidden"
TASK = "factorized"
PROTOCOL = "staged"
WORKERS = base.WORKERS
CAPACITY = base.CAPACITY
MESSAGE_SLOTS = base.MESSAGE_SLOTS
NULL_MESSAGE = base.NULL_MESSAGE
ACTION_COUNT = base.ACTION_COUNT
UPDATES = 3000
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = base.ENTROPY_INITIAL
ENTROPY_ZERO_AFTER = base.ENTROPY_ZERO_AFTER
CORRECT_REWARD = base.CORRECT_REWARD
WRONG_REWARD = base.WRONG_REWARD
SEEDS = tuple(range(76101, 76133))
SUPPORTS = ("full", "leave_one_out")
CHANNELS = ("live", "silent")
CONDITIONS = tuple(f"{support}_{channel}" for support in SUPPORTS for channel in CHANNELS)
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
TARGET_WORKER = 0
PARENT_CONDITION = "dual2_rotating_hidden_live_factorized_staged"
EVAL_SEED_OFFSET = 940000


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    parts = condition.rsplit("_", 1)
    require(len(parts) == 2 and parts[0] in SUPPORTS and parts[1] in CHANNELS, f"bad condition {condition}")
    return parts[0], parts[1]


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def heldout_goal(seed):
    # Four held-out combinations are balanced across the 32 seeds.
    return int((int(seed) - SEEDS[0]) % 4)


def episode_stream(seed, count, *, evaluation=False, update=0, support="full", heldout=None):
    require(support in SUPPORTS, "unknown support")
    ep = base.episode_stream(int(seed), PARTNER_MODE, VISIBILITY, TASK, count, evaluation=evaluation, update=update)
    if heldout is None:
        heldout = heldout_goal(seed)
    heldout = int(heldout)
    require(0 <= heldout < 4, "bad heldout goal")
    if support == "leave_one_out" and not evaluation:
        idx = base.goal_index(ep["goal"])
        replacement = (idx + 1) % 4
        ep["goal"] = np.array(ep["goal"], copy=True)
        ep["goal"][idx == heldout] = np.stack([replacement[idx == heldout] // 2, replacement[idx == heldout] % 2], axis=-1)
        ep["target_bits"] = base.target_bits(ep["goal"], TASK)
    ep["support"] = support
    ep["heldout_goal"] = heldout
    return ep


def balanced_eval_stream(seed, count, goal_kind, heldout):
    require(goal_kind in ("all", "seen", "heldout"), "bad evaluation goal kind")
    ep = base.episode_stream(int(seed), PARTNER_MODE, VISIBILITY, TASK, count, evaluation=True)
    values = np.arange(count, dtype=np.int64) % (4 if goal_kind == "all" else 3 if goal_kind == "seen" else 1)
    if goal_kind == "heldout":
        idx = np.full(count, int(heldout), dtype=np.int64)
    elif goal_kind == "seen":
        choices = np.asarray([i for i in range(4) if i != int(heldout)], dtype=np.int64)
        idx = choices[values]
    else:
        idx = values
    ep["goal"] = np.stack([idx // 2, idx % 2], axis=-1).astype(np.int8)
    ep["target_bits"] = base.target_bits(ep["goal"], TASK)
    ep["support"] = "evaluation"
    ep["heldout_goal"] = int(heldout)
    return ep


def prepare(parent_root, parent_hashes, parent_composable):
    return {
        "schema": "heldout_composition_study_v1",
        "horizon": HORIZON,
        "action_start": ACTION_START,
        "subtasks": SUBTASKS,
        "steps_per_subtask": STEPS_PER_SUBTASK,
        "form": FORM,
        "partner_mode": PARTNER_MODE,
        "visibility": VISIBILITY,
        "task": TASK,
        "protocol": PROTOCOL,
        "workers": WORKERS,
        "capacity": CAPACITY,
        "target_worker": TARGET_WORKER,
        "seeds": list(SEEDS),
        "supports": list(SUPPORTS),
        "channels": list(CHANNELS),
        "conditions": list(CONDITIONS),
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "entropy_initial": ENTROPY_INITIAL,
        "entropy_zero_after": ENTROPY_ZERO_AFTER,
        "checkpoints": list(CHECKPOINTS),
        "parent_condition": PARENT_CONDITION,
        "parent_root": str(parent_root),
        "parent_checkpoint_sha256": parent_hashes,
        "parent_composable": {str(k): bool(v) for k, v in parent_composable.items()},
        "heldout_assignment": {str(seed): heldout_goal(seed) for seed in SEEDS},
        "pairing": "live and silent arms share worlds, goals, partners, message uniforms and action uniforms within each support",
        "replacement": "replace worker 0; parent sender and workers 1-3 remain frozen",
        "controls": ["full support", "leave-one-goal-out support", "from-scratch silent"],
        "zero_shot_rule": "heldout natural team return >= 0.60 on a goal combination absent from child training",
        "no_teacher_or_language_prior": True,
    }
