"""Frozen design for comparing joint-history and slot-local receivers."""
from __future__ import annotations

import hashlib
import numpy as np

from research_program.action_dependent_signaling_study import design as base
from research_program.heldout_composition_study import design as heldout

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
SEEDS = tuple(range(76101, 76133))
REPRESENTATIONS = ("joint_history", "slot_local")
SUPPORTS = ("full", "leave_one_out")
CHANNELS = ("live", "silent")
CONDITIONS = tuple(f"{representation}_{support}_{channel}" for representation in REPRESENTATIONS for support in SUPPORTS for channel in CHANNELS)
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
TARGET_WORKER = 0
PARENT_CONDITION = "dual2_rotating_hidden_live_factorized_staged"
EVAL_SEED_OFFSET = 950000


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    representation = next((value for value in REPRESENTATIONS if condition.startswith(value + "_")), None)
    require(representation is not None, f"bad condition {condition}")
    support, channel = condition[len(representation) + 1 :].rsplit("_", 1)
    require(support in SUPPORTS and channel in CHANNELS, f"bad condition {condition}")
    return representation, support, channel


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def heldout_goal(seed):
    return heldout.heldout_goal(seed)


def episode_stream(seed, count, *, evaluation=False, update=0, support="full", heldout_goal_index=None):
    return heldout.episode_stream(seed, count, evaluation=evaluation, update=update, support=support, heldout=heldout_goal_index)


def balanced_eval_stream(seed, count, goal_kind, heldout_goal_index):
    return heldout.balanced_eval_stream(seed, count, goal_kind, heldout_goal_index)


def prepare(parent_root, parent_hashes, parent_composable):
    return {
        "schema": "receiver_representation_study_v1",
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
        "representations": list(REPRESENTATIONS),
        "supports": list(SUPPORTS),
        "channels": list(CHANNELS),
        "conditions": list(CONDITIONS),
        "seeds": list(SEEDS),
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
        "pairing": "representation arms share parent, worlds, goals, partners, message uniforms and action uniforms within each support/channel",
        "replacement": "replace worker 0; parent sender and workers 1-3 remain frozen",
        "representation_control": "joint_history indexes all received token history; slot_local indexes only the current staged slot; parameter tensor shape is identical",
        "zero_shot_rule": "heldout natural team return >= 0.60 on a goal combination absent from child training",
        "no_teacher_or_language_prior": True,
    }
