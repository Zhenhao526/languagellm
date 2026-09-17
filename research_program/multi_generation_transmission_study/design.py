"""Frozen design for a two-generation component-substitution chain."""
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
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
TARGET_WORKER = 0
PARENT_CONDITION = "dual2_rotating_hidden_live_factorized_staged"

# This is the pre-registered parent-composable stratum from the 32-seed parent
# batch.  The stratum is fixed before this experiment is executed.
SEEDS = (76103, 76104, 76105, 76106, 76112, 76113, 76120, 76122, 76129)
LINEAGES = ("worker_then_sender", "sender_then_worker")
CHANNELS = ("live", "silent")
GENERATIONS = (1, 2)
CONDITIONS = tuple(
    f"{lineage}_g1{g1}_g2{g2}"
    for lineage in LINEAGES
    for g1 in CHANNELS
    for g2 in CHANNELS
)
EVAL_SEED_OFFSET = 970000


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    parts = condition.split("_")
    require(len(parts) == 5 and parts[0] in ("worker", "sender") and parts[1] in ("then",), f"bad condition {condition}")
    # The split form is worker_then_sender_g1live_g2silent.
    lineage = "_".join(parts[:3])
    require(lineage in LINEAGES, f"bad lineage {lineage}")
    require(parts[3].startswith("g1") and parts[3][2:] in CHANNELS, f"bad generation 1 channel {condition}")
    require(parts[4].startswith("g2") and parts[4][2:] in CHANNELS, f"bad generation 2 channel {condition}")
    return lineage, parts[3][2:], parts[4][2:]


def stage_condition(condition, generation):
    lineage, g1, g2 = parse_condition(condition)
    require(generation in GENERATIONS, "bad generation")
    return lineage, (g1 if generation == 1 else g2)


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def episode_stream(seed, generation, count, *, evaluation=False, update=0):
    # Generation is part of the deterministic stream key, so each stage has a
    # distinct world sample while live/silent arms remain exactly paired.
    stream_seed = int(seed) + 1000003 * int(generation)
    return base.episode_stream(stream_seed, PARTNER_MODE, VISIBILITY, TASK, count, evaluation=evaluation, update=update)


def prepare(parent_root, parent_hashes, parent_analysis_sha256):
    return {
        "schema": "multi_generation_transmission_study_v1",
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
        "lineages": list(LINEAGES),
        "channels": list(CHANNELS),
        "generations": list(GENERATIONS),
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
        "parent_analysis_sha256": parent_analysis_sha256,
        "paired_design": "generation-1 live/silent and generation-2 live/silent arms share stage-specific worlds, goals, partners and uniforms",
        "lineage_design": {
            "worker_then_sender": "replace worker 0 with a slot-local receiver, then replace the sender while that receiver remains frozen",
            "sender_then_worker": "replace the sender, then replace worker 0 with a slot-local receiver while the new sender remains frozen",
        },
        "replacement": "one component is newly initialized at each generation; the complementary components are inherited and frozen during that stage",
        "composition_rule": "natural live return >= 0.60 and absolute recombined-minus-natural <= 0.02",
        "no_teacher_or_language_prior": True,
    }
