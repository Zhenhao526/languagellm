"""Frozen design for noisy compositional transmission and repair."""
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

# Fixed before executing this study: the parent-composable stratum from the
# 32-seed action-dependent batch.
SEEDS = (76103, 76104, 76105, 76106, 76112, 76113, 76120, 76122, 76129)
REPRESENTATIONS = ("joint_history", "slot_local")
ADAPTATIONS = ("worker_only", "coadapt")
NOISE_LEVELS = (0.0, 0.10, 0.25, 0.40)
NOISE_KEYS = {"p00": 0.0, "p10": 0.10, "p25": 0.25, "p40": 0.40}
CONDITIONS = tuple(
    f"{adaptation}_{representation}_{key}"
    for adaptation in ADAPTATIONS
    for representation in REPRESENTATIONS
    for key in NOISE_KEYS
)
EVAL_SEED_OFFSET = 980000
NOISE_STREAM_SALT = 981001


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_condition(condition):
    parts = condition.split("_")
    if parts[:2] == ["worker", "only"]:
        require(len(parts) == 5, f"bad condition {condition}")
        adaptation, representation, key = "worker_only", "_".join(parts[2:4]), parts[4]
    else:
        require(len(parts) == 4, f"bad condition {condition}")
        adaptation, representation, key = parts[0], "_".join(parts[1:3]), parts[3]
    require(adaptation in ADAPTATIONS, f"bad adaptation {adaptation}")
    require(representation in REPRESENTATIONS, f"bad representation {representation}")
    require(key in NOISE_KEYS, f"bad noise key {key}")
    return adaptation, representation, NOISE_KEYS[key], key


def noise_key(noise):
    return min(NOISE_KEYS, key=lambda key: abs(NOISE_KEYS[key] - float(noise)))


def entropy_coefficient(update):
    require(1 <= update <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def episode_stream(seed, count, *, evaluation=False, update=0):
    ep = base.episode_stream(int(seed), PARTNER_MODE, VISIBILITY, TASK, count, evaluation=evaluation, update=update)
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), NOISE_STREAM_SALT if evaluation else NOISE_STREAM_SALT - 1, int(update)]))
    ep["noise_uniforms"] = rng.random((count, MESSAGE_SLOTS))
    ep["noise_stream_salt"] = NOISE_STREAM_SALT
    return ep


def prepare(parent_root, parent_hashes, parent_analysis_sha256):
    return {
        "schema": "noise_repair_study_v1",
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
        "representations": list(REPRESENTATIONS),
        "adaptations": list(ADAPTATIONS),
        "noise_levels": list(NOISE_LEVELS),
        "noise_keys": NOISE_KEYS,
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
        "pairing": "all noise/adaptation/representation arms share stage-specific worlds, goals, partners, message uniforms, action uniforms and noise uniforms within each seed/update",
        "replacement": "replace worker 0; worker 1-3 remain frozen; worker_only freezes sender; coadapt updates hidden sender and worker 0 together",
        "noise": "independent binary slot flips after message delivery and before receiver state update",
        "composition_rule": "noisy natural return >= 0.60 and absolute noisy recombined-minus-natural <= 0.02",
        "compatibility": "evaluate frozen incumbent workers 1-3 under the final sender to distinguish repair from sender drift",
        "no_teacher_or_language_prior": True,
    }
