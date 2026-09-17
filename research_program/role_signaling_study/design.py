"""Frozen division-of-labour signaling game.

One randomly assigned scout privately observes a request sequence and sends
short discrete tokens.  The other agent is the only one allowed to harvest.
The setup removes simultaneous-action credit noise while retaining persistent
inventory, scarcity, private perception and a blackout period.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

HORIZON = 6
MESSAGE_ROUNDS = (0, 1)
ALPHABET_SIZE = 8
NULL_MESSAGE = ALPHABET_SIZE
ACTION_COUNT = 3                    # wait, take site 0, take site 1
AGENTS = (0, 1)
INFORMATIONS = ("PI", "FI")
SCARCITIES = ("scarce", "abundant")
MEMORIES = ("stateless", "recurrent")
CHANNELS = ("silent", "live")
TASKS = ("persistent", "switching")
SEEDS = (68101, 68102, 68103, 68104, 68105, 68106, 68107, 68108)
UPDATES = 2000
BATCH_SIZE = 128
HIDDEN = 32
LEARNING_RATE = 0.002
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-8
GRADIENT_CLIP = 5.0
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 750
EVAL_SEED_OFFSET = 900000
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
EXPOSE_OUTCOME = False

# own request, local patch, inventory, last-result, time, partner request,
# remote patch, remote inventory, FI flag, sender/worker role one-hot
OBS_DIM = 2 + 2 + 1 + 3 + HORIZON + 2 + 2 + 1 + 1 + 2
RECV_DIM = ALPHABET_SIZE + 1


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def parse_condition(condition: str):
    parts = condition.split("_")
    require(len(parts) == 5, f"Bad condition: {condition}")
    memory, scarcity, information, channel, task = parts
    require(memory in MEMORIES and scarcity in SCARCITIES
            and information in INFORMATIONS and channel in CHANNELS
            and task in TASKS, f"Unknown condition: {condition}")
    return memory, scarcity, information, channel, task


CONDITIONS = tuple(
    f"{memory}_{scarcity}_{information}_{channel}_{task}"
    for memory in MEMORIES for scarcity in SCARCITIES
    for information in INFORMATIONS for channel in CHANNELS
    for task in TASKS
)


def capacity(scarcity: str) -> int:
    require(scarcity in SCARCITIES, "Unknown scarcity")
    return 2 if scarcity == "scarce" else HORIZON


def task_index(task: str) -> int:
    require(task in TASKS, "Unknown task")
    return TASKS.index(task)


def entropy_coefficient(update: int) -> float:
    require(1 <= update <= UPDATES, "Invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_stream(seed: int, scarcity: str, task: str, count: int, *,
                   evaluation: bool = False, update: int = 0):
    require(count > 0, "count must be positive")
    require(scarcity in SCARCITIES and task in TASKS, "bad episode factor")
    # Capacity is deliberately absent from the key so scarcity arms share
    # worlds, private requests, roles and score uniforms exactly.
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), EVAL_SEED_OFFSET if evaluation else 0,
        task_index(task), int(update)]))
    site_type = rng.integers(0, 2, size=(count, 2), dtype=np.int8)
    sender = rng.integers(0, 2, size=count, dtype=np.int8)
    if task == "persistent":
        bits = rng.integers(0, 2, size=(count, 2), dtype=np.int8)
        goal = np.repeat(bits[:, :, None], HORIZON, axis=2)
    else:
        patterns = np.array([
            [0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1],
        ], dtype=np.int8)
        choices = np.array([4, 5, 6, 7] if evaluation else [0, 1, 2, 3])
        goal = np.stack([patterns[rng.choice(choices, size=count)],
                         patterns[rng.choice(choices, size=count)]], axis=1)
    return {
        "site_type": site_type, "site0_type": site_type[:, 0].copy(),
        "sender": sender, "goal": goal,
        "message_uniforms": rng.random((count, HORIZON, 2), dtype=np.float64),
        "action_uniforms": rng.random((count, HORIZON, 2), dtype=np.float64),
        "capacity": np.full(count, capacity(scarcity), dtype=np.int8),
        "seed": int(seed), "scarcity": scarcity, "task": task,
        "evaluation": bool(evaluation), "update": int(update),
    }


def encode_observation(*, own_goal: int, local_type: int, local_inventory: int,
                        last_result: int, time: int, partner_goal: int,
                        remote_type: int, remote_inventory: int,
                        information: str, sender_role: bool) -> np.ndarray:
    require(own_goal in (0, 1) and local_type in (0, 1), "bad resource id")
    require(last_result in (0, 1, 2) and 0 <= time < HORIZON,
            "bad observation field")
    x = np.zeros(OBS_DIM, dtype=np.float64); k = 0
    x[k + own_goal] = 1.; k += 2
    x[k + local_type] = 1.; k += 2
    x[k] = local_inventory / float(HORIZON); k += 1
    x[k + last_result] = 1.; k += 3
    x[k + time] = 1.; k += HORIZON
    if information == "FI":
        x[k + partner_goal] = 1.
    k += 2
    if information == "FI":
        x[k + remote_type] = 1.
    k += 2
    if information == "FI":
        x[k] = remote_inventory / float(HORIZON)
    k += 1
    x[k] = 1. if information == "FI" else 0.; k += 1
    x[k + (0 if sender_role else 1)] = 1.; k += 2
    require(k == OBS_DIM, "observation layout mismatch")
    return x


def prepare() -> dict:
    return {
        "schema": "role_signaling_communication_v1",
        "horizon": HORIZON, "message_rounds": list(MESSAGE_ROUNDS),
        "blackout_rounds": [t for t in range(HORIZON) if t not in MESSAGE_ROUNDS],
        "alphabet_size": ALPHABET_SIZE, "null_message": NULL_MESSAGE,
        "action_names": ["wait", "take_site0", "take_site1"],
        "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "tasks": list(TASKS), "updates": UPDATES, "batch_size": BATCH_SIZE,
        "hidden": HIDDEN, "learning_rate": LEARNING_RATE,
        "gradient_clip": GRADIENT_CLIP, "entropy_initial": ENTROPY_INITIAL,
        "entropy_zero_after": ENTROPY_ZERO_AFTER, "observation_dim": OBS_DIM,
        "receive_dim": RECV_DIM,
        "train_goal_patterns": [[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]],
        "heldout_goal_patterns": [[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
                                   [0, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1]],
        "resource_regimes": {"scarce": 2, "abundant": HORIZON},
        "role": "one random scout knows a private request sequence; one worker is the only agent allowed to harvest",
        "private_information": "scout request, local patch and inventory; partner request and remote patch are masked in PI",
        "outcome_feedback": "masked in the baseline to prevent reward correctness from becoming a second communication channel",
        "task_pressure": "worker reward is determined by the scout's private request",
        "channel": "one eight-way token at rounds 0 and 1, delivered before the next action; rounds 2-5 are blackout",
        "reward": "+1 for worker collection matching the scout request and available inventory; -0.25 for wrong/depleted take; wait 0; scout receives the same team reward",
        "network": "agent-specific 32-unit tanh recurrent core (or no recurrent edge), message 8-way head and action 3-way head",
        "no_teacher_or_language_prior": True,
        "pairing": "same initialization, worlds, requests, roles, message uniforms and action uniforms within seed/memory/scarcity/information/task block",
        "evaluation": "training-support and held-out switching patterns, greedy natural/closed/permuted controls",
        "automatic_followon_experiment": False,
    }
