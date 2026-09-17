"""Frozen design for the private-demand cooperation experiment.

The task makes information asymmetry consequential: agent ``a`` privately
knows the material it wants, but the reward for agent ``1-a``'s action is
defined by that request.  A successful token therefore has to carry a
partner-relevant convention, rather than merely correlate with an agent's own
action.
"""
from __future__ import annotations

from itertools import product
import hashlib
from pathlib import Path
import numpy as np

HORIZON = 6
MESSAGE_ROUNDS = (0, 1)
ALPHABET_SIZE = 8
NULL_MESSAGE = ALPHABET_SIZE
ACTION_COUNT = 3                         # wait, take site 0, take site 1
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
WRONG_REWARD = -1.0

# own request, local patch type, local inventory, last result, time, partner
# request, remote patch type, remote inventory, information flag
OBS_DIM = 2 + 2 + 1 + 3 + HORIZON + 2 + 2 + 1 + 1
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
    """Generate paired worlds, private requests and score uniforms.

    Scarcity is omitted from the RNG key so capacity arms are paired exactly.
    ``switching`` has four training patterns and four held-out patterns; the
    persistent task samples one private bit and repeats it for all rounds.
    """
    require(count > 0, "count must be positive")
    require(task in TASKS and scarcity in SCARCITIES, "bad episode factor")
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), EVAL_SEED_OFFSET if evaluation else 0,
        task_index(task), int(update)]))
    site_type = rng.integers(0, 2, size=(count, 2), dtype=np.int8)
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
        goal_a = patterns[rng.choice(choices, size=count)]
        goal_b = patterns[rng.choice(choices, size=count)]
        goal = np.stack([goal_a, goal_b], axis=1)
    return {
        "site_type": site_type,
        "site0_type": site_type[:, 0].copy(),
        "goal": goal,
        "message_uniforms": rng.random((count, HORIZON, 2), dtype=np.float64),
        "action_uniforms": rng.random((count, HORIZON, 2), dtype=np.float64),
        "capacity": np.full(count, capacity(scarcity), dtype=np.int8),
        "seed": int(seed), "scarcity": scarcity, "task": task,
        "evaluation": bool(evaluation), "update": int(update),
    }


def encode_observation(*, own_goal: int, local_type: int, local_inventory: int,
                        last_result: int, time: int, partner_goal: int,
                        remote_type: int, remote_inventory: int,
                        information: str) -> np.ndarray:
    require(own_goal in (0, 1) and local_type in (0, 1), "bad resource id")
    require(last_result in (0, 1, 2) and 0 <= time < HORIZON,
            "bad observation field")
    x = np.zeros(OBS_DIM, dtype=np.float64)
    k = 0
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
    x[k] = 1. if information == "FI" else 0.
    require(k + 1 == OBS_DIM, "observation layout mismatch")
    return x


def prepare() -> dict:
    return {
        "schema": "partner_demand_communication_v1",
        "horizon": HORIZON, "message_rounds": list(MESSAGE_ROUNDS),
        "blackout_rounds": [t for t in range(HORIZON) if t not in MESSAGE_ROUNDS],
        "alphabet_size": ALPHABET_SIZE, "null_message": NULL_MESSAGE,
        "action_names": ["wait", "take_site0", "take_site1"],
        "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "tasks": list(TASKS), "updates": UPDATES, "batch_size": BATCH_SIZE,
        "hidden": HIDDEN, "learning_rate": LEARNING_RATE,
        "gradient_clip": GRADIENT_CLIP, "entropy_initial": ENTROPY_INITIAL,
        "entropy_zero_after": ENTROPY_ZERO_AFTER,
        "observation_dim": OBS_DIM, "receive_dim": RECV_DIM,
        "train_goal_patterns": [[0, 0, 0, 1, 1, 1], [1, 1, 1, 0, 0, 0],
                                [0, 0, 1, 1, 0, 0], [1, 1, 0, 0, 1, 1]],
        "heldout_goal_patterns": [[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0],
                                   [0, 1, 1, 1, 0, 0], [1, 0, 0, 0, 1, 1]],
        "resource_regimes": {"scarce": 2, "abundant": HORIZON},
        "private_information": "own request, local patch and inventory; partner request and remote patch are masked in PI",
        "full_information": "both requests and both patch states are visible in FI",
        "task_pressure": "agent a privately requests a material; agent 1-a is rewarded for collecting it",
        "channel": "one eight-way token at rounds 0 and 1; token arrives before the next action and then the channel is blacked out",
        "reward": "+1 when an action satisfies the partner's current request and available inventory; -1 for a wrong/depleted take; wait 0",
        "simultaneous_resolution": "same-site requests share capacity; if one unit remains, lower-index agent receives it",
        "network": "agent-specific 32-unit tanh recurrent core (or no recurrent edge), message 8-way head and action 3-way head",
        "no_teacher_or_language_prior": True,
        "pairing": "same initial parameters, episode worlds, requests, message uniforms and action uniforms within seed/memory/scarcity/information/task block",
        "evaluation": "training-support and held-out switching patterns, each with greedy natural/closed/permuted channel controls",
        "automatic_followon_experiment": False,
    }
