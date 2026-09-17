"""Frozen design for the multi-round resource-scarcity communication game.

The environment exposes only task observations to policies.  The world state,
partner goal and remote-site state are retained in the researcher-side episode
record for evaluation and auditing.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
from pathlib import Path
import hashlib
import numpy as np

HORIZON = 5
MESSAGE_ROUNDS = (0, 1)
ALPHABET_SIZE = 8
NULL_MESSAGE = ALPHABET_SIZE
ACTION_COUNT = 3  # wait, take site 0, take site 1
AGENTS = (0, 1)
INFORMATIONS = ("PI", "FI")
SCARCITIES = ("scarce", "abundant")
MEMORIES = ("stateless", "recurrent")
CHANNELS = ("silent", "live")
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

# A compact observation includes private and public fields in a fixed layout.
# PI masks partner goal, remote site type and remote inventory with zeroes.
OBS_DIM = 2 + 2 + 1 + 3 + HORIZON + 2 + 2 + 1 + 1
RECV_DIM = ALPHABET_SIZE + 1

# Six message/action checkpoints are not needed: episodes are short and the
# channel closes after round 1.  These seeds are used for independent evaluation.
EVAL_SEED_OFFSET = 900000


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def parse_condition(condition: str):
    # condition format memory_scarcity_information_channel
    parts = condition.split("_")
    require(len(parts) == 4, f"Bad condition: {condition}")
    memory, scarcity, information, channel = parts
    require(memory in MEMORIES and scarcity in SCARCITIES
            and information in INFORMATIONS and channel in CHANNELS,
            f"Unknown condition: {condition}")
    return memory, scarcity, information, channel


CONDITIONS = tuple(
    f"{memory}_{scarcity}_{information}_{channel}"
    for memory in MEMORIES for scarcity in SCARCITIES
    for information in INFORMATIONS for channel in CHANNELS
)


def capacity(scarcity: str) -> int:
    require(scarcity in SCARCITIES, "Unknown scarcity")
    return 1 if scarcity == "scarce" else 3


def entropy_coefficient(update: int) -> float:
    require(1 <= update <= UPDATES, "Invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (update - 1) / ENTROPY_ZERO_AFTER)


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def episode_stream(seed: int, scarcity: str, count: int, *, evaluation=False, update: int = 0):
    """Generate paired exogenous episodes and score uniforms.

    Each site has an independent binary material type.  Each agent receives a
    private five-round goal sequence.  The sequence is
    sampled from a fixed support with both alternating and persistent demands;
    the held-out evaluation support uses the complementary transitions.
    """
    require(count > 0, "count must be positive")
    # Scarcity is deliberately excluded from the RNG key.  This makes scarce
    # and abundant arms share exactly the same worlds and score uniforms; only
    # the inventory capacity changes.  The evaluation bit is kept in the key
    # so held-out episodes cannot accidentally overlap training episodes.
    rng = np.random.default_rng(np.random.SeedSequence([
        seed, EVAL_SEED_OFFSET if evaluation else 0, int(update)]))
    # Site types are sampled independently.  With two complementary sites the
    # remote type would be inferable from the local type, making the PI/FI
    # manipulation illusory.  Independent patches preserve genuine private
    # information and also create episodes in which a demanded material is
    # absent or duplicated.
    site_type = rng.integers(0, 2, size=(count, 2), dtype=np.int8)
    # 0/1 are resource kinds.  Training and evaluation use the same marginal,
    # while evaluation reverses the transition-pattern subset.
    patterns = np.array([
        [0, 0, 1, 1, 0], [1, 1, 0, 0, 1],
        [0, 1, 0, 1, 0], [1, 0, 1, 0, 1],
        [0, 0, 0, 1, 1], [1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0], [1, 0, 0, 1, 1],
    ], dtype=np.int8)
    # Distinct pattern halves create a small sequence holdout without changing
    # the vocabulary or object categories.
    choices = np.arange(8)
    if evaluation:
        choices = choices[[2, 3, 6, 7]]
    else:
        choices = choices[[0, 1, 4, 5]]
    goal_a = patterns[rng.choice(choices, size=count)]
    # B has an independently sampled sequence, with a balanced marginal.
    goal_b = patterns[rng.choice(choices, size=count)]
    # Score uniforms are paired across live and silent arms.  One message and
    # one action uniform per agent and round; inactive post-blackout messages
    # are still consumed to keep streams aligned.
    message_u = rng.random((count, HORIZON, 2), dtype=np.float64)
    action_u = rng.random((count, HORIZON, 2), dtype=np.float64)
    return {
        "site_type": site_type,
        # Keep the first-site projection for backwards-compatible summaries;
        # new code should hash and consume the full site_type array.
        "site0_type": site_type[:, 0].copy(),
        "goal": np.stack([goal_a, goal_b], axis=1),
        "message_uniforms": message_u,
        "action_uniforms": action_u,
        "capacity": np.full(count, capacity(scarcity), dtype=np.int8),
        "seed": int(seed), "scarcity": scarcity, "evaluation": bool(evaluation), "update": int(update),
    }


def encode_observation(*, own_goal: int, local_type: int, local_inventory: int,
                        last_result: int, time: int, partner_goal: int,
                        remote_type: int, remote_inventory: int, information: str) -> np.ndarray:
    """Encode the official observation; no hidden state or reward enters PI."""
    require(own_goal in (0, 1) and local_type in (0, 1), "Bad resource id")
    require(last_result in (0, 1, 2) and 0 <= time < HORIZON, "Bad observation field")
    x = np.zeros(OBS_DIM, dtype=np.float64); k = 0
    x[own_goal] = 1.; k += 2
    x[k + local_type] = 1.; k += 2
    x[k] = local_inventory / 3.; k += 1
    x[k + last_result] = 1.; k += 3
    x[k + time] = 1.; k += HORIZON
    if information == "FI":
        x[k + partner_goal] = 1.
    k += 2
    if information == "FI":
        x[k + remote_type] = 1.
    k += 2
    if information == "FI":
        x[k] = remote_inventory / 3.
    k += 1
    x[k] = 1. if information == "FI" else 0.;
    require(k + 1 == OBS_DIM, "Observation layout mismatch")
    return x


def prepare() -> dict:
    """Return a JSON-serializable frozen design record."""
    patterns_train = [[0, 0, 1, 1, 0], [1, 1, 0, 0, 1],
                      [0, 0, 0, 1, 1], [1, 1, 1, 0, 0]]
    patterns_held = [[0, 1, 0, 1, 0], [1, 0, 1, 0, 1],
                     [0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]
    return {
        "schema": "temporal_scarcity_communication_v1",
        "horizon": HORIZON, "message_rounds": list(MESSAGE_ROUNDS),
        "blackout_rounds": [t for t in range(HORIZON) if t not in MESSAGE_ROUNDS],
        "alphabet_size": ALPHABET_SIZE, "null_message": NULL_MESSAGE,
        "action_names": ["wait", "take_site0", "take_site1"],
        "seeds": list(SEEDS), "conditions": list(CONDITIONS),
        "updates": UPDATES, "batch_size": BATCH_SIZE, "hidden": HIDDEN,
        "learning_rate": LEARNING_RATE, "gradient_clip": GRADIENT_CLIP,
        "entropy_initial": ENTROPY_INITIAL,
        "entropy_zero_after": ENTROPY_ZERO_AFTER,
        "observation_dim": OBS_DIM, "receive_dim": RECV_DIM,
        "train_goal_patterns": patterns_train,
        "heldout_goal_patterns": patterns_held,
        "resource_regimes": {"scarce": 1, "abundant": 3},
        "private_information": "own goal sequence and assigned site's current type/inventory; partner goal, remote site type and remote inventory are masked in PI",
        "full_information": "both goal sequences and both sites are visible in FI",
        "channel": "one eight-way token at rounds 0 and 1; receiver input is null after round 1",
        "reward": "per-step +1 for matching an unfilled private goal with available material; -0.25 for a wrong/depleted take; wait 0; team return is mean over agents and five rounds",
        "simultaneous_resolution": "same-site requests share capacity; if one unit remains, lower-index agent receives it",
        "network": "agent-specific tanh recurrent core (or no recurrent edge), message 8-way head and action 3-way head",
        "no_teacher_or_language_prior": True,
        "pairing": "same initial parameters, episode worlds, message uniforms and action uniforms within seed/memory/scarcity/information block",
        "evaluation": "training-support and held-out goal transition patterns, each with greedy natural/closed/permuted message controls",
        "automatic_followon_experiment": False,
    }
