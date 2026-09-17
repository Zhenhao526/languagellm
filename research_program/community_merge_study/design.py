"""Frozen design for community codebook conflict and role symmetry."""
from __future__ import annotations

import hashlib
import numpy as np

ATTRIBUTES = 2
VALUES = 3
MEANINGS = VALUES ** ATTRIBUTES
MEANING_PAIRS = tuple((a, b) for a in range(VALUES) for b in range(VALUES))
SCENE_SIZE = 4
ALPHABET_SIZE = 4
MESSAGE_LENGTH = 2
MESSAGE_STATE_COUNT = ALPHABET_SIZE ** MESSAGE_LENGTH + 1  # final row is silence
COMMUNITIES = 2
PARTNERS_PER_COMMUNITY = 2
PARTNERS = COMMUNITIES * PARTNERS_PER_COMMUNITY
POPULATIONS = ("aligned", "conflict")
VISIBILITIES = ("hidden", "visible")
ADAPTATIONS = ("fresh_only", "coadapt")
SUPPORTS = ("full", "heldout_combo", "heldout_value")
ROLES = ("alternating", "sender_only")
CHANNELS = ("live", "silent")
TASK = "referential_selection"
FORM = "quad2"
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
NULL_MESSAGE = -1
SEEDS = tuple(range(78501, 78510))
UPDATES = 3000
PARENT_UPDATES = UPDATES
CHILD_UPDATES = UPDATES
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
EVAL_SEED_OFFSET = 985000
STREAM_SALT = 985001
POLICY_SALT = 985002
PARTNER_SALT = 985003

PARENT_CONDITIONS = POPULATIONS
CHILD_CONDITIONS = tuple(
    f"{population}_{visibility}_{adaptation}_{support}_{role}"
    for population in POPULATIONS
    for visibility in VISIBILITIES
    for adaptation in ADAPTATIONS
    for support in SUPPORTS
    for role in ROLES
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_parent_condition(condition):
    require(condition in PARENT_CONDITIONS, f"bad parent condition {condition}")
    return condition


def parse_child_condition(condition):
    parts = condition.split("_")
    require(len(parts) >= 5, f"bad child condition {condition}")
    if parts[-2:] == ["sender", "only"]:
        role = "sender_only"
        core = parts[:-2]
    elif parts[-1] == "alternating":
        role = "alternating"
        core = parts[:-1]
    else:
        raise ValueError(f"bad child condition {condition}")
    population, visibility = core[:2]
    if core[2:4] == ["fresh", "only"]:
        adaptation = "fresh_only"
        cursor = 4
    elif core[2] == "coadapt":
        adaptation = "coadapt"
        cursor = 3
    else:
        raise ValueError(f"bad child condition {condition}")
    if core[cursor:cursor + 2] == ["heldout", "combo"]:
        support = "heldout_combo"
        cursor += 2
    elif core[cursor:cursor + 2] == ["heldout", "value"]:
        support = "heldout_value"
        cursor += 2
    elif core[cursor:cursor + 1] == ["full"]:
        support = "full"
        cursor += 1
    else:
        raise ValueError(f"bad child condition {condition}")
    require(cursor == len(core), f"bad child condition {condition}")
    require(
        population in POPULATIONS
        and visibility in VISIBILITIES
        and adaptation in ADAPTATIONS
        and support in SUPPORTS
        and role in ROLES,
        f"bad child condition {condition}",
    )
    return population, visibility, adaptation, support, role


def goal_index(goal):
    g = np.asarray(goal, dtype=np.int64)
    return g[..., 0] * VALUES + g[..., 1]


def attrs_from_meaning(index):
    i = np.asarray(index, dtype=np.int64)
    return np.stack([i // VALUES, i % VALUES], axis=-1).astype(np.int8)


def heldout_combo(seed):
    require(int(seed) in SEEDS, "unknown seed")
    return int((int(seed) - SEEDS[0]) % MEANINGS)


def heldout_value(seed):
    require(int(seed) in SEEDS, "unknown seed")
    return int((int(seed) - SEEDS[0]) % VALUES)


def surface_permutation(population, community):
    require(population in POPULATIONS and 0 <= int(community) < COMMUNITIES, "bad surface factors")
    if population == "aligned" or int(community) == 0:
        return np.arange(ALPHABET_SIZE, dtype=np.int8)
    # Two independent pair swaps create a visible but bijective community code.
    return np.asarray((1, 0, 3, 2), dtype=np.int8)


def inverse_permutation(permutation):
    p = np.asarray(permutation, dtype=np.int64)
    out = np.empty_like(p)
    out[p] = np.arange(len(p), dtype=np.int64)
    return out.astype(np.int8)


def message_index(message):
    m = np.asarray(message, dtype=np.int64)
    if m.ndim == 1:
        if np.any(m < 0):
            return MESSAGE_STATE_COUNT - 1
        return int(m[0] * ALPHABET_SIZE + m[1])
    out = m[..., 0] * ALPHABET_SIZE + m[..., 1]
    return np.where(np.any(m < 0, axis=-1), MESSAGE_STATE_COUNT - 1, out).astype(np.int64)


def entropy_coefficient(update):
    require(1 <= int(update) <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (int(update) - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def _scene_stream(rng, count):
    """Generate distinct four-object scenes while covering every first factor."""
    scene = np.empty((count, SCENE_SIZE), dtype=np.int64)
    for i in range(count):
        meanings = [factor * VALUES + int(rng.integers(VALUES)) for factor in range(VALUES)]
        remaining = np.asarray([m for m in range(MEANINGS) if m not in meanings], dtype=np.int64)
        meanings.append(int(rng.choice(remaining)))
        values = np.asarray(meanings, dtype=np.int64)
        rng.shuffle(values)
        scene[i] = values
    return scene


def _target_positions(scene, uniforms, support, combo, value):
    meanings = np.asarray(scene, dtype=np.int64)
    if support == "full":
        allowed = np.ones_like(meanings, dtype=bool)
    elif support == "heldout_combo":
        allowed = meanings != int(combo)
    else:
        allowed = (meanings // VALUES) != int(value)
    positions = np.empty(len(scene), dtype=np.int8)
    for i, row in enumerate(allowed):
        candidates = np.flatnonzero(row)
        require(len(candidates) > 0, "support produced an empty target set")
        positions[i] = candidates[min(int(float(uniforms[i]) * len(candidates)), len(candidates) - 1)]
    return positions


def _child_roles(count, seed, update, role):
    if role == "sender_only":
        return np.zeros(count, dtype=np.int8)
    # Role 0 = fresh sender, role 1 = fresh receiver.  Blocks contain both
    # roles for every partner, making the role contrast paired and balanced.
    return ((np.arange(count, dtype=np.int64) // PARTNERS + int(seed) + int(update)) % 2).astype(np.int8)


def episode_stream(
    seed,
    count,
    *,
    support="full",
    role="alternating",
    heldout_combo=None,
    heldout_value_=None,
    evaluation=False,
    update=0,
):
    require(count > 0 and support in SUPPORTS and role in ROLES, "bad episode factors")
    require(int(seed) in SEEDS or evaluation, "unknown seed")
    combo = heldout_combo if heldout_combo is not None else (heldout_combo_fn(seed) if int(seed) in SEEDS else 0)
    value = heldout_value_ if heldout_value_ is not None else (heldout_value(seed) if int(seed) in SEEDS else 0)
    require(0 <= int(combo) < MEANINGS and 0 <= int(value) < VALUES, "bad held-out factors")
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update), STREAM_SALT])
    )
    scene_meanings = _scene_stream(rng, count)
    target_uniform = rng.random(count)
    target_index = _target_positions(scene_meanings, target_uniform, support, combo, value)
    goal_meaning = scene_meanings[np.arange(count), target_index]
    partner_id = (np.arange(count, dtype=np.int64) % PARTNERS).astype(np.int8)
    fresh_role = _child_roles(count, seed, update, role)
    return {
        "scene_meanings": scene_meanings,
        "scene": attrs_from_meaning(scene_meanings),
        "target_index": target_index,
        "goal": attrs_from_meaning(goal_meaning),
        "goal_meaning": goal_meaning.astype(np.int8),
        "target_uniform": target_uniform,
        "partner_id": partner_id,
        "community_id": (partner_id // PARTNERS_PER_COMMUNITY).astype(np.int8),
        "fresh_role": fresh_role,
        "message_uniforms": rng.random((count, MESSAGE_LENGTH)),
        "action_uniforms": rng.random(count),
        "support": support,
        "role": role,
        "heldout_combo": int(combo),
        "heldout_value": int(value),
        "evaluation": bool(evaluation),
        "update": int(update),
    }


def heldout_combo_fn(seed):
    return heldout_combo(seed)


def _evaluation_scenes(rng, target_meanings):
    out = np.empty((len(target_meanings), SCENE_SIZE), dtype=np.int64)
    target_index = np.empty(len(target_meanings), dtype=np.int8)
    for i, target in enumerate(np.asarray(target_meanings, dtype=np.int64)):
        others = np.asarray([m for m in range(MEANINGS) if m != int(target)], dtype=np.int64)
        distractors = rng.choice(others, size=SCENE_SIZE - 1, replace=False)
        row = np.asarray([int(target), *[int(x) for x in distractors]], dtype=np.int64)
        rng.shuffle(row)
        out[i] = row
        target_index[i] = int(np.flatnonzero(row == int(target))[0])
    return out, target_index


def balanced_eval_stream(seed, count, goal_kind, support, combo, value, role="alternating"):
    require(goal_kind in ("all", "seen", "heldout_combo", "heldout_value"), "bad goal kind")
    require(support in SUPPORTS and role in ROLES, "bad eval factors")
    require(count > 0, "bad eval count")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET + 17, int(combo), int(value), STREAM_SALT]))
    if goal_kind == "all":
        meanings = np.arange(count, dtype=np.int64) % MEANINGS
    elif goal_kind == "heldout_combo":
        meanings = np.full(count, int(combo), dtype=np.int64)
    elif goal_kind == "heldout_value":
        meanings = int(value) * VALUES + (np.arange(count, dtype=np.int64) % VALUES)
    else:
        choices = np.arange(MEANINGS, dtype=np.int64)
        if support == "heldout_combo":
            choices = choices[choices != int(combo)]
        elif support == "heldout_value":
            choices = choices[(choices // VALUES) != int(value)]
        meanings = choices[np.arange(count, dtype=np.int64) % len(choices)]
    scene_meanings, target_index = _evaluation_scenes(rng, meanings)
    partner_id = (np.arange(count, dtype=np.int64) % PARTNERS).astype(np.int8)
    fresh_role = _child_roles(count, seed, 0, role)
    return {
        "scene_meanings": scene_meanings,
        "scene": attrs_from_meaning(scene_meanings),
        "target_index": target_index,
        "goal": attrs_from_meaning(meanings),
        "goal_meaning": meanings.astype(np.int8),
        "target_uniform": np.zeros(count, dtype=np.float64),
        "partner_id": partner_id,
        "community_id": (partner_id // PARTNERS_PER_COMMUNITY).astype(np.int8),
        "fresh_role": fresh_role,
        "message_uniforms": rng.random((count, MESSAGE_LENGTH)),
        "action_uniforms": rng.random(count),
        "support": "evaluation",
        "role": role,
        "heldout_combo": int(combo),
        "heldout_value": int(value),
        "evaluation": True,
        "update": 0,
    }


def prepare():
    return {
        "schema": "community_merge_study_v1",
        "task": TASK,
        "form": {"name": FORM, "alphabet_size": ALPHABET_SIZE, "length": MESSAGE_LENGTH, "message_state_count": MESSAGE_STATE_COUNT},
        "attributes": ATTRIBUTES,
        "values_per_attribute": VALUES,
        "meanings": MEANINGS,
        "scene_size": SCENE_SIZE,
        "communities": COMMUNITIES,
        "partners_per_community": PARTNERS_PER_COMMUNITY,
        "partners": PARTNERS,
        "populations": list(POPULATIONS),
        "visibilities": list(VISIBILITIES),
        "adaptations": list(ADAPTATIONS),
        "supports": list(SUPPORTS),
        "roles": list(ROLES),
        "channels": list(CHANNELS),
        "surface_rule": "aligned uses identity in both communities; conflict swaps (0,1) and (2,3) in community 1",
        "role_rule": "alternating balances fresh sender and fresh receiver for every partner; sender_only fixes fresh as sender",
        "partner_exposure": "every alternating child block contains both roles for all four partners",
        "message_symbols": ["@", "#", "￥", "%"],
        "goal_pairs": [list(x) for x in MEANING_PAIRS],
        "seeds": list(SEEDS),
        "parent_conditions": list(PARENT_CONDITIONS),
        "child_conditions": list(CHILD_CONDITIONS),
        "updates": UPDATES,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "entropy_initial": ENTROPY_INITIAL,
        "checkpoints": list(CHECKPOINTS),
        "heldout_combo_assignment": {str(seed): heldout_combo(seed) for seed in SEEDS},
        "heldout_value_assignment": {str(seed): heldout_value(seed) for seed in SEEDS},
        "primary_readouts": [
            "heldout_combo and heldout_value referential return",
            "role-conditioned live-minus-silent message effect",
            "fresh sender cross-partner code consistency",
            "slot recombination and community codebook convergence",
        ],
        "functional_rule": "held-out natural return >= 0.60",
        "no_teacher_or_language_prior": True,
        "parent_runs": len(SEEDS) * len(PARENT_CONDITIONS) * COMMUNITIES,
        "child_runs": len(SEEDS) * len(CHILD_CONDITIONS),
        "runs": len(SEEDS) * (len(PARENT_CONDITIONS) * COMMUNITIES + len(CHILD_CONDITIONS)),
        "runtime_policy": "float64 NumPy only",
    }
