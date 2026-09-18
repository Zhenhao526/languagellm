"""Frozen design for a learned-routing factor-sharing test."""
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
MESSAGE_STATE_COUNT = ALPHABET_SIZE ** MESSAGE_LENGTH + 1
COMMUNITIES = 2
PARTNERS_PER_COMMUNITY = 2
PARTNERS = COMMUNITIES * PARTNERS_PER_COMMUNITY
ARCHITECTURES = ("holistic", "factorized", "routed")
POPULATIONS = ("aligned", "conflict")
VISIBILITIES = ("hidden", "visible")
SUPPORTS = ("full", "heldout_combo", "heldout_value")
ROLES = ("alternating",)
CHANNELS = ("live", "silent")
TASK = "referential_selection"
FORM = "quad2"
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
NULL_MESSAGE = -1
# Fresh seed block: streams are independent of the earlier two-arm archive.
SEEDS = tuple(range(78701, 78710))
UPDATES = 3000
PARENT_UPDATES = UPDATES
CHILD_UPDATES = UPDATES
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
EVAL_SEED_OFFSET = 986000
STREAM_SALT = 986001
POLICY_SALT = 986002

PARENT_CONDITIONS = tuple(f"{architecture}_{population}" for architecture in ARCHITECTURES for population in POPULATIONS)
CHILD_CONDITIONS = tuple(
    f"{architecture}_{population}_{visibility}_{support}"
    for architecture in ARCHITECTURES
    for population in POPULATIONS
    for visibility in VISIBILITIES
    for support in SUPPORTS
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_parent_condition(condition):
    parts = condition.split("_")
    require(len(parts) == 2 and parts[0] in ARCHITECTURES and parts[1] in POPULATIONS, f"bad parent condition {condition}")
    return parts[0], parts[1]


def parse_child_condition(condition):
    parts = condition.split("_", 3)
    require(len(parts) == 4, f"bad child condition {condition}")
    architecture, population, visibility, support = parts
    require(architecture in ARCHITECTURES and population in POPULATIONS and visibility in VISIBILITIES and support in SUPPORTS, f"bad child condition {condition}")
    return architecture, population, visibility, support


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
    return np.asarray((1, 0, 3, 2), dtype=np.int8)


def inverse_permutation(permutation):
    p = np.asarray(permutation, dtype=np.int64)
    out = np.empty_like(p)
    out[p] = np.arange(len(p), dtype=np.int64)
    return out.astype(np.int8)


def message_index(message):
    m = np.asarray(message, dtype=np.int64)
    if m.ndim == 1:
        if np.any(m < 0): return MESSAGE_STATE_COUNT - 1
        return int(m[0] * ALPHABET_SIZE + m[1])
    out = m[..., 0] * ALPHABET_SIZE + m[..., 1]
    return np.where(np.any(m < 0, axis=-1), MESSAGE_STATE_COUNT - 1, out).astype(np.int64)


def decode_message_state(state):
    s = np.asarray(state, dtype=np.int64)
    token0 = np.where(s < ALPHABET_SIZE ** MESSAGE_LENGTH, s // ALPHABET_SIZE, -1)
    token1 = np.where(s < ALPHABET_SIZE ** MESSAGE_LENGTH, s % ALPHABET_SIZE, -1)
    return np.stack([token0, token1], axis=-1).astype(np.int64)


def entropy_coefficient(update):
    require(1 <= int(update) <= UPDATES, "invalid update")
    return ENTROPY_INITIAL * max(0.0, 1.0 - (int(update) - 1) / ENTROPY_ZERO_AFTER)


def array_sha(x):
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def _scene_stream(rng, count):
    scene = np.empty((count, SCENE_SIZE), dtype=np.int64)
    for i in range(count):
        meanings = [factor * VALUES + int(rng.integers(VALUES)) for factor in range(VALUES)]
        remaining = np.asarray([m for m in range(MEANINGS) if m not in meanings], dtype=np.int64)
        meanings.append(int(rng.choice(remaining)))
        row = np.asarray(meanings, dtype=np.int64)
        rng.shuffle(row)
        scene[i] = row
    return scene


def _target_positions(scene, uniforms, support, combo, value):
    meanings = np.asarray(scene, dtype=np.int64)
    if support == "full": allowed = np.ones_like(meanings, dtype=bool)
    elif support == "heldout_combo": allowed = meanings != int(combo)
    else: allowed = (meanings // VALUES) != int(value)
    positions = np.empty(len(scene), dtype=np.int8)
    for i, row in enumerate(allowed):
        candidates = np.flatnonzero(row)
        require(len(candidates) > 0, "support produced an empty target set")
        positions[i] = candidates[min(int(float(uniforms[i]) * len(candidates)), len(candidates) - 1)]
    return positions


def _child_roles(count, seed, update):
    return ((np.arange(count, dtype=np.int64) // PARTNERS + int(seed) + int(update)) % 2).astype(np.int8)


def episode_stream(seed, count, *, support="full", update=0, evaluation=False):
    require(count > 0 and support in SUPPORTS, "bad episode factors")
    require(int(seed) in SEEDS or evaluation, "unknown seed")
    combo = heldout_combo(seed) if int(seed) in SEEDS else 0
    value = heldout_value(seed) if int(seed) in SEEDS else 0
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update), STREAM_SALT]))
    scene_meanings = _scene_stream(rng, count)
    target_uniform = rng.random(count)
    target_index = _target_positions(scene_meanings, target_uniform, support, combo, value)
    goal_meaning = scene_meanings[np.arange(count), target_index]
    partner_id = (np.arange(count, dtype=np.int64) % PARTNERS).astype(np.int8)
    return {
        "scene_meanings": scene_meanings,
        "scene": attrs_from_meaning(scene_meanings),
        "target_index": target_index,
        "goal": attrs_from_meaning(goal_meaning),
        "goal_meaning": goal_meaning.astype(np.int8),
        "target_uniform": target_uniform,
        "partner_id": partner_id,
        "community_id": (partner_id // PARTNERS_PER_COMMUNITY).astype(np.int8),
        "fresh_role": _child_roles(count, seed, update),
        "message_uniforms": rng.random((count, MESSAGE_LENGTH)),
        "action_uniforms": rng.random(count),
        "support": support,
        "heldout_combo": combo,
        "heldout_value": value,
        "evaluation": bool(evaluation),
        "update": int(update),
    }


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


def balanced_eval_stream(seed, count, goal_kind, support, combo, value):
    require(goal_kind in ("all", "seen", "heldout_combo", "heldout_value"), "bad goal kind")
    require(support in SUPPORTS and count > 0, "bad eval factors")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), EVAL_SEED_OFFSET + 17, int(combo), int(value), STREAM_SALT]))
    if goal_kind == "all": meanings = np.arange(count, dtype=np.int64) % MEANINGS
    elif goal_kind == "heldout_combo": meanings = np.full(count, int(combo), dtype=np.int64)
    elif goal_kind == "heldout_value": meanings = int(value) * VALUES + (np.arange(count, dtype=np.int64) % VALUES)
    else:
        choices = np.arange(MEANINGS, dtype=np.int64)
        if support == "heldout_combo": choices = choices[choices != int(combo)]
        elif support == "heldout_value": choices = choices[(choices // VALUES) != int(value)]
        meanings = choices[np.arange(count, dtype=np.int64) % len(choices)]
    scene_meanings, target_index = _evaluation_scenes(rng, meanings)
    partner_id = (np.arange(count, dtype=np.int64) % PARTNERS).astype(np.int8)
    return {
        "scene_meanings": scene_meanings,
        "scene": attrs_from_meaning(scene_meanings),
        "target_index": target_index,
        "goal": attrs_from_meaning(meanings),
        "goal_meaning": meanings.astype(np.int8),
        "partner_id": partner_id,
        "community_id": (partner_id // PARTNERS_PER_COMMUNITY).astype(np.int8),
        "fresh_role": _child_roles(count, int(seed), 0),
        "message_uniforms": rng.random((count, MESSAGE_LENGTH)),
        "action_uniforms": rng.random(count),
        "support": support,
        "heldout_combo": int(combo),
        "heldout_value": int(value),
        "evaluation": True,
        "update": 0,
    }


def prepare():
    return {
        "schema":"routed_factorization_study_v1",
        "task":TASK,
        "form":{"name":FORM,"alphabet_size":ALPHABET_SIZE,"length":MESSAGE_LENGTH,"message_state_count":MESSAGE_STATE_COUNT},
        "architectures":list(ARCHITECTURES),
        "architecture_intervention":"holistic stores one sender/receiver row per complete meaning/message; factorized fixes slot j to attribute j; routed shares atomic factor tables but learns a soft slot-to-attribute routing matrix",
        "attributes":ATTRIBUTES,"values_per_attribute":VALUES,"meanings":MEANINGS,"scene_size":SCENE_SIZE,
        "communities":COMMUNITIES,"partners_per_community":PARTNERS_PER_COMMUNITY,"partners":PARTNERS,
        "populations":list(POPULATIONS),"visibilities":list(VISIBILITIES),"supports":list(SUPPORTS),"roles":list(ROLES),"channels":list(CHANNELS),
        "surface_rule":"aligned identity in both communities; conflict swaps (0,1) and (2,3) in community 1",
        "role_rule":"alternating balances fresh sender and fresh receiver for all four partners",
        "message_symbols":["@","#","￥","%"],"goal_pairs":[list(x) for x in MEANING_PAIRS],"seeds":list(SEEDS),
        "parent_conditions":list(PARENT_CONDITIONS),"child_conditions":list(CHILD_CONDITIONS),
        "updates":UPDATES,"batch_size":BATCH_SIZE,"learning_rate":LEARNING_RATE,"entropy_initial":ENTROPY_INITIAL,"checkpoints":list(CHECKPOINTS),
        "heldout_combo_assignment":{str(s):heldout_combo(s) for s in SEEDS},"heldout_value_assignment":{str(s):heldout_value(s) for s in SEEDS},
        "primary_readouts":["architecture effect on heldout combination/value return","live-minus-silent and natural-minus-permuted","cross-partner code consistency","slot-wise factor alignment and community conflict cost","routing alignment up to slot permutation"],
        "functional_rule":"held-out natural return >= 0.60","no_teacher_or_language_prior":True,
        "parent_runs":len(SEEDS)*len(PARENT_CONDITIONS),"child_runs":len(SEEDS)*len(CHILD_CONDITIONS),"runs":len(SEEDS)*(len(PARENT_CONDITIONS)+len(CHILD_CONDITIONS)),
        "runtime_policy":"float64 NumPy only",
    }
