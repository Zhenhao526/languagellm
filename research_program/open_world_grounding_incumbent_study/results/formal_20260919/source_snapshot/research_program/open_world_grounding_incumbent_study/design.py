"""Open-world cultural transmission with incumbent receiver grounding.

The first stage forms a single-new extension with a fresh sender and receiver
while the parent protocol remains frozen. The second stage freezes that
extension and replaces one role at a time. A receiver replacement is either
randomly initialized or copies only the incumbent receiver parameters. The
fresh receiver can see either canonical object surfaces or a stable reversal
of all attribute labels. The double-new meaning is absent from every training
support, so transfer is tested as zero-shot composition after cultural
learning rather than as inheritance of parameters.
"""
from __future__ import annotations

import hashlib

import numpy as np

ATTRIBUTES = 2
VALUES = 4
OLD_VALUES = 3
NEW_VALUE = 3
MEANINGS = VALUES ** ATTRIBUTES
OLD_MEANINGS = OLD_VALUES ** ATTRIBUTES
OLD_MEANING_INDICES = tuple(a * VALUES + b for a in range(OLD_VALUES) for b in range(OLD_VALUES))
MEANING_PAIRS = tuple((a, b) for a in range(VALUES) for b in range(VALUES))
SCENE_SIZE = 4
ALPHABET_SIZE = 4
MESSAGE_LENGTH = 2
MESSAGE_STATE_COUNT = ALPHABET_SIZE ** MESSAGE_LENGTH + 1
COMMUNITIES = 2
PARTNERS_PER_COMMUNITY = 2
PARTNERS = COMMUNITIES * PARTNERS_PER_COMMUNITY
ARCHITECTURES = ("holistic", "factorized")
POPULATIONS = ("aligned",)
VISIBILITIES = ("hidden",)
SUPPORTS = ("expanded_single_pair",)
TRANSFER_SUPPORTS = ("expanded_single_sender", "expanded_single_receiver",
                     "expanded_single_pair", "expanded_single_none")
OBJECT_SURFACE_MAPPINGS = ("identity", "reverse")
RECEIVER_INITIALIZATIONS = ("fresh", "incumbent_receiver")
ALL_SUPPORTS = ("old_combo",) + tuple(dict.fromkeys(SUPPORTS + TRANSFER_SUPPORTS))
EVAL_KINDS = ("old_seen", "old_combo", "new_single", "new_double", "all")
ROLES = ("alternating", "pair")
CHANNELS = ("live", "silent")
TASK = "open_world_referential_selection"
FORM = "quad2_open4"
CORRECT_REWARD = 1.0
WRONG_REWARD = -0.25
NULL_MESSAGE = -1

SEEDS = tuple(range(88001, 88010))
UPDATES = 3000
PARENT_UPDATES = UPDATES
CHILD_UPDATES = UPDATES
BATCH_SIZE = 512
LEARNING_RATE = 0.08
ROUTE_INIT_SCALE = 0.50
ROUTE_TEMPERATURE = 5.0
ENTROPY_INITIAL = 0.01
ENTROPY_ZERO_AFTER = 5000
CHECKPOINTS = (0, 500, 1000, 2000, 3000)
EVAL_SEED_OFFSET = 987000
STREAM_SALT = 987001
POLICY_SALT = 987002

PARENT_CONDITIONS = tuple(f"{architecture}_{population}"
                          for architecture in ARCHITECTURES
                          for population in POPULATIONS)
CHILD_CONDITIONS = tuple(
    f"{architecture}_{population}_{visibility}_{support}"
    for architecture in ARCHITECTURES
    for population in POPULATIONS
    for visibility in VISIBILITIES
    for support in SUPPORTS
)
TRANSFER_CONDITIONS = tuple(
    f"{architecture}_{population}_{visibility}_{mapping}_{initialization}_{support}"
    for architecture in ARCHITECTURES
    for population in POPULATIONS
    for visibility in VISIBILITIES
    for mapping in OBJECT_SURFACE_MAPPINGS
    for initialization in RECEIVER_INITIALIZATIONS
    for support in TRANSFER_SUPPORTS
)


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def parse_parent_condition(condition):
    for architecture in sorted(ARCHITECTURES, key=len, reverse=True):
        prefix = architecture + "_"
        if condition.startswith(prefix):
            population = condition[len(prefix):]
            require(population in POPULATIONS, f"bad parent condition {condition}")
            return architecture, population
    raise ValueError(f"bad parent condition {condition}")


def parse_child_condition(condition):
    architecture = next((a for a in sorted(ARCHITECTURES, key=len, reverse=True)
                         if condition.startswith(a + "_")), None)
    require(architecture is not None, f"bad child condition {condition}")
    parts = condition[len(architecture) + 1:].split("_", 2)
    require(len(parts) == 3, f"bad child condition {condition}")
    population, visibility, support = parts
    require(population in POPULATIONS and visibility in VISIBILITIES and support in SUPPORTS,
            f"bad child condition {condition}")
    return architecture, population, visibility, support


def parse_transfer_condition(condition):
    architecture = next((a for a in sorted(ARCHITECTURES, key=len, reverse=True)
                         if condition.startswith(a + "_")), None)
    require(architecture is not None, f"bad transfer condition {condition}")
    parts = condition[len(architecture) + 1:].split("_", 3)
    require(len(parts) == 4, f"bad transfer condition {condition}")
    population, visibility, mapping, remainder = parts
    initialization = next((item for item in sorted(RECEIVER_INITIALIZATIONS, key=len, reverse=True)
                           if remainder.startswith(item + "_")), None)
    require(initialization is not None, f"bad transfer condition {condition}")
    support = remainder[len(initialization) + 1:]
    require(population in POPULATIONS and visibility in VISIBILITIES
            and mapping in OBJECT_SURFACE_MAPPINGS
            and initialization in RECEIVER_INITIALIZATIONS
            and support in TRANSFER_SUPPORTS,
            f"bad transfer condition {condition}")
    return architecture, population, visibility, mapping, initialization, support


def attrs_from_meaning(index):
    i = np.asarray(index, dtype=np.int64)
    return np.stack([i // VALUES, i % VALUES], axis=-1).astype(np.int8)


def meaning_index(attrs):
    a = np.asarray(attrs, dtype=np.int64)
    return a[..., 0] * VALUES + a[..., 1]


def is_new_meaning(index):
    attrs = attrs_from_meaning(index)
    return np.any(attrs == NEW_VALUE, axis=-1)


def is_new_single(index):
    attrs = attrs_from_meaning(index)
    return np.sum(attrs == NEW_VALUE, axis=-1) == 1


def heldout_combo(seed):
    require(int(seed) in SEEDS, "unknown seed")
    return int(OLD_MEANING_INDICES[(int(seed) - SEEDS[0]) % OLD_MEANINGS])


def surface_permutation(population, community):
    require(population in POPULATIONS and 0 <= int(community) < COMMUNITIES,
            "bad surface factors")
    return np.arange(ALPHABET_SIZE, dtype=np.int8)


def inverse_permutation(permutation):
    p = np.asarray(permutation, dtype=np.int64)
    out = np.empty_like(p)
    out[p] = np.arange(len(p), dtype=np.int64)
    return out.astype(np.int8)


def object_surface_permutation(mapping):
    require(mapping in OBJECT_SURFACE_MAPPINGS, "bad object surface mapping")
    if mapping == "identity":
        return np.arange(VALUES, dtype=np.int8)
    # A stable bijection changes the visible label of every value while
    # preserving enough information for reward-based re-alignment.
    return np.asarray([NEW_VALUE, 2, 1, 0], dtype=np.int8)


def surface_scene_meanings(scene_meanings, mapping):
    scene = attrs_from_meaning(scene_meanings).astype(np.int64)
    scene = object_surface_permutation(mapping)[scene]
    return meaning_index(scene).astype(np.int64)


def message_index(message):
    m = np.asarray(message, dtype=np.int64)
    if m.ndim == 1:
        if np.any(m < 0):
            return MESSAGE_STATE_COUNT - 1
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


def _scene_stream(rng, count, expanded):
    universe = (np.arange(MEANINGS, dtype=np.int64) if expanded
                else np.asarray(OLD_MEANING_INDICES, dtype=np.int64))
    scene = np.empty((count, SCENE_SIZE), dtype=np.int64)
    for i in range(count):
        row = rng.choice(universe, size=SCENE_SIZE, replace=False)
        rng.shuffle(row)
        scene[i] = row
    return scene


def _target_positions(scene, uniforms, support, combo):
    meanings = np.asarray(scene, dtype=np.int64)
    allowed = np.ones_like(meanings, dtype=bool)
    if support == "old_combo":
        allowed &= meanings != int(combo)
    elif support in ALL_SUPPORTS and "single" in support:
        allowed &= meanings != int(NEW_VALUE * VALUES + NEW_VALUE)
    positions = np.empty(len(scene), dtype=np.int8)
    for i, row in enumerate(allowed):
        candidates = np.flatnonzero(row)
        require(len(candidates) > 0, "support produced an empty target set")
        positions[i] = candidates[min(int(float(uniforms[i]) * len(candidates)), len(candidates) - 1)]
    return positions


def _child_roles(count, seed, update):
    return ((np.arange(count, dtype=np.int64) // PARTNERS + int(seed) + int(update)) % 2).astype(np.int8)


def _fresh_role_for_support(base_role, goal_meaning, support):
    """Set the fresh role for new meanings while preserving old-role rotation."""
    role = np.asarray(base_role, dtype=np.int8).copy()
    new = is_new_meaning(goal_meaning)
    if support.endswith("_pair") or support.endswith("_none"):
        role[new] = 2
    elif support.endswith("_sender"):
        role[new] = 0
    elif support.endswith("_receiver"):
        role[new] = 1
    return role


def episode_stream(seed, count, *, support="old_world", update=0,
                   evaluation=False, surface_mapping="identity"):
    valid_supports = set(ALL_SUPPORTS) | {"old_world"}
    require(count > 0 and support in valid_supports and
            surface_mapping in OBJECT_SURFACE_MAPPINGS, "bad episode factors")
    require(int(seed) in SEEDS or evaluation, "unknown seed")
    combo = heldout_combo(seed) if int(seed) in SEEDS else 0
    expanded = support in ALL_SUPPORTS and support != "old_combo"
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), EVAL_SEED_OFFSET if evaluation else 0, int(update), STREAM_SALT
    ]))
    scene_meanings = _scene_stream(rng, count, expanded)
    target_uniform = rng.random(count)
    target_support = "old_combo" if support == "old_combo" else support
    target_index = _target_positions(scene_meanings, target_uniform, target_support, combo)
    goal_meaning = scene_meanings[np.arange(count), target_index]
    partner_id = (np.arange(count, dtype=np.int64) % PARTNERS).astype(np.int8)
    fresh_role = _child_roles(count, seed, update)
    fresh_role = _fresh_role_for_support(fresh_role, goal_meaning, support)
    return {
        "scene_meanings": scene_meanings,
        "scene": attrs_from_meaning(scene_meanings),
        "fresh_scene_meanings": surface_scene_meanings(scene_meanings, surface_mapping),
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
        "surface_mapping": surface_mapping,
        "heldout_combo": combo,
        "evaluation": bool(evaluation),
        "update": int(update),
    }


def _evaluation_scenes(rng, target_meanings, expanded):
    universe = (np.arange(MEANINGS, dtype=np.int64) if expanded
                else np.asarray(OLD_MEANING_INDICES, dtype=np.int64))
    out = np.empty((len(target_meanings), SCENE_SIZE), dtype=np.int64)
    target_index = np.empty(len(target_meanings), dtype=np.int8)
    for i, target in enumerate(np.asarray(target_meanings, dtype=np.int64)):
        target = int(target)
        attrs = attrs_from_meaning(target)
        # Force the target to be ambiguous under either single attribute:
        # one distractor shares attribute 0, one shares attribute 1, and a
        # fourth object prevents a one-feature shortcut from being reliable.
        modulus = VALUES if expanded else OLD_VALUES
        same_attr0 = int(attrs[0]) * VALUES + ((int(attrs[1]) + 1) % modulus)
        same_attr1 = ((int(attrs[0]) + 1) % modulus) * VALUES + int(attrs[1])
        required = [same_attr0, same_attr1]
        candidates = [int(x) for x in universe
                      if int(x) not in {target, *required}]
        require(len(candidates) >= SCENE_SIZE - 1 - len(required), "not enough distractors")
        distractors = required + [int(rng.choice(np.asarray(candidates, dtype=np.int64)))]
        row = np.asarray([int(target), *[int(x) for x in distractors]], dtype=np.int64)
        rng.shuffle(row)
        out[i] = row
        target_index[i] = int(np.flatnonzero(row == int(target))[0])
    return out, target_index


def balanced_eval_stream(seed, count, goal_kind, support, combo,
                         surface_mapping="identity"):
    require(goal_kind in EVAL_KINDS, "bad goal kind")
    require(support in ALL_SUPPORTS and count > 0 and
            surface_mapping in OBJECT_SURFACE_MAPPINGS, "bad eval factors")
    rng = np.random.default_rng(np.random.SeedSequence([
        int(seed), EVAL_SEED_OFFSET + 17, int(combo), STREAM_SALT
    ]))
    old = np.asarray(OLD_MEANING_INDICES, dtype=np.int64)
    new_single = np.asarray([m for m in range(MEANINGS) if is_new_single(m)], dtype=np.int64)
    if goal_kind == "all":
        meanings = np.arange(count, dtype=np.int64) % MEANINGS
    elif goal_kind == "old_seen":
        choices = old[old != int(combo)]
        meanings = choices[np.arange(count) % len(choices)]
    elif goal_kind == "old_combo":
        meanings = np.full(count, int(combo), dtype=np.int64)
    elif goal_kind == "new_single":
        meanings = new_single[np.arange(count) % len(new_single)]
    else:
        meanings = np.full(count, NEW_VALUE * VALUES + NEW_VALUE, dtype=np.int64)
    expanded = goal_kind in ("new_single", "new_double", "all") or support != "old_combo"
    scene_meanings, target_index = _evaluation_scenes(rng, meanings, expanded)
    partner_id = (np.arange(count, dtype=np.int64) % PARTNERS).astype(np.int8)
    role = _child_roles(count, int(seed), 0)
    role = _fresh_role_for_support(role, meanings, support)
    return {
        "scene_meanings": scene_meanings,
        "scene": attrs_from_meaning(scene_meanings),
        "fresh_scene_meanings": surface_scene_meanings(scene_meanings, surface_mapping),
        "target_index": target_index,
        "goal": attrs_from_meaning(meanings),
        "goal_meaning": meanings.astype(np.int8),
        "partner_id": partner_id,
        "community_id": (partner_id // PARTNERS_PER_COMMUNITY).astype(np.int8),
        "fresh_role": role,
        "message_uniforms": rng.random((count, MESSAGE_LENGTH)),
        "action_uniforms": rng.random(count),
        "support": support,
        "surface_mapping": surface_mapping,
        "heldout_combo": int(combo),
        "evaluation": True,
        "update": 0,
    }


def prepare():
    return {
        "schema": "open_world_grounding_incumbent_study_v1",
        "task": TASK,
        "form": {"name": FORM, "alphabet_size": ALPHABET_SIZE,
                 "length": MESSAGE_LENGTH, "message_state_count": MESSAGE_STATE_COUNT},
        "architectures": list(ARCHITECTURES),
        "architecture_intervention": "holistic indexes complete meanings; factorized fixes slot-to-attribute factors; tied_routed shares a learned route coordinate while factor tables remain role-specific",
        "attributes": ATTRIBUTES, "values_per_attribute": VALUES, "old_values": OLD_VALUES,
        "new_value": NEW_VALUE, "meanings": MEANINGS, "old_meanings": OLD_MEANINGS,
        "old_meaning_indices": list(OLD_MEANING_INDICES),
        "scene_size": SCENE_SIZE, "communities": COMMUNITIES,
        "partners_per_community": PARTNERS_PER_COMMUNITY, "partners": PARTNERS,
        "populations": list(POPULATIONS), "visibilities": list(VISIBILITIES),
        "supports": list(SUPPORTS), "transfer_supports": list(TRANSFER_SUPPORTS),
        "object_surface_mappings": list(OBJECT_SURFACE_MAPPINGS),
        "receiver_initializations": list(RECEIVER_INITIALIZATIONS),
        "all_supports": list(ALL_SUPPORTS), "eval_kinds": list(EVAL_KINDS),
        "roles": list(ROLES), "channels": list(CHANNELS),
        "parent_support": "old_world",
        "stages": ["closed_parent", "single_new_pair_expansion", "incumbent_replacement"],
        "child_intervention": "the expansion stage assigns both fresh roles for new-single meanings and leaves new-double out of training; transfer stages replace sender, receiver, both, or neither role while retaining the same single-new support",
        "grounding_intervention": "incumbent receivers see canonical object labels; fresh receivers see identity or a stable reverse attribute permutation",
        "receiver_initialization_intervention": "fresh policy keeps a random receiver; incumbent_receiver copies only the frozen incumbent receiver table while sender parameters remain fresh",
        "message_symbols": ["@", "#", "￥", "%"],
        "goal_pairs": [list(x) for x in MEANING_PAIRS], "seeds": list(SEEDS),
        "parent_conditions": list(PARENT_CONDITIONS), "child_conditions": list(CHILD_CONDITIONS),
        "transfer_conditions": list(TRANSFER_CONDITIONS),
        "updates": UPDATES, "batch_size": BATCH_SIZE, "learning_rate": LEARNING_RATE,
        "route_init_scale": ROUTE_INIT_SCALE, "route_temperature": ROUTE_TEMPERATURE,
        "entropy_initial": ENTROPY_INITIAL, "checkpoints": list(CHECKPOINTS),
    }
