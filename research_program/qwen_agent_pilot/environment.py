"""Small deterministic collection game for a three-context agent pilot."""
from __future__ import annotations

import random

AGENTS = ("A", "B", "C")
OBJECTS = ("berry", "wood", "stone")
ATTRIBUTES = ("red", "blue", "green")
DESTINATIONS = ("river camp", "hill camp", "forest camp")

# Partner position is relative to the requester: first or second other agent.
MEANING_POOL = (
    ("berry", "red", 0, "river camp"),
    ("wood", "blue", 1, "hill camp"),
    ("stone", "green", 0, "forest camp"),
    ("berry", "blue", 1, "forest camp"),
    ("wood", "green", 0, "river camp"),
    ("stone", "red", 1, "hill camp"),
)

BLOCKS = 2
EPISODES_PER_BLOCK = len(AGENTS) * len(MEANING_POOL)
PILOT_EPISODES = BLOCKS * EPISODES_PER_BLOCK


def helpers_for(owner: str) -> tuple[str, str]:
    if owner not in AGENTS:
        raise ValueError(f"unknown owner: {owner}")
    return tuple(agent for agent in AGENTS if agent != owner)


def _balanced_schedule(seed: int, block: int) -> list[tuple[str, int]]:
    """Every requester sees every meaning once in each block, in shuffled order."""
    pairs = [(owner, meaning_id) for owner in AGENTS
             for meaning_id in range(len(MEANING_POOL))]
    random.Random(seed * 65537 + block).shuffle(pairs)
    return pairs


def make_episode(index: int, seed: int = 20260919) -> dict:
    """Return one episode from the balanced two-block convention pilot."""
    if index < 0:
        raise ValueError("episode index must be nonnegative")
    if index >= PILOT_EPISODES:
        raise ValueError(f"pilot has exactly {PILOT_EPISODES} episodes")
    block = index // EPISODES_PER_BLOCK
    within_block = index % EPISODES_PER_BLOCK
    owner, meaning_idx = _balanced_schedule(seed, block)[within_block]
    obj, attr, partner_position, destination = MEANING_POOL[meaning_idx]
    helpers = helpers_for(owner)
    assigned_partner = helpers[partner_position]

    rng = random.Random(seed * 1000 + index)
    target = (obj, attr)
    distractors = [(obj, other) for other in ATTRIBUTES if other != attr]
    distractors.append(rng.choice([(o, a) for o in OBJECTS for a in ATTRIBUTES
                                   if o != obj and a != attr]))
    candidates = [(obj, attr), *distractors]
    rng.shuffle(candidates)
    scene = [
        {"item_id": f"I{position}", "object": item_obj, "attribute": item_attr}
        for position, (item_obj, item_attr) in enumerate(candidates)
    ]
    target_item_id = next(
        item["item_id"] for item in scene
        if (item["object"], item["attribute"]) == target
    )
    return {
        "episode": index,
        "block": block,
        "meaning_id": meaning_idx,
        "owner": owner,
        "helpers": helpers,
        "goal": {
            "object": obj,
            "attribute": attr,
            "partner": assigned_partner,
            "partner_position": partner_position,
            "destination": destination,
        },
        "scene": scene,
        "destinations": DESTINATIONS,
        "target_item_id": target_item_id,
    }


def score_episode(episode: dict, actions: dict[str, dict | None]) -> dict:
    """Score the joint collection action; no action gets direct semantic hints."""
    goal = episode["goal"]
    designated = goal["partner"]
    assigned_action = actions.get(designated)
    other = next(agent for agent in episode["helpers"] if agent != designated)
    other_action = actions.get(other)
    exact_item = (
        assigned_action is not None
        and assigned_action.get("item_id") == episode["target_item_id"]
    )
    exact_destination = (
        assigned_action is not None
        and assigned_action.get("destination") == goal["destination"]
    )
    unassigned_waited = other_action is None
    success = bool(exact_item and exact_destination and unassigned_waited)
    return {
        "success": success,
        "reward": 1.0 if success else -0.25,
        "designated_item_correct": bool(exact_item),
        "designated_destination_correct": bool(exact_destination),
        "designated_helper_correct": bool(exact_item and exact_destination),
        "unassigned_helper_waited": bool(unassigned_waited),
    }
