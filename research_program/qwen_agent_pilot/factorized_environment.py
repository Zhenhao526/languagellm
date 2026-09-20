"""Factorial three-agent collection environment for calibration v4."""
from __future__ import annotations

import itertools
import random

AGENTS = ("A", "B", "C")
OBJECTS = ("berry", "wood")
ATTRIBUTES = ("red", "blue")
DESTINATIONS = ("river camp", "hill camp")
BLOCKS = 2

# Eight payloads crossed with two relative helper positions give sixteen orders.
PAYLOADS = tuple(itertools.product(OBJECTS, ATTRIBUTES, DESTINATIONS))
MEANING_POOL = tuple(
    (obj, attr, partner_position, destination)
    for obj, attr, destination in PAYLOADS
    for partner_position in (0, 1)
)
MEANING_TO_ID = {meaning: index for index, meaning in enumerate(MEANING_POOL)}
PAYLOAD_TO_ID = {payload: index for index, payload in enumerate(PAYLOADS)}
EPISODES_PER_BLOCK = len(AGENTS) * len(MEANING_POOL)
TOTAL_EPISODES = BLOCKS * EPISODES_PER_BLOCK


def helpers_for(owner: str) -> tuple[str, str]:
    if owner not in AGENTS:
        raise ValueError(f"unknown requester: {owner}")
    return tuple(agent for agent in AGENTS if agent != owner)


def _balanced_schedule(seed: int, block: int) -> list[tuple[str, int]]:
    pairs = [(owner, meaning_id) for owner in AGENTS
             for meaning_id in range(len(MEANING_POOL))]
    random.Random(seed * 65537 + block).shuffle(pairs)
    return pairs


def make_episode(index: int, seed: int) -> dict:
    if index < 0 or index >= TOTAL_EPISODES:
        raise ValueError(f"v4 schedule has exactly {TOTAL_EPISODES} episodes")
    block = index // EPISODES_PER_BLOCK
    within_block = index % EPISODES_PER_BLOCK
    owner, meaning_id = _balanced_schedule(seed, block)[within_block]
    obj, attr, partner_position, destination = MEANING_POOL[meaning_id]
    helpers = helpers_for(owner)
    goal = {
        "object": obj,
        "attribute": attr,
        "partner": helpers[partner_position],
        "partner_position": partner_position,
        "destination": destination,
    }

    # Every binary object–attribute combination is present exactly once.
    candidates = list(itertools.product(OBJECTS, ATTRIBUTES))
    random.Random(seed * 1000 + index).shuffle(candidates)
    board = [
        {"item_id": f"I{position}", "object": item_obj, "attribute": item_attr}
        for position, (item_obj, item_attr) in enumerate(candidates)
    ]
    target_item_id = next(
        row["item_id"] for row in board
        if row["object"] == obj and row["attribute"] == attr
    )
    payload = (obj, attr, destination)
    return {
        "episode": index,
        "block": block,
        "owner": owner,
        "helpers": helpers,
        "meaning_id": meaning_id,
        "payload_id": PAYLOAD_TO_ID[payload],
        "goal": goal,
        "scene": board,
        "destinations": DESTINATIONS,
        "target_item_id": target_item_id,
    }


def score_episode(episode: dict, actions: dict[str, dict | None]) -> dict:
    goal = episode["goal"]
    assigned = goal["partner"]
    action_results = {}
    for agent in episode["helpers"]:
        action = actions.get(agent)
        should_act = agent == assigned
        action_results[agent] = {
            "should_act": should_act,
            "acted": action is not None,
            "responsibility_correct": (action is not None) == should_act,
            "item_correct": (
                action is not None and action.get("item_id") == episode["target_item_id"]
                if action is not None else None
            ),
            "destination_correct": (
                action is not None and action.get("destination") == goal["destination"]
                if action is not None else None
            ),
        }
    chosen = actions.get(assigned)
    item_correct = chosen is not None and chosen.get("item_id") == episode["target_item_id"]
    destination_correct = chosen is not None and chosen.get("destination") == goal["destination"]
    unassigned = next(agent for agent in episode["helpers"] if agent != assigned)
    waited = actions.get(unassigned) is None
    success = bool(item_correct and destination_correct and waited)
    return {
        "success": success,
        "reward": 1.0 if success else -0.25,
        "designated_item_correct": bool(item_correct),
        "designated_destination_correct": bool(destination_correct),
        "designated_helper_correct": bool(item_correct and destination_correct),
        "unassigned_helper_waited": waited,
        "helper_feedback": action_results,
    }
