"""Independent, exact finite-environment information bounds for v0.5.

Uses only Python's standard library and does not import the training environment.
Run: python redesign_v0.5/bounds_audit.py --output <json-path>
The bound assumes perfect perception of permitted observations; it does not
assume that a learned policy can attain the bound.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
from functools import lru_cache
from itertools import permutations, product
import json
from math import gcd
from pathlib import Path


MAPS = tuple(permutations(range(4), 2))  # (food location, water location)
MAP_INDEX = {world: index for index, world in enumerate(MAPS)}
DECODERS = tuple(product(range(4), repeat=2))
EMPTY = 2


def number(value: Fraction) -> dict:
    return {"fraction": str(value), "decimal": float(value)}


def static_channel_bounds() -> dict:
    """Enumerate all deterministic receiver functions and their subsets.

    A receiver function is (location for food goal, location for water goal).
    Sender-known goal: sender may choose a different message for each goal.
    Sender-hidden goal: one message must serve both equiprobable goals.
    Private random action-menu permutations only relabel the four locations.
    Randomized policies cannot improve these optima: hold the other policies
    fixed, choose the best deterministic response, and iterate at a maximizer.
    """
    masks = []
    for decoder in DECODERS:
        half = full = known = 0
        for index, world in enumerate(MAPS):
            correct = tuple(decoder[goal] == world[goal] for goal in range(2))
            if any(correct):
                half |= 1 << index
            if all(correct):
                full |= 1 << index
            for goal in range(2):
                if correct[goal]:
                    known |= 1 << (2 * index + goal)
        masks.append((half, full, known))

    size = 1 << len(DECODERS)
    half_masks, full_masks, known_masks = [0] * size, [0] * size, [0] * size
    optima = {"goal_known": {}, "goal_hidden": {}}
    for subset in range(1, size):
        bit = subset & -subset
        previous = subset ^ bit
        decoder_index = bit.bit_length() - 1
        half_masks[subset] = half_masks[previous] | masks[decoder_index][0]
        full_masks[subset] = full_masks[previous] | masks[decoder_index][1]
        known_masks[subset] = known_masks[previous] | masks[decoder_index][2]
        k = subset.bit_count()
        values = {
            "goal_known": known_masks[subset].bit_count(),
            "goal_hidden": half_masks[subset].bit_count() + full_masks[subset].bit_count(),
        }
        for condition, correct_count in values.items():
            if k not in optima[condition] or correct_count > optima[condition][k][0]:
                optima[condition][k] = (correct_count, subset)

    output = {}
    for condition, entries in optima.items():
        rows = {}
        for k in range(1, 26):
            correct_count, subset = max(entries[j] for j in range(1, min(k, 16) + 1))
            witness = [list(DECODERS[j]) for j in range(16) if subset & (1 << j)]
            rows[str(k)] = {
                **number(Fraction(correct_count, 24)),
                "receiver_location_pairs": witness,
                "actual_distinct_decoders": len(witness),
            }
        output[condition] = rows
    assert output["goal_known"]["1"]["fraction"] == "1/4"
    assert output["goal_hidden"]["1"]["fraction"] == "1/4"
    for k in range(4, 26):
        assert output["goal_known"][str(k)]["fraction"] == "1"
    assert output["goal_hidden"]["5"]["fraction"] == "17/24"
    assert output["goal_hidden"]["11"]["fraction"] != "1"
    for k in range(12, 26):
        assert output["goal_hidden"][str(k)]["fraction"] == "1"
    return output


def normalize(weights: tuple[int, ...]) -> tuple[int, ...]:
    common = 0
    for value in weights:
        common = gcd(common, value)
    assert common > 0
    return tuple(value // common for value in weights)


def collect(world: tuple[int, int], location: int) -> int:
    if location == world[0]:
        return 0
    if location == world[1]:
        return 1
    return EMPTY


def next_worlds(world: tuple[int, int], outcome: int) -> tuple[tuple[int, int], ...]:
    """Collected resource respawns uniformly, including its old location.

    The other resource does not move. Empty collection changes nothing.
    """
    if outcome == EMPTY:
        return (world,)
    other_location = world[1 - outcome]
    return tuple(
        (location, other_location) if outcome == 0 else (other_location, location)
        for location in range(4)
        if location != other_location
    )


def inventory_step(inventory: tuple[int, int], outcome: int, goal: int):
    updated = list(inventory)
    if outcome != EMPTY:
        updated[outcome] = min(2, updated[outcome] + 1)
    reward = int(updated[goal] > 0)
    updated[goal] -= reward
    return tuple(updated), reward


@lru_cache(None)
def belief_branches(belief: tuple[int, ...], action: int):
    """Posterior after observing the actual pickup, then unobserved respawn."""
    total = sum(belief)
    branches = []
    for outcome in range(3):
        mass = 0
        updated = [0] * len(MAPS)
        for index, weight in enumerate(belief):
            world = MAPS[index]
            if not weight or collect(world, action) != outcome:
                continue
            mass += weight
            for new_world in next_worlds(world, outcome):
                updated[MAP_INDEX[new_world]] += weight
        if mass:
            # Each old state within this outcome has the same number of
            # successors (3 for food/water, 1 for empty), so normalization
            # cancels that common denominator exactly.
            branches.append((outcome, Fraction(mass, total), normalize(tuple(updated))))
    assert sum(branch[1] for branch in branches) == 1
    return tuple(branches)


@lru_cache(None)
def no_message_value(remaining: int, belief: tuple[int, ...], inventory: tuple[int, int]):
    """Optimal expected total satisfied needs, before observing the next goal."""
    if remaining == 0:
        return Fraction(0)
    goal_values = []
    for goal in range(2):
        action_values = []
        for action in range(4):
            value = Fraction(0)
            for outcome, probability, new_belief in belief_branches(belief, action):
                new_inventory, reward = inventory_step(inventory, outcome, goal)
                value += probability * (
                    reward + no_message_value(remaining - 1, new_belief, new_inventory)
                )
            action_values.append(value)
        goal_values.append(max(action_values))
    return sum(goal_values, Fraction(0)) / 2


@lru_cache(None)
def full_information_value(remaining: int, world: tuple[int, int], inventory: tuple[int, int]):
    if remaining == 0:
        return Fraction(0)
    goal_values = []
    for goal in range(2):
        action_values = []
        for action in range(4):
            outcome = collect(world, action)
            new_inventory, reward = inventory_step(inventory, outcome, goal)
            successors = next_worlds(world, outcome)
            continuation = sum(
                (full_information_value(remaining - 1, state, new_inventory) for state in successors),
                Fraction(0),
            ) / len(successors)
            action_values.append(reward + continuation)
        goal_values.append(max(action_values))
    return sum(goal_values, Fraction(0)) / 2


def dynamic_bounds() -> dict:
    start_belief = (1,) * len(MAPS)
    rows = {}
    for horizon in (1, 2, 3):
        no_message = no_message_value(horizon, start_belief, (0, 0))
        full_information = sum(
            (full_information_value(horizon, world, (0, 0)) for world in MAPS),
            Fraction(0),
        ) / len(MAPS)
        assert full_information == horizon
        rows[str(horizon)] = {
            "no_message_total_reward": number(no_message),
            "no_message_mean_reward": number(no_message / horizon),
            "full_information_total_reward": number(full_information),
            "full_information_mean_reward": number(full_information / horizon),
        }
    assert rows["1"]["no_message_mean_reward"]["fraction"] == "1/4"
    assert rows["3"]["no_message_mean_reward"]["decimal"] > 0.25
    return {
        "assumptions": [
            "12 equiprobable initial maps: distinct food and water locations; other two locations empty",
            "Fixed observer/collector roles for the entire segment; observer has no environmental action",
            "IID equiprobable private food/water goal each round, independent of map and all history",
            "Collector knows own current goal, inventory, prior chosen locations and exact pickup outcomes",
            "Initial inventory (0,0), cap 2 each; pickup precedes consumption of one requested unit",
            "Pickup happens even if its inventory is full; overflow is discarded",
            "Collected resource respawns uniformly in the 3 locations excluding the other resource; old location allowed",
            "Respawn is unobserved by collector; uncollected resource stays; empty pickup changes neither position",
            "No additional observations of map, observer action, message, movement, or respawn randomness",
            "Action menus relabel physical locations bijectively and are visible to collector only",
        ],
        "by_horizon": rows,
        "no_message_dp_cache": no_message_value.cache_info()._asdict(),
        "belief_transition_cache": belief_branches.cache_info()._asdict(),
        "full_information_dp_cache": full_information_value.cache_info()._asdict(),
    }


def independent_history_tree(horizon: int) -> Fraction:
    """Second exact solver: unnormalized probability histories, without caches.

    This deliberately does not use belief_branches, normalization,
    next_worlds, inventory_step, or either Bellman-value implementation above.
    Each branch retains its joint probability mass rather than a posterior.
    """
    def solve(remaining, atoms, inventory):
        if not remaining:
            return Fraction(0)
        goal_values = []
        for goal in (0, 1):
            values = []
            for location in range(4):
                observations = {None: {}, 0: {}, 1: {}}
                for (food_location, water_location), probability in atoms.items():
                    outcome = 0 if location == food_location else 1 if location == water_location else None
                    grouped = observations[outcome]
                    if outcome is None:
                        world = (food_location, water_location)
                        grouped[world] = grouped.get(world, Fraction(0)) + probability
                    else:
                        excluded = water_location if outcome == 0 else food_location
                        for replacement in range(4):
                            if replacement == excluded:
                                continue
                            world = (replacement, water_location) if outcome == 0 else (food_location, replacement)
                            grouped[world] = grouped.get(world, Fraction(0)) + probability / 3
                value = Fraction(0)
                for outcome, new_atoms in observations.items():
                    if not new_atoms:
                        continue
                    held = list(inventory)
                    if outcome is not None:
                        held[outcome] = min(held[outcome] + 1, 2)
                    reward = 1 if held[goal] else 0
                    held[goal] -= reward
                    value += reward * sum(new_atoms.values(), Fraction(0))
                    value += solve(remaining - 1, new_atoms, tuple(held))
                values.append(value)
            goal_values.append(max(values))
        return (goal_values[0] + goal_values[1]) / 2

    initial = {(food, water): Fraction(1, 12) for food in range(4) for water in range(4) if food != water}
    return solve(horizon, initial, (0, 0))


def run_audit() -> dict:
    # Independently check every elementary stochastic transition.
    assert len(MAPS) == 12 and len(DECODERS) == 16
    transition_checks = 0
    for world in MAPS:
        for action in range(4):
            outcome = collect(world, action)
            successors = next_worlds(world, outcome)
            assert len(successors) == (1 if outcome == EMPTY else 3)
            assert len(set(successors)) == len(successors)
            for successor in successors:
                assert successor in MAP_INDEX
                if outcome != EMPTY:
                    assert successor[1 - outcome] == world[1 - outcome]
            transition_checks += 1
    history_tree = independent_history_tree(3)
    assert history_tree == Fraction(265, 216)
    assert history_tree == no_message_value(3, (1,) * 12, (0, 0))
    return {
        "status": "passed",
        "scope": "Independent mathematical environment audit; does not validate a training implementation",
        "static_worlds": len(MAPS),
        "receiver_functions": len(DECODERS),
        "receiver_function_subsets_enumerated": (1 << len(DECODERS)) - 1,
        "elementary_dynamic_transitions_checked": transition_checks,
        "stage_a": static_channel_bounds(),
        "persistent_three_rounds": dynamic_bounds(),
        "independent_history_tree_total_reward": number(history_tree),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_audit()
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "goal_hidden_K5": result["stage_a"]["goal_hidden"]["5"],
        "persistent_by_horizon": result["persistent_three_rounds"]["by_horizon"],
        "no_message_dp_cache": result["persistent_three_rounds"]["no_message_dp_cache"],
        "output": str(args.output) if args.output else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
