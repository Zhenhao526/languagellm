"""Independent NumPy checks for the static v0.7 task; no policy imports.

This module deliberately does not load camp.py or the runner. Its public
functions accept arrays or explicit design metadata supplied by the report
script, so dataset counts and message capacities are never inferred from a
successful run alone.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, context):
    require(np.array_equal(actual, expected), f"{context}: arrays differ")


def near(actual, expected, context, tolerance=1e-7):
    require(np.allclose(actual, expected, atol=tolerance, rtol=0),
            f"{context}: {actual} != {expected}")


def describe(values):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and values.size > 0, "Descriptive sample must be nonempty")
    require(np.isfinite(values).all(), "Descriptive sample contains nonfinite values")
    return {"n": int(values.size), "mean": float(values.mean()),
            "min": float(values.min()), "max": float(values.max()),
            "values": values.tolist()}


def selected_mean(reward, mask):
    mask = np.asarray(mask, dtype=bool)
    count = int(mask.sum())
    return {"n": count, "reward_sum": int(np.asarray(reward)[mask].sum()),
            "mean_reward": float(np.asarray(reward)[mask].mean()) if count else None}


def split_audit(n_sites, split_maps):
    """Verify explicit matching splits and marginal resource/location coverage."""
    all_maps = set(permutations(range(n_sites), 2))
    heldout_sets, records = [], []
    for split_id, maps in split_maps.items():
        heldout = {tuple(map(int, m)) for m in maps}
        require(heldout.issubset(all_maps), "Heldout maps must be legal")
        train = all_maps - heldout
        require(len(heldout) == n_sites == len(maps), "Incorrect holdout size or duplicate maps")
        require(all((water, food) in heldout for food, water in heldout), "Holdout must contain both directions of each matching edge")
        for resource in (0, 1):
            equal(np.bincount([m[resource] for m in heldout], minlength=n_sites),
                  np.ones(n_sites, dtype=int), "Heldout resource/location balance")
            equal(np.bincount([m[resource] for m in train], minlength=n_sites),
                  np.full(n_sites, n_sites - 2), "Training resource/location balance")
        heldout_sets.append(heldout)
        records.append({"split_id": str(split_id), "train_maps": [list(m) for m in sorted(train)],
                        "heldout_maps": [list(m) for m in sorted(heldout)],
                        "train_resource_location_frequency": n_sites - 2,
                        "heldout_resource_location_frequency": 1})
    require(sum(len(s) for s in heldout_sets) == len(set().union(*heldout_sets)),
            "Different matching splits must have disjoint heldout maps")
    return {"status": "passed", "n_sites": n_sites, "all_maps": [list(m) for m in sorted(all_maps)],
            "splits": records, "distinct_heldout_maps": len(set().union(*heldout_sets))}


def audit_static_arrays(arrays, *, n_sites, vocab, length, history_events,
                        episodes, mode, blocked, photo_metadata, photo_split="test"):
    """Recover one-step outcomes independently from map, need and action.

    Array names follow the established static trace vocabulary. The report adapter
    must map any changed v0.7 names explicitly before calling this function.
    No hidden-map labels are fed into a model; using them here is an offline
    correctness check of the saved outcomes.
    """
    a = arrays
    required = {"scout", "episode", "positions", "photo_ids", "goals", "menu",
                "sent", "delivered", "action", "place", "reward", "step", "inventory",
                "history", "gathered", "overflow", "next_inventory", "next_positions", "refill_uniform"}
    require(required.issubset(a), f"Missing static trace columns: {required - a.keys()}")
    n = int(episodes)
    require(n > 0 and n % 2 == 0, "Evaluation requires equally sized two directions")
    require(all(len(value) == n for value in a.values()), "Inconsistent static trace lengths")
    require(a["positions"].shape == (n, 2), "Map must contain two resource locations")
    require(a["photo_ids"].shape == (n, 2), "Photo IDs must contain two resources")
    require(a["menu"].shape == (n, n_sites), "Incorrect action menu shape")
    equal(np.sort(a["menu"], axis=1), np.tile(np.arange(n_sites), (n, 1)), "Menu permutation")
    require(np.issubdtype(a["positions"].dtype, np.integer), "Resource positions must be integers")
    require(((a["positions"] >= 0) & (a["positions"] < n_sites)).all(), "Resource position out of range")
    require((a["positions"][:, 0] != a["positions"][:, 1]).all(), "Resources cannot occupy the same site")
    for column, limit in (("scout", 2), ("goals", 2), ("action", n_sites)):
        require(a[column].shape == (n,), f"Incorrect {column} shape")
        require(np.issubdtype(a[column].dtype, np.integer), f"{column} must be integer")
        require(((a[column] >= 0) & (a[column] < limit)).all(), f"{column} out of range")
    equal(a["place"], a["menu"][np.arange(n), a["action"]], "Action menu routing")
    for direction in (0, 1):
        ix = a["scout"] == direction
        require(int(ix.sum()) == n // 2, "Incorrect direction count")
        equal(np.sort(a["episode"][ix]), np.arange(n // 2), "Direction episode IDs")
    for column in ("sent", "delivered"):
        require(a[column].shape == (n, length), f"Incorrect {column} message shape")
        require(np.issubdtype(a[column].dtype, np.integer), f"{column} must contain integer tokens")
        require(((a[column] >= 0) & (a[column] < vocab)).all(), f"{column} symbol out of range")
    if blocked or mode == "blank":
        equal(a["delivered"], np.zeros_like(a["delivered"]), "Blocked/blank delivery")
    elif mode == "shuffle":
        # Static context includes the receiver's own goal; inventory and history
        # are fixed zeros. Shuffle must preserve whole-message counts per stratum.
        for direction in (0, 1):
            for goal in (0, 1):
                ix = (a["scout"] == direction) & (a["goals"] == goal)
                sent, sn = np.unique(a["sent"][ix], axis=0, return_counts=True)
                delivered, dn = np.unique(a["delivered"][ix], axis=0, return_counts=True)
                equal(sent, delivered, "Conditional shuffled message values")
                equal(sn, dn, "Conditional shuffled message counts")
    else:
        equal(a["sent"], a["delivered"], "Unmodified delivery")
    picked = (a["place"][:, None] == a["positions"]).astype(np.int64)
    reward = picked[np.arange(n), a["goals"]]
    equal(reward, a["reward"], "Recomputed static reward")
    if "gathered" in a:
        equal(picked, a["gathered"], "Recomputed pickup")
    # The complete trace must contain these fields even though this experiment
    # has no dynamics; the required-column check above prevents silent omission.
    for column, shape in (("step", (n,)), ("inventory", (n, 2)),
                          ("history", (n, history_events * (n_sites + 3))),
                          ("overflow", (n, 2))):
        if column in a:
            equal(a[column], np.zeros(shape), f"Static {column}")
    if "next_inventory" in a:
        after = picked.copy()
        after[np.arange(n), a["goals"]] -= reward
        equal(after, a["next_inventory"], "Post-consumption inventory")
    if "next_positions" in a:
        equal(a["positions"], a["next_positions"], "Static map must not respawn")
    for resource, category in enumerate(("food", "water")):
        ids = a["photo_ids"][:, resource]
        require(np.issubdtype(ids.dtype, np.integer), "Photo IDs must be integers")
        require(((ids >= 0) & (ids < len(photo_metadata))).all(), "Photo ID out of range")
        require(all(photo_metadata[int(i)]["category"] == category and
                    photo_metadata[int(i)]["split"] == photo_split for i in np.unique(ids)),
                "Incorrect photo category or evaluation split")
    return reward, {"status": "passed", "steps_checked": n,
                    "checks": ["shapes", "map", "menu", "directions", "message_values_and_delivery",
                               "reward", "static_context", "photo_category_and_split"]}


def static_metrics(arrays, reward, *, n_sites, heldout_maps):
    """Check balanced world/need/direction evaluation and retain exact counts."""
    a = arrays
    maps = list(permutations(range(n_sites), 2))
    require(len(reward) % (4 * len(maps)) == 0, "Evaluation size cannot balance map/goal/direction cells")
    per_cell = len(reward) // (4 * len(maps))
    map_rows, cells = [], []
    for food, water in maps:
        map_mask = (a["positions"] == (food, water)).all(axis=1)
        map_rows.append({"food_site": food, "water_site": water, **selected_mean(reward, map_mask)})
        for direction in (0, 1):
            for goal in (0, 1):
                ix = map_mask & (a["scout"] == direction) & (a["goals"] == goal)
                require(int(ix.sum()) == per_cell, "Evaluation is not balanced per map/goal/direction")
                cells.append({"food_site": food, "water_site": water,
                              "direction": direction, "goal": goal, **selected_mean(reward, ix)})
    heldout = {tuple(m) for m in heldout_maps}
    require(heldout.issubset(set(maps)), "Metric heldout map set contains invalid maps")
    out_mask = np.asarray([tuple(m) in heldout for m in a["positions"]], dtype=bool)
    return {"mean_reward": float(np.mean(reward)), "episodes": len(reward), "horizon": 1,
            "cases_per_map_goal_direction": per_cell,
            "direction_means": [selected_mean(reward, a["scout"] == d) for d in (0, 1)],
            "goal_means": [selected_mean(reward, a["goals"] == g) for g in (0, 1)],
            "map_subsets": {"train_maps": selected_mean(reward, ~out_mask),
                            "heldout_maps": selected_mean(reward, out_mask)},
            "map_rows": map_rows, "balanced_cells": cells}
