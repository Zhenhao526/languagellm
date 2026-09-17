"""Independent cross-team token substitution checks for v0.33.

The formal endpoint protocol files contain one receiver table and one pair of
sender messages per team.  This analysis keeps each receiver fixed, replaces
one or both sender tokens with tokens emitted by another team, and measures
held-out target J.  It is deliberately separate from the production scorer:
the result tests whether the learned code is population-compatible rather
than merely successful for the pair that trained together.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("dual_same_full", "dual_complementary")
TEAMS = ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1))
PRIVATE_TYPES = (0, 1, 0, 1)
MAPS = tuple((food, water) for food in range(6) for water in range(6) if food != water)
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TARGET_PAIRS = ((0, 1), (0, 3), (1, 2), (1, 4), (2, 0), (2, 5),
                (3, 2), (3, 4), (4, 0), (4, 5), (5, 1), (5, 3))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def target_ids(partition: int) -> np.ndarray:
    q = PANELS[partition - 1]
    index = {pair: i for i, pair in enumerate(MAPS)}
    return np.asarray(sorted(index[(q[i], q[j])] for i, j in TARGET_PAIRS), np.int64)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def score(receiver: dict[str, np.ndarray], food: dict[str, np.ndarray], water: dict[str, np.ndarray], mask: np.ndarray) -> float:
    decoder = np.argmax(receiver["receiver_logits"], axis=-1)
    code = 7 * food["tokens"][:, 0] + water["tokens"][:, 1]
    action = decoder[code]
    return float(np.mean(np.all(action[mask] == receiver["positions"][mask], axis=-1)))


def run(out: Path) -> dict:
    rows = []
    for condition in CONDITIONS:
        for seed, partition in itertools.product(SEEDS, PARTITIONS):
            folder = out / "social" / f"s{seed}_p{partition}_{condition}"
            raw = [load_npz(folder / f"protocol_2400_team{team}.npz") for team in range(4)]
            target = np.isin(raw[0]["map_id"], target_ids(partition))
            if not all(np.array_equal(target, np.isin(item["map_id"], target_ids(partition))) for item in raw[1:]):
                raise AssertionError("paired target mask mismatch")
            values = {key: [] for key in ("within", "food_swap", "water_swap", "same_type_food", "same_type_water", "all_cross")}
            for receiver in range(4):
                within = lambda food, water: score(raw[receiver], raw[food], raw[water], target)
                values["within"].append(within(receiver, receiver))
                same_food = [team for team in range(4) if team != receiver and PRIVATE_TYPES[TEAMS[team][1]] == PRIVATE_TYPES[TEAMS[receiver][1]]][0]
                same_water = [team for team in range(4) if team != receiver and PRIVATE_TYPES[TEAMS[team][2]] == PRIVATE_TYPES[TEAMS[receiver][2]]][0]
                values["same_type_food"].append(within(same_food, receiver))
                values["same_type_water"].append(within(receiver, same_water))
                values["food_swap"].append(within((receiver + 1) % 4, receiver))
                values["water_swap"].append(within(receiver, (receiver + 1) % 4))
                values["all_cross"].append(float(np.mean([within(food, water) for food, water in itertools.product(range(4), repeat=2)])))
            for metric, team_values in values.items():
                rows.append({"seed": seed, "partition": partition, "condition": condition,
                             "metric": metric, "value": float(np.mean(team_values))})

    summary = {}
    for condition in CONDITIONS:
        summary[condition] = {}
        for metric in ("within", "food_swap", "water_swap", "same_type_food", "same_type_water", "all_cross"):
            values = np.asarray([row["value"] for row in rows if row["condition"] == condition and row["metric"] == metric], np.float64)
            summary[condition][metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                                          "values": values.tolist()}
        summary[condition]["delta_same_type_food"] = float(summary[condition]["same_type_food"]["mean"] - summary[condition]["within"]["mean"])
        summary[condition]["delta_same_type_water"] = float(summary[condition]["same_type_water"]["mean"] - summary[condition]["within"]["mean"])
        summary[condition]["delta_all_cross"] = float(summary[condition]["all_cross"]["mean"] - summary[condition]["within"]["mean"])
    record = {"status": "complete", "formal": True, "source_sha256": sha(ROOT / "cross_team.py"),
              "endpoint": 2400, "seeds": list(SEEDS), "partitions": list(PARTITIONS),
              "conditions": list(CONDITIONS), "rows": rows, "summary": summary,
              "definitions": {"within": "each receiver with its trained team tokens",
                              "food_swap": "food token from the next team, water token retained",
                              "water_swap": "water token from the next team, food token retained",
                              "same_type_food": "food token from a different team with the same private type",
                              "same_type_water": "water token from a different team with the same private type",
                              "all_cross": "mean over all 16 food/water team combinations"}}
    (out / "cross_team.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = run(args.out.resolve())
    print(json.dumps({"status": record["status"], "source_sha256": record["source_sha256"], "conditions": record["conditions"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
