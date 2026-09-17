"""Independent sender-token compatibility checks across topology schedules."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
SCHEDULES = ("A", "B", "C")
PRIVATE_TYPES = (0, 1, 0, 1)
TEAMS = {
    "A": ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)),
    "B": ((0, 2, 3), (1, 3, 0), (2, 0, 1), (3, 1, 2)),
    "C": ((0, 3, 1), (1, 0, 2), (2, 1, 3), (3, 2, 0)),
}
MAPS = tuple((food, water) for food in range(6) for water in range(6) if food != water)
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TARGET_PAIRS = ((0, 1), (0, 3), (1, 2), (1, 4), (2, 0), (2, 5), (3, 2), (3, 4), (4, 0), (4, 5), (5, 1), (5, 3))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def target_ids(partition: int) -> np.ndarray:
    q = PANELS[partition - 1]
    index = {pair: i for i, pair in enumerate(MAPS)}
    return np.asarray(sorted(index[(q[i], q[j])] for i, j in TARGET_PAIRS), np.int64)


def score(receiver: dict[str, np.ndarray], food: dict[str, np.ndarray], water: dict[str, np.ndarray], mask: np.ndarray) -> float:
    decoder = np.argmax(receiver["receiver_logits"], axis=-1)
    code = 7 * food["tokens"][:, 0] + water["tokens"][:, 1]
    action = decoder[code]
    return float(np.mean(np.all(action[mask] == receiver["positions"][mask], axis=-1)))


def one_schedule(out: Path, seed: int, partition: int, schedule: str) -> list[dict]:
    folder = out / "social" / f"s{seed}_p{partition}_dual_complementary"
    prefix = "protocol" if schedule == "A" else f"transfer_{schedule}"
    raw = [load(folder / f"{prefix}_2400_team{team}.npz") for team in range(4)]
    target = np.isin(raw[0]["map_id"], target_ids(partition))
    values = {key: [] for key in ("within", "same_type_food", "same_type_water", "all_cross")}
    for receiver in range(4):
        local = lambda food, water: score(raw[receiver], raw[food], raw[water], target)
        values["within"].append(local(receiver, receiver))
        receiver_food = TEAMS[schedule][receiver][1]
        receiver_water = TEAMS[schedule][receiver][2]
        same_food = [team for team in range(4) if team != receiver and PRIVATE_TYPES[TEAMS[schedule][team][1]] == PRIVATE_TYPES[receiver_food]][0]
        same_water = [team for team in range(4) if team != receiver and PRIVATE_TYPES[TEAMS[schedule][team][2]] == PRIVATE_TYPES[receiver_water]][0]
        values["same_type_food"].append(local(same_food, receiver))
        values["same_type_water"].append(local(receiver, same_water))
        values["all_cross"].append(float(np.mean([local(food, water) for food, water in itertools.product(range(4), repeat=2)])))
    return [{"seed": seed, "partition": partition, "schedule": schedule, "metric": metric, "value": float(np.mean(team_values))} for metric, team_values in values.items()]


def mean_sd(values):
    array = np.asarray(values, np.float64)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1)), "values": array.tolist()}


def run(out: Path) -> dict:
    rows = []
    for seed, partition, schedule in itertools.product(SEEDS, PARTITIONS, SCHEDULES):
        rows.extend(one_schedule(out, seed, partition, schedule))
    summary = {}
    for schedule in SCHEDULES:
        summary[schedule] = {metric: mean_sd([row["value"] for row in rows if row["schedule"] == schedule and row["metric"] == metric]) for metric in ("within", "same_type_food", "same_type_water", "all_cross")}
        summary[schedule]["delta_all_cross"] = float(summary[schedule]["all_cross"]["mean"] - summary[schedule]["within"]["mean"])
    old_path = PROJECT / "redesign_v0.33/results/team_001/cross_team.json"
    old = json.loads(old_path.read_text())
    old_all_cross = old["summary"]["dual_complementary"]["all_cross"]["mean"]
    record = {"status": "complete", "formal": True, "source_sha256": sha(ROOT / "cross_topology.py"), "seeds": list(SEEDS), "partitions": list(PARTITIONS), "schedules": list(SCHEDULES), "rows": rows, "summary": summary,
              "v033_fixed_all_cross": old_all_cross, "rotating_A_all_cross_gain_vs_v033": float(summary["A"]["all_cross"]["mean"] - old_all_cross),
              "definitions": {"within": "receiver with its native schedule tokens", "same_type_food": "replace the food sender token with a different team of the same private type", "same_type_water": "replace the water sender token with a different team of the same private type", "all_cross": "mean over all 16 food/water team token combinations", "A": "native endpoint schedule", "B": "seen by rotating training", "C": "held-out endpoint schedule"},
              "input": str(old_path.resolve()), "input_sha256": sha(old_path)}
    (out / "cross_topology.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    record = run(args.out.resolve()); print(json.dumps({"status": record["status"], "source_sha256": record["source_sha256"], "summary": {schedule: {metric: value["mean"] for metric, value in metrics.items() if isinstance(value, dict)} for schedule, metrics in record["summary"].items()}}, ensure_ascii=False))


if __name__ == "__main__": main()
