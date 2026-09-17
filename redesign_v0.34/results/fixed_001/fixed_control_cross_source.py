"""Cross-team token compatibility for the same-namespace fixed-A control."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
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


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def target_ids(partition: int):
    order = PANELS[partition - 1]
    index = {pair: i for i, pair in enumerate(MAPS)}
    return np.asarray(sorted(index[(order[i], order[j])] for i, j in TARGET_PAIRS), dtype=np.int64)


def score(receiver, food, water, mask):
    decoder = np.argmax(receiver["receiver_logits"], axis=-1)
    code = 7 * food["tokens"][:, 0] + water["tokens"][:, 1]
    action = decoder[code]
    return float(np.mean(np.all(action[mask] == receiver["positions"][mask], axis=-1)))


def one_schedule(out: Path, seed: int, partition: int, schedule: str):
    folder = out / "social" / f"s{seed}_p{partition}_dual_complementary"
    prefix = "protocol" if schedule == "A" else f"transfer_{schedule}"
    raw = [npz(folder / f"{prefix}_2400_team{team}.npz") for team in range(4)]
    mask = np.isin(raw[0]["map_id"], target_ids(partition))
    values = {key: [] for key in ("within", "same_type_food", "same_type_water", "all_cross")}
    for receiver in range(4):
        local = lambda food, water: score(raw[receiver], raw[food], raw[water], mask)
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
    values = np.asarray(values, dtype=np.float64)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "values": values.tolist()}


def run(out: Path):
    complete = read(out / "training_complete.json")
    if complete.get("status") != "complete" or complete.get("control") != "fixed_A":
        raise ValueError("fixed-A training completion required")
    rows = []
    for seed, partition, schedule in itertools.product(SEEDS, PARTITIONS, SCHEDULES):
        rows.extend(one_schedule(out, seed, partition, schedule))
    summary = {}
    for schedule in SCHEDULES:
        summary[schedule] = {
            metric: mean_sd([row["value"] for row in rows if row["schedule"] == schedule and row["metric"] == metric])
            for metric in ("within", "same_type_food", "same_type_water", "all_cross")
        }
        summary[schedule]["delta_all_cross"] = float(summary[schedule]["all_cross"]["mean"] - summary[schedule]["within"]["mean"])

    rotating_out = out.parent / "rotation_001"
    rotating = read(rotating_out / "cross_topology.json")
    rotating_rows = {(row["seed"], row["partition"], row["schedule"], row["metric"]): row["value"] for row in rotating["rows"]}
    fixed_rows = {(row["seed"], row["partition"], row["schedule"], row["metric"]): row["value"] for row in rows}
    paired = []
    for seed, partition, schedule, metric in itertools.product(SEEDS, PARTITIONS, SCHEDULES, ("within", "same_type_food", "same_type_water", "all_cross")):
        fixed = fixed_rows[(seed, partition, schedule, metric)]
        rotating_value = rotating_rows[(seed, partition, schedule, metric)]
        paired.append({"seed": seed, "partition": partition, "schedule": schedule, "metric": metric, "fixed": fixed, "rotating": rotating_value, "rotating_minus_fixed": rotating_value - fixed})
    paired_summary = {
        schedule: {
            metric: mean_sd([row["rotating_minus_fixed"] for row in paired if row["schedule"] == schedule and row["metric"] == metric])
            for metric in ("within", "same_type_food", "same_type_water", "all_cross")
        }
        for schedule in SCHEDULES
    }
    record = {
        "status": "complete",
        "formal": True,
        "control": "fixed_A",
        "namespace": "34034",
        "seeds": list(SEEDS),
        "partitions": list(PARTITIONS),
        "schedules": list(SCHEDULES),
        "rows": rows,
        "summary": summary,
        "paired_with_rotation": {"same_namespace": True, "summary": paired_summary, "rows": paired},
        "definitions": {
            "within": "receiver with its native schedule tokens",
            "same_type_food": "replace food token with a different team of the same private type",
            "same_type_water": "replace water token with a different team of the same private type",
            "all_cross": "mean over all 16 food/water team token combinations",
            "primary_unit": "seed×panel; panels are averaged within seed for primary inference",
        },
        "source_sha256": sha(ROOT / "fixed_control_cross.py"),
        "input_training_complete_sha256": sha(out / "training_complete.json"),
        "input_rotating_cross_sha256": sha(rotating_out / "cross_topology.json"),
        "limits": [
            "Compatibility is an endpoint intervention on saved sender tokens and receiver tables.",
            "It measures grounded readout under token replacement; it does not establish a shared lexicon or compositional grammar.",
        ],
    }
    write(out / "fixed_control_cross.json", record)
    shutil.copy2(ROOT / "fixed_control_cross.py", out / "fixed_control_cross_source.py")
    return record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    started = time.monotonic()
    try:
        result = run(args.out.resolve())
        print(json.dumps({"status": result["status"], "source_sha256": result["source_sha256"], "summary": {s: {m: result["summary"][s][m]["mean"] for m in ("within", "same_type_food", "same_type_water", "all_cross")} for s in SCHEDULES}, "seconds": time.monotonic() - started}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns()
        write(args.out.resolve() / f"fixed_control_cross_failure_{stamp}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "fixed_control_cross.py")})
        raise


if __name__ == "__main__":
    main()
