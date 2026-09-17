"""Independent endpoint transfer analysis for the v0.34 topology probe."""
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


def endpoint_score(raw: dict[str, np.ndarray], partition: int) -> dict[str, float]:
    mask = np.isin(raw["map_id"], target_ids(partition))
    decoder = np.argmax(raw["receiver_logits"], axis=-1)
    code = 7 * raw["tokens"][:, 0] + raw["tokens"][:, 1]
    action = decoder[code]
    correct = action[mask] == raw["positions"][mask]
    return {"J": float(np.mean(np.all(correct, axis=-1))), "food": float(np.mean(correct[:, 0])), "water": float(np.mean(correct[:, 1]))}


def mean_sd(values):
    array = np.asarray(values, np.float64)
    return {"mean": float(array.mean()), "sd": float(array.std(ddof=1)), "values": array.tolist()}


def run(out: Path) -> dict:
    rows = []
    for seed, partition, schedule in itertools.product(SEEDS, PARTITIONS, SCHEDULES):
        folder = out / "social" / f"s{seed}_p{partition}_dual_complementary"
        prefix = "protocol" if schedule == "A" else f"transfer_{schedule}"
        team_scores = [endpoint_score(load(folder / f"{prefix}_2400_team{team}.npz"), partition) for team in range(4)]
        for team, score in enumerate(team_scores):
            rows.append({"seed": seed, "partition": partition, "schedule": schedule, "team": team, **score})

    summary = {}
    for schedule in SCHEDULES:
        summary[schedule] = {key: mean_sd([row[key] for row in rows if row["schedule"] == schedule]) for key in ("J", "food", "water")}
    # Average within seed×partition first, so no team slot is treated as an
    # independent source in the topology comparison.
    run_rows = []
    for seed, partition, schedule in itertools.product(SEEDS, PARTITIONS, SCHEDULES):
        selected = [row for row in rows if row["seed"] == seed and row["partition"] == partition and row["schedule"] == schedule]
        run_rows.append({"seed": seed, "partition": partition, "schedule": schedule,
                         **{key: float(np.mean([row[key] for row in selected])) for key in ("J", "food", "water")}})
    paired = {}
    for schedule in ("B", "C"):
        paired[schedule] = {key: mean_sd([b[key] - a[key] for b, a in zip(
            sorted([row for row in run_rows if row["schedule"] == schedule], key=lambda r: (r["seed"], r["partition"])),
            sorted([row for row in run_rows if row["schedule"] == "A"], key=lambda r: (r["seed"], r["partition"]))
        )]) for key in ("J", "food", "water")}
    # Compare the native A endpoint to the earlier fixed-topology v0.33 batch
    # at the seed level.  This is a descriptive cross-batch replication check,
    # not an additional independent experiment.
    old_path = PROJECT / "redesign_v0.33/results/team_001/analysis.json"
    old = json.loads(old_path.read_text())
    old_rows = {row["seed"]: row for row in old["seed_rows"] if row["condition"] == "dual_complementary"}
    new_rows = {}
    for seed in SEEDS:
        selected = [row for row in run_rows if row["seed"] == seed and row["schedule"] == "A"]
        new_rows[seed] = float(np.mean([row["J"] for row in selected]))
    replication = [{"seed": seed, "v033_fixed_J": old_rows[seed]["scores"]["target12"]["pooled"]["J"], "v034_rotating_native_A_J": new_rows[seed], "difference": new_rows[seed] - old_rows[seed]["scores"]["target12"]["pooled"]["J"]} for seed in SEEDS]
    record = {"status": "complete", "formal": True, "endpoint": 2400, "source_sha256": sha(ROOT / "rotation_transfer.py"),
              "seeds": list(SEEDS), "partitions": list(PARTITIONS), "schedules": list(SCHEDULES), "rows": rows,
              "run_rows": run_rows, "summary": summary, "paired_vs_A": paired, "replication_vs_v033": replication,
              "definitions": {"A": "native endpoint; fixed training schedule and even-step rotating schedule",
                              "B": "seen only on odd rotating training updates",
                              "C": "held-out schedule, never used during training",
                              "J": "greedy two-resource target12 success averaged over four team slots"},
              "input": str(old_path.resolve()), "input_sha256": sha(old_path)}
    (out / "rotation_transfer.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    record = run(args.out.resolve()); print(json.dumps({"status": record["status"], "source_sha256": record["source_sha256"], "summary": record["summary"]}, ensure_ascii=False))


if __name__ == "__main__": main()
