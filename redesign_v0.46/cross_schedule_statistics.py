"""Paired statistics for the v0.46 crossed newcomer adaptation batch."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CULTURES = (
    "static_role__fixed_A",
    "static_role__rotating_AB",
    "static_role__random_ABC",
    "random_role__fixed_A",
    "random_role__rotating_AB",
    "random_role__random_ABC",
)
SCHEDULES = ("fixed_A", "rotating_AB", "random_ABC")
UPDATES = (0, 100, 300)
BOOTSTRAP_SEED = 46047
BOOTSTRAP_REPS = 20000


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "values": a.tolist()}


def bootstrap_delta(a, b, rng):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError((a.shape, b.shape))
    delta = b - a
    indices = rng.integers(0, delta.size, size=(BOOTSTRAP_REPS, delta.size))
    means = delta[indices].mean(axis=1)
    return {"n": int(delta.size), "mean": float(delta.mean()), "sd": float(delta.std(ddof=1)), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], "values": delta.tolist()}


def run(out: Path):
    analysis_path = out / "cross_schedule_analysis.json"
    data = read(analysis_path)
    if data.get("status") != "complete" or data.get("runs") != 1296:
        raise AssertionError("complete v0.46 analysis required")
    rows = data["rows"]
    if len(rows) != 279936:
        raise AssertionError(f"row coverage {len(rows)}")

    chain_values = defaultdict(list)
    for row in rows:
        if row["role_permutation"] != "012":
            continue
        key = (row["resident_culture"], row["adaptation_schedule"], row["seed"], row["partition"], row["assignment"], row["update"])
        chain_values[key].append(float(row["equivariant"]["J"]))
    for key, values in chain_values.items():
        if len(values) != 12:
            raise AssertionError((key, len(values)))
    endpoint = {}
    for culture in CULTURES:
        endpoint[culture] = {}
        for adaptation in SCHEDULES:
            endpoint[culture][adaptation] = [float(np.mean(chain_values[(culture, adaptation, seed, part, assignment, 300)])) for seed in (34101,34102,34103,34104) for part in (1,2,3) for assignment in ("012","021","102","120","201","210")]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    paired_schedule = {}
    for culture in CULTURES:
        paired_schedule[culture] = {}
        for left, right in (("fixed_A", "rotating_AB"), ("fixed_A", "random_ABC"), ("rotating_AB", "random_ABC")):
            paired_schedule[culture][f"{right}_minus_{left}"] = bootstrap_delta(endpoint[culture][left], endpoint[culture][right], rng)

    gains = {}
    for culture in CULTURES:
        gains[culture] = {}
        for adaptation in SCHEDULES:
            start = [float(np.mean(chain_values[(culture, adaptation, seed, part, assignment, 0)])) for seed in (34101,34102,34103,34104) for part in (1,2,3) for assignment in ("012","021","102","120","201","210")]
            gains[culture][adaptation] = summarize(np.asarray(endpoint[culture][adaptation]) - np.asarray(start))

    # Pooled main effect of role randomization, paired within visual group,
    # resident condition and adaptation schedule.
    role_effect = {}
    for adaptation in SCHEDULES:
        paired = []
        for condition in ("fixed_A", "rotating_AB", "random_ABC"):
            static = endpoint["static_role__" + condition][adaptation]
            random = endpoint["random_role__" + condition][adaptation]
            paired.extend(np.asarray(random) - np.asarray(static))
        role_effect[adaptation] = summarize(paired)

    record = {
        "status": "complete",
        "formal": True,
        "probe": "cross_adaptation_schedules",
        "runs": 1296,
        "endpoint": {culture: {adaptation: summarize(values) for adaptation, values in schedules.items()} for culture, schedules in endpoint.items()},
        "paired_schedule_effects": paired_schedule,
        "recovery_gains": gains,
        "role_randomization_effect": role_effect,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPS, "unit": "seed×partition×resource_assignment chain, averaged over four eval teams/topologies"},
        "source_sha256": sha(Path(__file__).resolve()),
        "analysis_sha256": sha(analysis_path),
        "limits": ["Paired endpoint effects summarize the 72 visual-group chains per culture; they are descriptive uncertainty intervals, not a substitute for a preregistered hierarchical model.", "The newcomer and residents share frozen visual frontends and the resident parameters are frozen during adaptation."],
    }
    (out / "cross_schedule_statistics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"], "runs": record["runs"], "schedule_effect_cells": len(CULTURES) * 3, "bootstrap_repetitions": BOOTSTRAP_REPS}, ensure_ascii=False))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.out.resolve())


if __name__ == "__main__":
    main()
