"""Paired bootstrap statistics for the v0.47 reset-granularity batch."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CULTURES = (
    "static_role__fixed_A", "static_role__rotating_AB", "static_role__random_ABC",
    "random_role__fixed_A", "random_role__rotating_AB", "random_role__random_ABC",
)
RESET_MODES = ("sender_only", "receiver_only", "both")
UPDATES = (0, 100, 300)
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
ASSIGNMENTS = ("012", "021", "102", "120", "201", "210")
BOOTSTRAP_SEED = 47047
BOOTSTRAP_REPS = 20000


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values):
    a = np.asarray(values, dtype=float)
    return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)) if a.size > 1 else 0.0, "values": a.tolist()}


def bootstrap_delta(left, right, rng):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError((left.shape, right.shape))
    delta = right - left
    indices = rng.integers(0, delta.size, size=(BOOTSTRAP_REPS, delta.size))
    means = delta[indices].mean(axis=1)
    return {"n": int(delta.size), "mean": float(delta.mean()), "sd": float(delta.std(ddof=1)), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], "values": delta.tolist()}


def run(out: Path):
    analysis_path = out / "reset_granularity_analysis.json"
    data = read(analysis_path)
    if data.get("status") != "complete" or data.get("runs") != 1296 or tuple(data.get("reset_modes", ())) != RESET_MODES:
        raise AssertionError("complete v0.47 analysis required")
    if len(data.get("rows", [])) != 279936:
        raise AssertionError(f"row coverage {len(data.get('rows', []))}")
    chain_values = defaultdict(list)
    for row in data["rows"]:
        if row["role_permutation"] == "012":
            key = (row["resident_culture"], row["reset_mode"], row["seed"], row["partition"], row["assignment"], row["update"])
            chain_values[key].append(float(row["equivariant"]["J"]))
    for key, values in chain_values.items():
        if len(values) != 12:
            raise AssertionError((key, len(values)))
    endpoint = {}
    start = {}
    for culture, reset_mode in itertools.product(CULTURES, RESET_MODES):
        endpoint[culture, reset_mode] = [float(np.mean(chain_values[(culture, reset_mode, seed, part, assignment, 300)])) for seed in SEEDS for part in PARTITIONS for assignment in ASSIGNMENTS]
        start[culture, reset_mode] = [float(np.mean(chain_values[(culture, reset_mode, seed, part, assignment, 0)])) for seed in SEEDS for part in PARTITIONS for assignment in ASSIGNMENTS]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    effects = {}
    for culture in CULTURES:
        effects[culture] = {}
        for left, right in (("sender_only", "receiver_only"), ("sender_only", "both"), ("receiver_only", "both")):
            effects[culture][f"{right}_minus_{left}"] = bootstrap_delta(endpoint[culture, left], endpoint[culture, right], rng)
    gains = {culture: {mode: summarize(np.asarray(endpoint[culture, mode]) - np.asarray(start[culture, mode])) for mode in RESET_MODES} for culture in CULTURES}
    role_effect = {}
    for reset_mode in RESET_MODES:
        pooled = []
        for condition in ("fixed_A", "rotating_AB", "random_ABC"):
            pooled.extend(np.asarray(endpoint["random_role__" + condition, reset_mode]) - np.asarray(endpoint["static_role__" + condition, reset_mode]))
        role_effect[reset_mode] = summarize(pooled)
    record = {
        "status": "complete", "formal": True, "probe": "reset_granularity", "runs": 1296,
        "endpoint": {culture: {mode: summarize(endpoint[culture, mode]) for mode in RESET_MODES} for culture in CULTURES},
        "paired_reset_effects": effects, "recovery_gains": gains, "role_randomization_effect": role_effect,
        "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPS, "unit": "seed×partition×resource_assignment chain, averaged over four eval teams/topologies"},
        "source_sha256": sha(Path(__file__).resolve()), "analysis_sha256": sha(analysis_path),
        "limits": ["Paired endpoint effects summarize 72 visual-group chains per resident culture; intervals are descriptive.", "The newcomer inherits the resident visual frontend and all unreset communication modules."],
    }
    (out / "reset_granularity_statistics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"], "runs": record["runs"], "reset_effect_cells": len(CULTURES) * 3, "bootstrap_repetitions": BOOTSTRAP_REPS}, ensure_ascii=False))
    return record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); run(args.out.resolve())


if __name__ == "__main__":
    main()
