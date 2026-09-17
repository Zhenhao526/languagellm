"""Paired bootstrap statistics for the v0.48 co-adaptation batch."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


CULTURES = ("static_role__fixed_A", "static_role__rotating_AB", "static_role__random_ABC", "random_role__fixed_A", "random_role__rotating_AB", "random_role__random_ABC")
RESIDENT_MODES = ("resident_frozen", "resident_sender_sparse", "resident_receiver_sparse", "resident_both_sparse")
UPDATES = (0, 100, 300, 600)
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
ASSIGNMENTS = ("012", "021", "102", "120", "201", "210")
BOOTSTRAP_SEED = 48047
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
    analysis_path = out / "coadaptation_analysis.json"; data = read(analysis_path)
    if data.get("status") != "complete" or data.get("runs") != 1728 or tuple(data.get("resident_adaptation_modes", ())) != RESIDENT_MODES:
        raise AssertionError("complete v0.48 analysis required")
    if len(data.get("rows", [])) != 497664:
        raise AssertionError(f"row coverage {len(data.get('rows', []))}")
    chain_values = defaultdict(list)
    for row in data["rows"]:
        if row["role_permutation"] == "012":
            key = (row["resident_culture"], row["resident_mode"], row["seed"], row["partition"], row["assignment"], row["update"])
            chain_values[key].append(float(row["equivariant"]["J"]))
    for key, values in chain_values.items():
        if len(values) != 12:
            raise AssertionError((key, len(values)))
    endpoint, start = {}, {}
    for culture, resident_mode in itertools.product(CULTURES, RESIDENT_MODES):
        endpoint[culture, resident_mode] = [float(np.mean(chain_values[(culture, resident_mode, seed, part, assignment, 600)])) for seed in SEEDS for part in PARTITIONS for assignment in ASSIGNMENTS]
        start[culture, resident_mode] = [float(np.mean(chain_values[(culture, resident_mode, seed, part, assignment, 0)])) for seed in SEEDS for part in PARTITIONS for assignment in ASSIGNMENTS]
    rng = np.random.default_rng(BOOTSTRAP_SEED); effects = {}
    comparisons = (("resident_frozen", "resident_sender_sparse"), ("resident_frozen", "resident_receiver_sparse"), ("resident_frozen", "resident_both_sparse"), ("resident_sender_sparse", "resident_receiver_sparse"), ("resident_sender_sparse", "resident_both_sparse"), ("resident_receiver_sparse", "resident_both_sparse"))
    for culture in CULTURES:
        effects[culture] = {}
        for left, right in comparisons:
            effects[culture][f"{right}_minus_{left}"] = bootstrap_delta(endpoint[culture, left], endpoint[culture, right], rng)
    gains = {culture: {mode: summarize(np.asarray(endpoint[culture, mode]) - np.asarray(start[culture, mode])) for mode in RESIDENT_MODES} for culture in CULTURES}
    role_effect = {}
    for resident_mode in RESIDENT_MODES:
        pooled = []
        for condition in ("fixed_A", "rotating_AB", "random_ABC"):
            pooled.extend(np.asarray(endpoint["random_role__" + condition, resident_mode]) - np.asarray(endpoint["static_role__" + condition, resident_mode]))
        role_effect[resident_mode] = summarize(pooled)
    drift_summary = {}
    summary = data["summary"]
    for culture, mode in itertools.product(CULTURES, RESIDENT_MODES):
        cell = summary[culture][mode]["600"]
        drift_summary[culture, mode] = {"mean": cell["resident_communication_drift_mean"], "max": cell["resident_communication_drift_max"]}
    record = {"status": "complete", "formal": True, "probe": "coadaptation_horizon", "runs": 1728, "endpoint": {culture: {mode: summarize(endpoint[culture, mode]) for mode in RESIDENT_MODES} for culture in CULTURES}, "paired_resident_effects": effects, "recovery_gains": gains, "role_randomization_effect": role_effect, "resident_drift_endpoint": {culture: {mode: drift_summary[culture, mode] for mode in RESIDENT_MODES} for culture in CULTURES}, "bootstrap": {"seed": BOOTSTRAP_SEED, "repetitions": BOOTSTRAP_REPS, "unit": "seed×partition×resource_assignment chain, averaged over four eval teams/topologies"}, "source_sha256": sha(Path(__file__).resolve()), "analysis_sha256": sha(analysis_path), "limits": ["Paired endpoint effects summarize 72 matched visual groups per resident culture; intervals are descriptive.", "Resident adaptation is capped at 30 steps and uses a lower learning rate; this is a bounded co-adaptation probe, not unrestricted social learning."]}
    (out / "coadaptation_statistics.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": record["status"], "runs": record["runs"], "resident_effect_cells": len(CULTURES) * len(comparisons), "bootstrap_repetitions": BOOTSTRAP_REPS}, ensure_ascii=False)); return record


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); run(args.out.resolve())


if __name__ == "__main__": main()
