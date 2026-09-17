"""Independent replay audit for the FI message locality probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from . import runner


def require(ok, message):
    if not ok:
        raise ValueError(message)


def compare(expected, actual, prefix=""):
    require(set(expected) == set(actual), prefix + "keys differ")
    max_error = 0.0
    for key in expected:
        a, b = expected[key], actual[key]
        if isinstance(a, dict):
            max_error = max(max_error, compare(a, b, prefix + key + "/"))
        elif isinstance(a, list):
            require(len(a) == len(b), prefix + key + " length differs")
            for i, (x, y) in enumerate(zip(a, b)):
                if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                    err = abs(float(x) - float(y)); max_error = max(max_error, err); require(np.isclose(x, y, atol=2e-13, rtol=0), prefix + f"{key}[{i}] differs")
                else:
                    require(x == y, prefix + f"{key}[{i}] differs")
        elif isinstance(a, (int, float)):
            err = abs(float(a) - float(b)); max_error = max(max_error, err); require(np.isclose(a, b, atol=2e-13, rtol=0), prefix + key + " differs")
        else:
            require(a == b, prefix + key + " differs")
    return max_error


def replay_seed(payload):
    seed, prepared, saved_rows = payload
    arrays = runner.fi.make_arrays(prepared["partitions"]["new_layouts"]); max_error = 0.0; checked = 0
    grouped = {(r["schedule"], r["trained_channel"], r["mode"]): r for r in saved_rows}
    require(len(grouped) == 32, f"Seed {seed} row count")
    for schedule in runner.SCHEDULES:
        for channel in runner.CHANNELS:
            cond = runner.condition(schedule, channel); checkpoint = runner.SOURCE / "execution" / f"seed_{seed}_{cond}" / "checkpoint_6000.npz"
            nets = runner.fi.load_networks(checkpoint); natural = None
            for mode in runner.MODES:
                row = grouped[schedule, channel, mode]; require(runner.sha(checkpoint) == row["checkpoint_sha256"], "Checkpoint hash mismatch")
                actual = runner.evaluate_mode(nets, arrays, prepared["partitions"]["new_layouts"], channel, mode)
                max_error = max(max_error, compare(row["metrics"], actual, f"{seed}/{schedule}/{channel}/{mode}/"))
                if mode == "natural": natural = actual
                expected_delta = row["delta_from_natural"]
                actual_delta = runner.delta_metrics(natural, actual)
                max_error = max(max_error, compare(expected_delta, actual_delta, f"{seed}/{schedule}/{channel}/{mode}/delta/"))
                checked += 1
    return checked, max_error


def main(source, output):
    source = Path(source).resolve(); output = Path(output).resolve(); output.mkdir(parents=False, exist_ok=False)
    runner.verify(source); saved = runner.read(source / "execution/results.json")
    require(saved.get("status") == "completed_json_only_locality_probe", "Invalid locality result")
    rows = saved.get("rows", []); require(len(rows) == runner.budget()["rows"], "Incomplete locality rows")
    prepared = runner.read(source / "prepared.json")
    by_seed = {seed: [r for r in rows if int(r["seed"]) == seed] for seed in runner.SEEDS}
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        checks = pool.map(replay_seed, [(seed, prepared, by_seed[seed]) for seed in runner.SEEDS])
    checked = sum(c for c, _ in checks); max_error = max(e for _, e in checks)
    verification = dict(status="passed", source=str(source), policy_rows=64, modes=len(runner.MODES), rows=len(rows), evaluations=512,
                        worlds=runner.budget()["worlds"], model_forwards=runner.budget()["model_forwards"], optimizer_updates=0,
                        checkpoint_hashes_checked=64, rows_replayed=checked, max_abs_error=float(max_error),
                        route_modes=list(runner.MODES), first_window_recomputed=True, second_window_recomputed=True,
                        intervention=runner.config()["route_definition"])
    path = output / "verification.json"; path.write_text(json.dumps(verification, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    receipt = dict(status="passed", verification_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), model_forwards=verification["model_forwards"], optimizer_updates=0)
    (output / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps(verification, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--source", required=True); parser.add_argument("--output", required=True); args = parser.parse_args(); main(args.source, args.output)

