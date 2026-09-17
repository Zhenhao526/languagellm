"""Independent analysis for the same-namespace fixed-A topology control.

The production runner is never imported.  This script reuses the NumPy-only
scoring routines from ``analyze_results.py`` and verifies the saved endpoint
tables against the sealed control run before making any comparison with the
rotating batch.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
SCHEDULES = ("A", "B", "C")
CONDITION = "dual_complementary"
TIMES = (0, 100, 600, 1200, 2100, 2400)
WORLD_KEYS = ("map_id", "photo_ids", "positions", "shown")


def _load_numpy_scoring():
    path = ROOT / "analyze_results.py"
    spec = importlib.util.spec_from_file_location("v034_numpy_scoring", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load NumPy scoring source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SCORING = _load_numpy_scoring()


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Checks:
    def __init__(self):
        self.count = 0
        self.comparisons = 0
        self.max_error = 0.0
        self.coverage = Counter()

    def require(self, value, label: str):
        self.count += 1
        if not bool(value):
            raise AssertionError(label)

    def exact(self, actual, expected, label: str):
        if isinstance(expected, dict):
            self.require(set(actual) == set(expected), label + " keys")
            for key in expected:
                self.exact(actual[key], expected[key], label + "/" + str(key))
        elif isinstance(expected, (list, tuple)):
            self.require(len(actual) == len(expected), label + " length")
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.exact(a, b, label + "/" + str(i))
        elif isinstance(expected, np.ndarray):
            self.require(np.array_equal(actual, expected), label)
        else:
            self.require(actual == expected, label)

    def close(self, actual, expected, label: str):
        if isinstance(expected, dict):
            self.require(set(actual) == set(expected), label + " keys")
            for key in expected:
                self.close(actual[key], expected[key], label + "/" + str(key))
        elif isinstance(expected, (list, tuple)):
            self.require(len(actual) == len(expected), label + " length")
            for i, (a, b) in enumerate(zip(actual, expected)):
                self.close(a, b, label + "/" + str(i))
        elif isinstance(expected, (bool, str)) or expected is None:
            self.exact(actual, expected, label)
        elif isinstance(expected, (int, float, np.number)):
            a, b = float(actual), float(expected)
            self.require(np.isfinite(a) and np.isfinite(b), label + " finite")
            self.comparisons += 1
            self.max_error = max(self.max_error, abs(a - b))
            self.require(np.isclose(a, b, rtol=1e-9, atol=1e-10), label + " numerical")
        else:
            self.exact(actual, expected, label)

    def add(self, key: str, value: int = 1):
        self.coverage[key] += value


def mean_sd(values):
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)), "values": arr.tolist()}


def nested_mean(rows):
    if isinstance(rows[0], dict):
        return {key: nested_mean([row[key] for row in rows]) for key in rows[0]}
    return float(np.mean(rows))


def auc(values):
    y = np.asarray(values, dtype=np.float64)
    x = np.asarray(TIMES, dtype=np.float64)
    return float(np.trapezoid(y, x) / (x[-1] - x[0]))


def check_binding(invocation, complete, checks: Checks):
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            file_path = Path(path)
            checks.require(file_path.is_file(), key + " bound path exists")
            checks.exact(sha(file_path), digest, key + " bound hash")


def check_world(raw, worlds, checks: Checks, label: str):
    for key in WORLD_KEYS:
        checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"])
    checks.require(raw["tokens"].shape == (n, 2), label + " token shape")
    checks.require(raw["tokens"].dtype.kind in "iu", label + " token dtype")
    checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), label + " token domain")
    checks.require(raw["sender_log_probs"].shape == (n, 49), label + " sender shape")
    checks.require(np.isfinite(raw["sender_log_probs"]).all(), label + " sender finite")
    checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1.0, atol=1e-6), label + " sender normalization")
    checks.require(raw["receiver_logits"].shape == (49, 2, 6), label + " receiver shape")
    checks.require(np.isfinite(raw["receiver_logits"]).all(), label + " receiver finite")


def agreement(folder: Path, checks: Checks):
    raw = npz(folder / "agreement_2400.npz")
    expected = {
        f"{view}_type{k}_{suffix}"
        for view in ("full", "food_only", "water_only")
        for k in (0, 1)
        for suffix in ("a", "b")
    }
    checks.require(set(raw) == expected, "agreement inventory")
    for key, value in raw.items():
        checks.require(len(value) == 180, "agreement row count")
    full = [float(np.all(raw[f"full_type{k}_a"] == raw[f"full_type{k}_b"], axis=-1).mean()) for k in (0, 1)]
    food = [float(np.mean(raw[f"food_only_type{k}_a"] == raw[f"food_only_type{k}_b"])) for k in (0, 1)]
    water = [float(np.mean(raw[f"water_only_type{k}_a"] == raw[f"water_only_type{k}_b"])) for k in (0, 1)]
    token0 = [float(np.mean(raw[f"full_type{k}_a"][:, 0] == raw[f"full_type{k}_b"][:, 0])) for k in (0, 1)]
    token1 = [float(np.mean(raw[f"full_type{k}_a"][:, 1] == raw[f"full_type{k}_b"][:, 1])) for k in (0, 1)]
    return {
        "full_message_agreement": float(np.mean(full)),
        "full_token0_agreement": float(np.mean(token0)),
        "full_token1_agreement": float(np.mean(token1)),
        "food_token_agreement": float(np.mean(food)),
        "water_token_agreement": float(np.mean(water)),
    }


def summary_metrics(score):
    out = {}
    for split in ("train12", "target12", "held18", "common30"):
        pooled = score[split]["pooled"]
        out[split] = {key: float(pooled[key]) for key in ("J", "food", "water", "Q")}
    return out


def run(out: Path):
    checks = Checks()
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    checks.require(invocation.get("formal") is True, "formal invocation")
    checks.exact(invocation["seeds"], list(SEEDS), "seeds")
    checks.exact(invocation["partitions"], list(PARTITIONS), "partitions")
    checks.exact(invocation["conditions"], [CONDITION], "single control condition")
    checks.exact(invocation["updates"], 2400, "update count")
    checks.exact(invocation["private_types"], [0, 1, 0, 1], "private types")
    checks.exact(complete["status"], "complete", "training terminal status")
    checks.exact(complete["control"], "fixed_A", "control label")
    checks.exact(complete["social_runs"], 12, "social run count")
    check_binding(invocation, complete, checks)

    wrapper = read(out / "wrapper_receipt.json")
    checks.exact(wrapper["status"], "complete", "wrapper status")
    checks.exact(wrapper["control"], "fixed_A", "wrapper control")
    checks.exact(wrapper["schedule"], "A_every_update", "wrapper schedule")
    checks.exact(wrapper["source_sha256"], sha(ROOT / "run_fixed_control.py"), "wrapper source hash")
    checks.exact(wrapper["production_runner_sha256"], sha(ROOT / "run_support.py"), "runner source hash")

    test_worlds = npz(out / "test_worlds.npz")
    checks.exact(len(test_worlds["map_id"]), 180, "test world count")
    checks.exact(test_worlds["positions"], SCORING.MAPS[test_worlds["map_id"]], "test positions")

    rows = []
    schedule_rows = []
    raw_hashes = {}
    for seed, panel in itertools.product(SEEDS, PARTITIONS):
        folder = out / "social" / f"s{seed}_p{panel}_{CONDITION}"
        cfg = read(folder / "config.json")
        checks.exact(cfg["condition"], CONDITION, "config condition")
        checks.exact(cfg["training_schedule"], "A_only_fixed_control", "config fixed schedule")
        checks.exact(cfg["wrapper"], "run_fixed_control.py", "config wrapper")
        curve = read(folder / "curve.json")
        checks.exact([item["update"] for item in curve], list(TIMES), "curve checkpoints")
        result = read(folder / "result.json")
        checks.exact(result["status"], "complete", "social terminal status")

        team_curves = {team: [] for team in range(4)}
        endpoint_by_team = {}
        for item in curve:
            update = int(item["update"])
            for team in range(4):
                path = folder / f"protocol_{update:04d}_team{team}.npz"
                rel = str(path.relative_to(out))
                checks.require(rel in complete["files"], "protocol completion binding")
                raw = npz(path)
                check_world(raw, test_worlds, checks, f"protocol {seed}/{panel}/{update}/{team}")
                metric = SCORING.score(raw, panel, CONDITION)
                checks.close(metric, item["scores"][f"team{team}"], "saved score recheck")
                team_curves[team].append({"update": update, "scores": summary_metrics(metric)})
                raw_hashes[rel] = sha(path)
                checks.add("protocol_tables")
                checks.add("protocol_worlds", 180)
                if update == 2400:
                    checks.close(metric, result["scores"][f"team{team}"], "terminal score recheck")
                    endpoint_by_team[team] = metric

            agree = agreement(folder, checks)
            checks.add("agreement_files")
            for team in range(4):
                team_curves[team][-1]["agreement"] = agree

        endpoint_mean = nested_mean([endpoint_by_team[team] for team in range(4)])
        trajectory = []
        for index, update in enumerate(TIMES):
            values = [team_curves[team][index]["scores"] for team in range(4)]
            trajectory.append({"update": update, "scores": nested_mean(values)})
        endpoint_agreement = agreement(folder, checks)
        rows.append({
            "seed": seed,
            "partition": panel,
            "condition": CONDITION,
            "schedule": "A",
            "team_scores": {f"team{team}": summary_metrics(endpoint_by_team[team]) for team in range(4)},
            "scores": summary_metrics(endpoint_mean),
            "trajectory": trajectory,
            "auc_target_J": auc([item["scores"]["target12"]["J"] for item in trajectory]),
            "agreement": endpoint_agreement,
        })

        for schedule in SCHEDULES:
            prefix = "protocol" if schedule == "A" else f"transfer_{schedule}"
            team_metrics = []
            for team in range(4):
                path = folder / f"{prefix}_2400_team{team}.npz"
                rel = str(path.relative_to(out))
                checks.require(rel in complete["files"], f"{schedule} completion binding")
                raw = npz(path)
                check_world(raw, test_worlds, checks, f"{schedule} transfer {seed}/{panel}/{team}")
                metric = SCORING.score(raw, panel, CONDITION)
                team_metrics.append(metric)
                raw_hashes[rel] = sha(path)
                checks.add("transfer_tables")
            averaged = nested_mean(team_metrics)
            schedule_rows.append({
                "seed": seed,
                "partition": panel,
                "schedule": schedule,
                "scores": summary_metrics(averaged),
                "auc_target_J": rows[-1]["auc_target_J"] if schedule == "A" else None,
            })

    def aggregate_schedule(schedule: str):
        selected = [row for row in schedule_rows if row["schedule"] == schedule]
        return {
            split: {
                key: mean_sd([row["scores"][split][key] for row in selected])
                for key in ("J", "food", "water", "Q")
            }
            for split in ("train12", "target12", "held18", "common30")
        }

    schedule_summary = {schedule: aggregate_schedule(schedule) for schedule in SCHEDULES}

    # Collapse panels to the prespecified independent seed unit.
    seed_summary = {}
    for seed in SEEDS:
        seed_summary[str(seed)] = {}
        for schedule in SCHEDULES:
            selected = [r for r in schedule_rows if r["seed"] == seed and r["schedule"] == schedule]
            seed_summary[str(seed)][schedule] = {
                split: {key: float(np.mean([r["scores"][split][key] for r in selected])) for key in ("J", "food", "water", "Q")}
                for split in ("train12", "target12", "held18", "common30")
            }

    rotating_out = out.parent / "rotation_001"
    rotation_invocation = read(rotating_out / "invocation.json")
    rotation_complete = read(rotating_out / "training_complete.json")
    checks.exact(invocation["source_hashes"], rotation_invocation["source_hashes"], "paired source namespace")
    checks.exact(invocation["input_hashes"], rotation_invocation["input_hashes"], "paired input namespace")
    rotation_transfer = read(rotating_out / "rotation_transfer.json")
    checks.require(rotation_transfer.get("formal") is True, "rotating transfer formal")
    rotating_rows = {(row["seed"], row["partition"], row["schedule"]): row for row in rotation_transfer["run_rows"]}
    fixed_rows = {(row["seed"], row["partition"], row["schedule"]): row for row in schedule_rows}
    paired_rows = []
    for seed, panel in itertools.product(SEEDS, PARTITIONS):
        record = {"seed": seed, "partition": panel}
        for schedule in SCHEDULES:
            fixed = fixed_rows[(seed, panel, schedule)]["scores"]["target12"]
            rotating = rotating_rows[(seed, panel, schedule)]
            record[schedule] = {
                "fixed": {key: float(fixed[key]) for key in ("J", "food", "water")},
                "rotating": {key: float(rotating[key]) for key in ("J", "food", "water")},
                "rotating_minus_fixed": {key: float(rotating[key] - fixed[key]) for key in ("J", "food", "water")},
            }
        paired_rows.append(record)
    paired_summary = {
        schedule: {
            key: mean_sd([row[schedule]["rotating_minus_fixed"][key] for row in paired_rows])
            for key in ("J", "food", "water")
        }
        for schedule in SCHEDULES
    }

    rotation_analysis = read(rotating_out / "analysis.json")
    rotation_comp_agreement = rotation_analysis["aggregate"][CONDITION]["agreement"][-1]["agreement"]
    fixed_agreement = mean_sd([row["agreement"]["full_message_agreement"] for row in rows])

    record = {
        "status": "complete",
        "formal": True,
        "control": "fixed_A",
        "namespace": "34034",
        "endpoint": 2400,
        "seeds": list(SEEDS),
        "partitions": list(PARTITIONS),
        "schedules": list(SCHEDULES),
        "rows": rows,
        "schedule_rows": schedule_rows,
        "schedule_summary": schedule_summary,
        "seed_summary": seed_summary,
        "paired_with_rotation": {
            "same_namespace": True,
            "rows": paired_rows,
            "summary": paired_summary,
            "interpretation": "rotating minus fixed-A; seed×panel is the paired unit, panels are averaged within seed for primary inference",
        },
        "agreement": {
            "fixed_A_endpoint": fixed_agreement,
            "rotating_endpoint": rotation_comp_agreement,
            "rotating_minus_fixed_full_message": float(rotation_comp_agreement["full_message_agreement"] - fixed_agreement["mean"]),
        },
        "source_sha256": sha(ROOT / "fixed_control_analysis.py"),
        "scoring_dependency_sha256": sha(ROOT / "analyze_results.py"),
        "training_complete_sha256": sha(out / "training_complete.json"),
        "rotation_transfer_sha256": sha(rotating_out / "rotation_transfer.json"),
        "checked_raw_sha256": raw_hashes,
        "checks": checks.count,
        "scalar_comparisons": checks.comparisons,
        "maximum_metric_absolute_difference": checks.max_error,
        "coverage": dict(checks.coverage),
        "production_modules_imported": False,
        "model_calls": 0,
        "limits": [
            "Fixed-A has one condition and 12 social runs; rotating has three conditions and 36 social runs.",
            "The primary causal contrast uses the same seed×panel fixture and averages panels within seed.",
            "This is a topology control in the two-sender protocol, not evidence of a population-wide lexicon or compositional grammar.",
        ],
    }
    write(out / "fixed_control_analysis.json", record)
    qa = {
        "passed": True,
        "formal": True,
        "status": "passed_same_namespace_control_recheck",
        "checks": checks.count,
        "scalar_comparisons": checks.comparisons,
        "maximum_metric_absolute_difference": checks.max_error,
        "coverage": dict(checks.coverage),
        "analysis_source_sha256": sha(ROOT / "fixed_control_analysis.py"),
        "scoring_dependency_sha256": sha(ROOT / "analyze_results.py"),
        "analysis_sha256": sha(out / "fixed_control_analysis.json"),
        "checked_raw_sha256": raw_hashes,
        "production_modules_imported": False,
        "model_calls": 0,
        "seconds": 0.0,
    }
    write(out / "fixed_control_raw_validation.json", qa)
    shutil.copy2(ROOT / "fixed_control_analysis.py", out / "fixed_control_analysis_source.py")
    return record, qa


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    started = time.monotonic()
    try:
        record, qa = run(out)
        qa["seconds"] = time.monotonic() - started
        write(out / "fixed_control_raw_validation.json", qa)
        print(json.dumps({key: record[key] for key in ("status", "checks", "scalar_comparisons", "maximum_metric_absolute_difference", "coverage")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns()
        write(out / f"fixed_control_analysis_failure_{stamp}.json", {"passed": False, "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "fixed_control_analysis.py")})
        raise


if __name__ == "__main__":
    main()
