"""Independent NumPy analysis for the v0.35 zero-shot identity probe."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
V034 = ROOT.parent / "redesign_v0.34"
ROTATION = V034 / "results" / "rotation_001"
FIXED = V034 / "results" / "fixed_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB")
SCHEDULES = ("A", "B", "C")
ROLES = ("receiver", "food_sender", "water_sender")
PRIVATE_TYPES = (0, 1, 0, 1)
TEAMS = {
    "A": ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)),
    "B": ((0, 2, 3), (1, 3, 0), (2, 0, 1), (3, 1, 2)),
    "C": ((0, 3, 1), (1, 0, 2), (2, 1, 3), (3, 2, 0)),
}
ROLE_INDEX = {"receiver": 0, "food_sender": 1, "water_sender": 2}
MAPS = tuple((food, water) for food in range(6) for water in range(6) if food != water)
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TARGET_PAIRS = ((0, 1), (0, 3), (1, 2), (1, 4), (2, 0), (2, 5), (3, 2), (3, 4), (4, 0), (4, 5), (5, 1), (5, 3))
TRAIN_PAIRS = ((0, 2), (0, 5), (1, 0), (1, 3), (2, 1), (2, 4), (3, 1), (3, 5), (4, 2), (4, 3), (5, 0), (5, 4))


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def target_ids(partition: int, pairs=TARGET_PAIRS):
    order = PANELS[partition - 1]
    index = {pair: i for i, pair in enumerate(MAPS)}
    return np.asarray(sorted(index[(order[i], order[j])] for i, j in pairs), dtype=np.int64)


def score(raw, partition: int):
    decoder = np.argmax(raw["receiver_logits"], axis=-1)
    codes = 7 * raw["tokens"][:, 0] + raw["tokens"][:, 1]
    action = decoder[codes]
    correct = action == raw["positions"]
    target = np.isin(raw["map_id"], target_ids(partition))
    train = np.isin(raw["map_id"], target_ids(partition, TRAIN_PAIRS))
    held = np.isin(raw["map_id"], np.setdiff1d(np.arange(30, dtype=np.int64), target_ids(partition, TRAIN_PAIRS)))
    return {
        split: {
            key: float(np.mean(correct[mask, 0 if key == "food" else 1])) if key in ("food", "water") else float(np.mean(np.all(correct[mask], axis=-1)))
            for key in ("J", "food", "water")
        }
        for split, mask in (("train12", train), ("target12", target), ("held18", held))
    }


def mean_sd(values):
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)), "values": arr.tolist()}


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
        self.require(np.array_equal(actual, expected), label)

    def close(self, actual, expected, label: str):
        a = np.asarray(actual)
        b = np.asarray(expected)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
        self.comparisons += int(a.size)
        self.max_error = max(self.max_error, error)
        self.require(np.allclose(a, b, atol=1e-6, rtol=1e-6), label + " numerical")

    def add(self, key: str, value: int = 1):
        self.coverage[key] += value


def check_world(raw, worlds, checks: Checks, label: str):
    for key in ("map_id", "photo_ids", "positions", "shown"):
        checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"])
    checks.require(raw["tokens"].shape == (n, 2) and raw["tokens"].dtype.kind in "iu", label + " tokens")
    checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), label + " token domain")
    checks.require(raw["sender_log_probs"].shape == (n, 49) and np.isfinite(raw["sender_log_probs"]).all(), label + " joint log table")
    checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1.0, atol=1e-6), label + " joint normalization")
    checks.require(raw["sender_log_probs_food"].shape == (n, 7) and raw["sender_log_probs_water"].shape == (n, 7), label + " component log tables")
    checks.require(np.allclose(np.exp(raw["sender_log_probs_food"]).sum(-1), 1.0, atol=1e-6), label + " food normalization")
    checks.require(np.allclose(np.exp(raw["sender_log_probs_water"]).sum(-1), 1.0, atol=1e-6), label + " water normalization")
    checks.close(raw["sender_log_probs"], (raw["sender_log_probs_food"][:, :, None] + raw["sender_log_probs_water"][:, None, :]).reshape(n, 49), label + " joint factorization")
    checks.require(raw["receiver_logits"].shape == (49, 2, 6) and np.isfinite(raw["receiver_logits"]).all(), label + " receiver table")
    checks.add("episode_worlds", n)


def run(out: Path):
    checks = Checks()
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    episodes_record = read(out / "episodes.json")
    checks.exact(invocation["conditions"], list(CONDITIONS), "condition inventory")
    checks.exact(invocation["schedules"], list(SCHEDULES), "schedule inventory")
    checks.exact(invocation["roles"], list(ROLES), "role inventory")
    checks.exact(invocation["endpoint_rows_expected"], 864, "expected endpoint rows")
    checks.exact(complete["status"], "complete", "probe completion")
    checks.exact(complete["formal"], True, "formal probe")
    checks.exact(complete["endpoint_rows"], 864, "endpoint rows")
    checks.exact(episodes_record["status"], "complete", "episode manifest")
    checks.exact(episodes_record["count"], 864, "episode manifest count")
    checks.exact(len(episodes_record["rows"]), 864, "episode manifest rows")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " binding")
        for path, digest in invocation[key].items():
            file_path = Path(path)
            checks.require(file_path.is_file(), key + " bound path exists")
            checks.exact(sha(file_path), digest, key + " bound hash")
    worlds = npz(out / "test_worlds.npz")
    checks.exact(len(worlds["map_id"]), 180, "test world count")

    # Baseline completion manifests bind every endpoint file used below.
    complete_by_condition = {"fixed_A": (FIXED, read(FIXED / "training_complete.json")), "rotating_AB": (ROTATION, read(ROTATION / "training_complete.json"))}
    rows = []
    baseline_hashes = {}
    for index, entry in enumerate(episodes_record["rows"]):
        condition = entry["condition"]
        checks.require(condition in CONDITIONS, "episode condition")
        seed, panel, schedule, role, held = entry["seed"], entry["partition"], entry["schedule"], entry["role"], entry["held_identity"]
        checks.require((seed, panel, schedule, role, held) in itertools.product(SEEDS, PARTITIONS, SCHEDULES, ROLES, range(4)), "episode identity tuple")
        team = TEAMS[schedule][entry["slot"]]
        role_index = ROLE_INDEX[role]
        checks.exact(team[role_index], held, "held identity slot")
        checks.exact(entry["private_type"], PRIVATE_TYPES[held], "newcomer private type")
        expected_replacement = 4 + PRIVATE_TYPES[held]
        checks.exact(entry[f"{role.replace('_sender', '')}_model"] if role == "receiver" else entry["food_model"] if role == "food_sender" else entry["water_model"], expected_replacement, "replacement model id")
        episode_path = out / entry["path"]
        checks.require(episode_path.is_file(), "episode path exists")
        raw = npz(episode_path)
        check_world(raw, worlds, checks, f"episode {index}")
        condition_folder = episode_path.parent
        tables = npz(condition_folder / "model_tables.npz")
        replacement = expected_replacement
        receiver_model = entry["receiver_model"]; food_model = entry["food_model"]; water_model = entry["water_model"]
        checks.close(raw["tokens"][:, 0], tables["food_tokens"][food_model], "food model table")
        checks.close(raw["tokens"][:, 1], tables["water_tokens"][water_model], "water model table")
        checks.close(raw["receiver_logits"], tables["receiver_logits"][receiver_model], "receiver model table")
        baseline_path = Path(entry["baseline_path"])
        base_out, base_complete = complete_by_condition[condition]
        checks.require(baseline_path.is_file(), "baseline path exists")
        rel = str(baseline_path.relative_to(base_out))
        checks.require(rel in base_complete["files"], "baseline completion binding")
        baseline_hashes[str(baseline_path)] = sha(baseline_path)
        checks.exact(sha(baseline_path), base_complete["files"][rel], "baseline hash")
        baseline = npz(baseline_path)
        check_world(baseline, worlds, checks, f"baseline {index}")
        newcomer_score = score(raw, panel)
        baseline_score = score(baseline, panel)
        delta = {split: {key: newcomer_score[split][key] - baseline_score[split][key] for key in ("J", "food", "water")} for split in ("train12", "target12", "held18")}
        token_match = None
        if role == "food_sender":
            token_match = float(np.mean(raw["tokens"][:, 0] == baseline["tokens"][:, 0]))
        elif role == "water_sender":
            token_match = float(np.mean(raw["tokens"][:, 1] == baseline["tokens"][:, 1]))
        rows.append({
            **{key: entry[key] for key in ("seed", "partition", "condition", "schedule", "role", "held_identity", "private_type", "slot", "receiver_model", "food_model", "water_model")},
            "newcomer": newcomer_score,
            "baseline": baseline_score,
            "delta": delta,
            "token_match_to_baseline": token_match,
        })
        checks.add("episode_rows")
    # Ensure the manifest contains exactly one row for every factorial cell.
    keys = [(r["condition"], r["seed"], r["partition"], r["schedule"], r["role"], r["held_identity"]) for r in rows]
    checks.exact(len(set(keys)), 864, "factorial episode uniqueness")

    def summarize(selected):
        return {
            "newcomer": {split: {key: mean_sd([row["newcomer"][split][key] for row in selected]) for key in ("J", "food", "water")} for split in ("train12", "target12", "held18")},
            "baseline": {split: {key: mean_sd([row["baseline"][split][key] for row in selected]) for key in ("J", "food", "water")} for split in ("train12", "target12", "held18")},
            "delta": {split: {key: mean_sd([row["delta"][split][key] for row in selected]) for key in ("J", "food", "water")} for split in ("train12", "target12", "held18")},
            "token_match_to_baseline": mean_sd([row["token_match_to_baseline"] for row in selected if row["token_match_to_baseline"] is not None]) if any(row["token_match_to_baseline"] is not None for row in selected) else None,
            "n": len(selected),
        }

    summary = {condition: {schedule: {role: summarize([row for row in rows if row["condition"] == condition and row["schedule"] == schedule and row["role"] == role]) for role in ROLES} for schedule in SCHEDULES} for condition in CONDITIONS}
    type_summary = {condition: {private_type: summarize([row for row in rows if row["condition"] == condition and row["private_type"] == private_type]) for private_type in (0, 1)} for condition in CONDITIONS}

    seed_summary = {}
    for condition, schedule, role, seed in itertools.product(CONDITIONS, SCHEDULES, ROLES, SEEDS):
        selected = [row for row in rows if row["condition"] == condition and row["schedule"] == schedule and row["role"] == role and row["seed"] == seed]
        seed_summary[f"{condition}/{schedule}/{role}/{seed}"] = summarize(selected)

    paired = []
    for key in itertools.product(SEEDS, PARTITIONS, SCHEDULES, ROLES, range(4)):
        seed, panel, schedule, role, held = key
        fixed = next(row for row in rows if row["condition"] == "fixed_A" and row["seed"] == seed and row["partition"] == panel and row["schedule"] == schedule and row["role"] == role and row["held_identity"] == held)
        rotating = next(row for row in rows if row["condition"] == "rotating_AB" and row["seed"] == seed and row["partition"] == panel and row["schedule"] == schedule and row["role"] == role and row["held_identity"] == held)
        paired.append({"seed": seed, "partition": panel, "schedule": schedule, "role": role, "held_identity": held, "rotating_minus_fixed": {split: {key: rotating["newcomer"][split][key] - fixed["newcomer"][split][key] for key in ("J", "food", "water")} for split in ("train12", "target12", "held18")}})
    paired_summary = {schedule: {role: {split: {key: mean_sd([row["rotating_minus_fixed"][split][key] for row in paired if row["schedule"] == schedule and row["role"] == role]) for key in ("J", "food", "water")} for split in ("train12", "target12", "held18")} for role in ROLES} for schedule in SCHEDULES}

    record = {
        "status": "complete",
        "formal": True,
        "probe": "zero_shot_identity_holdout",
        "namespace": "v0.34-34034",
        "seeds": list(SEEDS),
        "partitions": list(PARTITIONS),
        "conditions": list(CONDITIONS),
        "schedules": list(SCHEDULES),
        "roles": list(ROLES),
        "rows": rows,
        "summary": summary,
        "private_type_summary": type_summary,
        "seed_summary": seed_summary,
        "paired_rotating_minus_fixed": paired_summary,
        "paired_rows": paired,
        "baseline_hashes": baseline_hashes,
        "checks": checks.count,
        "scalar_comparisons": checks.comparisons,
        "maximum_metric_absolute_difference": checks.max_error,
        "source_sha256": sha(ROOT / "identity_analysis.py"),
        "probe_training_complete_sha256": sha(out / "training_complete.json"),
        "limits": [
            "The newcomer is zero-shot: its communication modules are freshly reset and never socially trained.",
            "The same private visual encoder type is retained, so this isolates social identity rather than perceptual competence.",
            "The role replacement is an endpoint intervention; it does not model online cultural adaptation.",
        ],
    }
    write(out / "identity_analysis.json", record)
    qa = {
        "passed": True,
        "status": "passed_zero_shot_identity_recheck",
        "formal": True,
        "checks": checks.count,
        "scalar_comparisons": checks.comparisons,
        "maximum_metric_absolute_difference": checks.max_error,
        "coverage": dict(checks.coverage),
        "analysis_source_sha256": sha(ROOT / "identity_analysis.py"),
        "analysis_sha256": sha(out / "identity_analysis.json"),
        "checked_baseline_sha256": baseline_hashes,
        "model_calls": 0,
        "production_modules_imported": False,
    }
    write(out / "identity_raw_validation.json", qa)
    shutil.copy2(ROOT / "identity_analysis.py", out / "identity_analysis_source.py")
    return record, qa


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    started = time.monotonic()
    try:
        record, qa = run(args.out.resolve())
        qa["seconds"] = time.monotonic() - started
        write(args.out.resolve() / "identity_raw_validation.json", qa)
        print(json.dumps({key: record[key] for key in ("status", "checks", "scalar_comparisons", "maximum_metric_absolute_difference")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(args.out.resolve() / f"identity_analysis_failure_{stamp}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "identity_analysis.py")}); raise


if __name__ == "__main__": main()
