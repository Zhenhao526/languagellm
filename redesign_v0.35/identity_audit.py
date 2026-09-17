"""Raw-table audit for the v0.35 zero-shot identity endpoint probe."""
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


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Checks:
    def __init__(self):
        self.count = 0
        self.coverage = Counter()

    def require(self, value, label: str):
        self.count += 1
        if not bool(value):
            raise AssertionError(label)

    def exact(self, actual, expected, label: str):
        self.require(np.array_equal(actual, expected), label)

    def close(self, actual, expected, label: str):
        a = np.asarray(actual); b = np.asarray(expected)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.require(np.allclose(a, b, rtol=1e-6, atol=1e-6), label + " numerical")

    def add(self, key: str, value: int = 1):
        self.coverage[key] += value


def check_binding(invocation, complete, checks: Checks):
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            file_path = Path(path)
            checks.require(file_path.is_file(), key + " bound file exists")
            checks.exact(sha(file_path), digest, key + " bound hash")


def check_raw(raw, worlds, checks: Checks, label: str):
    for key in ("map_id", "photo_ids", "positions", "shown"):
        checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"])
    checks.require(raw["tokens"].shape == (n, 2) and raw["tokens"].dtype.kind in "iu", label + " token shape")
    checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), label + " token domain")
    for key in ("sender_log_probs_food", "sender_log_probs_water"):
        checks.require(raw[key].shape == (n, 7) and np.isfinite(raw[key]).all(), label + " component table")
        checks.require(np.allclose(np.exp(raw[key]).sum(-1), 1.0, atol=1e-6), label + " component normalization")
    checks.require(raw["sender_log_probs"].shape == (n, 49) and np.isfinite(raw["sender_log_probs"]).all(), label + " joint table")
    checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1.0, atol=1e-6), label + " joint normalization")
    checks.require(np.allclose(raw["sender_log_probs"], (raw["sender_log_probs_food"][:, :, None] + raw["sender_log_probs_water"][:, None, :]).reshape(n, 49), atol=1e-6, rtol=1e-6), label + " factorization")
    checks.require(raw["receiver_logits"].shape == (49, 2, 6) and np.isfinite(raw["receiver_logits"]).all(), label + " receiver table")
    checks.add("episode_worlds", n)


def run(out: Path):
    checks = Checks()
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    episodes = read(out / "episodes.json")
    checks.exact(invocation["conditions"], list(CONDITIONS), "conditions")
    checks.exact(invocation["schedules"], list(SCHEDULES), "schedules")
    checks.exact(invocation["roles"], list(ROLES), "roles")
    checks.exact(invocation["endpoint_rows_expected"], 864, "expected rows")
    checks.exact(complete["status"], "complete", "completion")
    checks.exact(complete["probe"], "zero_shot_identity_holdout", "probe")
    checks.exact(complete["endpoint_rows"], 864, "endpoint rows")
    checks.exact(episodes["count"], 864, "manifest count")
    check_binding(invocation, complete, checks)
    worlds = npz(out / "test_worlds.npz")
    checks.exact(len(worlds["map_id"]), 180, "world count")
    base_complete = {"fixed_A": (FIXED, read(FIXED / "training_complete.json")), "rotating_AB": (ROTATION, read(ROTATION / "training_complete.json"))}
    for condition, seed, panel in itertools.product(CONDITIONS, SEEDS, PARTITIONS):
        folder = out / "social" / f"s{seed}_p{panel}_{condition}"
        cfg = read(folder / "config.json")
        checks.exact(cfg["resident_ids"], [0, 1, 2, 3], "resident ids")
        checks.exact(cfg["newcomer_model_ids"], {"type0": 4, "type1": 5}, "newcomer ids")
        tables = npz(folder / "model_tables.npz")
        checks.require(tables["food_log_probs"].shape == (6, 180, 7), "food model table shape")
        checks.require(tables["water_log_probs"].shape == (6, 180, 7), "water model table shape")
        checks.require(tables["food_tokens"].shape == (6, 180) and tables["water_tokens"].shape == (6, 180), "token model table shape")
        checks.require(tables["receiver_logits"].shape == (6, 49, 2, 6), "receiver model table shape")
        checks.require(np.isfinite(tables["food_log_probs"]).all() and np.isfinite(tables["water_log_probs"]).all() and np.isfinite(tables["receiver_logits"]).all(), "model table finite")
        checks.exact(tables["food_tokens"], np.argmax(tables["food_log_probs"], axis=-1), "food greedy table")
        checks.exact(tables["water_tokens"], np.argmax(tables["water_log_probs"], axis=-1), "water greedy table")
        for schedule, role, held in itertools.product(SCHEDULES, ROLES, range(4)):
            entry = next(row for row in episodes["rows"] if row["condition"] == condition and row["seed"] == seed and row["partition"] == panel and row["schedule"] == schedule and row["role"] == role and row["held_identity"] == held)
            slot = [i for i, team in enumerate(TEAMS[schedule]) if team[ROLE_INDEX[role]] == held]
            checks.exact(entry["slot"], slot[0], "slot identity")
            checks.exact(entry["private_type"], PRIVATE_TYPES[held], "private type")
            replacement = 4 + PRIVATE_TYPES[held]
            if role == "receiver": checks.exact(entry["receiver_model"], replacement, "receiver replacement")
            if role == "food_sender": checks.exact(entry["food_model"], replacement, "food replacement")
            if role == "water_sender": checks.exact(entry["water_model"], replacement, "water replacement")
            path = out / entry["path"]
            checks.require(path.is_file(), "episode exists")
            check_raw(npz(path), worlds, checks, f"episode {condition}/{seed}/{panel}/{schedule}/{role}/{held}")
            raw = npz(path)
            food_model, water_model, receiver_model = entry["food_model"], entry["water_model"], entry["receiver_model"]
            checks.close(raw["tokens"][:, 0], tables["food_tokens"][food_model], "food endpoint linkage")
            checks.close(raw["tokens"][:, 1], tables["water_tokens"][water_model], "water endpoint linkage")
            checks.close(raw["sender_log_probs_food"], tables["food_log_probs"][food_model], "food log linkage")
            checks.close(raw["sender_log_probs_water"], tables["water_log_probs"][water_model], "water log linkage")
            checks.close(raw["receiver_logits"], tables["receiver_logits"][receiver_model], "receiver endpoint linkage")
            baseline = Path(entry["baseline_path"])
            base_out, base_tc = base_complete[condition]
            checks.require(baseline.is_file(), "baseline exists")
            rel = str(baseline.relative_to(base_out))
            checks.require(rel in base_tc["files"], "baseline bound")
            checks.exact(sha(baseline), base_tc["files"][rel], "baseline hash")
            checks.add("episode_rows")
    checks.exact(len({(row["condition"], row["seed"], row["partition"], row["schedule"], row["role"], row["held_identity"]) for row in episodes["rows"]}), 864, "factorial uniqueness")
    result = {
        "passed": True,
        "status": "passed_zero_shot_identity_raw_audit",
        "formal": True,
        "probe": "zero_shot_identity_holdout",
        "checks": checks.count,
        "coverage": dict(checks.coverage),
        "source_hashes": invocation["source_hashes"],
        "input_hashes": invocation["input_hashes"],
        "training_complete_sha256": sha(out / "training_complete.json"),
        "audit_source_sha256": sha(ROOT / "identity_audit.py"),
        "exclusions": ["No online newcomer adaptation or gradient replay is included; this audit covers saved model tables, endpoint linkage, worlds, token domains, probability normalization and baseline hashes."],
    }
    write(out / "identity_audit.json", result)
    shutil.copy2(ROOT / "identity_audit.py", out / "identity_audit_source.py")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    started = time.monotonic()
    try:
        result = run(args.out.resolve()); result["seconds"] = time.monotonic() - started; write(args.out.resolve() / "identity_audit.json", result)
        print(json.dumps({key: result[key] for key in ("passed", "status", "checks", "coverage", "seconds")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(args.out.resolve() / f"identity_audit_failure_{stamp}.json", {"passed": False, "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "identity_audit.py")}); raise


if __name__ == "__main__": main()
