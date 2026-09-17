"""Independent replay audit for the v0.40 triad replacement traces."""
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
V039_OUT = ROOT.parent / "redesign_v0.39" / "results" / "triad_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
FORMATION_CONDITIONS = ("origin_fixed_A", "origin_rotating_AB", "origin_random_ABC")
TRANSMISSION_CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
GENERATIONS = (1, 2, 3, 4)
UPDATES = 300
BATCH = 240
RESOURCES = 3
VOCAB = 7
SITES = 6
BASELINE = np.float32(0.1)
ENTROPY_WEIGHT = np.float32(0.02)
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Checks:
    def __init__(self):
        self.count = 0
        self.comparisons = 0
        self.max_error = 0.0
        self.coverage = Counter()

    def require(self, value, label):
        self.count += 1
        if not bool(value):
            raise AssertionError(label)

    def exact(self, actual, expected, label):
        self.require(np.array_equal(actual, expected), label)

    def close(self, actual, expected, label, atol=1e-6, rtol=1e-6):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size)
        error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0
        self.max_error = max(self.max_error, error)
        self.require(np.allclose(a, b, atol=atol, rtol=rtol), label + " numerical")

    def add(self, key, value=1):
        self.coverage[key] += value


def expected_schedule(condition, seed, part, generation, step):
    global_step = (generation - 1) * UPDATES + step
    if condition == "fixed_A":
        return "A", global_step
    if condition == "rotating_AB":
        return ("A" if global_step % 2 == 0 else "B"), global_step
    rng = np.random.default_rng(np.random.SeedSequence([39040, int(seed), int(part), int(global_step), 77]))
    return str(rng.choice(np.asarray(SCHEDULES))), global_step


def draw(probabilities, uniforms):
    return np.minimum((np.cumsum(probabilities, axis=-1) < uniforms[:, None]).sum(axis=-1), probabilities.shape[-1] - 1)


def expected_fixture(seed, part, slot, global_step):
    indices = np.random.default_rng(
        np.random.SeedSequence([39039, int(seed), int(part), int(slot), int(global_step), 1])
    ).integers(0, 480, size=BATCH, dtype=np.int64)
    uniforms = np.random.default_rng(
        np.random.SeedSequence([39039, int(seed), int(part), int(slot), int(global_step), 2])
    ).random((BATCH, 2 * RESOURCES), dtype=np.float32)
    return indices, uniforms


def check_trace(path, train_worlds, seed, part, formation, transmission, generation, update, slot, checks):
    data = npz(path)
    checks.require(data["schedule_id"].shape == (1,), "schedule id shape")
    schedule = str(data["schedule_id"][0])
    expected, global_step = expected_schedule(transmission, seed, part, generation, update - 1)
    checks.exact(schedule, expected, "schedule replay")
    checks.exact(data["global_step"], np.asarray([global_step], dtype=np.int64), "global step")
    checks.exact(data["slot"], np.asarray([slot], dtype=np.int64), "slot")
    checks.exact(data["team"], np.asarray(TEAMS[schedule][slot], dtype=np.int64), "team topology")
    expected_idx, expected_uniforms = expected_fixture(seed, part, slot, global_step)
    checks.exact(data["world__indices"], expected_idx, "fixture indices")
    checks.exact(data["world__uniforms"], expected_uniforms, "fixture uniforms")
    checks.exact(data["trace__uniforms"], expected_uniforms, "trace uniforms")
    indices = data["world__indices"]
    checks.require(indices.shape == (BATCH,) and ((indices >= 0) & (indices < len(train_worlds["map_id"]))).all(), "fixture domain")
    positions = train_worlds["positions"][indices]
    checks.exact(data["trace__positions"], positions, "trace positions")

    token_prob = data["trace__token_probabilities"]
    action_prob = data["trace__action_probabilities"]
    checks.require(token_prob.shape == (BATCH, RESOURCES, VOCAB) and np.isfinite(token_prob).all(), "token probabilities")
    checks.require(action_prob.shape == (BATCH, RESOURCES, SITES) and np.isfinite(action_prob).all(), "action probabilities")
    checks.close(token_prob.sum(-1), np.ones((BATCH, RESOURCES)), "token normalization")
    checks.close(action_prob.sum(-1), np.ones((BATCH, RESOURCES)), "action normalization")
    expected_messages = np.stack([draw(token_prob[:, resource], expected_uniforms[:, resource]) for resource in range(RESOURCES)], axis=1)
    expected_actions = np.stack([draw(action_prob[:, resource], expected_uniforms[:, RESOURCES + resource]) for resource in range(RESOURCES)], axis=1)
    checks.exact(data["trace__messages"], expected_messages, "token replay")
    checks.exact(data["trace__actions"], expected_actions, "action replay")
    checks.require(((data["trace__messages"] >= 0) & (data["trace__messages"] < VOCAB)).all(), "message domain")
    checks.require(((data["trace__actions"] >= 0) & (data["trace__actions"] < SITES)).all(), "action domain")
    success = (expected_actions == positions).astype(np.float32)
    checks.close(data["trace__success"], success, "success replay")
    reward = success.sum(axis=1) / np.float32(6.0) + np.float32(0.5) * success.prod(axis=1)
    advantage = reward - BASELINE
    checks.close(data["trace__reward"], reward, "reward replay")
    checks.close(data["trace__advantage"], advantage, "advantage replay")
    checks.close(data["trace__baseline"], BASELINE, "baseline")
    checks.close(data["trace__entropy_weight"], ENTROPY_WEIGHT, "entropy weight")

    token_logp = np.log(np.maximum(token_prob, np.finfo(np.float32).tiny))
    action_logp = np.log(np.maximum(action_prob, np.finfo(np.float32).tiny))
    selected_sender = np.take_along_axis(token_logp, expected_messages[:, :, None], axis=-1).squeeze(-1)
    selected_receiver = np.take_along_axis(action_logp, expected_actions[:, :, None], axis=-1).squeeze(-1)
    checks.close(data["trace__sender_logp"], selected_sender, "sender logp", atol=2e-6)
    checks.close(data["trace__receiver_logp"], selected_receiver.sum(axis=1), "receiver logp", atol=2e-6)
    sender_entropy = -(token_prob * token_logp).sum(axis=-1)
    receiver_entropy = -(action_prob * action_logp).sum(axis=-1).sum(axis=1)
    checks.close(data["trace__sender_entropy"], sender_entropy, "sender entropy", atol=2e-6)
    checks.close(data["trace__receiver_entropy"], receiver_entropy, "receiver entropy", atol=2e-6)
    checks.add("trace_files")
    checks.add("trace_rows", BATCH)


def run(out: Path):
    checks = Checks()
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    runs = read(out / "runs.json")
    checks.exact(invocation["formal"], True, "formal invocation")
    checks.exact(invocation["seeds"], list(SEEDS), "seeds")
    checks.exact(invocation["partitions"], list(PARTITIONS), "partitions")
    checks.exact(invocation["formation_conditions"], list(FORMATION_CONDITIONS), "formation conditions")
    checks.exact(invocation["transmission_conditions"], list(TRANSMISSION_CONDITIONS), "transmission conditions")
    checks.exact(invocation["generations"], 4, "generations")
    checks.exact(invocation["updates_per_generation"], UPDATES, "updates")
    checks.exact(invocation["schedule_seed_namespace"], 39040, "schedule namespace")
    checks.exact(complete["status"], "complete", "completion")
    checks.exact(complete["formal"], True, "formal completion")
    checks.exact(complete["probe"], "triad_transfer", "probe")
    checks.exact(complete["runs"], 108, "runs")
    checks.exact(complete["replacement_events"], 432, "replacement events")
    checks.exact(runs["count"], 108, "run manifest")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            checks.require(Path(path).is_file(), key + " bound path")
            checks.exact(sha(Path(path)), digest, key + " bound hash")
    worlds = npz(out / "test_worlds.npz")
    checks.exact(len(worlds["map_id"]), 120, "test worlds")
    checks.exact(worlds["map_id"], np.arange(120, dtype=np.int64), "test map order")
    for part in PARTITIONS:
        train_worlds = npz(out / f"train_worlds_p{part}.npz")
        checks.exact(len(train_worlds["map_id"]), 480, f"train worlds p{part}")
        checks.require(np.all((train_worlds["map_id"] >= 0) & (train_worlds["map_id"] < 120)), f"train map domain p{part}")

    expected_protocols = 0
    expected_traces = 0
    expected_trace_rows = 0
    for seed, part, formation, transmission in itertools.product(SEEDS, PARTITIONS, FORMATION_CONDITIONS, TRANSMISSION_CONDITIONS):
        chain = out / "chains" / f"s{seed}_p{part}_{formation}__{transmission}"
        cfg = read(chain / "config.json")
        checks.exact(cfg["formation_condition"], formation, "chain formation")
        checks.exact(cfg["transmission_condition"], transmission, "chain transmission")
        checks.require((chain / "generation_000_before.pt").is_file(), "initial state")
        curve = read(chain / "curve.json")
        checks.exact([int(item["generation"]) for item in curve], [0, 1, 2, 3, 4], "curve generation order")
        train_worlds = npz(out / f"train_worlds_p{part}.npz")
        for generation in GENERATIONS:
            folder = chain / f"generation_{generation:02d}"
            gcfg = read(folder / "config.json")
            checks.exact(gcfg["generation"], generation, "generation config")
            checks.exact(gcfg["formation_condition"], formation, "generation formation")
            checks.exact(gcfg["transmission_condition"], transmission, "generation transmission")
            checks.exact(gcfg["schedule_seed_namespace"], 39040, "generation namespace")
            checks.require((folder / "generation_after.pt").is_file(), "after state")
            log_path = folder / "training.jsonl"
            lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
            checks.exact(len(lines), UPDATES, "training rows")
            observed = Counter()
            for index, row in enumerate(lines, start=1):
                expected, global_step = expected_schedule(transmission, seed, part, generation, index - 1)
                checks.exact(row["update"], index, "training update")
                checks.exact(row["global_step"], global_step, "training global step")
                checks.exact(row["schedule"], expected, "training schedule")
                checks.require(np.isfinite(float(row["loss"])), "training loss finite")
                checks.require(all(np.isfinite(float(value)) for value in row["norms"].values()), "training norms finite")
                observed[expected] += 1
            checks.exact(sum(observed.values()), UPDATES, "schedule count total")
            for schedule in SCHEDULES:
                checks.exact(observed[schedule], sum(expected_schedule(transmission, seed, part, generation, step)[0] == schedule for step in range(UPDATES)), "schedule count")
            for update in (1, UPDATES):
                for slot in range(4):
                    path = folder / f"train_{update:04d}_slot{slot}.npz"
                    rel = str(path.relative_to(out))
                    checks.require(rel in complete["files"], "trace completion hash")
                    check_trace(path, train_worlds, seed, part, formation, transmission, generation, update, slot, checks)
                    expected_traces += 1
                    expected_trace_rows += BATCH
            for schedule in SCHEDULES:
                for slot in range(4):
                    path = folder / f"protocol_{schedule}_{generation:04d}_team{slot}.npz"
                    rel = str(path.relative_to(out))
                    checks.require(rel in complete["files"], "protocol completion hash")
                    checks.require(path.is_file(), "protocol file")
                    expected_protocols += 1
        # Generation zero is the sealed v0.39 endpoint before any newcomer
        # replacement. It has protocol tables but no training traces.
        folder = chain / "generation_00"
        for schedule in SCHEDULES:
            for slot in range(4):
                path = folder / f"protocol_{schedule}_0000_team{slot}.npz"
                rel = str(path.relative_to(out))
                checks.require(rel in complete["files"], "generation zero protocol completion hash")
                checks.require(path.is_file(), "generation zero protocol file")
                expected_protocols += 1

    actual_traces = len(list((out / "chains").glob("*/generation_*/train_*_slot*.npz")))
    actual_protocols = len(list((out / "chains").glob("*/generation_*/protocol_*_team*.npz")))
    checks.exact(actual_traces, expected_traces, "trace count")
    checks.exact(actual_protocols, expected_protocols, "protocol count")
    # Two checkpoints × four slots are stored for each of 108 chains and four
    # replacement generations: 108×4×2×4 = 3456 files. Each has 240 rows,
    # giving the declared 829440 replay rows.
    checks.exact(expected_traces, 3456, "formal trace count")
    checks.exact(expected_trace_rows, 829440, "formal trace rows")
    checks.exact(expected_protocols, 6480, "formal protocol count")
    checks.exact(complete["trace_files_expected"], 3456, "declared trace count")
    record = {
        "status": "complete",
        "formal": True,
        "probe": "triad_transfer",
        "checks": checks.count,
        "scalar_comparisons": checks.comparisons,
        "maximum_replay_absolute_difference": checks.max_error,
        "protocol_tables": expected_protocols,
        "trace_files": expected_traces,
        "trace_rows": expected_trace_rows,
        "source_sha256": sha(ROOT / "triad_transfer_audit.py"),
        "training_complete_sha256": sha(out / "training_complete.json"),
        "coverage": dict(checks.coverage),
        "production_modules_imported": False,
        "model_calls": 0,
    }
    write(out / "triad_transfer_audit.json", record)
    shutil.copy2(ROOT / "triad_transfer_audit.py", out / "triad_transfer_audit_source.py")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        record = run(args.out.resolve())
        record["seconds"] = time.monotonic() - started
        write(args.out.resolve() / "triad_transfer_audit.json", record)
        print(json.dumps({key: record[key] for key in ("status", "checks", "scalar_comparisons", "maximum_replay_absolute_difference", "protocol_tables", "trace_files", "trace_rows")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns()
        write(args.out.resolve() / f"triad_transfer_audit_failure_{stamp}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "triad_transfer_audit.py")})
        raise


if __name__ == "__main__":
    main()
