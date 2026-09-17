"""Independent replay audit for v0.41 two-token formation traces."""
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
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
ASSIGNMENTS = ("canonical", "cyclic")
SCHEDULES = ("A", "B", "C")
CHECKPOINTS = (0, 100, 600, 1200)
UPDATES = 1200
BATCH = 240
RESOURCES = 3
TOKENS = 2
VOCAB = 7
SITES = 6
SCHEDULE_NAMESPACE = 41040
FIXTURE_NAMESPACE = 41041
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


def expected_schedule(condition, seed, part, step):
    if condition == "fixed_A":
        return "A"
    if condition == "rotating_AB":
        return "A" if step % 2 == 0 else "B"
    rng = np.random.default_rng(np.random.SeedSequence([SCHEDULE_NAMESPACE, int(seed), int(part), int(step), 77]))
    return str(rng.choice(np.asarray(SCHEDULES)))


def draw(probabilities, uniforms):
    return np.minimum((np.cumsum(probabilities, axis=-1) < uniforms[:, None]).sum(axis=-1), probabilities.shape[-1] - 1)


def expected_fixture(seed, part, slot, step):
    indices = np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE, int(seed), int(part), int(slot), int(step), 1])).integers(0, 480, size=BATCH, dtype=np.int64)
    uniforms = np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE, int(seed), int(part), int(slot), int(step), 2])).random((BATCH, 9), dtype=np.float32)
    return indices, uniforms


def check_trace(path, worlds, seed, part, condition, assignment, step, slot, checks):
    data = npz(path)
    schedule = str(data["schedule_id"][0])
    expected = expected_schedule(condition, seed, part, step)
    checks.exact(data["global_step"], np.asarray([step], dtype=np.int64), "global step")
    checks.exact(schedule, expected, "schedule replay")
    checks.exact(data["slot"], np.asarray([slot], dtype=np.int64), "slot")
    checks.exact(data["team"], np.asarray(TEAMS[schedule][slot], dtype=np.int64), "team")
    indices, uniforms = expected_fixture(seed, part, slot, step)
    checks.exact(data["world__indices"], indices, "fixture indices")
    checks.exact(data["world__uniforms"], uniforms, "fixture uniforms")
    checks.exact(data["trace__uniforms"], uniforms, "trace uniforms")
    positions = worlds["positions"][indices]
    checks.exact(data["trace__positions"], positions, "positions")
    token_prob = data["trace__token_probabilities"]
    action_prob = data["trace__action_probabilities"]
    checks.require(token_prob.shape == (BATCH, RESOURCES, TOKENS, VOCAB) and np.isfinite(token_prob).all(), "token probabilities")
    checks.require(action_prob.shape == (BATCH, RESOURCES, SITES) and np.isfinite(action_prob).all(), "action probabilities")
    checks.close(token_prob.sum(-1), np.ones((BATCH, RESOURCES, TOKENS)), "token normalization")
    checks.close(action_prob.sum(-1), np.ones((BATCH, RESOURCES)), "action normalization")
    messages = np.stack(
        [
            np.stack(
                [draw(token_prob[:, resource, token], uniforms[:, resource if token == 0 else RESOURCES + resource]) for token in range(TOKENS)],
                axis=1,
            )
            for resource in range(RESOURCES)
        ],
        axis=1,
    )
    actions = np.stack([draw(action_prob[:, resource], uniforms[:, 2 * RESOURCES + resource]) for resource in range(RESOURCES)], axis=1)
    checks.exact(data["trace__messages"], messages, "sequential token replay")
    checks.exact(data["trace__actions"], actions, "action replay")
    checks.require(((messages >= 0) & (messages < VOCAB)).all(), "message domain")
    checks.require(((actions >= 0) & (actions < SITES)).all(), "action domain")
    success = (actions == positions).astype(np.float32)
    checks.close(data["trace__success"], success, "success")
    reward = success.sum(axis=1) / np.float32(6.0) + np.float32(0.5) * success.prod(axis=1)
    checks.close(data["trace__reward"], reward, "reward")
    checks.close(data["trace__advantage"], reward - BASELINE, "advantage")
    checks.close(data["trace__baseline"], BASELINE, "baseline")
    checks.close(data["trace__entropy_weight"], ENTROPY_WEIGHT, "entropy weight")
    logp_token = np.log(np.maximum(token_prob, np.finfo(np.float32).tiny))
    logp_action = np.log(np.maximum(action_prob, np.finfo(np.float32).tiny))
    selected_sender = np.empty((BATCH, RESOURCES, TOKENS), dtype=np.float32)
    for resource in range(RESOURCES):
        for token in range(TOKENS):
            selected_sender[:, resource, token] = np.take_along_axis(logp_token[:, resource, token], messages[:, resource, token, None], axis=-1).squeeze(-1)
    selected_receiver = np.stack([np.take_along_axis(logp_action[:, resource], actions[:, resource, None], axis=-1).squeeze(-1) for resource in range(RESOURCES)], axis=1)
    checks.close(data["trace__sender_logp"], selected_sender, "sender logp", atol=2e-6)
    checks.close(data["trace__receiver_logp"], selected_receiver.sum(axis=1), "receiver logp", atol=2e-6)
    checks.close(data["trace__sender_entropy"], -(token_prob * logp_token).sum(-1), "sender entropy", atol=2e-6)
    checks.close(data["trace__receiver_entropy"], -(action_prob * logp_action).sum(-1).sum(1), "receiver entropy", atol=2e-6)
    checks.coverage["trace_files"] += 1
    checks.coverage["trace_rows"] += BATCH


def run(out: Path):
    checks = Checks()
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    runs = read(out / "runs.json")
    checks.exact(invocation["formal"], True, "formal invocation")
    checks.exact(invocation["seeds"], list(SEEDS), "seeds")
    checks.exact(invocation["partitions"], list(PARTITIONS), "partitions")
    checks.exact(invocation["conditions"], list(CONDITIONS), "conditions")
    checks.exact(tuple(invocation["assignments"].keys()), ASSIGNMENTS, "assignments")
    checks.exact(invocation["schedule_namespace"], SCHEDULE_NAMESPACE, "schedule namespace")
    checks.exact(invocation["fixture_namespace"], FIXTURE_NAMESPACE, "fixture namespace")
    checks.exact(invocation["checkpoints"], list(CHECKPOINTS), "checkpoints")
    checks.exact(complete["status"], "complete", "completion")
    checks.exact(complete["formal"], True, "formal completion")
    checks.exact(complete["probe"], "two_token_formation", "probe")
    checks.exact(complete["runs"], 72, "runs")
    checks.exact(runs["count"], 72, "run manifest")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            checks.require(Path(path).is_file(), key + " bound path")
            checks.exact(sha(Path(path)), digest, key + " bound hash")

    worlds = {assignment: npz(out / f"test_worlds_{assignment}.npz") for assignment in ASSIGNMENTS}
    train_worlds = {
        (part, assignment): npz(out / f"train_worlds_p{part}_{assignment}.npz")
        for part in PARTITIONS
        for assignment in ASSIGNMENTS
    }
    for assignment in ASSIGNMENTS:
        checks.exact(worlds[assignment]["map_id"], np.arange(120, dtype=np.int64), assignment + " map order")
        checks.exact(len(worlds[assignment]["positions"]), 120, assignment + " test world count")
    expected_protocols = 0
    expected_traces = 0
    expected_trace_rows = 0
    for seed, part, assignment, condition in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS, CONDITIONS):
        chain = out / "social" / f"s{seed}_p{part}_{assignment}_{condition}"
        cfg = read(chain / "config.json")
        checks.exact(cfg["assignment"], assignment, "config assignment")
        checks.exact(cfg["condition"], condition, "config condition")
        checks.exact(cfg["resource_permutation"], invocation["assignments"][assignment], "config permutation")
        checks.exact(cfg["checkpoints"], list(CHECKPOINTS), "config checkpoints")
        checks.require((chain / "final.pt").is_file(), "final state")
        curve = read(chain / "curve.json")
        checks.exact([int(item["update"]) for item in curve], list(CHECKPOINTS), "curve updates")
        log_path = chain / "training.jsonl"
        lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
        checks.exact(len(lines), UPDATES, "training rows")
        observed = Counter()
        for index, row in enumerate(lines):
            expected = expected_schedule(condition, seed, part, index)
            checks.exact(row["update"], index + 1, "training update")
            checks.exact(row["global_step"], index, "training global step")
            checks.exact(row["schedule"], expected, "training schedule")
            checks.require(np.isfinite(float(row["loss"])), "loss finite")
            checks.require(all(np.isfinite(float(value)) for value in row["norms"].values()), "gradient norms finite")
            observed[expected] += 1
        checks.exact(sum(observed.values()), UPDATES, "schedule totals")
        for schedule in SCHEDULES:
            checks.exact(observed[schedule], sum(expected_schedule(condition, seed, part, step) == schedule for step in range(UPDATES)), "schedule count")
        for update in (1, UPDATES):
            for slot in range(4):
                path = chain / f"train_{update:04d}_slot{slot}.npz"
                rel = str(path.relative_to(out))
                checks.require(rel in complete["files"], "trace completion hash")
                check_trace(path, train_worlds[part, assignment], seed, part, condition, assignment, update - 1, slot, checks)
                expected_traces += 1
                expected_trace_rows += BATCH
        for update in CHECKPOINTS:
            for schedule in SCHEDULES:
                for slot in range(4):
                    path = chain / f"protocol_{schedule}_{update:04d}_team{slot}.npz"
                    rel = str(path.relative_to(out))
                    checks.require(rel in complete["files"], "protocol completion hash")
                    checks.require(path.is_file(), "protocol file")
                    expected_protocols += 1
    actual_traces = len(list((out / "social").glob("*/train_*_slot*.npz")))
    actual_protocols = len(list((out / "social").glob("*/protocol_*_team*.npz")))
    checks.exact(actual_traces, expected_traces, "trace count")
    checks.exact(actual_protocols, expected_protocols, "protocol count")
    checks.exact(expected_traces, 576, "formal trace count")
    checks.exact(expected_trace_rows, 138240, "formal trace rows")
    checks.exact(expected_protocols, 3456, "formal protocol count")
    checks.exact(complete["trace_files_expected"], 576, "declared trace count")
    checks.exact(complete["trace_rows_expected"], 138240, "declared trace rows")
    record = {
        "status": "complete",
        "formal": True,
        "probe": "two_token_formation",
        "checks": checks.count,
        "scalar_comparisons": checks.comparisons,
        "maximum_replay_absolute_difference": checks.max_error,
        "protocol_tables": expected_protocols,
        "trace_files": expected_traces,
        "trace_rows": expected_trace_rows,
        "source_sha256": sha(ROOT / "two_token_audit.py"),
        "training_complete_sha256": sha(out / "training_complete.json"),
        "coverage": dict(checks.coverage),
        "production_modules_imported": False,
        "model_calls": 0,
    }
    write(out / "two_token_audit.json", record)
    shutil.copy2(ROOT / "two_token_audit.py", out / "two_token_audit_source.py")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        record = run(args.out.resolve())
        record["seconds"] = time.monotonic() - started
        write(args.out.resolve() / "two_token_audit.json", record)
        print(json.dumps({key: record[key] for key in ("status", "checks", "scalar_comparisons", "maximum_replay_absolute_difference", "protocol_tables", "trace_files", "trace_rows")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns()
        write(args.out.resolve() / f"two_token_audit_failure_{stamp}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "two_token_audit.py")})
        raise


if __name__ == "__main__":
    main()
