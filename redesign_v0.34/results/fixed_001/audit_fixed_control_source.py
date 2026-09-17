"""Bounded raw-table and stochastic-trace audit for the fixed-A control."""
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
TIMES = (0, 100, 600, 1200, 2100, 2400)
TRACE_STEPS = (0, 2100, 2399)
WORLD_KEYS = ("map_id", "photo_ids", "positions", "shown")
TEAMS = ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1))


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

    def add(self, key: str, value: int = 1):
        self.coverage[key] += value


def draw(probabilities, uniforms):
    return np.minimum((np.cumsum(probabilities, axis=-1) < uniforms[..., None]).sum(-1), probabilities.shape[-1] - 1)


def check_binding(invocation, complete, checks: Checks):
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            file_path = Path(path)
            checks.require(file_path.is_file(), key + " bound file exists")
            checks.exact(sha(file_path), digest, key + " bound file hash")


def check_protocol(raw, worlds, checks: Checks, label: str):
    for key in WORLD_KEYS:
        checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"])
    checks.require(raw["tokens"].shape == (n, 2) and raw["tokens"].dtype.kind in "iu", label + " token shape")
    checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), label + " token domain")
    checks.require(raw["sender_log_probs"].shape == (n, 49) and np.isfinite(raw["sender_log_probs"]).all(), label + " sender table")
    checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1.0, atol=1e-6), label + " sender normalization")
    checks.require(raw["receiver_logits"].shape == (49, 2, 6) and np.isfinite(raw["receiver_logits"]).all(), label + " receiver table")
    checks.add("endpoint_protocol_worlds", n)


def check_dual_trace(fields, train_worlds, checks: Checks, label: str):
    required = {"indices", "uniforms", "messages", "token_food", "token_water", "token_probabilities_food", "token_probabilities_water", "action_probabilities", "actions", "positions", "success", "reward", "advantage"}
    checks.require(required.issubset(fields), label + " required fields")
    idx = fields["indices"]
    checks.require(idx.ndim == 1 and len(idx) == 240 and np.issubdtype(idx.dtype, np.integer), label + " index shape")
    checks.require(((idx >= 0) & (idx < len(train_worlds["map_id"]))).all(), label + " index domain")
    positions = train_worlds["positions"][idx]
    checks.exact(fields["positions"], positions, label + " positions")
    uniforms = fields["uniforms"]
    checks.require(uniforms.shape == (240, 4) and np.isfinite(uniforms).all() and ((uniforms >= 0) & (uniforms <= 1)).all(), label + " uniforms")
    token_food = fields["token_food"]
    token_water = fields["token_water"]
    checks.require(token_food.shape == (240,) and token_water.shape == (240,), label + " token vectors")
    checks.require(((token_food >= 0) & (token_food < 7)).all() and ((token_water >= 0) & (token_water < 7)).all(), label + " token domain")
    checks.exact(fields["messages"], np.stack((token_food, token_water), axis=1), label + " message stack")
    for key, token in (("token_probabilities_food", token_food), ("token_probabilities_water", token_water)):
        probs = fields[key]
        checks.require(probs.shape == (240, 7) and np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-6), label + " token probabilities")
        checks.exact(draw(probs, uniforms[:, 0 if key.endswith("food") else 1]), token, label + " categorical token replay")
    action_probs = fields["action_probabilities"]
    actions = fields["actions"]
    checks.require(action_probs.shape == (240, 2, 6) and np.isfinite(action_probs).all() and np.allclose(action_probs.sum(-1), 1, atol=1e-6), label + " action probabilities")
    checks.require(actions.shape == (240, 2) and actions.dtype.kind in "iu" and ((actions >= 0) & (actions < 6)).all(), label + " actions")
    checks.exact(draw(action_probs, uniforms[:, 2:4]), actions, label + " categorical action replay")
    success = (actions == positions).astype(np.float32)
    checks.exact(fields["success"], success, label + " success")
    reward = (.25 * success.sum(1) + .5 * success.prod(1)).astype(np.float32)
    checks.require(np.allclose(fields["reward"], reward, atol=1e-6, rtol=1e-6), label + " reward")
    checks.require(np.allclose(fields["advantage"], reward - .5, atol=1e-6, rtol=1e-6), label + " advantage")
    checks.add("training_trace_rows", 240)


def check_trace_file(path: Path, train_worlds, checks: Checks, label: str):
    raw = npz(path)
    world_slots = {}
    for key, value in raw.items():
        if key.startswith("world__"):
            _, slot, field = key.split("__")
            world_slots.setdefault(slot, {})[field] = value
    checks.exact(sorted(world_slots), [f"slot{i}" for i in range(4)], label + " world slots")
    for slot in sorted(world_slots):
        fixture = world_slots[slot]
        checks.require(fixture["indices"].shape == (240,), label + " indices")
        checks.require(fixture["uniforms"].shape == (240, 4), label + " world uniforms")
    prefixes = sorted({"__".join(key.split("__")[:3]) for key in raw if key.startswith("trace__")})
    checks.exact(len(prefixes), 4, label + " trace slot count")
    for prefix in prefixes:
        _, slot, role = prefix.split("__")
        fields = {key[len(prefix) + 2:]: value for key, value in raw.items() if key.startswith(prefix + "__")}
        fields["indices"] = world_slots[slot]["indices"]
        checks.exact(fields["uniforms"], world_slots[slot]["uniforms"], label + "/" + prefix + " uniforms")
        check_dual_trace(fields, train_worlds, checks, label + "/" + prefix)
    checks.add("trace_files")


def check_agreement(path: Path, checks: Checks):
    raw = npz(path)
    expected = {f"{view}_type{k}_{suffix}" for view in ("full", "food_only", "water_only") for k in (0, 1) for suffix in ("a", "b")}
    checks.require(set(raw) == expected, "agreement inventory")
    for value in raw.values():
        checks.require(len(value) == 180 and np.isfinite(value).all(), "agreement rows")
    checks.add("agreement_files")


def run(out: Path):
    checks = Checks()
    invocation = read(out / "invocation.json")
    complete = read(out / "training_complete.json")
    checks.require(invocation.get("formal") is True, "formal invocation")
    checks.exact(invocation["seeds"], list(SEEDS), "seeds")
    checks.exact(invocation["partitions"], list(PARTITIONS), "partitions")
    checks.exact(invocation["conditions"], ["dual_complementary"], "condition inventory")
    checks.exact(invocation["updates"], 2400, "updates")
    checks.exact(complete["status"], "complete", "training terminal status")
    checks.exact(complete["control"], "fixed_A", "control")
    checks.exact(complete["social_runs"], 12, "social runs")
    check_binding(invocation, complete, checks)
    wrapper = read(out / "wrapper_receipt.json")
    checks.exact(wrapper["source_sha256"], sha(ROOT / "run_fixed_control.py"), "wrapper binding")
    checks.exact(wrapper["production_runner_sha256"], sha(ROOT / "run_support.py"), "runner binding")
    receipt = read(out / "terminal_receipt.json")
    checks.exact(receipt["status"], "complete", "terminal receipt")
    checks.exact(receipt["exit_code"], 0, "terminal exit")
    checks.require(not (out / ".venv").exists(), "no bundled environment")

    test_worlds = npz(out / "test_worlds.npz")
    train_worlds = npz(out / "train_worlds.npz")
    checks.exact(len(test_worlds["map_id"]), 180, "test worlds")
    checks.exact(len(train_worlds["map_id"]), 720, "train worlds")
    checks.exact(test_worlds["positions"], np.asarray(list(itertools.permutations(range(6), 2)), np.int64)[test_worlds["map_id"]], "test positions")
    checks.exact(train_worlds["positions"], np.asarray(list(itertools.permutations(range(6), 2)), np.int64)[train_worlds["map_id"]], "train positions")

    for seed, panel in itertools.product(SEEDS, PARTITIONS):
        folder = out / "social" / f"s{seed}_p{panel}_dual_complementary"
        cfg = read(folder / "config.json")
        checks.exact(cfg["teams"], [list(team) for team in TEAMS], "A team schedule")
        checks.exact(cfg["training_schedule"], "A_only_fixed_control", "fixed schedule config")
        checks.exact(cfg["wrapper"], "run_fixed_control.py", "fixed wrapper config")
        curve = read(folder / "curve.json")
        checks.exact([item["update"] for item in curve], list(TIMES), "curve checkpoints")
        for update in TIMES:
            for team in range(4):
                path = folder / f"protocol_{update:04d}_team{team}.npz"
                rel = str(path.relative_to(out))
                checks.require(rel in complete["files"], "protocol completion hash")
                check_protocol(npz(path), test_worlds, checks, f"protocol {seed}/{panel}/{update}/{team}")
            check_agreement(folder / f"agreement_{update:04d}.npz", checks)
        for step in TRACE_STEPS:
            path = folder / f"train_{step + 1:04d}.npz"
            check_trace_file(path, train_worlds, checks, f"trace {seed}/{panel}/{step + 1}")
        for schedule in ("B", "C"):
            for team in range(4):
                path = folder / f"transfer_{schedule}_2400_team{team}.npz"
                rel = str(path.relative_to(out))
                checks.require(rel in complete["files"], "transfer completion hash")
                check_protocol(npz(path), test_worlds, checks, f"transfer {schedule} {seed}/{panel}/{team}")
        checks.add("social_runs")
    result = {
        "passed": True,
        "status": "passed_same_namespace_fixed_A_trace_audit",
        "formal": True,
        "control": "fixed_A",
        "checks": checks.count,
        "coverage": dict(checks.coverage),
        "source_hashes": invocation["source_hashes"],
        "input_hashes": invocation["input_hashes"],
        "training_complete_sha256": sha(out / "training_complete.json"),
        "audit_source_sha256": sha(ROOT / "audit_fixed_control.py"),
        "exclusions": [
            "No complete gradient/Adam replay.",
            "The audit checks categorical trace replay, reward arithmetic, frozen world identity, completion hashes and endpoint transfer tables.",
        ],
    }
    write(out / "fixed_control_audit.json", result)
    shutil.copy2(ROOT / "audit_fixed_control.py", out / "audit_fixed_control_source.py")
    return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args()
    started = time.monotonic()
    try:
        result = run(args.out.resolve())
        result["seconds"] = time.monotonic() - started
        write(args.out.resolve() / "fixed_control_audit.json", result)
        print(json.dumps({key: result[key] for key in ("passed", "status", "checks", "coverage", "seconds")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns()
        write(args.out.resolve() / f"fixed_control_audit_failure_{stamp}.json", {"passed": False, "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "audit_fixed_control.py")})
        raise


if __name__ == "__main__": main()
