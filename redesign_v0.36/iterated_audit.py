"""Bounded trace and endpoint audit for v0.36 iterated replacement."""
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
SCHEDULES = ("A", "B", "C")
GENERATIONS = tuple(range(9))
TEAMS = {
    "A": ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)),
    "B": ((0, 2, 3), (1, 3, 0), (2, 0, 1), (3, 1, 2)),
    "C": ((0, 3, 1), (1, 0, 2), (2, 1, 3), (3, 2, 0)),
}


def read(path: Path): return json.loads(path.read_text())
def write(path: Path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def npz(path: Path):
    with np.load(path, allow_pickle=False) as z: return {key: z[key] for key in z.files}


class Checks:
    def __init__(self): self.count = 0; self.coverage = Counter()
    def require(self, value, label):
        self.count += 1
        if not bool(value): raise AssertionError(label)
    def exact(self, actual, expected, label): self.require(np.array_equal(actual, expected), label)
    def add(self, key, value=1): self.coverage[key] += value


def expected_schedule(condition, seed, part, generation, global_step, step):
    if condition == "fixed_A": return "A"
    if condition == "rotating_AB": return "A" if global_step % 2 == 0 else "B"
    rng = np.random.default_rng(np.random.SeedSequence([34036, int(seed), int(part), int(generation), int(step), 77]))
    return str(rng.choice(np.asarray(SCHEDULES)))


def check_raw(raw, worlds, checks, label):
    for key in ("map_id", "photo_ids", "positions", "shown"): checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"]); checks.require(raw["tokens"].shape == (n, 2) and raw["tokens"].dtype.kind in "iu", label + " tokens"); checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), label + " token domain")
    for key in ("sender_log_probs_food", "sender_log_probs_water"):
        checks.require(raw[key].shape == (n, 7) and np.isfinite(raw[key]).all(), label + " component table"); checks.require(np.allclose(np.exp(raw[key]).sum(-1), 1, atol=1e-6), label + " component normalization")
    checks.require(raw["sender_log_probs"].shape == (n, 49) and np.isfinite(raw["sender_log_probs"]).all(), label + " joint table"); checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1, atol=1e-6), label + " joint normalization")
    checks.require(np.allclose(raw["sender_log_probs"], (raw["sender_log_probs_food"][:, :, None] + raw["sender_log_probs_water"][:, None, :]).reshape(n, 49), atol=1e-6, rtol=1e-6), label + " factorization")
    checks.require(raw["receiver_logits"].shape == (49, 2, 6) and np.isfinite(raw["receiver_logits"]).all(), label + " receiver"); checks.add("protocol_worlds", n)


def draw(probs, uniforms): return np.minimum((np.cumsum(probs, axis=-1) < uniforms[..., None]).sum(-1), probs.shape[-1] - 1)


def check_trace(path: Path, train_worlds, condition, seed, part, generation, checks):
    raw = npz(path); checks.require(raw["schedule_id"].shape == (1,), "trace schedule id shape"); schedule = str(raw["schedule_id"][0]); checks.require(schedule in SCHEDULES, "trace schedule domain"); global_step = int(raw["global_step"][0]); step = global_step - (generation - 1) * 300
    checks.exact(np.asarray(schedule), np.asarray(expected_schedule(condition, seed, part, generation, global_step, step)), "trace schedule replay")
    slots = {}
    for key, value in raw.items():
        if key.startswith("world__"):
            _, slot, field = key.split("__"); slots.setdefault(slot, {})[field] = value
    checks.exact(sorted(slots), [f"slot{i}" for i in range(4)], "trace world slots")
    prefixes = sorted({"__".join(key.split("__")[:3]) for key in raw if key.startswith("trace__")}); checks.exact(len(prefixes), 4, "trace prefix count")
    for prefix in prefixes:
        _, slot, role = prefix.split("__"); slot_index = int(slot.removeprefix("slot")); expected_role = f"r{TEAMS[schedule][slot_index][0]}_sf{TEAMS[schedule][slot_index][1]}_sw{TEAMS[schedule][slot_index][2]}"; checks.exact(role, expected_role, "trace schedule role")
        fields = {key[len(prefix) + 2:]: value for key, value in raw.items() if key.startswith(prefix + "__")}; indices = slots[slot]["indices"]; uniforms = slots[slot]["uniforms"]
        checks.require(indices.shape == (240,) and np.issubdtype(indices.dtype, np.integer) and ((indices >= 0) & (indices < len(train_worlds["map_id"]))).all(), "trace indices"); checks.require(uniforms.shape == (240, 4) and np.isfinite(uniforms).all() and ((uniforms >= 0) & (uniforms <= 1)).all(), "trace uniforms")
        positions = train_worlds["positions"][indices]; checks.exact(fields["uniforms"], uniforms, "trace uniform linkage"); checks.exact(fields["positions"], positions, "trace positions"); checks.require(fields["token_food"].shape == (240,) and fields["token_water"].shape == (240,), "trace tokens"); checks.exact(fields["messages"], np.stack((fields["token_food"], fields["token_water"]), axis=1), "trace message stack")
        for key, col, token_key in (("token_probabilities_food", 0, "token_food"), ("token_probabilities_water", 1, "token_water")):
            probs = fields[key]; checks.require(probs.shape == (240, 7) and np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-6), "trace token probability"); checks.exact(draw(probs, uniforms[:, col]), fields[token_key], "trace token replay")
        probs = fields["action_probabilities"]; checks.require(probs.shape == (240, 2, 6) and np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-6), "trace action probability"); actions = fields["actions"]; checks.exact(draw(probs, uniforms[:, 2:4]), actions, "trace action replay")
        success = (actions == positions).astype(np.float32); checks.exact(fields["success"], success, "trace success"); reward = (.25 * success.sum(1) + .5 * success.prod(1)).astype(np.float32); checks.require(np.allclose(fields["reward"], reward, atol=1e-6, rtol=1e-6), "trace reward"); checks.require(np.allclose(fields["advantage"], reward - .5, atol=1e-6, rtol=1e-6), "trace advantage"); checks.add("trace_rows", 240)
    checks.add("trace_files")


def run(out: Path):
    checks = Checks(); invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); runs = read(out / "runs.json")
    checks.exact(invocation["formal"], True, "formal invocation"); checks.exact(invocation["seeds"], list(SEEDS), "seeds"); checks.exact(invocation["partitions"], list(PARTITIONS), "partitions"); checks.exact(invocation["conditions"], list(CONDITIONS), "conditions"); checks.exact(invocation["generations"], 8, "generations"); checks.exact(invocation["updates_per_generation"], 300, "updates")
    checks.exact(complete["status"], "complete", "completion"); checks.exact(complete["formal"], True, "formal completion"); checks.exact(complete["probe"], "iterated_replacement", "probe"); checks.exact(complete["runs"], 36, "runs"); checks.exact(runs["count"], 36, "run manifest")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            p = Path(path); checks.require(p.is_file(), key + " bound path"); checks.exact(sha(p), digest, key + " bound hash")
    worlds = npz(out / "test_worlds.npz"); train_worlds = npz(out / "train_worlds.npz"); checks.exact(len(worlds["map_id"]), 180, "test worlds"); checks.exact(len(train_worlds["map_id"]), 720, "train worlds")
    for seed, part, condition in itertools.product(SEEDS, PARTITIONS, CONDITIONS):
        chain = out / "chains" / f"s{seed}_p{part}_{condition}"; cfg = read(chain / "config.json"); checks.exact(cfg["condition"], condition, "chain condition"); checks.exact(cfg["replacement_order"], [0, 1, 2, 3, 0, 1, 2, 3], "replacement order"); checks.require((chain / "generation_000_before.pt").is_file(), "initial state")
        curve = read(chain / "curve.json"); checks.exact([item["generation"] for item in curve], list(GENERATIONS), "curve generations")
        for generation in GENERATIONS:
            item = curve[generation]; folder = chain / f"generation_{generation:02d}"; checks.exact(item["generation"], generation, "generation linkage")
            for schedule in SCHEDULES:
                for slot in range(4):
                    path = folder / f"protocol_{schedule}_{generation:04d}_team{slot}.npz"; rel = str(path.relative_to(out)); checks.require(rel in complete["files"], "protocol completion binding"); check_raw(npz(path), worlds, checks, f"protocol {condition}/{seed}/{part}/{generation}/{schedule}/{slot}"); checks.add("protocol_tables")
            if generation > 0:
                checks.exact(item["replaced_identity"], [0, 1, 2, 3, 0, 1, 2, 3][generation - 1], "replacement identity"); counts = item["schedule_counts"]; checks.exact(sorted(counts), list(SCHEDULES), "schedule count keys"); checks.exact(sum(counts.values()), 300, "schedule count total")
                for name, global_step in (("train_0001.npz", (generation - 1) * 300), ("train_0300.npz", generation * 300 - 1)):
                    path = folder / name; rel = str(path.relative_to(out)); checks.require(rel in complete["files"], "trace completion binding"); check_trace(path, train_worlds, condition, seed, part, generation, checks); checks.exact(npz(path)["global_step"], np.asarray([global_step], dtype=np.int64), "global step")
                checks.require((folder / "generation_after.pt").is_file(), "generation state")
            else: checks.exact(item["replaced_identity"], None, "generation zero identity")
            checks.add("generations")
        checks.add("chains")
    result = {"passed": True, "status": "passed_iterated_replacement_trace_audit", "formal": True, "probe": "iterated_replacement", "checks": checks.count, "coverage": dict(checks.coverage), "source_hashes": invocation["source_hashes"], "input_hashes": invocation["input_hashes"], "training_complete_sha256": sha(out / "training_complete.json"), "audit_source_sha256": sha(ROOT / "iterated_audit.py"), "exclusions": ["No optimizer-state replay; the audit checks endpoint tables, categorical training traces, schedule replay, reward arithmetic and completion bindings."]}
    write(out / "iterated_audit.json", result); shutil.copy2(ROOT / "iterated_audit.py", out / "iterated_audit_source.py"); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); started = time.monotonic()
    try:
        result = run(args.out.resolve()); result["seconds"] = time.monotonic() - started; write(args.out.resolve() / "iterated_audit.json", result); print(json.dumps({key: result[key] for key in ("passed", "status", "checks", "coverage", "seconds")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(args.out.resolve() / f"iterated_audit_failure_{stamp}.json", {"passed": False, "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "iterated_audit.py")}); raise


if __name__ == "__main__": main()
