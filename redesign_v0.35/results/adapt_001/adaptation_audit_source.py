"""Bounded raw trace audit for online newcomer adaptation."""
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
OUT = ROOT / "results" / "adapt_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB")
SCHEDULES = ("A", "B", "C")
HELD_IDENTITIES = (0, 1)
CHECKPOINTS = (0, 100, 300, 600)
TEAMS = {
    "A": ((0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)),
    "B": ((0, 2, 3), (1, 3, 0), (2, 0, 1), (3, 1, 2)),
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
    def close(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected); self.require(a.shape == b.shape, label + " shape"); self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite"); self.require(np.allclose(a, b, atol=1e-6, rtol=1e-6), label + " numerical")
    def add(self, key, value=1): self.coverage[key] += value


def draw(probs, uniforms):
    return np.minimum((np.cumsum(probs, axis=-1) < uniforms[..., None]).sum(-1), probs.shape[-1] - 1)


def check_binding(invocation, complete, checks):
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            p = Path(path); checks.require(p.is_file(), key + " bound path"); checks.exact(sha(p), digest, key + " bound hash")


def check_raw(raw, worlds, checks, label):
    for key in ("map_id", "photo_ids", "positions", "shown"): checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"])
    checks.require(raw["tokens"].shape == (n, 2) and raw["tokens"].dtype.kind in "iu", label + " tokens")
    checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), label + " token domain")
    for key in ("sender_log_probs_food", "sender_log_probs_water"):
        checks.require(raw[key].shape == (n, 7) and np.isfinite(raw[key]).all(), label + " component table")
        checks.require(np.allclose(np.exp(raw[key]).sum(-1), 1, atol=1e-6), label + " component normalization")
    checks.require(raw["sender_log_probs"].shape == (n, 49) and np.isfinite(raw["sender_log_probs"]).all(), label + " joint table")
    checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1, atol=1e-6), label + " joint normalization")
    checks.require(np.allclose(raw["sender_log_probs"], (raw["sender_log_probs_food"][:, :, None] + raw["sender_log_probs_water"][:, None, :]).reshape(n, 49), atol=1e-6, rtol=1e-6), label + " factorization")
    checks.require(raw["receiver_logits"].shape == (49, 2, 6) and np.isfinite(raw["receiver_logits"]).all(), label + " receiver")
    checks.add("protocol_worlds", n)


def check_trace(path: Path, train_worlds, condition, update, checks):
    raw = npz(path); slots = {}
    for key, value in raw.items():
        if key.startswith("world__"):
            _, slot, field = key.split("__"); slots.setdefault(slot, {})[field] = value
    checks.exact(sorted(slots), [f"slot{i}" for i in range(4)], "trace world slots")
    expected_schedule = "A" if condition == "fixed_A" else ("A" if (update - 1) % 2 == 0 else "B")
    expected_teams = TEAMS[expected_schedule]
    prefixes = sorted({"__".join(key.split("__")[:3]) for key in raw if key.startswith("trace__")})
    checks.exact(len(prefixes), 4, "trace prefix count")
    for prefix in prefixes:
        _, slot, role = prefix.split("__")
        # The serialized slot label is ``slot0``/…; strip the textual prefix
        # before indexing the schedule table.
        slot_index = int(slot.removeprefix("slot"))
        expected_role = f"r{expected_teams[slot_index][0]}_sf{expected_teams[slot_index][1]}_sw{expected_teams[slot_index][2]}"
        checks.exact(role, expected_role, "trace schedule role")
        fields = {key[len(prefix) + 2:]: value for key, value in raw.items() if key.startswith(prefix + "__")}
        indices = slots[slot]["indices"]; uniforms = slots[slot]["uniforms"]
        checks.require(indices.shape == (240,) and np.issubdtype(indices.dtype, np.integer) and ((indices >= 0) & (indices < len(train_worlds["map_id"]))).all(), "trace indices")
        checks.require(uniforms.shape == (240, 4) and np.isfinite(uniforms).all() and ((uniforms >= 0) & (uniforms <= 1)).all(), "trace uniforms")
        checks.exact(fields["uniforms"], uniforms, "trace uniform linkage")
        positions = train_worlds["positions"][indices]; checks.exact(fields["positions"], positions, "trace positions")
        checks.require(fields["token_food"].shape == (240,) and fields["token_water"].shape == (240,), "trace tokens")
        checks.exact(fields["messages"], np.stack((fields["token_food"], fields["token_water"]), axis=1), "trace message stack")
        for key, uniform_col, token_key in (("token_probabilities_food", 0, "token_food"), ("token_probabilities_water", 1, "token_water")):
            probs = fields[key]; checks.require(probs.shape == (240, 7) and np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-6), "trace token probability"); checks.exact(draw(probs, uniforms[:, uniform_col]), fields[token_key], "trace token replay")
        probs = fields["action_probabilities"]; checks.require(probs.shape == (240, 2, 6) and np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-6), "trace action probability")
        actions = fields["actions"]; checks.exact(draw(probs, uniforms[:, 2:4]), actions, "trace action replay")
        success = (actions == positions).astype(np.float32); checks.exact(fields["success"], success, "trace success")
        reward = (.25 * success.sum(1) + .5 * success.prod(1)).astype(np.float32); checks.require(np.allclose(fields["reward"], reward, atol=1e-6, rtol=1e-6), "trace reward"); checks.require(np.allclose(fields["advantage"], reward - .5, atol=1e-6, rtol=1e-6), "trace advantage")
        checks.add("trace_rows", 240)
    checks.add("trace_files")


def run(out: Path):
    checks = Checks(); invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); runs = read(out / "runs.json")
    checks.exact(invocation["formal"], True, "formal invocation"); checks.exact(invocation["seeds"], list(SEEDS), "seeds"); checks.exact(invocation["partitions"], list(PARTITIONS), "partitions"); checks.exact(invocation["conditions"], list(CONDITIONS), "conditions"); checks.exact(invocation["held_identities"], list(HELD_IDENTITIES), "held identities"); checks.exact(invocation["updates"], 600, "updates"); checks.exact(invocation["checkpoints"], list(CHECKPOINTS), "checkpoints")
    checks.exact(complete["status"], "complete", "completion"); checks.exact(complete["formal"], True, "formal completion"); checks.exact(complete["probe"], "online_newcomer_adaptation", "probe"); checks.exact(complete["runs"], 48, "runs"); checks.exact(runs["count"], 48, "runs manifest")
    check_binding(invocation, complete, checks); worlds = npz(out / "test_worlds.npz"); train_worlds = npz(out / "train_worlds.npz"); checks.exact(len(worlds["map_id"]), 180, "test world rows"); checks.exact(len(train_worlds["map_id"]), 720, "train world rows")
    for condition, seed, panel, held in itertools.product(CONDITIONS, SEEDS, PARTITIONS, HELD_IDENTITIES):
        folder = out / "social" / f"s{seed}_p{panel}_{condition}_id{held}"; cfg = read(folder / "config.json"); checks.exact(cfg["condition"], condition, "config condition"); checks.exact(cfg["held_identity"], held, "config held"); checks.exact(cfg["updates"], 600, "config updates"); checks.exact(cfg["checkpoints"], list(CHECKPOINTS), "config checkpoints")
        curve = read(folder / "curve.json"); checks.exact([x["update"] for x in curve], list(CHECKPOINTS), "curve updates")
        for update in CHECKPOINTS:
            for schedule in SCHEDULES:
                for slot in range(4):
                    path = folder / f"protocol_{schedule}_{update:04d}_team{slot}.npz"; rel = str(path.relative_to(out)); checks.require(rel in complete["files"], "protocol completion binding"); check_raw(npz(path), worlds, checks, f"protocol {condition}/{seed}/{panel}/{held}/{schedule}/{update}/{slot}"); checks.add("protocol_tables")
            if update in (0, 600):
                path = folder / f"train_{update if update else 1:04d}.npz"; rel = str(path.relative_to(out)); checks.require(rel in complete["files"], "trace completion binding"); check_trace(path, train_worlds, condition, update if update else 1, checks)
        checks.require((folder / "final_newcomer.pt").is_file(), "final newcomer state")
        checks.add("runs")
    result = {"passed": True, "status": "passed_online_adaptation_trace_audit", "formal": True, "probe": "online_newcomer_adaptation", "checks": checks.count, "coverage": dict(checks.coverage), "source_hashes": invocation["source_hashes"], "input_hashes": invocation["input_hashes"], "training_complete_sha256": sha(out / "training_complete.json"), "audit_source_sha256": sha(ROOT / "adaptation_audit.py"), "exclusions": ["No optimizer-state replay; the audit checks saved endpoint tables, categorical training traces, reward arithmetic, schedule identity, completion hashes and frozen input bindings."]}
    write(out / "adaptation_audit.json", result); shutil.copy2(ROOT / "adaptation_audit.py", out / "adaptation_audit_source.py"); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); started = time.monotonic()
    try:
        result = run(args.out.resolve()); result["seconds"] = time.monotonic() - started; write(args.out.resolve() / "adaptation_audit.json", result); print(json.dumps({key: result[key] for key in ("passed", "status", "checks", "coverage", "seconds")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(args.out.resolve() / f"adaptation_audit_failure_{stamp}.json", {"passed": False, "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "adaptation_audit.py")}); raise


if __name__ == "__main__": main()
