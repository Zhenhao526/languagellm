"""Bounded trace and endpoint audit for v0.39 triad formation."""
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
SEEDS = (34101, 34102, 34103, 34104); PARTITIONS = (1, 2, 3); CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC"); SCHEDULES = ("A", "B", "C"); CHECKPOINTS = (0, 100, 600, 1200); VOCAB = 7; RESOURCES = 3; BATCH = 240; MAPS = 120
TEAMS = {"A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)), "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)), "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0))}


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


def expected_schedule(condition, seed, part, step):
    if condition == "fixed_A": return "A"
    if condition == "rotating_AB": return "A" if step % 2 == 0 else "B"
    return str(np.random.default_rng(np.random.SeedSequence([39039, int(seed), int(part), int(step), 77])).choice(np.asarray(SCHEDULES)))


def draw(probs, uniforms): return np.minimum((np.cumsum(probs, axis=-1) < uniforms[..., None]).sum(-1), probs.shape[-1] - 1)


def check_trace(path, train_worlds, condition, seed, part, step, checks):
    raw = npz(path); checks.require(raw["schedule_id"].shape == (1,), "schedule shape"); schedule = str(raw["schedule_id"][0]); checks.require(schedule in SCHEDULES, "schedule domain"); checks.exact(np.asarray(schedule), np.asarray(expected_schedule(condition, seed, part, step)), "schedule replay"); checks.exact(raw["global_step"], np.asarray([step], dtype=np.int64), "global step"); indices = raw["world__indices"]; uniforms = raw["world__uniforms"]; checks.require(indices.shape == (BATCH,) and np.issubdtype(indices.dtype, np.integer) and ((indices >= 0) & (indices < len(train_worlds["map_id"]))).all(), "trace indices"); checks.require(uniforms.shape == (BATCH, 6) and np.isfinite(uniforms).all() and ((uniforms >= 0) & (uniforms <= 1)).all(), "trace uniforms")
    positions = train_worlds["positions"][indices]; fields = {key[len("trace__"):]: value for key, value in raw.items() if key.startswith("trace__")}; checks.exact(fields["positions"], positions, "positions linkage"); checks.exact(fields["uniforms"], uniforms, "uniform linkage"); checks.require(fields["messages"].shape == (BATCH, RESOURCES) and fields["actions"].shape == (BATCH, RESOURCES), "message/action shape"); checks.require(((fields["messages"] >= 0) & (fields["messages"] < VOCAB)).all(), "message domain")
    probs = fields["token_probabilities"]; checks.require(probs.shape == (BATCH, RESOURCES, VOCAB) and np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-6), "token probs")
    for resource in range(RESOURCES): checks.exact(draw(probs[:, resource], uniforms[:, resource]), fields["messages"][:, resource], f"token replay {resource}")
    actions_probs = fields["action_probabilities"]; checks.require(actions_probs.shape == (BATCH, RESOURCES, 6) and np.isfinite(actions_probs).all() and np.allclose(actions_probs.sum(-1), 1, atol=1e-6), "action probs")
    for resource in range(RESOURCES): checks.exact(draw(actions_probs[:, resource], uniforms[:, RESOURCES + resource]), fields["actions"][:, resource], f"action replay {resource}")
    success = (fields["actions"] == positions).astype(np.float32); checks.exact(fields["success"], success, "success"); reward = (success.sum(axis=1) / 6.0 + .5 * success.prod(axis=1)).astype(np.float32); checks.require(np.allclose(fields["reward"], reward, atol=1e-6, rtol=1e-6), "reward"); checks.require(np.allclose(fields["advantage"], reward - .1, atol=1e-6, rtol=1e-6), "advantage"); checks.require(fields["sender_logp"].shape == (BATCH, RESOURCES) and fields["receiver_logp"].shape == (BATCH,), "logp shape"); checks.add("trace_rows", BATCH); checks.add("trace_files")


def run(out: Path):
    checks = Checks(); invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); runs = read(out / "runs.json"); checks.exact(invocation["formal"], True, "formal invocation"); checks.exact(invocation["seeds"], list(SEEDS), "seeds"); checks.exact(invocation["partitions"], list(PARTITIONS), "partitions"); checks.exact(invocation["conditions"], list(CONDITIONS), "conditions"); checks.exact(invocation["updates"], 1200, "updates"); checks.exact(invocation["checkpoints"], list(CHECKPOINTS), "checkpoints"); checks.exact(complete["status"], "complete", "completion"); checks.exact(complete["formal"], True, "formal completion"); checks.exact(complete["probe"], "triad_origin", "probe"); checks.exact(complete["runs"], 36, "runs"); checks.exact(runs["count"], 36, "run manifest")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items(): checks.require(Path(path).is_file(), key + " bound path"); checks.exact(sha(Path(path)), digest, key + " bound hash")
    worlds = npz(out / "test_worlds.npz"); checks.exact(len(worlds["map_id"]), MAPS, "test worlds")
    for part in PARTITIONS: checks.exact(len(npz(out / f"train_worlds_p{part}.npz")["map_id"]), 480, f"train worlds p{part}")
    for seed, part, condition in itertools.product(SEEDS, PARTITIONS, CONDITIONS):
        chain = out / "social" / f"s{seed}_p{part}_{condition}"; cfg = read(chain / "config.json"); checks.exact(cfg["condition"], condition, "config condition"); curve = read(chain / "curve.json"); checks.exact([x["update"] for x in curve], list(CHECKPOINTS), "curve updates"); train_worlds = npz(out / f"train_worlds_p{part}.npz")
        for item in curve:
            update = int(item["update"]); checks.exact(sorted(item["scores"]), list(SCHEDULES), "score schedule keys")
            for schedule in SCHEDULES:
                for slot in range(4):
                    path = chain / f"protocol_{schedule}_{update:04d}_team{slot}.npz"; rel = str(path.relative_to(out)); checks.require(rel in complete["files"], "protocol binding"); raw = npz(path); checks.exact(raw["map_id"], worlds["map_id"], "protocol map"); checks.exact(raw["photo_ids"], worlds["photo_ids"], "protocol photos"); checks.exact(raw["positions"], worlds["positions"], "protocol positions"); checks.exact(raw["shown"], worlds["shown"], "protocol shown"); checks.require(raw["tokens"].shape == (MAPS, RESOURCES), "protocol tokens"); checks.require(raw["receiver_logits"].shape == (VOCAB ** RESOURCES, RESOURCES, 6), "protocol receiver"); checks.add("protocol_tables")
            if update == 1200:
                for slot in range(4): checks.require((chain / f"train_1200_slot{slot}.npz").is_file(), "terminal trace")
        for name, step in (("train_0001", 0), ("train_1200", 1199)):
            for slot in range(4): check_trace(chain / f"{name}_slot{slot}.npz", train_worlds, condition, seed, part, step, checks)
        checks.add("runs")
    result = {"passed": True, "status": "passed_triad_trace_audit", "formal": True, "probe": "triad_origin", "checks": checks.count, "coverage": dict(checks.coverage), "source_hashes": invocation["source_hashes"], "input_hashes": invocation["input_hashes"], "training_complete_sha256": sha(out / "training_complete.json"), "audit_source_sha256": sha(ROOT / "triad_audit.py"), "exclusions": ["No optimizer-state replay; audit checks endpoint tables, categorical traces, reward arithmetic, schedule replay and completion bindings."]}; write(out / "triad_audit.json", result); shutil.copy2(ROOT / "triad_audit.py", out / "triad_audit_source.py"); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); started = time.monotonic()
    try:
        result = run(args.out.resolve()); result["seconds"] = time.monotonic() - started; write(args.out.resolve() / "triad_audit.json", result); print(json.dumps({k: result[k] for k in ("passed", "status", "checks", "coverage", "seconds")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(args.out.resolve() / f"triad_audit_failure_{stamp}.json", {"passed": False, "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "triad_audit.py")}); raise


if __name__ == "__main__": main()
