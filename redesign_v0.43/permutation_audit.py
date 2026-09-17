"""Independent NumPy audit for the v0.43 permutation-formation batch.

This audit deliberately does not import the production trainer or any model
code.  It replays the deterministic fixture, schedule, role-order and
categorical sampling rules from the recorded arrays, then checks the stored
policy traces and metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
OUT = ROOT / "results" / "permutation_001"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
ROLE_MODES = ("static_role", "random_role")
ASSIGNMENTS = tuple("".join(map(str, p)) for p in itertools.permutations(range(3)))
SCHEDULES = ("A", "B", "C")
ROLE_PERMS = tuple(itertools.permutations(range(3)))
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}
BATCH, RESOURCES, TOKENS, VOCAB, SITES, UPDATES = 240, 3, 2, 7, 6, 600
FIXTURE_NAMESPACE, SCHEDULE_NAMESPACE, ROLE_NAMESPACE = 43041, 43040, 43042
TRACE_STEPS = (0, UPDATES - 1)


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Audit:
    def __init__(self):
        self.checks = 0
        self.comparisons = 0
        self.max_error = 0.0
        self.coverage = defaultdict(int)

    def require(self, value, label):
        self.checks += 1
        if not bool(value):
            raise AssertionError(label)

    def exact(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.array_equal(a, b), label)

    def close(self, actual, expected, label, atol=2e-5, rtol=2e-5):
        a, b = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.max_error = max(self.max_error, err)
        self.require(np.allclose(a, b, atol=atol, rtol=rtol), label)


def fixture(seed, part, slot, step, n_worlds):
    indices = np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE, seed, part, slot, step, 1])).integers(0, n_worlds, size=BATCH, dtype=np.int64)
    uniforms = np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE, seed, part, slot, step, 2])).random((BATCH, RESOURCES * TOKENS + RESOURCES), dtype=np.float32)
    return indices, uniforms


def schedule_for(condition, seed, part, step):
    if condition == "fixed_A":
        return "A"
    if condition == "rotating_AB":
        return "A" if step % 2 == 0 else "B"
    if condition == "random_ABC":
        return str(np.random.default_rng(np.random.SeedSequence([SCHEDULE_NAMESPACE, seed, part, step, 77])).choice(np.asarray(SCHEDULES)))
    raise ValueError(condition)


def role_for(mode, seed, part, step, slot):
    if mode == "static_role":
        return (0, 1, 2)
    return tuple(int(x) for x in np.random.default_rng(np.random.SeedSequence([ROLE_NAMESPACE, seed, part, step, slot, 77])).permutation(RESOURCES))


def categorical(probabilities, uniforms):
    # This is the production _draw rule: the first cumulative probability
    # strictly greater than u is selected (the sum of cumsum < u).
    return (np.cumsum(probabilities, axis=-1) < uniforms[..., None]).sum(axis=-1).clip(max=probabilities.shape[-1] - 1)


def audit_trace(audit, path, seed, part, assignment, mode, condition, worlds):
    d = load_npz(path)
    name = path.name
    step = int(d["global_step"][0])
    slot = int(d["slot"][0])
    schedule = str(d["schedule_id"][0])
    audit.require(step in TRACE_STEPS, f"{name} trace step")
    audit.exact(d["global_step"], np.asarray([step], dtype=np.int64), f"{name} scalar step")
    audit.require(slot in range(4), f"{name} slot")
    audit.require(schedule in SCHEDULES, f"{name} schedule")
    expected_schedule = schedule_for(condition, seed, part, step)
    audit.require(schedule == expected_schedule, f"{name} deterministic schedule")
    expected_team = np.asarray(TEAMS[schedule][slot], dtype=np.int64)
    audit.exact(d["team"], expected_team, f"{name} team")
    expected_role = np.asarray(role_for(mode, seed, part, step, slot), dtype=np.int64)
    audit.exact(d["role_permutation"], expected_role, f"{name} role permutation")
    idx, uniforms = fixture(seed, part, slot, step, len(worlds["map_id"]))
    audit.exact(d["world__indices"], idx, f"{name} fixture indices")
    audit.exact(d["world__uniforms"], uniforms, f"{name} fixture uniforms")
    audit.exact(d["trace__uniforms"], uniforms, f"{name} trace uniforms")
    original = {key: worlds[key][idx] for key in ("map_id", "photo_ids", "positions", "shown")}
    audit.exact(d["world__original_positions"], original["positions"], f"{name} original positions")
    target = original["positions"][:, expected_role]
    audit.exact(d["trace__positions"], target, f"{name} permuted positions")
    msg = d["trace__messages"]
    token_probs = d["trace__token_probabilities"]
    action_logits = d["trace__action_logits"]
    action_probs = d["trace__action_probabilities"]
    actions = d["trace__actions"]
    audit.require(msg.shape == (BATCH, RESOURCES, TOKENS), f"{name} message shape")
    audit.require(token_probs.shape == (BATCH, RESOURCES, TOKENS, VOCAB), f"{name} token probability shape")
    audit.require(action_logits.shape == (BATCH, RESOURCES, SITES), f"{name} action logit shape")
    audit.require(action_probs.shape == (BATCH, RESOURCES, SITES), f"{name} action probability shape")
    audit.require(actions.shape == (BATCH, RESOURCES), f"{name} action shape")
    audit.require(np.isin(msg, np.arange(VOCAB)).all(), f"{name} token domain")
    audit.require(np.isin(actions, np.arange(SITES)).all(), f"{name} action domain")
    audit.require(np.isfinite(token_probs).all() and np.isfinite(action_logits).all() and np.isfinite(action_probs).all(), f"{name} finite policy arrays")
    audit.close(token_probs.sum(axis=-1), np.ones((BATCH, RESOURCES, TOKENS), dtype=np.float32), f"{name} token probability sums", atol=3e-5)
    audit.close(action_probs.sum(axis=-1), np.ones((BATCH, RESOURCES), dtype=np.float32), f"{name} action probability sums", atol=3e-5)
    expected_t0 = categorical(token_probs[:, :, 0, :], uniforms[:, :RESOURCES])
    expected_t1 = categorical(token_probs[:, :, 1, :], uniforms[:, RESOURCES:2 * RESOURCES])
    audit.exact(msg[:, :, 0], expected_t0, f"{name} token0 categorical replay")
    audit.exact(msg[:, :, 1], expected_t1, f"{name} token1 categorical replay")
    expected_actions = categorical(action_probs, uniforms[:, 2 * RESOURCES:])
    audit.exact(actions, expected_actions, f"{name} action categorical replay")
    correct = (actions == target).astype(np.float32)
    reward = (correct.sum(axis=1) / np.float32(6.0) + np.float32(0.5) * correct.prod(axis=1)).astype(np.float32)
    audit.close(d["trace__success"], correct, f"{name} success replay", atol=1e-6, rtol=1e-6)
    audit.close(d["trace__reward"], reward, f"{name} reward replay", atol=1e-6, rtol=1e-6)
    baseline = float(d["trace__baseline"])
    audit.require(baseline == 0.1, f"{name} baseline")
    audit.close(d["trace__advantage"], reward - np.float32(baseline), f"{name} advantage replay", atol=1e-6, rtol=1e-6)
    log_token = np.log(np.maximum(token_probs, np.finfo(np.float32).tiny))
    expected_sender_logp = np.take_along_axis(log_token, msg[..., None], axis=-1).squeeze(-1)
    expected_sender_entropy = -(token_probs * log_token).sum(axis=-1)
    log_action = np.log(np.maximum(action_probs, np.finfo(np.float32).tiny))
    expected_receiver_logp = np.take_along_axis(log_action, actions[..., None], axis=-1).squeeze(-1).sum(axis=-1)
    expected_receiver_entropy = -(action_probs * log_action).sum(axis=-1).sum(axis=-1)
    audit.close(d["trace__sender_logp"], expected_sender_logp, f"{name} sender logp replay", atol=3e-5, rtol=3e-5)
    audit.close(d["trace__sender_entropy"], expected_sender_entropy, f"{name} sender entropy replay", atol=3e-5, rtol=3e-5)
    audit.close(d["trace__receiver_logp"], expected_receiver_logp, f"{name} receiver logp replay", atol=3e-5, rtol=3e-5)
    audit.close(d["trace__receiver_entropy"], expected_receiver_entropy, f"{name} receiver entropy replay", atol=3e-5, rtol=3e-5)
    audit.require(float(d["trace__entropy_weight"]) == 0.02, f"{name} entropy weight")
    audit.coverage["traces"] += 1
    audit.coverage["trace_rows"] += BATCH


def verify_chain(audit, chain, seed, part, assignment, mode, condition):
    cfg = read(chain / "config.json")
    audit.require(cfg["seed"] == seed and cfg["partition"] == part and cfg["assignment"] == assignment and cfg["role_mode"] == mode and cfg["condition"] == condition, f"{chain.name} config")
    worlds = load_npz(OUT / f"train_worlds_p{part}_{assignment}.npz")
    # Six-site training partition has 60 ordered maps and eight image
    # combinations per map (480 grounded worlds); the endpoint test bank has
    # 120 maps with one held-out image combination.
    audit.require(len(worlds["map_id"]) == 480, f"{chain.name} world count")
    lines = (chain / "training.jsonl").read_text().splitlines()
    audit.require(len(lines) == UPDATES, f"{chain.name} training log count")
    for step, line in enumerate(lines):
        row = json.loads(line)
        audit.require(row["update"] == step + 1 and row["global_step"] == step and row["role_mode"] == mode, f"{chain.name} log index")
        audit.require(row["schedule"] == schedule_for(condition, seed, part, step), f"{chain.name} log schedule")
        expected_roles = {str(slot): list(role_for(mode, seed, part, step, slot)) for slot in range(4)}
        audit.require(row["role_permutations"] == expected_roles, f"{chain.name} log roles")
        audit.require(np.isfinite(float(row["loss"])) and all(np.isfinite(float(x)) for x in row["norms"].values()), f"{chain.name} finite optimization log")
        audit.coverage["updates"] += 1
    traces = sorted(chain.glob("train_*.npz"))
    audit.require(len(traces) == 8, f"{chain.name} trace count")
    for path in traces:
        audit_trace(audit, path, seed, part, assignment, mode, condition, worlds)
    protocols = sorted(chain.glob("protocol_*.npz"))
    audit.require(len(protocols) == 48, f"{chain.name} protocol count")
    for path in protocols:
        d = load_npz(path)
        audit.exact(d["role_permutations"], np.asarray(ROLE_PERMS, dtype=np.int64), f"{path.name} role table")
        audit.require(d["tokens"].shape == (6, 120, 3, 2) and d["receiver_logits"].shape == (6, 120, 3, 6), f"{path.name} protocol shapes")
        audit.require(np.isin(d["tokens"], np.arange(VOCAB)).all() and np.isfinite(d["receiver_logits"]).all(), f"{path.name} protocol values")
        audit.coverage["protocols"] += 1


def run(out: Path):
    audit = Audit()
    invocation = read(OUT / "invocation.json")
    complete = read(OUT / "training_complete.json")
    audit.require(invocation.get("formal") is True and invocation.get("version") == "v0.43-permutation-formation", "formal invocation")
    audit.require(complete.get("formal") is True and complete.get("status") == "complete" and complete.get("runs") == 432, "formal completion")
    audit.require(complete.get("updates_per_run") == UPDATES, "updates per run")
    audit.exact(invocation["role_permutations"], np.asarray(ROLE_PERMS).tolist(), "invocation role permutations")
    # Recheck every recorded source and input hash without importing any of it.
    for kind in ("source_hashes", "input_hashes"):
        for path, expected in invocation[kind].items():
            p = Path(path)
            audit.require(p.exists(), f"{kind} exists {path}")
            audit.require(sha(p) == expected, f"{kind} hash {path}")
            audit.coverage[f"{kind}_hashes"] += 1
    chain_count = 0
    for seed, part, assignment, mode, condition in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS, ROLE_MODES, CONDITIONS):
        verify_chain(audit, OUT / "social" / f"s{seed}_p{part}_{assignment}_{mode}_{condition}", seed, part, assignment, mode, condition)
        chain_count += 1
    audit.require(chain_count == 432, "chain coverage")
    audit.require(audit.coverage["traces"] == 432 * 8, "trace coverage")
    audit.require(audit.coverage["protocols"] == 432 * 48, "protocol coverage")
    result = {
        "status": "complete", "formal": True, "probe": "permutation_formation", "runs": chain_count,
        "checks": audit.checks, "scalar_comparisons": audit.comparisons, "maximum_replay_absolute_difference": audit.max_error,
        "coverage": dict(audit.coverage), "source_sha256": sha(ROOT / "permutation_audit.py"),
        "production_modules_imported": False, "model_calls": 0,
        "limits": ["Audit replays recorded traces and deterministic fixtures; it does not independently retrain weights.", "Protocol endpoint checks are covered by permutation_analysis.json."],
    }
    out.mkdir(parents=True, exist_ok=True)
    write_path = out / "permutation_audit.json"
    write_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    shutil.copy2(ROOT / "permutation_audit.py", out / "permutation_audit_source.py")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    result = run(args.out.resolve())
    result["seconds"] = time.monotonic() - started
    (args.out.resolve() / "permutation_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "runs": result["runs"], "checks": result["checks"], "scalar_comparisons": result["scalar_comparisons"], "maximum_replay_absolute_difference": result["maximum_replay_absolute_difference"], "seconds": result["seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
