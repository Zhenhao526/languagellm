"""Independent deterministic audit for the v0.48 co-adaptation batch."""
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
OUT = ROOT / "results" / "coadaptation_001"
V043 = PROJECT / "redesign_v0.43"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
ASSIGNMENTS = ("012", "021", "102", "120", "201", "210")
ROLE_MODES = ("static_role", "random_role")
CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
CULTURES = tuple(f"{mode}__{condition}" for mode in ROLE_MODES for condition in CONDITIONS)
RESIDENT_MODES = ("resident_frozen", "resident_sender_sparse", "resident_receiver_sparse", "resident_both_sparse")
SCHEDULES = ("A", "B", "C")
ROLE_PERMS = tuple(itertools.permutations(range(3)))
ROLE_KEYS = tuple("".join(map(str, p)) for p in ROLE_PERMS)
UPDATES = 600
TRACE_STEPS = (0, UPDATES - 1)
BATCH, TRAIN_WORLDS, RESOURCES, TOKENS = 240, 480, 3, 2
SCHEDULE_NAMESPACE, FIXTURE_NAMESPACE, ROLE_NAMESPACE, RESET_NAMESPACE = 48040, 48041, 48042, 48043
TEAMS = {
    "A": ((0, 1, 2, 3), (1, 2, 3, 0), (2, 3, 0, 1), (3, 0, 1, 2)),
    "B": ((0, 2, 1, 3), (1, 3, 0, 2), (2, 0, 3, 1), (3, 1, 2, 0)),
    "C": ((0, 3, 2, 1), (1, 0, 3, 2), (2, 1, 0, 3), (3, 2, 1, 0)),
}
RESIDENT_ROLES = {
    "resident_frozen": (),
    "resident_sender_sparse": ("sender",),
    "resident_receiver_sparse": ("receiver",),
    "resident_both_sparse": ("sender", "receiver"),
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


class Checks:
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
        a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
        self.require(a.shape == b.shape, label + " shape")
        self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size)
        err = float(np.max(np.abs(a - b))) if a.size else 0.0
        self.max_error = max(self.max_error, err)
        self.require(np.allclose(a, b, atol=atol, rtol=rtol), label)


def schedule_for(seed, part, step):
    return str(np.random.default_rng(np.random.SeedSequence([SCHEDULE_NAMESPACE, seed, part, step, 77])).choice(np.asarray(SCHEDULES)))


def role_for(mode, seed, part, step, slot):
    if mode == "static_role":
        return (0, 1, 2)
    return tuple(int(x) for x in np.random.default_rng(np.random.SeedSequence([ROLE_NAMESPACE, seed, part, step, slot, 77])).permutation(RESOURCES))


def fixture(seed, part, slot, step):
    idx = np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE, seed, part, slot, step, 1])).integers(0, TRAIN_WORLDS, size=BATCH, dtype=np.int64)
    uni = np.random.default_rng(np.random.SeedSequence([FIXTURE_NAMESPACE, seed, part, slot, step, 2])).random((BATCH, RESOURCES * TOKENS + RESOURCES), dtype=np.float32)
    return idx, uni


def train_ids(partition):
    maps = np.asarray(list(itertools.permutations(range(6), RESOURCES)), dtype=np.int64)
    panels = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
    rank = {site: i for i, site in enumerate(panels[partition - 1])}
    return np.asarray([i for i, triple in enumerate(maps) if rank[int(triple[0])] < rank[int(triple[1])]], dtype=np.int64)


def metric(actions, positions, map_id, partition):
    correct = np.asarray(actions) == np.asarray(positions)
    train = np.isin(map_id, train_ids(partition))
    return {split: {"J": float(np.mean(np.all(correct[mask], axis=-1))), "resource0": float(np.mean(correct[mask, 0])), "resource1": float(np.mean(correct[mask, 1])), "resource2": float(np.mean(correct[mask, 2]))} for split, mask in (("train60", train), ("target60", ~train), ("all120", np.ones(len(correct), dtype=bool)))}


def replay_trace(audit, path, seed, part, mode, train_worlds):
    d = load(path); step = int(d["global_step"][0]); slot = int(d["slot"][0]); schedule = str(d["schedule_id"][0])
    audit.require(step in TRACE_STEPS, f"{path.name} step"); audit.require(schedule == schedule_for(seed, part, step), f"{path.name} schedule"); audit.exact(d["team"], np.asarray(TEAMS[schedule][slot], dtype=np.int64), f"{path.name} team"); audit.exact(d["role_permutation"], np.asarray(role_for(mode, seed, part, step, slot), dtype=np.int64), f"{path.name} role")
    idx, uni = fixture(seed, part, slot, step); audit.exact(d["world__indices"], idx, f"{path.name} indices"); audit.exact(d["world__uniforms"], uni, f"{path.name} uniforms"); audit.exact(d["trace__uniforms"], uni, f"{path.name} trace uniforms")
    original = train_worlds["positions"][idx]; target = original[:, np.asarray(d["role_permutation"], dtype=np.int64)]; audit.exact(d["world__original_positions"], original, f"{path.name} original positions"); audit.exact(d["trace__positions"], target, f"{path.name} target positions")
    msg, tp, ap, actions = d["trace__messages"], d["trace__token_probabilities"], d["trace__action_probabilities"], d["trace__actions"]
    audit.require(msg.shape == (BATCH, 3, 2) and tp.shape == (BATCH, 3, 2, 7) and d["trace__action_logits"].shape == (BATCH, 3, 6) and ap.shape == (BATCH, 3, 6) and actions.shape == (BATCH, 3), f"{path.name} shapes")
    audit.require(np.isin(msg, np.arange(7)).all() and np.isin(actions, np.arange(6)).all(), f"{path.name} domains"); audit.require(np.isfinite(tp).all() and np.isfinite(ap).all(), f"{path.name} finite probs")
    audit.close(tp.sum(-1), np.ones((BATCH, 3, 2), dtype=np.float32), f"{path.name} token sums", atol=3e-5); audit.close(ap.sum(-1), np.ones((BATCH, 3), dtype=np.float32), f"{path.name} action sums", atol=3e-5)
    def draw(prob, u):
        # torch.cumsum on the production float32 tensor accumulates each
        # prefix in higher precision and casts to float32.  This NumPy form
        # reproduces its boundary decisions without importing production code.
        cdf = np.asarray(np.cumsum(prob.astype(np.float64), axis=-1), dtype=np.float32)
        return cdf.astype(np.float64).__lt__(u[..., None]).sum(-1).clip(max=prob.shape[-1] - 1)
    audit.exact(msg[:, :, 0], draw(tp[:, :, 0, :], uni[:, :3]), f"{path.name} token0 draw"); audit.exact(msg[:, :, 1], draw(tp[:, :, 1, :], uni[:, 3:6]), f"{path.name} token1 draw"); audit.exact(actions, draw(ap, uni[:, 6:]), f"{path.name} action draw")
    correct = (actions == target).astype(np.float32); reward = correct.sum(1) / np.float32(6.0) + np.float32(0.5) * correct.prod(1)
    audit.close(d["trace__success"], correct, f"{path.name} success", atol=1e-6, rtol=1e-6); audit.close(d["trace__reward"], reward, f"{path.name} reward", atol=1e-6, rtol=1e-6); audit.close(d["trace__advantage"], reward - np.float32(float(d["trace__baseline"])), f"{path.name} advantage", atol=1e-6, rtol=1e-6); audit.require(float(d["trace__baseline"]) == .1, f"{path.name} baseline"); audit.require(float(d["trace__entropy_weight"]) == .02, f"{path.name} entropy weight")
    logp = np.log(np.maximum(tp, np.finfo(np.float32).tiny)); expected_sender = np.take_along_axis(logp, msg[..., None], axis=-1).squeeze(-1); expected_sender_ent = -(tp * logp).sum(-1); loga = np.log(np.maximum(ap, np.finfo(np.float32).tiny)); expected_receiver = np.take_along_axis(loga, actions[..., None], axis=-1).squeeze(-1).sum(-1); expected_receiver_ent = -(ap * loga).sum(-1).sum(-1)
    audit.close(d["trace__sender_logp"], expected_sender, f"{path.name} sender logp", atol=4e-5, rtol=4e-5); audit.close(d["trace__sender_entropy"], expected_sender_ent, f"{path.name} sender entropy", atol=4e-5, rtol=4e-5); audit.close(d["trace__receiver_logp"], expected_receiver, f"{path.name} receiver logp", atol=4e-5, rtol=4e-5); audit.close(d["trace__receiver_entropy"], expected_receiver_ent, f"{path.name} receiver entropy", atol=4e-5, rtol=4e-5)
    audit.coverage["traces"] += 1; audit.coverage["trace_rows"] += BATCH


def run(out: Path):
    audit = Checks(); invocation = read(OUT / "invocation.json"); complete = read(OUT / "training_complete.json")
    audit.require(invocation.get("formal") is True and invocation.get("version") == "v0.48-coadaptation-horizon", "formal invocation"); audit.require(invocation.get("adaptation_schedule") == "random_ABC", "adaptation schedule"); audit.require(invocation.get("newcomer_reset_mode") == "both", "newcomer reset"); audit.require(tuple(invocation.get("resident_adaptation_modes", ())) == RESIDENT_MODES, "resident mode coverage"); audit.require(complete.get("formal") is True and complete.get("status") == "complete" and complete.get("runs") == 1728 and complete.get("updates_per_run") == UPDATES, "formal completion")
    for kind in ("source_hashes", "input_hashes"):
        for path, digest in invocation[kind].items():
            p = Path(path); audit.require(p.is_file(), f"{kind} exists {path}"); audit.require(sha(p) == digest, f"{kind} hash {path}"); audit.coverage[f"{kind}_hashes"] += 1
    groups = 0
    for seed, part, assignment, culture, resident_mode in itertools.product(SEEDS, PARTITIONS, ASSIGNMENTS, CULTURES, RESIDENT_MODES):
        mode, condition = culture.split("__", 1); chain = OUT / "social" / f"s{seed}_p{part}_{assignment}_{culture}__resident_{resident_mode}"; cfg = read(chain / "config.json")
        audit.require(cfg["seed"] == seed and cfg["partition"] == part and cfg["assignment"] == assignment and cfg["resident_culture"] == culture and cfg["resident_adaptation_mode"] == resident_mode and cfg["adaptation_schedule"] == "random_ABC" and cfg["newcomer_reset_mode"] == "both", f"{chain.name} config"); audit.require(cfg["resident_update_interval"] == 20 and cfg["resident_update_budget"] == 30, f"{chain.name} budget"); audit.require(cfg["reset_record"]["reset_mode"] == "both" and cfg["reset_record"]["reset_modules"] == ["send_context", "send_embedding", "send_recur", "send_out", "receive_embedding", "actor"], f"{chain.name} reset modules")
        logs = (chain / "training.jsonl").read_text().splitlines(); audit.require(len(logs) == UPDATES, f"{chain.name} log count")
        expected_resident_keys = [f"resident{identity}_{role}" for identity in (1, 2, 3) for role in RESIDENT_ROLES[resident_mode]]
        for step, line in enumerate(logs):
            row = json.loads(line); expected_step = (step % 20 == 0 and step // 20 < 30); audit.require(row["update"] == step + 1 and row["global_step"] == step and row["schedule"] == schedule_for(seed, part, step) and row["role_mode"] == mode and row["resident_adaptation_mode"] == resident_mode and row["resident_update"] == expected_step, f"{chain.name} log {step}"); expected = {str(slot): list(role_for(mode, seed, part, step, slot)) for slot in range(4)}; audit.require(row["role_permutations"] == expected, f"{chain.name} log roles {step}"); audit.require(np.isfinite(float(row["loss"])) and all(np.isfinite(float(v)) for v in row["norms"].values()), f"{chain.name} log finite")
            for key in expected_resident_keys:
                audit.require(key in row["norms"], f"{chain.name} resident norm key")
            for key, value in row["norms"].items():
                if key.startswith("resident") and not expected_step:
                    audit.require(float(value) == 0.0, f"{chain.name} resident frozen step {step}")
            audit.coverage["updates"] += 1
        train_worlds = load(V043 / "results" / "permutation_001" / f"train_worlds_p{part}_{assignment}.npz"); test_worlds = load(V043 / "results" / "permutation_001" / f"test_worlds_{assignment}.npz")
        traces = sorted(chain.glob("train_*.npz")); audit.require(len(traces) == 8, f"{chain.name} trace count"); [replay_trace(audit, p, seed, part, mode, train_worlds) for p in traces]
        protocols = sorted(chain.glob("protocol_*.npz")); audit.require(len(protocols) == 48, f"{chain.name} protocol count"); curve = read(chain / "curve.json"); audit.require([x["update"] for x in curve] == [0, 100, 300, 600], f"{chain.name} curve checkpoints")
        for item in curve:
            drift = item["resident_drift"]; audit.require(np.isfinite(float(drift["resident_communication_drift_mean"])) and np.isfinite(float(drift["resident_communication_drift_max"])), f"{chain.name} drift finite"); audit.require(float(drift["resident_communication_drift_mean"]) >= 0 and float(drift["resident_communication_drift_max"]) >= 0, f"{chain.name} drift sign")
            if resident_mode == "resident_frozen": audit.require(float(drift["resident_communication_drift_max"]) <= 1e-12, f"{chain.name} frozen drift")
            update = int(item["update"])
            for schedule in SCHEDULES:
                for slot in range(4):
                    raw = load(chain / f"protocol_{schedule}_{update:04d}_team{slot}.npz"); audit.exact(raw["map_id"], test_worlds["map_id"], f"{chain.name} protocol maps"); audit.exact(raw["photo_ids"], test_worlds["photo_ids"], f"{chain.name} protocol photos"); audit.exact(raw["positions"], test_worlds["positions"], f"{chain.name} protocol positions"); audit.exact(raw["shown"], test_worlds["shown"], f"{chain.name} protocol shown"); audit.exact(raw["role_permutations"], np.asarray(ROLE_PERMS, dtype=np.int64), f"{chain.name} protocol roles"); audit.require(raw["tokens"].shape == (6, 120, 3, 2) and raw["actions"].shape == (6, 120, 3), f"{chain.name} protocol arrays"); audit.require(np.isin(raw["tokens"], np.arange(7)).all() and np.isin(raw["actions"], np.arange(6)).all(), f"{chain.name} protocol domain"); audit.coverage["protocol_files"] += 1
        audit.coverage["chains"] += 1; groups += 1
    audit.require(groups == 1728, "group coverage"); audit.require(audit.coverage["traces"] == 1728 * 8, "trace coverage"); audit.require(audit.coverage["protocol_files"] == 1728 * 48, "protocol coverage")
    result = {"status": "complete", "formal": True, "probe": "coadaptation_horizon", "runs": groups, "checks": audit.checks, "scalar_comparisons": audit.comparisons, "maximum_replay_absolute_difference": audit.max_error, "coverage": dict(audit.coverage), "source_sha256": sha(ROOT / "coadaptation_audit.py"), "production_modules_imported": False, "model_calls": 0, "limits": ["Audit replays deterministic fixtures and stored traces/actions; it does not independently retrain weights.", "The audit verifies the resident update budget and stored drift fields, not causal validity of the learning rule."]}
    out.mkdir(parents=True, exist_ok=True); (out / "coadaptation_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"); shutil.copy2(ROOT / "coadaptation_audit.py", out / "coadaptation_audit_source.py"); return result


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); started = time.monotonic(); result = run(args.out.resolve()); result["seconds"] = time.monotonic() - started; (args.out.resolve() / "coadaptation_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"); print(json.dumps({"status": result["status"], "runs": result["runs"], "checks": result["checks"], "scalar_comparisons": result["scalar_comparisons"], "maximum_replay_absolute_difference": result["maximum_replay_absolute_difference"], "seconds": result["seconds"]}, ensure_ascii=False))


if __name__ == "__main__": main()
