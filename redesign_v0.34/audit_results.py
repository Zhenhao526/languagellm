"""Bounded independent replay audit for v0.33 team traces."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
CONDITIONS = ("single_full", "dual_same_full", "dual_complementary")
PRIVATE_TYPES = (0, 1, 0, 1)


def read(path): return json.loads(Path(path).read_text())
def write(path, value): Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def npz(path):
    with np.load(path, allow_pickle=False) as z: return {key: z[key] for key in z.files}


class Checks:
    def __init__(self): self.count = 0; self.coverage = {}
    def require(self, ok, label):
        self.count += 1
        if not bool(ok): raise AssertionError(label)
    def exact(self, actual, expected, label): self.require(np.array_equal(actual, expected), label)
    def close(self, actual, expected, label): self.require(np.allclose(actual, expected, atol=1e-6, rtol=1e-6), label)
    def add(self, key, value=1): self.coverage[key] = self.coverage.get(key, 0) + value


def draw(probabilities, uniforms):
    return np.minimum((np.cumsum(probabilities, axis=-1) < uniforms[..., None]).sum(-1), probabilities.shape[-1] - 1)


def check_source_binding(invocation, complete, checks):
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            checks.require(Path(path).is_file(), "bound file exists")
            checks.exact(sha(Path(path)), digest, "bound file hash")


def check_protocol(raw, worlds, checks):
    n = len(worlds["map_id"])
    for key in ("map_id", "photo_ids", "positions", "shown"):
        checks.exact(raw[key], worlds[key], "protocol world/" + key)
    checks.require(raw["tokens"].shape == (n, 2) and raw["tokens"].dtype.kind in "iu", "protocol token shape")
    checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), "protocol token domain")
    checks.require(raw["sender_log_probs"].shape == (n, 49) and np.isfinite(raw["sender_log_probs"]).all(), "protocol sender table")
    checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1, atol=1e-6), "protocol sender normalization")
    checks.require(raw["receiver_logits"].shape == (49, 2, 6) and np.isfinite(raw["receiver_logits"]).all(), "protocol receiver table")
    checks.add("endpoint_protocol_worlds", n)


def check_trace(fields, train_worlds, checks):
    required = {"indices", "uniforms", "positions", "actions", "success", "reward", "advantage"}
    if not required.issubset(fields): raise AssertionError("trace fields")
    idx = fields["indices"]
    checks.require(idx.ndim == 1 and len(idx) == 240 and np.issubdtype(idx.dtype, np.integer), "trace index shape")
    positions = train_worlds["positions"][idx]
    checks.exact(fields["positions"], positions, "trace positions")
    checks.require(fields["uniforms"].shape == (240, 4) and np.isfinite(fields["uniforms"]).all() and ((fields["uniforms"] >= 0) & (fields["uniforms"] <= 1)).all(), "trace uniforms")
    actions = fields["actions"]; success = (actions == positions).astype(np.float32)
    checks.exact(fields["success"], success, "trace action success")
    reward = (.25 * success.sum(1) + .5 * success.prod(1)).astype(np.float32)
    checks.close(fields["reward"], reward, "trace reward"); checks.close(fields["advantage"], reward - .5, "trace advantage")
    if "action_probabilities" in fields:
        probs = fields["action_probabilities"]; checks.require(probs.shape == (240, 2, 6) and np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-6), "trace action probabilities")
        checks.exact(draw(probs, fields["uniforms"][:, 2:4]), actions, "trace action categorical replay")
    if "token_probabilities" in fields:
        probs = fields["token_probabilities"]; checks.require(probs.shape == (240, 2, 7) and np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-6), "trace token probabilities")
        checks.exact(draw(probs, fields["uniforms"][:, :2]), fields["messages"], "trace single token replay")
    else:
        for key, token_key in (("token_probabilities_food", "token_food"), ("token_probabilities_water", "token_water")):
            probs = fields[key]; checks.require(probs.shape == (240, 7) and np.isfinite(probs).all() and np.allclose(probs.sum(-1), 1, atol=1e-6), "trace dual token probabilities")
            column = 0 if token_key == "token_food" else 1
            checks.exact(draw(probs, fields["uniforms"][:, column]), fields[token_key], "trace dual token categorical replay")
    checks.add("training_trace_rows", 240)


def audit(out, checks):
    invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); check_source_binding(invocation, complete, checks)
    seeds, panels = invocation["seeds"], invocation["partitions"]; times = [0, 100, 600, 1200, 2100, 2400] if invocation["formal"] else [0, 40]
    checks.exact(invocation["conditions"], list(CONDITIONS), "conditions"); checks.exact(invocation["private_types"], list(PRIVATE_TYPES), "private types")
    test_worlds = npz(out / "test_worlds.npz"); train_worlds = npz(out / "train_worlds.npz")
    checks.exact(len(test_worlds["map_id"]), 180, "test rows"); checks.exact(len(train_worlds["map_id"]), 720, "train rows")
    for seed, panel in itertools.product(seeds, panels):
        folders = {condition: out / "social" / f"s{seed}_p{panel}_{condition}" for condition in CONDITIONS}
        for condition, folder in folders.items():
            config = read(folder / "config.json"); checks.exact(config["condition"], condition, "config condition"); checks.exact(config["teams"], [[0, 1, 2], [1, 2, 3], [2, 3, 0], [3, 0, 1]] if condition != "single_full" else [[0, 1], [1, 2], [2, 3], [3, 0]], "team schedule")
            manifest = read(out / "cache" / f"s{seed}_p{panel}" / "view_manifest.json")
            for error in manifest["full_view_errors"].values(): checks.require(np.isfinite(error) and error <= 3e-5, "full view inheritance")
            curve = read(folder / "curve.json"); checks.exact([item["update"] for item in curve], times, "curve checkpoints")
            for t in times:
                for slot in range(4):
                    key = f"team{slot}"; path = folder / f"protocol_{t:04d}_{key}.npz"; raw = npz(path); check_protocol(raw, test_worlds, checks); checks.require(str(path.relative_to(out)) in complete["files"], "protocol completion binding")
                agreement = npz(folder / f"agreement_{t:04d}.npz"); checks.require(set(agreement) == {f"{view}_type{k}_{suffix}" for view in ("full", "food_only", "water_only") for k in (0, 1) for suffix in ("a", "b")}, "agreement inventory"); checks.add("agreement_files")
            for step in (0, 2100, 2399) if invocation["formal"] else (0, 39):
                path = folder / f"train_{step + 1:04d}.npz"; raw = npz(path)
                groups = sorted({key.split("__", 3)[0] + "__" + key.split("__", 3)[1] + "__" + key.split("__", 3)[2] for key in raw if key.startswith("trace__")})
                for prefix in groups:
                    fields = {key[len(prefix) + 2:]: value for key, value in raw.items() if key.startswith(prefix + "__")}
                    slot = prefix.split("__")[1]
                    fields["indices"] = raw[f"world__{slot}__indices"]
                    check_trace(fields, train_worlds, checks)
                checks.add("trace_files")
        # The same slot fixtures are paired across all three conditions.
        for step in (1, 2101, 2400) if invocation["formal"] else (1, 40):
            arrays = []
            for condition in CONDITIONS:
                raw = npz(folders[condition] / f"train_{step:04d}.npz"); arrays.append({key: value for key, value in raw.items() if key.startswith("world__")})
            for key in arrays[0]:
                for other in arrays[1:]: checks.exact(arrays[0][key], other[key], "condition paired world"); checks.add("paired_world_checks")
    checks.add("social_runs", len(seeds) * len(panels) * len(CONDITIONS)); return invocation, complete


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", type=Path, required=True); args = ap.parse_args(); out = args.out.resolve(); checks = Checks(); started = time.monotonic()
    invocation, complete = audit(out, checks); result = dict(passed=True, status="passed_v033_team_trace_audit", formal=invocation["formal"], checks=checks.count, coverage=checks.coverage, source_hashes=invocation["source_hashes"], input_hashes=invocation["input_hashes"], training_complete_sha256=sha(out / "training_complete.json"), audit_source_sha256=sha(Path(__file__)), seconds=time.monotonic() - started, exclusions=["No complete gradient/Adam replay.", "Inherited visual/private preparation is bound by hashes; full-view equality and masked-view inventory are checked."])
    write(out / "audit_execution.json", result); shutil.copy2(Path(__file__), out / "audit_results_source.py"); print(json.dumps({key: result[key] for key in ("passed", "checks", "coverage", "seconds")}, ensure_ascii=False))


if __name__ == "__main__": main()
