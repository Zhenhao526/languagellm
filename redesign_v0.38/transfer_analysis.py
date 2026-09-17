"""Independent NumPy reanalysis of the v0.38 origin-transfer factorial."""
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
FORMATION_CONDITIONS = ("origin_fixed_A", "origin_rotating_AB", "origin_random_ABC")
TRANSMISSION_CONDITIONS = ("fixed_A", "rotating_AB", "random_ABC")
SCHEDULES = ("A", "B", "C")
GENERATIONS = tuple(range(5))
REPLACEMENT_ORDER = (0, 1, 2, 3)
MAPS = tuple((food, water) for food in range(6) for water in range(6) if food != water)
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TARGET_PAIRS = ((0, 1), (0, 3), (1, 2), (1, 4), (2, 0), (2, 5), (3, 2), (3, 4), (4, 0), (4, 5), (5, 1), (5, 3))
TRAIN_PAIRS = ((0, 2), (0, 5), (1, 0), (1, 3), (2, 1), (2, 4), (3, 1), (3, 5), (4, 2), (4, 3), (5, 0), (5, 4))


def read(path: Path): return json.loads(path.read_text())
def write(path: Path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
def sha(path: Path): return hashlib.sha256(path.read_bytes()).hexdigest()
def npz(path: Path):
    with np.load(path, allow_pickle=False) as z: return {key: z[key] for key in z.files}


def ids(partition, pairs):
    order = PANELS[partition - 1]; index = {pair: i for i, pair in enumerate(MAPS)}
    return np.asarray(sorted(index[(order[i], order[j])] for i, j in pairs), dtype=np.int64)


def score(raw, partition):
    decoder = np.argmax(raw["receiver_logits"], axis=-1)
    action = decoder[7 * raw["tokens"][:, 0] + raw["tokens"][:, 1]]
    correct = action == raw["positions"]
    train = np.isin(raw["map_id"], ids(partition, TRAIN_PAIRS)); target = np.isin(raw["map_id"], ids(partition, TARGET_PAIRS)); held = np.isin(raw["map_id"], np.setdiff1d(np.arange(30, dtype=np.int64), ids(partition, TRAIN_PAIRS)))
    return {split: {"J": float(np.mean(np.all(correct[mask], axis=-1))), "food": float(np.mean(correct[mask, 0])), "water": float(np.mean(correct[mask, 1]))} for split, mask in (("train12", train), ("target12", target), ("held18", held))}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0, "values": values.tolist()}


def auc(values):
    return float(np.trapezoid(np.asarray(values, dtype=float), np.arange(len(values), dtype=float)) / (len(values) - 1))


class Checks:
    def __init__(self): self.count = 0; self.comparisons = 0; self.max_error = 0.0; self.coverage = Counter()
    def require(self, value, label):
        self.count += 1
        if not bool(value): raise AssertionError(label)
    def exact(self, actual, expected, label): self.require(np.array_equal(actual, expected), label)
    def close(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected); self.require(a.shape == b.shape, label + " shape"); self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size); error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0; self.max_error = max(self.max_error, error); self.require(np.allclose(a, b, rtol=1e-6, atol=1e-6), label + " numerical")
    def add(self, key, value=1): self.coverage[key] += value


def check_raw(raw, worlds, checks, label):
    for key in ("map_id", "photo_ids", "positions", "shown"): checks.exact(raw[key], worlds[key], label + "/" + key)
    n = len(worlds["map_id"]); checks.require(raw["tokens"].shape == (n, 2) and raw["tokens"].dtype.kind in "iu", label + " tokens"); checks.require(((raw["tokens"] >= 0) & (raw["tokens"] < 7)).all(), label + " token domain")
    for key in ("sender_log_probs_food", "sender_log_probs_water"):
        checks.require(raw[key].shape == (n, 7) and np.isfinite(raw[key]).all(), label + " component table"); checks.require(np.allclose(np.exp(raw[key]).sum(-1), 1, atol=1e-6), label + " component normalization")
    checks.require(raw["sender_log_probs"].shape == (n, 49) and np.isfinite(raw["sender_log_probs"]).all(), label + " joint table"); checks.require(np.allclose(np.exp(raw["sender_log_probs"]).sum(-1), 1, atol=1e-6), label + " joint normalization")
    checks.require(np.allclose(raw["sender_log_probs"], (raw["sender_log_probs_food"][:, :, None] + raw["sender_log_probs_water"][:, None, :]).reshape(n, 49), atol=1e-6, rtol=1e-6), label + " factorization")
    checks.require(raw["receiver_logits"].shape == (49, 2, 6) and np.isfinite(raw["receiver_logits"]).all(), label + " receiver"); checks.add("protocol_worlds", n)


def summarize(rows):
    return {"n": len(rows), "target_J": mean_sd([x["target_J"] for x in rows]), "held_J": mean_sd([x["held_J"] for x in rows]), "train_J": mean_sd([x["train_J"] for x in rows]), "token_change_prev": mean_sd([x["token_change_prev"] for x in rows if x["generation"] > 0]) if any(x["generation"] > 0 for x in rows) else None, "token_change_base": mean_sd([x["token_change_base"] for x in rows])}


def run(out: Path):
    checks = Checks(); invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); runs = read(out / "runs.json")
    checks.exact(invocation["formal"], True, "formal invocation"); checks.exact(invocation["seeds"], list(SEEDS), "seeds"); checks.exact(invocation["partitions"], list(PARTITIONS), "partitions"); checks.exact(invocation["formation_conditions"], list(FORMATION_CONDITIONS), "formation conditions"); checks.exact(invocation["transmission_conditions"], list(TRANSMISSION_CONDITIONS), "transmission conditions"); checks.exact(invocation["generations"], 4, "generations"); checks.exact(invocation["updates_per_generation"], 300, "updates")
    checks.exact(complete["status"], "complete", "completion"); checks.exact(complete["formal"], True, "formal completion"); checks.exact(complete["probe"], "origin_transfer", "probe"); checks.exact(complete["runs"], 108, "runs"); checks.exact(runs["count"], 108, "run manifest")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            p = Path(path); checks.require(p.is_file(), key + " bound path"); checks.exact(sha(p), digest, key + " bound hash")
    worlds = npz(out / "test_worlds.npz"); train_worlds = npz(out / "train_worlds.npz"); checks.exact(len(worlds["map_id"]), 180, "test worlds"); checks.exact(len(train_worlds["map_id"]), 720, "train worlds")
    rows = []; chain_rows = []
    for seed, part, formation, transmission in itertools.product(SEEDS, PARTITIONS, FORMATION_CONDITIONS, TRANSMISSION_CONDITIONS):
        chain = out / "chains" / f"s{seed}_p{part}_{formation}__{transmission}"; cfg = read(chain / "config.json"); checks.exact(cfg["formation_condition"], formation, "formation linkage"); checks.exact(cfg["transmission_condition"], transmission, "transmission linkage"); checks.exact(cfg["replacement_order"], list(REPLACEMENT_ORDER), "replacement order"); checks.require((chain / "generation_000_before.pt").is_file(), "initial state")
        curve = read(chain / "curve.json"); checks.exact([item["generation"] for item in curve], list(GENERATIONS), "curve generations")
        chain_entry = {"seed": seed, "partition": part, "formation_condition": formation, "transmission_condition": transmission, "generations": []}; base_tokens = {}; previous_tokens = {}
        for generation in GENERATIONS:
            gen_folder = chain / f"generation_{generation:02d}"; item = curve[generation]; generation_row = {"generation": generation, "replaced_identity": item["replaced_identity"], "schedules": {}}
            for schedule in SCHEDULES:
                team_scores = []; team_tokens = []
                for slot in range(4):
                    path = gen_folder / f"protocol_{schedule}_{generation:04d}_team{slot}.npz"; rel = str(path.relative_to(out)); checks.require(rel in complete["files"], "protocol completion hash"); raw = npz(path); check_raw(raw, worlds, checks, f"{formation}/{transmission}/{seed}/{part}/{generation}/{schedule}/{slot}"); metric = score(raw, part); saved = item["scores"][schedule][f"team{slot}"]
                    for split in ("train12", "target12", "held18"):
                        for key in ("J", "food", "water"): checks.close(metric[split][key], saved[split]["pooled"][key], "saved metric")
                    team_scores.append(metric); team_tokens.append(raw["tokens"]); checks.add("protocol_tables")
                tokens = np.stack(team_tokens, axis=0)
                if generation == 0: base_tokens[schedule] = tokens; previous_tokens[schedule] = tokens; change_prev = 0.0
                else: change_prev = float(np.mean(np.any(tokens != previous_tokens[schedule], axis=-1))); previous_tokens[schedule] = tokens
                change_base = float(np.mean(np.any(tokens != base_tokens[schedule], axis=-1)))
                pooled = {split: {key: float(np.mean([metric[split][key] for metric in team_scores])) for key in ("J", "food", "water")} for split in ("train12", "target12", "held18")}
                row = {"seed": seed, "partition": part, "formation_condition": formation, "transmission_condition": transmission, "generation": generation, "replaced_identity": item["replaced_identity"], "schedule": schedule, "train_J": pooled["train12"]["J"], "target_J": pooled["target12"]["J"], "held_J": pooled["held18"]["J"], "token_change_prev": change_prev, "token_change_base": change_base}
                rows.append(row); generation_row["schedules"][schedule] = {key: row[key] for key in ("train_J", "target_J", "held_J", "token_change_prev", "token_change_base")}
            chain_entry["generations"].append(generation_row); checks.add("generations")
        chain_rows.append(chain_entry); checks.add("chains")
    summary = {formation: {transmission: {str(generation): {schedule: summarize([x for x in rows if x["formation_condition"] == formation and x["transmission_condition"] == transmission and x["generation"] == generation and x["schedule"] == schedule]) for schedule in SCHEDULES} for generation in GENERATIONS} for transmission in TRANSMISSION_CONDITIONS} for formation in FORMATION_CONDITIONS}
    retention = {}
    formation_auc = {}
    for formation in FORMATION_CONDITIONS:
        retention[formation] = {}; formation_auc[formation] = {}
        for transmission in TRANSMISSION_CONDITIONS:
            retention[formation][transmission] = {}; formation_auc[formation][transmission] = {}
            for schedule in SCHEDULES:
                values = []
                for seed, part in itertools.product(SEEDS, PARTITIONS):
                    curve_values = [next(x for x in rows if x["seed"] == seed and x["partition"] == part and x["formation_condition"] == formation and x["transmission_condition"] == transmission and x["generation"] == g and x["schedule"] == schedule)["target_J"] for g in GENERATIONS]
                    values.append(curve_values)
                arr = np.asarray(values); formation_auc[formation][transmission][schedule] = mean_sd([auc(v) for v in arr]); retention[formation][transmission][schedule] = mean_sd(arr[:, -1] - arr[:, 0])
    record = {"status": "complete", "formal": True, "probe": "origin_transfer", "seeds": list(SEEDS), "partitions": list(PARTITIONS), "formation_conditions": list(FORMATION_CONDITIONS), "transmission_conditions": list(TRANSMISSION_CONDITIONS), "schedules": list(SCHEDULES), "generations": list(GENERATIONS), "updates_per_generation": 300, "rows": rows, "chain_rows": chain_rows, "summary": summary, "retention": retention, "formation_auc": formation_auc, "checks": checks.count, "scalar_comparisons": checks.comparisons, "maximum_metric_absolute_difference": checks.max_error, "source_sha256": sha(ROOT / "transfer_analysis.py"), "probe_training_complete_sha256": sha(out / "training_complete.json"), "coverage": dict(checks.coverage), "limits": ["Formation endpoints inherit frozen private visual encoders from v0.37.", "Only communication modules are optimized during replacement.", "The factorial tests a four-agent toy social-learning bottleneck, not human language history.", "Token changes are surface fingerprints, not semantic distance."]}
    write(out / "transfer_analysis.json", record); qa = {"passed": True, "status": "passed_origin_transfer_recheck", "formal": True, "checks": checks.count, "scalar_comparisons": checks.comparisons, "maximum_metric_absolute_difference": checks.max_error, "coverage": dict(checks.coverage), "analysis_source_sha256": sha(ROOT / "transfer_analysis.py"), "analysis_sha256": sha(out / "transfer_analysis.json"), "production_modules_imported": False, "model_calls": 0}; write(out / "transfer_raw_validation.json", qa); shutil.copy2(ROOT / "transfer_analysis.py", out / "transfer_analysis_source.py"); return record, qa


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); started = time.monotonic()
    try:
        record, qa = run(args.out.resolve()); qa["seconds"] = time.monotonic() - started; write(args.out.resolve() / "transfer_raw_validation.json", qa); print(json.dumps({k: record[k] for k in ("status", "checks", "scalar_comparisons", "maximum_metric_absolute_difference")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(args.out.resolve() / f"transfer_analysis_failure_{stamp}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "transfer_analysis.py")}); raise


if __name__ == "__main__": main()
