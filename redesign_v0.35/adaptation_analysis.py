"""Independent NumPy reanalysis of online newcomer adaptation."""
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
V034 = ROOT.parent / "redesign_v0.34"
SEEDS = (34101, 34102, 34103, 34104)
PARTITIONS = (1, 2, 3)
CONDITIONS = ("fixed_A", "rotating_AB")
SCHEDULES = ("A", "B", "C")
CHECKPOINTS = (0, 100, 300, 600)
HELD_IDENTITIES = (0, 1)
MAPS = tuple((food, water) for food in range(6) for water in range(6) if food != water)
PANELS = ((0, 1, 2, 3, 4, 5), (0, 2, 1, 4, 3, 5), (0, 3, 1, 5, 2, 4))
TARGET_PAIRS = ((0, 1), (0, 3), (1, 2), (1, 4), (2, 0), (2, 5), (3, 2), (3, 4), (4, 0), (4, 5), (5, 1), (5, 3))
TRAIN_PAIRS = ((0, 2), (0, 5), (1, 0), (1, 3), (2, 1), (2, 4), (3, 1), (3, 5), (4, 2), (4, 3), (5, 0), (5, 4))


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npz(path: Path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def ids(partition, pairs):
    order = PANELS[partition - 1]
    index = {pair: i for i, pair in enumerate(MAPS)}
    return np.asarray(sorted(index[(order[i], order[j])] for i, j in pairs), dtype=np.int64)


def score(raw, partition):
    decoder = np.argmax(raw["receiver_logits"], axis=-1)
    action = decoder[7 * raw["tokens"][:, 0] + raw["tokens"][:, 1]]
    correct = action == raw["positions"]
    train = np.isin(raw["map_id"], ids(partition, TRAIN_PAIRS))
    target = np.isin(raw["map_id"], ids(partition, TARGET_PAIRS))
    held = np.isin(raw["map_id"], np.setdiff1d(np.arange(30, dtype=np.int64), ids(partition, TRAIN_PAIRS)))
    return {split: {"J": float(np.mean(np.all(correct[mask], axis=-1))), "food": float(np.mean(correct[mask, 0])), "water": float(np.mean(correct[mask, 1]))} for split, mask in (("train12", train), ("target12", target), ("held18", held))}


def mean_sd(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "sd": float(values.std(ddof=1)), "values": values.tolist()}


def auc(values):
    return float(np.trapezoid(np.asarray(values, dtype=float), np.asarray(CHECKPOINTS, dtype=float)) / 600.0)


class Checks:
    def __init__(self):
        self.count = 0; self.comparisons = 0; self.max_error = 0.0; self.coverage = Counter()
    def require(self, value, label):
        self.count += 1
        if not bool(value): raise AssertionError(label)
    def exact(self, actual, expected, label): self.require(np.array_equal(actual, expected), label)
    def close(self, actual, expected, label):
        a, b = np.asarray(actual), np.asarray(expected)
        self.require(a.shape == b.shape, label + " shape"); self.require(np.isfinite(a).all() and np.isfinite(b).all(), label + " finite")
        self.comparisons += int(a.size); error = float(np.max(np.abs(a.astype(float) - b.astype(float)))) if a.size else 0.0; self.max_error = max(self.max_error, error)
        self.require(np.allclose(a, b, rtol=1e-6, atol=1e-6), label + " numerical")
    def add(self, key, value=1): self.coverage[key] += value


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


def run(out: Path):
    checks = Checks(); invocation = read(out / "invocation.json"); complete = read(out / "training_complete.json"); runs = read(out / "runs.json")
    checks.exact(invocation["conditions"], list(CONDITIONS), "conditions"); checks.exact(invocation["seeds"], list(SEEDS), "seeds"); checks.exact(invocation["partitions"], list(PARTITIONS), "partitions"); checks.exact(invocation["updates"], 600, "updates"); checks.exact(invocation["checkpoints"], list(CHECKPOINTS), "checkpoints")
    checks.exact(complete["status"], "complete", "completion"); checks.exact(complete["probe"], "online_newcomer_adaptation", "probe"); checks.exact(complete["runs"], 48, "run count"); checks.exact(runs["count"], 48, "run manifest count")
    for key in ("source_hashes", "input_hashes"):
        checks.exact(invocation[key], complete[key], key + " identity")
        for path, digest in invocation[key].items():
            p = Path(path); checks.require(p.is_file(), key + " bound path"); checks.exact(sha(p), digest, key + " bound hash")
    worlds = npz(out / "test_worlds.npz"); checks.exact(len(worlds["map_id"]), 180, "test worlds")
    rows = []
    for run_entry in runs["rows"]:
        seed, panel, condition, held = run_entry["seed"], run_entry["partition"], run_entry["condition"], run_entry["held_identity"]
        folder = out / "social" / f"s{seed}_p{panel}_{condition}_id{held}"; config = read(folder / "config.json")
        checks.exact(config["held_identity"], held, "config held identity"); checks.exact(config["updates"], 600, "config updates"); checks.exact(config["checkpoints"], list(CHECKPOINTS), "config checkpoints")
        curve = read(folder / "curve.json"); checks.exact([item["update"] for item in curve], list(CHECKPOINTS), "curve updates")
        values = {schedule: [] for schedule in SCHEDULES}
        for item in curve:
            update = int(item["update"])
            for schedule in SCHEDULES:
                team_scores = []
                for slot in range(4):
                    path = folder / f"protocol_{schedule}_{update:04d}_team{slot}.npz"; rel = str(path.relative_to(out)); checks.require(rel in complete["files"], "protocol completion hash")
                    raw = npz(path); check_raw(raw, worlds, checks, f"{condition}/{seed}/{panel}/{held}/{schedule}/{update}/{slot}")
                    metric = score(raw, panel); saved = curve[[x["update"] for x in curve].index(update)]["scores"][schedule][f"team{slot}"]
                    for split in ("train12", "target12", "held18"):
                        for key in ("J", "food", "water"): checks.close(metric[split][key], saved[split]["pooled"][key], "saved metric recheck")
                    team_scores.append(metric); checks.add("protocol_tables")
                values[schedule].append({"update": update, "scores": {split: {key: float(np.mean([m[split][key] for m in team_scores])) for key in ("J", "food", "water")} for split in ("train12", "target12", "held18")}})
        for schedule in SCHEDULES:
            rows.append({"seed": seed, "partition": panel, "condition": condition, "held_identity": held, "private_type": int(held % 2), "schedule": schedule, "curve": values[schedule], "auc_target_J": auc([item["scores"]["target12"]["J"] for item in values[schedule]]), "final_target_J": values[schedule][-1]["scores"]["target12"]["J"], "initial_target_J": values[schedule][0]["scores"]["target12"]["J"]})

    def summarize(selected):
        return {"n": len(selected), "initial_target_J": mean_sd([row["initial_target_J"] for row in selected]), "final_target_J": mean_sd([row["final_target_J"] for row in selected]), "auc_target_J": mean_sd([row["auc_target_J"] for row in selected]), "curve": {str(update): mean_sd([row["curve"][i]["scores"]["target12"]["J"] for row in selected]) for i, update in enumerate(CHECKPOINTS)}}

    summary = {condition: {schedule: summarize([row for row in rows if row["condition"] == condition and row["schedule"] == schedule]) for schedule in SCHEDULES} for condition in CONDITIONS}
    by_type = {condition: {str(private_type): {schedule: summarize([row for row in rows if row["condition"] == condition and row["private_type"] == private_type and row["schedule"] == schedule]) for schedule in SCHEDULES} for private_type in (0, 1)} for condition in CONDITIONS}
    seed_summary = {}
    for condition, schedule, seed in itertools.product(CONDITIONS, SCHEDULES, SEEDS):
        seed_summary[f"{condition}/{schedule}/{seed}"] = summarize([row for row in rows if row["condition"] == condition and row["schedule"] == schedule and row["seed"] == seed])
    # Pair rotating and fixed newcomer curves at the same seed×panel×held cell.
    paired = []
    for seed, panel, held, schedule in itertools.product(SEEDS, PARTITIONS, HELD_IDENTITIES, SCHEDULES):
        fixed = next(row for row in rows if row["condition"] == "fixed_A" and row["seed"] == seed and row["partition"] == panel and row["held_identity"] == held and row["schedule"] == schedule)
        rotating = next(row for row in rows if row["condition"] == "rotating_AB" and row["seed"] == seed and row["partition"] == panel and row["held_identity"] == held and row["schedule"] == schedule)
        paired.append({"seed": seed, "partition": panel, "held_identity": held, "schedule": schedule, "curve": [{"update": update, "rotating_minus_fixed": rotating["curve"][i]["scores"]["target12"]["J"] - fixed["curve"][i]["scores"]["target12"]["J"]} for i, update in enumerate(CHECKPOINTS)]})
    paired_summary = {schedule: {str(update): mean_sd([row["curve"][i]["rotating_minus_fixed"] for row in paired if row["schedule"] == schedule]) for i, update in enumerate(CHECKPOINTS)} for schedule in SCHEDULES}
    record = {"status": "complete", "formal": True, "probe": "online_newcomer_adaptation", "seeds": list(SEEDS), "partitions": list(PARTITIONS), "conditions": list(CONDITIONS), "schedules": list(SCHEDULES), "held_identities": list(HELD_IDENTITIES), "checkpoints": list(CHECKPOINTS), "rows": rows, "summary": summary, "private_type_summary": by_type, "seed_summary": seed_summary, "paired_rotating_minus_fixed": paired_summary, "paired_rows": paired, "checks": checks.count, "scalar_comparisons": checks.comparisons, "maximum_metric_absolute_difference": checks.max_error, "source_sha256": sha(ROOT / "adaptation_analysis.py"), "probe_training_complete_sha256": sha(out / "training_complete.json"), "limits": ["Only the newcomer communication modules are optimized; residents and private visual encoders are frozen.", "The endpoint is a fixed four-agent social setting and does not include births, deaths or online population replacement.", "The primary interpretation uses learning curves and grounded J; it does not equate adaptation with a human-like lexicon."]}
    write(out / "adaptation_analysis.json", record)
    qa = {"passed": True, "status": "passed_online_adaptation_recheck", "formal": True, "checks": checks.count, "scalar_comparisons": checks.comparisons, "maximum_metric_absolute_difference": checks.max_error, "coverage": dict(checks.coverage), "analysis_source_sha256": sha(ROOT / "adaptation_analysis.py"), "analysis_sha256": sha(out / "adaptation_analysis.json"), "production_modules_imported": False, "model_calls": 0}
    write(out / "adaptation_raw_validation.json", qa); shutil.copy2(ROOT / "adaptation_analysis.py", out / "adaptation_analysis_source.py")
    return record, qa


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--out", type=Path, required=True); args = parser.parse_args(); started = time.monotonic()
    try:
        record, qa = run(args.out.resolve()); qa["seconds"] = time.monotonic() - started; write(args.out.resolve() / "adaptation_raw_validation.json", qa); print(json.dumps({key: record[key] for key in ("status", "checks", "scalar_comparisons", "maximum_metric_absolute_difference")}, ensure_ascii=False))
    except Exception as error:
        stamp = time.time_ns(); write(args.out.resolve() / f"adaptation_analysis_failure_{stamp}.json", {"status": "failed", "error": repr(error), "traceback": traceback.format_exc(), "source_sha256": sha(ROOT / "adaptation_analysis.py")}); raise


if __name__ == "__main__": main()
