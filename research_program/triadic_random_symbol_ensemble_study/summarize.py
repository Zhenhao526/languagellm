"""Summarize the audited random-permutation response grid."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import dataset, metrics

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / "research_program/triadic_content_response_study/results/content_001"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return dataset.read(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def statistics(values):
    return metrics.statistics(values)


def diagnostics(probabilities, actions, row):
    p = np.asarray(probabilities, dtype=np.float64); a = np.asarray(actions, dtype=np.int16)
    n = len(p); ix = np.arange(n); listeners = row["listeners"]; candidates = row["candidate_receiver_actions"]
    cp = p[ix[:, None], listeners[:, None], candidates]
    target = cp[ix, row["donor_endpoint"]]
    other = cp.copy(); other[ix, row["donor_endpoint"]] = -np.inf
    margin = target - other.max(axis=1)
    hit4 = np.argmax(cp, axis=1) == row["donor_endpoint"]
    hit17 = a[ix, listeners] == row["target_receiver_actions"]
    off = row["host_endpoint"] != row["donor_endpoint"]
    require(np.any(off), "Off-diagonal rows")
    return dict(target_probability=float(target[off].mean()), margin=float(margin[off].mean()),
                hit4=float(hit4[off].mean()), hit17=float(hit17[off].mean()),
                target_probability_all=float(target.mean()), margin_all=float(margin.mean()),
                hit4_all=float(hit4.mean()), hit17_all=float(hit17.mean()))


def summarize(run, audit=None):
    run = Path(run).resolve(); _plan, _prepared, _maps = dataset.verify(run)
    result = read(run / "execution" / "results.json")
    require(result["status"] == "completed", "Completed ensemble execution required")
    _content_plan, _content_prepared, arrays = dataset.source_arrays()
    row = __import__("research_program.triadic_content_response_study.dataset", fromlist=["flatten_rows"]).flatten_rows(arrays)
    records = []
    for rec in result["records"]:
        if rec["live"]:
            saved = load_npz(rec["path"]); p, a = saved["action_probabilities"], saved["action_indices"]
        else:
            source = load_npz(rec["source_natural_path"]); p, a = source["action_probabilities"][row["host_indices"]], source["action_indices"][row["host_indices"]]
        records.append(dict(seed=rec["seed"], condition=rec["condition"], rule=rec["rule"], mode=rec["mode"],
                            live=rec["live"], update=rec["update"], M=float(rec["metrics"]["M"]),
                            **diagnostics(p, a, row)))
    expected_count = len(metrics.SEEDS) * len(metrics.CONDITIONS) * len(metrics.MODES) * len(metrics.STEPS)
    require(len(records) == expected_count, "Complete ensemble summary grid")

    keys = ("M", "target_probability", "margin", "hit4", "hit17")
    trajectory = {}; endpoint = {}
    for mode in metrics.MODES:
        trajectory[mode] = {}; endpoint[mode] = {}
        for condition in metrics.CONDITIONS:
            trajectory[mode][condition] = {}
            for step in metrics.STEPS:
                selected = [r for r in records if r["mode"] == mode and r["condition"] == condition and r["update"] == step]
                require(len(selected) == 16, "Sixteen seeds per checkpoint")
                trajectory[mode][condition][str(step)] = {key: statistics([r[key] for r in selected]) for key in keys}
            selected = [r for r in records if r["mode"] == mode and r["condition"] == condition and r["update"] == 6000]
            endpoint[mode][condition] = {key: statistics([r[key] for r in selected]) for key in keys}

    ensemble_trajectory = {}
    for condition in metrics.CONDITIONS:
        ensemble_trajectory[condition] = {}
        for step in metrics.STEPS:
            by_seed = []
            for seed in metrics.SEEDS:
                selected = [r["M"] for r in records if r["seed"] == seed and r["condition"] == condition and r["update"] == step]
                require(len(selected) == len(metrics.MODES), "Six permutations per seed/checkpoint")
                by_seed.append(float(np.mean(selected)))
            ensemble_trajectory[condition][str(step)] = statistics(by_seed)

    ensemble_endpoint = {}
    for condition in metrics.CONDITIONS:
        ensemble_endpoint[condition] = {}
        for key in keys:
            by_seed = []
            for seed in metrics.SEEDS:
                selected = [r[key] for r in records if r["seed"] == seed and r["condition"] == condition and r["update"] == 6000]
                require(len(selected) == len(metrics.MODES), "Six permutations per seed endpoint")
                by_seed.append(float(np.mean(selected)))
            ensemble_endpoint[condition][key] = statistics(by_seed)

    content_summary_path = CONTENT / "summary_001" / "summary.json"
    content_summary = read(content_summary_path)
    baseline = {(int(r["seed"]), r["condition"], r["update"]): r for r in content_summary["records"]}
    require(len(baseline) == 384, "Complete natural baseline")
    by = {(r["seed"], r["condition"], r["mode"], r["update"]): r for r in records}
    paired = {}
    for mode in metrics.MODES:
        paired[mode] = {}
        for rule in metrics.RULES:
            values = [float(baseline[seed, rule + "_PL_live", 6000]["M"] - by[seed, rule + "_PL_live", mode, 6000]["M"]) for seed in metrics.SEEDS]
            paired[mode][rule] = statistics(values)
        values = [float(np.mean([baseline[seed, rule + "_PL_live", 6000]["M"] - by[seed, rule + "_PL_live", mode, 6000]["M"] for rule in metrics.RULES])) for seed in metrics.SEEDS]
        paired[mode]["rule_averaged"] = dict(statistics=statistics(values), by_seed=values,
                                             definition="mean over strict/reciprocal of natural same-group M minus permuted M")
    paired["ensemble"] = {}
    for rule in metrics.RULES:
        values = [float(np.mean([baseline[seed, rule + "_PL_live", 6000]["M"] - by[seed, rule + "_PL_live", mode, 6000]["M"] for mode in metrics.MODES])) for seed in metrics.SEEDS]
        paired["ensemble"][rule] = statistics(values)
    values = [float(np.mean([paired[mode]["rule_averaged"]["by_seed"][i] for mode in metrics.MODES])) for i in range(16)]
    paired["ensemble"]["rule_averaged"] = dict(statistics=statistics(values), by_seed=values,
                                                definition="mean over six permutations and strict/reciprocal of natural same-group M minus permuted M")

    summary = dict(
        schema="triadic_random_symbol_ensemble_summary_v1", run=str(run), plan_sha256=dataset.sha(run / "plan.json"),
        content_baseline_path=str(content_summary_path), content_baseline_sha256=dataset.sha(content_summary_path),
        audit_path=str(audit) if audit else None, groups=48, backgrounds=36, host_donor_cells=16,
        modes=list(metrics.MODES), random_permutations={mode: list(dataset.PERMUTATIONS[mode]) for mode in metrics.MODES},
        steps=list(metrics.STEPS), record_count=len(records), trajectory=trajectory, endpoint=endpoint,
        ensemble_trajectory=ensemble_trajectory, ensemble_endpoint=ensemble_endpoint, paired_selectivity=paired,
        primary=result["primary"], records=records,
        interpretation="The ensemble quantifies response sensitivity across six fixed symbol bijections. It does not establish a public lexicon, compositionality, or language origin.")
    out = run / "summary_001"; require(not out.exists(), "Never overwrite summary output"); out.mkdir()
    dataset.write(out / "summary.json", summary)
    inputs = {str(run / name): dataset.sha(run / name) for name in ("plan.json", "prepared.json", "execution/results.json")}
    inputs[str(content_summary_path)] = dataset.sha(content_summary_path)
    if audit:
        ap = Path(audit).resolve(); inputs[str(ap)] = dataset.sha(ap)
    receipt = dict(status="passed", at=time.time(), inputs_sha256=inputs,
                   output_sha256=dataset.sha(out / "summary.json"), model_forwards=0,
                   checkpoint_reads=0, npz_reads=2304, primary_exact_match=True,
                   scope="All 2304 policy-time records; 1152 live permutation NPZ plus 1152 silent natural aliases; no refit or posthoc case selection.")
    dataset.write(out / "receipt.json", receipt)
    return summary, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True); parser.add_argument("--audit"); args = parser.parse_args()
    summary, receipt = summarize(args.run, args.audit)
    print(json.dumps({"status": receipt["status"], "output_sha256": receipt["output_sha256"], "primary": summary["primary"]}, ensure_ascii=False))
