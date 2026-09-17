"""Summarize the packet-identity controls without model or checkpoint reads."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import dataset, metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
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
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1 and len(x) == 16 and np.isfinite(x).all(), "Sixteen paired seed values required")
    return metrics.statistics(x)


def diagnostics(probabilities, actions, row):
    p = np.asarray(probabilities, dtype=np.float64)
    a = np.asarray(actions, dtype=np.int16)
    n = len(p); ix = np.arange(n); listeners = row["listeners"]
    candidates = row["candidate_receiver_actions"]
    cp = p[ix[:, None], listeners[:, None], candidates]
    target = cp[ix, row["donor_endpoint"]]
    other = cp.copy(); other[ix, row["donor_endpoint"]] = -np.inf
    margin = target - other.max(axis=1)
    hit4 = np.argmax(cp, axis=1) == row["donor_endpoint"]
    hit17 = a[ix, listeners] == row["target_receiver_actions"]
    off = row["host_endpoint"] != row["donor_endpoint"]
    require(np.any(off), "Off-diagonal rows required")
    return dict(target_probability=float(target[off].mean()), margin=float(margin[off].mean()),
                hit4=float(hit4[off].mean()), hit17=float(hit17[off].mean()),
                target_probability_all=float(target.mean()), margin_all=float(margin.mean()),
                hit4_all=float(hit4.mean()), hit17_all=float(hit17.mean()))


def _index(records):
    return {(r["seed"], r["condition"], r.get("mode"), r["update"]): r for r in records}


def _stats_block(records, keys):
    return {key: statistics([r[key] for r in records]) for key in keys}


def summarize(run, audit=None):
    run = Path(run).resolve()
    plan, prepared, static_maps = dataset.verify(run)
    result = read(run / "execution" / "results.json")
    require(result["status"] == "completed", "Completed control execution required")
    _content_plan, _content_prepared, content_arrays = dataset.source_arrays()
    row = __import__("research_program.triadic_content_response_study.dataset", fromlist=["flatten_rows"]).flatten_rows(content_arrays)

    records = []
    for rec in result["records"]:
        if rec["live"]:
            saved = load_npz(rec["path"])
            probabilities, actions = saved["action_probabilities"], saved["action_indices"]
        else:
            source = load_npz(rec["source_natural_path"])
            probabilities = source["action_probabilities"][row["host_indices"]]
            actions = source["action_indices"][row["host_indices"]]
        d = diagnostics(probabilities, actions, row)
        records.append(dict(seed=rec["seed"], condition=rec["condition"], rule=rec["rule"],
                            mode=rec["mode"], live=rec["live"], update=rec["update"],
                            M=float(rec["metrics"]["M"]), **d))
    require(len(records) == 768, "Complete control summary grid")

    keys = ("M", "target_probability", "margin", "hit4", "hit17")
    trajectory = {}
    endpoint = {}
    for mode in metrics.MODES:
        trajectory[mode] = {}
        endpoint[mode] = {}
        for condition in metrics.CONDITIONS:
            trajectory[mode][condition] = {}
            for step in metrics.STEPS:
                selected = [r for r in records if r["mode"] == mode and r["condition"] == condition and r["update"] == step]
                require(len(selected) == 16, "Sixteen seeds per control checkpoint")
                trajectory[mode][condition][str(step)] = _stats_block(selected, keys)
            selected = [r for r in records if r["mode"] == mode and r["condition"] == condition and r["update"] == 6000]
            endpoint[mode][condition] = _stats_block(selected, keys)

    content_summary_path = CONTENT / "summary_001" / "summary.json"
    content_summary = read(content_summary_path)
    baseline = {(r["seed"], r["condition"], r["update"]): r for r in content_summary["records"]}
    require(len(baseline) == 384, "Complete same-group content baseline")
    by = {(r["seed"], r["condition"], r["mode"], r["update"]): r for r in records}

    paired = {"endpoint_cycle": {}, "cross_group_cycle": {}, "endpoint_minus_cross": {}}
    for rule in metrics.RULES:
        endpoint_values = []
        cross_values = []
        for seed in metrics.SEEDS:
            same = float(baseline[seed, rule + "_PL_live", 6000]["M"])
            endpoint_values.append(same - by[seed, rule + "_PL_live", "endpoint_cycle", 6000]["M"])
            cross_values.append(same - by[seed, rule + "_PL_live", "cross_group_cycle", 6000]["M"])
        paired["endpoint_cycle"][rule] = statistics(endpoint_values)
        paired["cross_group_cycle"][rule] = statistics(cross_values)
        paired["endpoint_minus_cross"][rule] = statistics(np.asarray(endpoint_values) - np.asarray(cross_values))

    average_endpoint = []
    average_cross = []
    average_difference = []
    by_seed = []
    for seed in metrics.SEEDS:
        e = []; c = []
        for rule in metrics.RULES:
            same = float(baseline[seed, rule + "_PL_live", 6000]["M"])
            e.append(same - by[seed, rule + "_PL_live", "endpoint_cycle", 6000]["M"])
            c.append(same - by[seed, rule + "_PL_live", "cross_group_cycle", 6000]["M"])
        endpoint_value = float(np.mean(e)); cross_value = float(np.mean(c))
        average_endpoint.append(endpoint_value); average_cross.append(cross_value)
        average_difference.append(endpoint_value - cross_value)
        by_seed.append(dict(seed=seed, endpoint_cycle_selectivity=endpoint_value,
                            cross_group_selectivity=cross_value,
                            endpoint_minus_cross=endpoint_value - cross_value))

    paired["rule_averaged"] = dict(
        endpoint_cycle=statistics(average_endpoint),
        cross_group_cycle=statistics(average_cross),
        endpoint_minus_cross=statistics(average_difference),
        by_seed=by_seed,
        definition="For each seed, average strict and reciprocal live values of same-group baseline M minus the indicated control M at update6000.")

    summary = dict(
        schema="triadic_packet_identity_control_summary_v1",
        run=str(run), plan_sha256=dataset.sha(run / "plan.json"),
        content_baseline_path=str(content_summary_path),
        content_baseline_sha256=dataset.sha(content_summary_path),
        audit_path=str(audit) if audit else None,
        groups=48, backgrounds=36, host_donor_cells=16, modes=list(metrics.MODES),
        steps=list(metrics.STEPS), record_count=len(records), trajectory=trajectory, endpoint=endpoint,
        paired_selectivity=paired, records=records, primary=result["primary"],
        interpretation=("A packet-endpoint selectivity contrast measures whether a live response changes when the packet is replaced "
                        "by another endpoint from the same group. The cross-group control uses the next group in the same "
                        "sender-listener×attribute layer. Neither contrast establishes a public lexicon, compositionality, or human-language origin."))
    out = run / "summary_001"
    require(not out.exists(), "Never overwrite summary output")
    out.mkdir()
    dataset.write(out / "summary.json", summary)
    inputs = {str(run / "plan.json"): dataset.sha(run / "plan.json"),
              str(run / "prepared.json"): dataset.sha(run / "prepared.json"),
              str(run / "execution" / "results.json"): dataset.sha(run / "execution" / "results.json"),
              str(content_summary_path): dataset.sha(content_summary_path)}
    if audit:
        audit_path = Path(audit).resolve()
        inputs[str(audit_path)] = dataset.sha(audit_path)
    receipt = dict(status="passed", at=time.time(), inputs_sha256=inputs,
                   output_sha256=dataset.sha(out / "summary.json"), model_forwards=0,
                   checkpoint_reads=0, npz_reads=768, primary_exact_match=True,
                   scope="All 768 policy-time-control records; 384 live control NPZ files and 384 silent natural aliases; no refit or posthoc case selection.")
    dataset.write(out / "receipt.json", receipt)
    return summary, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--audit")
    args = parser.parse_args()
    summary, receipt = summarize(args.run, args.audit)
    print(json.dumps({"status": receipt["status"], "output_sha256": receipt["output_sha256"],
                      "primary": summary["primary"]}, ensure_ascii=False))
