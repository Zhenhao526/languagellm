"""Summarize the completed four-choice probe without model or checkpoint reads."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from . import dataset, metrics


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return dataset.read_json(path)


def load_npz(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def diagnostics(probabilities, actions, row):
    p = np.asarray(probabilities, dtype=np.float64); a = np.asarray(actions, dtype=np.int16)
    n = len(p); ix = np.arange(n); listeners = row["listeners"]
    candidates = row["candidate_receiver_actions"]
    cp = p[ix[:, None], listeners[:, None], candidates]
    target = cp[ix, row["donor_endpoint"]]
    margin = target - np.max(np.where(np.arange(4)[None, :] == row["donor_endpoint"][:, None], -np.inf, cp), axis=1)
    target_action = row["target_receiver_actions"]
    hit4 = np.argmax(cp, axis=1) == row["donor_endpoint"]
    hit17 = a[ix, listeners] == target_action
    off = row["host_endpoint"] != row["donor_endpoint"]
    return dict(target_probability=float(target[off].mean()),
                margin=float(margin[off].mean()), hit4=float(hit4[off].mean()), hit17=float(hit17[off].mean()),
                target_probability_all=float(target.mean()), margin_all=float(margin.mean()),
                hit4_all=float(hit4.mean()), hit17_all=float(hit17.mean()))


def stats(values):
    x = np.asarray(values, dtype=np.float64)
    return dict(n=int(len(x)), mean=float(x.mean()), sample_sd=float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                min=float(x.min()), max=float(x.max()))


def summarize(run, audit=None):
    run = Path(run).resolve(); plan, prepared, static = __import__("research_program.triadic_content_response_study.dataset", fromlist=["verify"]).verify(run)
    execution = run / "execution"; result = read(execution / "results.json")
    require(result["status"] == "completed", "Completed probe required")
    row = dataset.flatten_rows(static)
    records = []
    for rec in result["records"]:
        if rec["live"]:
            saved = load_npz(rec["path"])
            p = saved["action_probabilities"]; a = saved["action_indices"]
        else:
            source = load_npz(rec["source_natural_path"])
            p = source["action_probabilities"][row["host_indices"]]
            a = source["action_indices"][row["host_indices"]]
        d = diagnostics(p, a, row)
        records.append(dict(seed=rec["seed"], condition=rec["condition"], rule=rec["rule"], live=rec["live"], update=rec["update"], M=rec["metrics"]["M"], **d))
    require(len(records) == 384, "Complete summary record grid")
    condition_order = list(metrics.CONDITIONS)
    trajectory = {}
    for condition in condition_order:
        trajectory[condition] = {}
        for step in metrics.STEPS:
            selected = [r for r in records if r["condition"] == condition and r["update"] == step]
            trajectory[condition][str(step)] = {key: stats([r[key] for r in selected]) for key in ("M", "target_probability", "margin", "hit4", "hit17")}
    endpoint = {}
    for condition in condition_order:
        selected = [r for r in records if r["condition"] == condition and r["update"] == 6000]
        endpoint[condition] = {key: stats([r[key] for r in selected]) for key in ("M", "target_probability", "margin", "hit4", "hit17")}
    primary_by_key = {}
    for key in ("M", "target_probability", "margin", "hit4", "hit17"):
        vals = []
        for seed in metrics.SEEDS:
            def get(condition):
                return next(r[key] for r in records if r["seed"] == seed and r["condition"] == condition and r["update"] == 6000)
            vals.append((get("reciprocal_PL_live") - get("reciprocal_PL_silent")
                         - get("strict_PL_live") + get("strict_PL_silent")))
        primary_by_key[key] = metrics.statistics(vals)
    layer_endpoint = {}
    for condition in condition_order:
        selected = [r for r in result["records"] if r["condition"] == condition and r["update"] == 6000]
        layer_endpoint[condition] = np.mean(np.asarray([r["metrics"]["layer_means"] for r in selected], dtype=np.float64), axis=0).tolist()
    summary = dict(schema="triadic_content_response_summary_v1", run=str(run), plan_sha256=dataset.sha(run / "plan.json"),
                   audit_path=str(audit) if audit else None, groups=48, backgrounds=36, host_donor_cells=16,
                   steps=list(metrics.STEPS), record_count=len(records), trajectory=trajectory, endpoint=endpoint,
                   endpoint_interaction=primary_by_key, layer_endpoint_means=layer_endpoint,
                   records=records, primary=result["primary"],
                   interpretation="M and the hit rates are limited four-choice/full-action probes; they do not establish a public lexicon, compositionality, or human-language origin.")
    out = run / "summary_001"; require(not out.exists(), "Never overwrite summary output"); out.mkdir()
    dataset.write_json(out / "summary.json", summary)
    inputs = {str(execution / "results.json"): dataset.sha(execution / "results.json"), str(run / "plan.json"): dataset.sha(run / "plan.json"), str(run / "prepared.json"): dataset.sha(run / "prepared.json")}
    if audit:
        ap = Path(audit); inputs[str(ap)] = dataset.sha(ap)
    receipt = dict(status="passed", at=time.time(), inputs_sha256=inputs, output_sha256=dataset.sha(out / "summary.json"),
                   model_forwards=0, checkpoint_reads=0, npz_reads=384,
                   primary_exact_match=True, scope="All 384 policy-time records; 192 live probe NPZ plus 192 silent natural aliases; no refit or posthoc case selection.")
    dataset.write_json(out / "receipt.json", receipt)
    return summary, receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--run", required=True); parser.add_argument("--audit")
    args = parser.parse_args(); summary, receipt = summarize(args.run, args.audit)
    print(json.dumps({"status": receipt["status"], "output_sha256": receipt["output_sha256"], "primary": summary["primary"]}, ensure_ascii=False))
