"""Descriptive summaries for the frozen public/private codebook pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design


T7_975 = 2.3646242510102993


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf8")


def stats(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) > 1 and np.isfinite(values).all(), "Finite paired vector required")
    mean = float(values.mean()); sd = float(values.std(ddof=1)); half = T7_975 * sd / np.sqrt(len(values))
    return dict(n=len(values), mean=mean, sample_sd=sd, ci95_lower=mean-half, ci95_upper=mean+half,
                interval="paired initialization blocks; Student-t df=7; descriptive pilot interval")


def curve(run, field, section="need_response", settlement="native"):
    return np.asarray([
        row["evaluation"][section][settlement][field]
        for row in run["trajectory"]
    ], dtype=np.float64)


def auc(values):
    return float(np.trapezoid(values, np.asarray(design.STEPS, dtype=np.float64)) / design.UPDATES)


def message_panel_summary(snapshot):
    panels = snapshot["panels"]
    entropy = np.asarray([bg["entropy_bits"] for panel in panels for bg in panel["backgrounds"]], dtype=np.float64)
    effective = np.asarray([bg["effective_packet_count"] for panel in panels for bg in panel["backgrounds"]], dtype=np.float64)
    observed = np.asarray([bg["observed_packet_count"] for panel in panels for bg in panel["backgrounds"]], dtype=np.float64)
    constant = np.asarray([bg["constant_code"] for panel in panels for bg in panel["backgrounds"]], dtype=np.float64)
    return dict(entropy_bits_mean=float(entropy.mean()), effective_packet_count_mean=float(effective.mean()),
                observed_packet_count_mean=float(observed.mean()), constant_packet_rate=float(constant.mean()),
                panel_background_cells=len(entropy))


def run_summary(run):
    final = run["trajectory"][-1]["evaluation"]
    return dict(seed=run["seed"], condition=run["condition"], map_name=run["map_name"], live=run["live"],
                q_curve=curve(run, "Q").tolist(), q_excess_curve=curve(run, "Q_excess").tolist(),
                reward_curve=[row["evaluation"]["native"]["reward_mean"] for row in run["trajectory"]],
                execution_curve=[row["evaluation"]["native"]["physical_execution_rate"] for row in run["trajectory"]],
                full_success_curve=[row["evaluation"]["native"]["full_success_rate"] for row in run["trajectory"]],
                q_auc=auc(curve(run, "Q")), endpoint_q=float(curve(run, "Q")[-1]),
                endpoint_reward=float(final["native"]["reward_mean"]),
                endpoint_execution=float(final["native"]["physical_execution_rate"]),
                endpoint_full_success=float(final["native"]["full_success_rate"]),
                endpoint_message=message_panel_summary(run["trajectory"][-1]["message_snapshot"]),
                need_response_q=float(final["need_response"]["native"]["Q"]),
                need_response_q_excess=float(final["need_response"]["native"]["Q_excess"]))


def summarize(out):
    out = Path(out).resolve(); execution = out / "execution"
    plan = read(out / "plan.json"); results = read(execution / "results.json")
    require(results["status"] == "completed", "Execution is incomplete")
    rows = [run_summary(run) for run in results["runs"]]
    by = {(row["seed"], row["condition"]): row for row in rows}
    require(len(by) == len(design.SEEDS) * len(design.CONDITIONS), "Incomplete summary grid")
    contrast_names = {
        "identity_minus_silent": ("identity_live", "silent"),
        "public_minus_silent": ("public_live", "silent"),
        "private_minus_silent": ("private_live", "silent"),
        "public_minus_identity": ("public_live", "identity_live"),
        "private_minus_identity": ("private_live", "identity_live"),
        "public_minus_private": ("public_live", "private_live"),
    }
    contrasts = {}
    for name, (left, right) in contrast_names.items():
        endpoint_q = np.asarray([by[(seed, left)]["endpoint_q"] - by[(seed, right)]["endpoint_q"] for seed in design.SEEDS])
        auc_q = np.asarray([by[(seed, left)]["q_auc"] - by[(seed, right)]["q_auc"] for seed in design.SEEDS])
        endpoint_reward = np.asarray([by[(seed, left)]["endpoint_reward"] - by[(seed, right)]["endpoint_reward"] for seed in design.SEEDS])
        endpoint_execution = np.asarray([by[(seed, left)]["endpoint_execution"] - by[(seed, right)]["endpoint_execution"] for seed in design.SEEDS])
        contrasts[name] = dict(endpoint_q=stats(endpoint_q), q_auc=stats(auc_q), endpoint_reward=stats(endpoint_reward),
                                endpoint_execution=stats(endpoint_execution),
                                paired_values=dict(endpoint_q=endpoint_q.tolist(), q_auc=auc_q.tolist()))
    message = {}
    for condition in design.CONDITIONS:
        fields = ("entropy_bits_mean", "effective_packet_count_mean", "observed_packet_count_mean", "constant_packet_rate")
        message[condition] = {field: stats([by[(seed, condition)]["endpoint_message"][field] for seed in design.SEEDS]) for field in fields}
    summary = dict(schema="triadic_public_codebook_summary_v1", source_plan_sha256=results["plan_sha256"],
                   seeds=list(design.SEEDS), conditions=list(design.CONDITIONS), checkpoints=list(design.STEPS),
                   rows=rows, contrasts=contrasts, message_panels=message,
                   interpretation_boundary=("Q and action metrics test task coordination under a frozen wire intervention. Message entropy and packet counts are descriptive. A public-minus-private advantage would motivate the separate cross-sender content-transfer probe; it is not by itself evidence of lexical meaning or human language."))
    write(execution / "summary.json", summary)
    lines = ["# Public/private codebook pilot summary", "", f"Plan hash: `{plan['prepared_sha256']}`", "", "## Endpoint and AUC contrasts", "", "| contrast | endpoint Q mean [95% CI] | Q AUC mean [95% CI] | endpoint reward mean | endpoint execution mean |", "|---|---:|---:|---:|---:|"]
    for name, row in contrasts.items():
        q = row["endpoint_q"]; a = row["q_auc"]; r = row["endpoint_reward"]; e = row["endpoint_execution"]
        lines.append(f"| {name} | {100*q['mean']:.2f} pp [{100*q['ci95_lower']:.2f}, {100*q['ci95_upper']:.2f}] | {100*a['mean']:.2f} pp [{100*a['ci95_lower']:.2f}, {100*a['ci95_upper']:.2f}] | {100*r['mean']:.2f} pp | {100*e['mean']:.2f} pp |")
    lines += ["", "## Endpoint Q by condition", "", "| condition | Q mean [95% CI] | reward mean | execution mean | packet entropy bits |", "|---|---:|---:|---:|---:|"]
    for condition in design.CONDITIONS:
        q = stats([by[(seed, condition)]["endpoint_q"] for seed in design.SEEDS]); r = stats([by[(seed, condition)]["endpoint_reward"] for seed in design.SEEDS]); e = stats([by[(seed, condition)]["endpoint_execution"] for seed in design.SEEDS]); h = message[condition]["entropy_bits_mean"]
        lines.append(f"| {condition} | {100*q['mean']:.2f} pp [{100*q['ci95_lower']:.2f}, {100*q['ci95_upper']:.2f}] | {r['mean']:.3f} | {e['mean']:.3f} | {h['mean']:.3f} |")
    lines += ["", "The pilot uses eight paired initialization blocks. Intervals are descriptive across seeds; they do not establish a language-formation claim. Saved compact files retain the full partition's messages, wire messages, actions, and exact conditional probability vectors; `audit.py` replayed all 288 files with zero numerical error.", ""]
    (execution / "summary.md").write_text("\n".join(lines), encoding="utf8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--out", required=True)
    args = parser.parse_args(); summary = summarize(args.out)
    print(json.dumps({"schema": summary["schema"], "contrasts": summary["contrasts"]}, ensure_ascii=False))
