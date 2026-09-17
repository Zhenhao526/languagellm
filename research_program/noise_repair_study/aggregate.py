"""Compact analysis of noisy transmission and sender-worker repair."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import design


def ci(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return [float(x.mean()), float(x.mean())] if len(x) else [None, None]
    h = 2.13145 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]

def mean_or_none(values):
    return float(np.mean(values)) if values else None


def load(path):
    rows = json.loads(Path(path).read_text())["results"]
    return {(int(row["seed"]), row["adaptation"], row["representation"], row["noise_key"]): row for row in rows}


def metric(row, population, name):
    return float(row["final"][population][name]["team_return_mean"])


def summary(row):
    nw = row["final"]["new_worker"]
    inc = row["final"]["incumbent_workers_mean"]
    natural = float(nw["natural_at_train_noise"]["team_return_mean"])
    clean = float(nw["clean_natural"]["team_return_mean"])
    recombined = float(nw["recombined_at_train_noise"]["team_return_mean"])
    return {
        "seed": int(row["seed"]),
        "adaptation": row["adaptation"],
        "representation": row["representation"],
        "noise_key": row["noise_key"],
        "noise_p": float(row["noise_p"]),
        "natural": natural,
        "clean": clean,
        "silent": float(nw["silent"]["team_return_mean"]),
        "permuted": float(nw["permuted_at_train_noise"]["team_return_mean"]),
        "recombined_minus_natural": recombined - natural,
        "noise_degradation": natural - clean,
        "live_minus_silent": natural - float(nw["silent"]["team_return_mean"]),
        "incumbent_natural": float(inc["natural_at_train_noise"]["team_return_mean"]),
        "incumbent_clean": float(inc["clean_natural"]["team_return_mean"]),
        "incumbent_silent": float(inc["silent"]["team_return_mean"]),
        "incumbent_recombined_minus_natural": float(inc["recombined_at_train_noise"]["team_return_mean"] - inc["natural_at_train_noise"]["team_return_mean"]),
        "sender_sequence_exact": bool(row["final"]["sender_sequences"] == row["parent_sender_sequences"]),
        "sender_sequences": row["final"]["sender_sequences"],
        "parent_sender_sequences": row["parent_sender_sequences"],
        "composable": bool(natural >= 0.60 and abs(recombined - natural) <= 0.02),
    }


def analyze(path):
    by = load(path)
    rows = [summary(row) for row in by.values()]
    groups = []
    for adaptation in design.ADAPTATIONS:
        for representation in design.REPRESENTATIONS:
            for noise_key, noise_p in design.NOISE_KEYS.items():
                values = [r for r in rows if r["adaptation"] == adaptation and r["representation"] == representation and r["noise_key"] == noise_key]
                groups.append({
                    "adaptation": adaptation,
                    "representation": representation,
                    "noise_key": noise_key,
                    "noise_p": noise_p,
                    "n": len(values),
                    "natural_mean": mean_or_none([r["natural"] for r in values]),
                    "natural_ci95_t": ci([r["natural"] for r in values]),
                    "clean_mean": mean_or_none([r["clean"] for r in values]),
                    "silent_mean": mean_or_none([r["silent"] for r in values]),
                    "live_minus_silent_mean": mean_or_none([r["live_minus_silent"] for r in values]),
                    "noise_degradation_mean": mean_or_none([r["noise_degradation"] for r in values]),
                    "recombined_minus_natural_mean": mean_or_none([r["recombined_minus_natural"] for r in values]),
                    "incumbent_natural_mean": mean_or_none([r["incumbent_natural"] for r in values]),
                    "incumbent_clean_mean": mean_or_none([r["incumbent_clean"] for r in values]),
                    "incumbent_recombined_minus_natural_mean": mean_or_none([r["incumbent_recombined_minus_natural"] for r in values]),
                    "sender_sequence_exact_count": int(sum(r["sender_sequence_exact"] for r in values)),
                    "composable_count": int(sum(r["composable"] for r in values)),
                })
    effects = []
    for representation in design.REPRESENTATIONS:
        for noise_key, noise_p in design.NOISE_KEYS.items():
            for metric_name in ("natural", "incumbent_natural", "noise_degradation", "recombined_minus_natural"):
                pairs = []
                for seed in design.SEEDS:
                    worker = next((r for r in rows if r["seed"] == seed and r["adaptation"] == "worker_only" and r["representation"] == representation and r["noise_key"] == noise_key), None)
                    coadapt = next((r for r in rows if r["seed"] == seed and r["adaptation"] == "coadapt" and r["representation"] == representation and r["noise_key"] == noise_key), None)
                    if worker is not None and coadapt is not None:
                        pairs.append(coadapt[metric_name] - worker[metric_name])
                effects.append({"representation": representation, "noise_key": noise_key, "noise_p": noise_p, "metric": "coadapt_minus_worker_only_" + metric_name, "values": pairs, "mean": mean_or_none(pairs), "ci95_t": ci(pairs)})
    return {"schema": "noise_repair_analysis_v1", "rule": {"natural_min": 0.60, "absolute_recombined_gap_max": 0.02}, "rows": rows, "groups": groups, "effects": effects}


def write_md(path, data):
    fmt=lambda x: "n/a" if x is None else f"{x:.3f}"
    lines = ["# Noisy transmission and protocol repair", "", "The nine parent-composable seeds are fixed before this batch. `natural` is evaluated at the training noise rate; `incumbent` averages frozen workers 1–3.", "", "| adaptation | receiver | noise | n | new natural | clean | silent | live−silent | recombined−natural | incumbent natural | composable |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["groups"]:
        lines.append(f"| `{row['adaptation']}` | `{row['representation']}` | {row['noise_p']:.2f} | {row['n']} | {fmt(row['natural_mean'])} | {fmt(row['clean_mean'])} | {fmt(row['silent_mean'])} | {fmt(row['live_minus_silent_mean'])} | {fmt(row['recombined_minus_natural_mean'])} | {fmt(row['incumbent_natural_mean'])} | {row['composable_count']}/{row['n']} |")
    lines += ["", "| receiver | noise | coadapt−worker-only natural | coadapt−worker-only incumbent | coadapt−worker-only recombined gap |", "|---|---:|---:|---:|---:|"]
    effects = {(x["representation"], x["noise_key"], x["metric"]): x for x in data["effects"]}
    for representation in design.REPRESENTATIONS:
        for noise_key, noise_p in design.NOISE_KEYS.items():
            a = effects[(representation, noise_key, "coadapt_minus_worker_only_natural")]
            b = effects[(representation, noise_key, "coadapt_minus_worker_only_incumbent_natural")]
            c = effects[(representation, noise_key, "coadapt_minus_worker_only_recombined_minus_natural")]
            def effect_fmt(x):
                if x['mean'] is None: return 'n/a'
                return f"{x['mean']:.3f} [{x['ci95_t'][0]:.3f},{x['ci95_t'][1]:.3f}]"
            lines.append(f"| `{representation}` | {noise_p:.2f} | {effect_fmt(a)} | {effect_fmt(b)} | {effect_fmt(c)} |")
    lines += ["", "A coadaptation gain with a simultaneous incumbent loss is evidence of renegotiation rather than faithful transmission. This is a conditional stability study on parent-composable protocols, not an emergence-rate estimate."]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--results", required=True); parser.add_argument("--out", required=True); parser.add_argument("--markdown", required=True); args = parser.parse_args()
    data = analyze(args.results)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    write_md(args.markdown, data)
    print(json.dumps({"status": "written", "rows": len(data["rows"]), "groups": len(data["groups"])}, ensure_ascii=False))
