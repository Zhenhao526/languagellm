"""Compact analysis for the ternary compositionality extension."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import design


def ci(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2: return [float(x.mean()), float(x.mean())] if len(x) else [None, None]
    h = 2.13145 * float(x.std(ddof=1)) / math.sqrt(len(x)); return [float(x.mean() - h), float(x.mean() + h)]


def load(path):
    rows = json.loads(Path(path).read_text())["results"]
    return {(int(row["seed"]), row["form"], row["task"], row["protocol"], row["channel"]): row for row in rows}


def summary(row):
    final = row["final"]
    return {"seed": int(row["seed"]), "form": row["form"], "task": row["task"], "protocol": row["protocol"], "channel": row["channel"], "natural": float(final["natural_mean_all_workers"]), "recombined_minus_natural": final["recombined_minus_natural_mean"], "composable": bool(final["composable"]), "worker_natural": float(np.mean([v["natural"]["team_return_mean"] for v in final["per_worker"].values()]))}


def analyze(path):
    by = load(path); rows = [summary(value) for value in by.values()]; groups = []
    for form in design.FORMS:
        for task in design.TASKS:
            for protocol in design.PROTOCOLS:
                for channel in design.CHANNELS:
                    values = [r for r in rows if r["form"] == form and r["task"] == task and r["protocol"] == protocol and r["channel"] == channel]
                    groups.append({"form": form, "task": task, "protocol": protocol, "channel": channel, "n": len(values), "natural_mean": float(np.mean([r["natural"] for r in values])), "natural_ci95_t": ci([r["natural"] for r in values]), "recombined_minus_natural_mean": float(np.mean([r["recombined_minus_natural"] for r in values if r["recombined_minus_natural"] is not None])) if any(r["recombined_minus_natural"] is not None for r in values) else None, "composable_count": int(sum(r["composable"] for r in values))})
    effects = []
    for form in design.FORMS:
        for task in design.TASKS:
            for protocol in design.PROTOCOLS:
                pairs = [next(r for r in rows if r["seed"] == seed and r["form"] == form and r["task"] == task and r["protocol"] == protocol and r["channel"] == "live")["natural"] - next(r for r in rows if r["seed"] == seed and r["form"] == form and r["task"] == task and r["protocol"] == protocol and r["channel"] == "silent")["natural"] for seed in design.SEEDS]
                effects.append({"form": form, "task": task, "protocol": protocol, "effect": "live-minus-silent", "values": pairs, "mean": float(np.mean(pairs)), "ci95_t": ci(pairs)})
    staged_vs_sim = []
    for form in design.FORMS:
        for task in design.TASKS:
            pairs = [next(r for r in rows if r["seed"] == seed and r["form"] == form and r["task"] == task and r["protocol"] == "staged" and r["channel"] == "live")["natural"] - next(r for r in rows if r["seed"] == seed and r["form"] == form and r["task"] == task and r["protocol"] == "simultaneous" and r["channel"] == "live")["natural"] for seed in design.SEEDS]
            staged_vs_sim.append({"form": form, "task": task, "values": pairs, "mean": float(np.mean(pairs)), "ci95_t": ci(pairs)})
    return {"schema": "ternary_composition_analysis_v1", "rule": {"natural_min": 0.60, "absolute_recombined_gap_max": 0.02}, "rows": rows, "groups": groups, "effects": effects, "staged_minus_simultaneous": staged_vs_sim}


def write_md(path, data):
    lines = ["# Ternary attribute and message-capacity extension", "", "The live/silent contrast is paired by seed. `mono9` has the same 9-symbol capacity as two ternary slots but no slot recombination readout; `tri3` exposes two 3-valued slots.", "", "| form | task | protocol | live natural | silent natural | live−silent | recombined−natural | composable |", "|---|---|---|---:|---:|---:|---:|---:|"]
    groups = {(r["form"], r["task"], r["protocol"], r["channel"]): r for r in data["groups"]}; effects = {(r["form"], r["task"], r["protocol"]): r for r in data["effects"]}
    for form in design.FORMS:
        for task in design.TASKS:
            for protocol in design.PROTOCOLS:
                live = groups[(form, task, protocol, "live")]; silent = groups[(form, task, protocol, "silent")]; effect = effects[(form, task, protocol)]
                gap = "n/a" if live["recombined_minus_natural_mean"] is None else f"{live['recombined_minus_natural_mean']:.3f}"
                lines.append(f"| `{form}` | `{task}` | `{protocol}` | {live['natural_mean']:.3f} | {silent['natural_mean']:.3f} | {effect['mean']:.3f} [{effect['ci95_t'][0]:.3f},{effect['ci95_t'][1]:.3f}] | {gap} | {live['composable_count']}/{live['n']} |")
    lines += ["", "| form | task | staged−simultaneous natural |", "|---|---|---:|"]
    for row in data["staged_minus_simultaneous"]:
        lines.append(f"| `{row['form']}` | `{row['task']}` | {row['mean']:.3f} [{row['ci95_t'][0]:.3f},{row['ci95_t'][1]:.3f}] |")
    lines += ["", "The ternary extension tests scaling to three-valued attributes and distinguishes a capacity-matched atomic token from an exposed slot structure."]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--results", required=True); parser.add_argument("--out", required=True); parser.add_argument("--markdown", required=True); args = parser.parse_args(); data = analyze(args.results); Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n"); write_md(args.markdown, data); print(json.dumps({"status": "written", "rows": len(data["rows"]), "groups": len(data["groups"])}, ensure_ascii=False))
