"""Compact analysis for the two-generation transmission chain."""
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


def load(path):
    rows = json.loads(Path(path).read_text())["results"]
    return {(int(row["seed"]), int(row["generation"]), row["lineage"], row["g1_channel"], row["channel"]): row for row in rows}


def summary(row):
    modes = row["final"]["modes"]
    natural = float(modes["natural"]["team_return_mean"])
    recombined = float(modes["recombined"]["team_return_mean"])
    return {"seed": int(row["seed"]), "generation": int(row["generation"]), "lineage": row["lineage"], "g1_channel": row["g1_channel"], "channel": row["channel"], "natural": natural, "silent": float(modes["closed"]["team_return_mean"]), "permuted": float(modes["permuted"]["team_return_mean"]), "recombined_minus_natural": recombined - natural, "semantic_success": float(row["final"]["codebook"]["semantic_success_mean"]), "sender_sequences": row["final"]["codebook"]["sender_sequences"], "composable": bool(natural >= 0.60 and abs(recombined - natural) <= 0.02)}


def analyze(path):
    by = load(path)
    rows = [summary(row) for row in by.values()]
    groups = []
    for lineage in design.LINEAGES:
        for generation in design.GENERATIONS:
            for channel in design.CHANNELS:
                values = [r for r in rows if r["lineage"] == lineage and r["generation"] == generation and r["channel"] == channel]
                groups.append({"lineage": lineage, "generation": generation, "channel": channel, "n": len(values), "natural_mean": float(np.mean([r["natural"] for r in values])), "natural_ci95_t": ci([r["natural"] for r in values]), "recombined_minus_natural_mean": float(np.mean([r["recombined_minus_natural"] for r in values])), "recombined_minus_natural_ci95_t": ci([r["recombined_minus_natural"] for r in values]), "composable_count": int(sum(r["composable"] for r in values)), "composable_rate": float(np.mean([r["composable"] for r in values])), "semantic_success_mean": float(np.mean([r["semantic_success"] for r in values]))})
    effects = []
    for lineage in design.LINEAGES:
        for generation in design.GENERATIONS:
            for control in ("silent", "permuted"):
                pairs = []
                for seed in sorted({r["seed"] for r in rows}):
                    for g1 in design.CHANNELS:
                        candidates = [r for r in rows if r["seed"] == seed and r["lineage"] == lineage and r["generation"] == generation and r["g1_channel"] == g1 and r["channel"] == "live"]
                        if not candidates:
                            continue
                        live = candidates[0]
                        compare = live["silent"] if control == "silent" else live["permuted"]
                        pairs.append(live["natural"] - compare)
                effects.append({"lineage": lineage, "generation": generation, "effect": "live-minus-" + control, "values": pairs, "mean": float(np.mean(pairs)), "ci95_t": ci(pairs)})
    transitions = []
    for lineage in design.LINEAGES:
        for g1 in design.CHANNELS:
            for g2 in design.CHANNELS:
                for metric in ("natural", "recombined_minus_natural"):
                    a = [r[metric] for r in rows if r["lineage"] == lineage and r["generation"] == 1 and r["g1_channel"] == g1 and r["channel"] == g1]
                    b = [r[metric] for r in rows if r["lineage"] == lineage and r["generation"] == 2 and r["g1_channel"] == g1 and r["channel"] == g2]
                    if not a or not b:
                        continue
                    transitions.append({"lineage": lineage, "g1_channel": g1, "g2_channel": g2, "metric": metric, "generation2_minus_generation1_mean": float(np.mean(b) - np.mean(a)), "generation1_mean": float(np.mean(a)), "generation2_mean": float(np.mean(b))})
    return {"schema": "multi_generation_transmission_analysis_v1", "parent_composable_seeds": list(design.SEEDS), "rule": {"natural_min": 0.60, "absolute_recombined_gap_max": 0.02}, "rows": rows, "groups": groups, "effects": effects, "transitions": transitions}


def write_md(path, data):
    lines = ["# Two-generation compositional transmission", "", "The sample is the nine pre-registered composable parent seeds. Each row is evaluated with natural, closed/silent, partner-permuted and slot-recombined messages.", "", "| lineage | generation | channel | n | natural | recombined−natural | composable |", "|---|---:|---|---:|---:|---:|---:|"]
    for row in data["groups"]:
        lines.append(f"| `{row['lineage']}` | {row['generation']} | `{row['channel']}` | {row['n']} | {row['natural_mean']:.3f} | {row['recombined_minus_natural_mean']:.3f} | {row['composable_count']}/{row['n']} |")
    lines += ["", "| lineage | generation | live−silent | live−permuted |", "|---|---:|---:|---:|"]
    for lineage in design.LINEAGES:
        for generation in design.GENERATIONS:
            effect = {(x["lineage"], x["generation"], x["effect"]): x for x in data["effects"]}
            a = effect.get((lineage, generation, "live-minus-silent"))
            b = effect.get((lineage, generation, "live-minus-permuted"))
            if a is None or b is None:
                lines.append(f"| `{lineage}` | {generation} | n/a | n/a |")
            else:
                lines.append(f"| `{lineage}` | {generation} | {a['mean']:.3f} [{a['ci95_t'][0]:.3f},{a['ci95_t'][1]:.3f}] | {b['mean']:.3f} [{b['ci95_t'][0]:.3f},{b['ci95_t'][1]:.3f}] |")
    lines += ["", "The generation-2 rows condition on the corresponding generation-1 endpoint. They therefore test cumulative transmission rather than fresh emergence."]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--results", required=True); parser.add_argument("--out", required=True); parser.add_argument("--markdown", required=True); args = parser.parse_args()
    data = analyze(args.results)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    write_md(args.markdown, data)
    print(json.dumps({"status": "written", "rows": len(data["rows"]), "groups": len(data["groups"])}, ensure_ascii=False))
