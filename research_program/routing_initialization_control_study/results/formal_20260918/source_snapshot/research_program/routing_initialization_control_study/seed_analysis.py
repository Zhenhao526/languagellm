"""Seed-level contrasts and route-mismatch summaries for the initialization control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _ci(values):
    x = np.asarray(list(values), dtype=float)
    mean = float(x.mean()) if len(x) else float("nan")
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * sd / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return {"n": int(len(x)), "mean": mean, "sd": sd,
            "ci95": [mean - half, mean + half]}


def _metric(row, kind="heldout_combo", mode="natural"):
    return float(row["final"][kind][mode]["team_return_mean"])


def _route(row, side):
    value = row["final"].get(f"{side}_route")
    return None if value is None else np.asarray(value["matrix"], dtype=float)


def _row_summary(row):
    sender = _route(row, "sender")
    receiver = _route(row, "receiver")
    mismatch = None if sender is None else float(np.abs(sender - receiver).mean())
    return {
        "seed": int(row["seed"]), "architecture": row["architecture"],
        "population": row["population"], "visibility": row["visibility"],
        "support": row["support"],
        "all_natural": _metric(row, "all"),
        "heldout_combo_natural": _metric(row, "heldout_combo"),
        "heldout_value_natural": _metric(row, "heldout_value"),
        "heldout_combo_message_gap": _metric(row, "heldout_combo", "natural") - _metric(row, "heldout_combo", "permuted"),
        "heldout_combo_live_silent": _metric(row, "heldout_combo", "natural") - _metric(row, "heldout_combo", "silent"),
        "sender_route_best_alignment": None if sender is None else float(row["final"]["sender_route"]["best_alignment"]),
        "receiver_route_best_alignment": None if receiver is None else float(row["final"]["receiver_route"]["best_alignment"]),
        "route_mismatch_mean_abs": mismatch,
        "sender_route_entropy": None if sender is None else float(row["final"]["sender_route"]["mean_entropy"]),
        "receiver_route_entropy": None if receiver is None else float(row["final"]["receiver_route"]["mean_entropy"]),
    }


def analyse(payload):
    children = payload["children"]
    rows = [_row_summary(row) for row in children]
    keyed = {(r["seed"], r["architecture"], r["population"], r["visibility"], r["support"]): r for r in rows}
    seeds = sorted({r["seed"] for r in rows})
    conditions = sorted({(r["population"], r["visibility"], r["support"]) for r in rows})
    contrasts = []
    for population, visibility, support in conditions:
        for left, right in (("tied_routed", "routed"),
                            ("tied_routed", "sync_routed"),
                            ("sync_routed", "routed"),
                            ("tied_routed", "holistic"),
                            ("factorized", "tied_routed")):
            paired = []
            for seed in seeds:
                a = keyed.get((seed, left, population, visibility, support))
                b = keyed.get((seed, right, population, visibility, support))
                if a is not None and b is not None:
                    paired.append({
                        "seed": seed,
                        "all": a["all_natural"] - b["all_natural"],
                        "heldout_combo": a["heldout_combo_natural"] - b["heldout_combo_natural"],
                        "heldout_value": a["heldout_value_natural"] - b["heldout_value_natural"],
                    })
            if paired:
                contrasts.append({
                    "left": left, "right": right, "population": population,
                    "visibility": visibility, "support": support,
                    "paired_seed_differences": paired,
                    "all": _ci(x["all"] for x in paired),
                    "heldout_combo": _ci(x["heldout_combo"] for x in paired),
                    "heldout_value": _ci(x["heldout_value"] for x in paired),
                })
    summaries = []
    for architecture in sorted({r["architecture"] for r in rows}):
        for population, visibility, support in conditions:
            subset = [r for r in rows if r["architecture"] == architecture and r["population"] == population
                      and r["visibility"] == visibility and r["support"] == support]
            if not subset:
                continue
            rec = {"architecture": architecture, "population": population,
                   "visibility": visibility, "support": support, "n": len(subset)}
            for key in ("all_natural", "heldout_combo_natural", "heldout_value_natural",
                        "heldout_combo_message_gap", "heldout_combo_live_silent",
                        "sender_route_best_alignment", "receiver_route_best_alignment",
                        "route_mismatch_mean_abs", "sender_route_entropy", "receiver_route_entropy"):
                values = [r[key] for r in subset if r[key] is not None]
                rec[key] = _ci(values) if values else None
            rec["heldout_combo_functional_count"] = sum(r["heldout_combo_natural"] >= 0.60 for r in subset)
            rec["heldout_value_functional_count"] = sum(r["heldout_value_natural"] >= 0.60 for r in subset)
            summaries.append(rec)
    return {"schema": "routing_initialization_control_seed_analysis_v1", "seed_rows": rows,
            "condition_summary": summaries, "contrasts": contrasts,
            "run_counts": {"children": len(children), "seeds": len(seeds)}}


def markdown(data):
    lines = ["# Routing initialization control seed analysis", "", f"Children: {data['run_counts']['children']}; seeds: {data['run_counts']['seeds']}.", "", "## Key aligned/hidden held-out-combination cells", ""]
    for row in data["condition_summary"]:
        if row["population"] == "aligned" and row["visibility"] == "hidden" and row["support"] == "heldout_combo":
            mismatch = "n/a" if row["route_mismatch_mean_abs"] is None else f"{row['route_mismatch_mean_abs']['mean']:.3f}"
            lines.append(f"- {row['architecture']}: combo {row['heldout_combo_natural']['mean']:.3f} [{row['heldout_combo_natural']['ci95'][0]:.3f}, {row['heldout_combo_natural']['ci95'][1]:.3f}], functional {row['heldout_combo_functional_count']}/{row['n']}, route mismatch {mismatch}")
    lines += ["", "## Paired contrasts", ""]
    for row in data["contrasts"]:
        if row["population"] == "aligned" and row["visibility"] == "hidden" and row["support"] == "heldout_combo":
            c = row["heldout_combo"]
            lines.append(f"- {row['left']} − {row['right']}: {c['mean']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    data = analyse(json.loads(Path(args.results).read_text()))
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    Path(args.markdown).write_text(markdown(data))


if __name__ == "__main__":
    main()
