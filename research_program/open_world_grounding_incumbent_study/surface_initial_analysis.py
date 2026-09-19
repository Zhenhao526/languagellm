"""Supplemental paired analysis of receiver initialization by surface mapping.

This analysis reads only the frozen formal transfer result table. It does not
change training, evaluation, endpoints, or the archived frozen analysis code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

T95_DF8 = 2.306004135204166
SUPPORT = "expanded_single_receiver"


def summarize(values):
    x = np.asarray(values, dtype=np.float64)
    mean = float(x.mean())
    if len(x) < 2:
        ci = [mean, mean]
    else:
        se = float(x.std(ddof=1) / np.sqrt(len(x)))
        ci = [float(mean - T95_DF8 * se), float(mean + T95_DF8 * se)]
    return {"n": int(len(x)), "mean": mean, "ci95_t_df8": ci,
            "values_by_seed": x.tolist()}


def analyze(payload):
    rows = payload["transfers"]
    lookup = {}
    for row in rows:
        key = (row["architecture"], row["surface_mapping"],
               row["receiver_initialization"], row["support"], int(row["seed"]))
        if key in lookup:
            raise ValueError(f"duplicate transfer cell {key}")
        lookup[key] = row
    seeds = sorted({int(row["seed"]) for row in rows})
    if len(seeds) != 9:
        raise ValueError(f"expected 9 seeds, got {len(seeds)}")

    def endpoint(architecture, mapping, initialization, seed, phase):
        row = lookup[(architecture, mapping, initialization, SUPPORT, seed)]
        cell = row[phase]["new_double"]
        return {"natural": float(cell["natural"]["team_return_mean"]),
                "recombined": float(cell["recombined"]["team_return_mean"])}

    cells = {}
    contrasts = {}
    for architecture in ("factorized", "holistic"):
        for mapping in ("identity", "reverse"):
            for initialization in ("fresh", "incumbent_receiver"):
                values = [endpoint(architecture, mapping, initialization, seed, phase)
                          for seed in seeds for phase in ("initial", "final")]
                initial = values[::2]
                final = values[1::2]
                gains = [f["natural"] - i["natural"] for i, f in zip(initial, final)]
                cells[f"{architecture}/{mapping}/{initialization}"] = {
                    "support": SUPPORT,
                    "initial_natural": summarize([x["natural"] for x in initial]),
                    "final_natural": summarize([x["natural"] for x in final]),
                    "repair_gain": summarize(gains),
                    "final_recombined": summarize([x["recombined"] for x in final]),
                    "functional_final_count_at_0_60": int(sum(x["natural"] >= 0.60 for x in final)),
                }

        for phase in ("initial", "final"):
            incumbent_swap = [endpoint(architecture, "reverse", "incumbent_receiver", s, phase)["natural"]
                              - endpoint(architecture, "identity", "incumbent_receiver", s, phase)["natural"]
                              for s in seeds]
            fresh_swap = [endpoint(architecture, "reverse", "fresh", s, phase)["natural"]
                          - endpoint(architecture, "identity", "fresh", s, phase)["natural"]
                          for s in seeds]
            interaction = [a - b for a, b in zip(incumbent_swap, fresh_swap)]
            contrasts[f"{architecture}/incumbent_receiver:reverse-minus-identity/{phase}:new_double"] = summarize(incumbent_swap)
            contrasts[f"{architecture}/fresh_receiver:reverse-minus-identity/{phase}:new_double"] = summarize(fresh_swap)
            contrasts[f"{architecture}/surface-by-initialization-interaction/{phase}:new_double"] = summarize(interaction)

        contrasts[f"{architecture}/incumbent_receiver/reverse:repair_gain"] = cells[
            f"{architecture}/reverse/incumbent_receiver"]["repair_gain"]
    return {
        "schema": "open_world_grounding_incumbent_surface_initial_analysis_v1",
        "source_results": "transfer_results.json",
        "support": SUPPORT,
        "endpoint": "new_double natural and donor-recombined mean team return",
        "seeds": seeds,
        "interval_method": "paired Student t interval, df=8, critical value 2.3060041352",
        "functional_threshold": 0.60,
        "transfers": len(rows),
        "cells": cells,
        "contrasts": contrasts,
    }


def markdown(summary):
    lines = [
        "# Receiver initialization × surface mapping: supplemental analysis",
        "",
        "This paired analysis reads the frozen 288-row transfer result table. It uses a two-sided Student t interval with 8 degrees of freedom across nine seed-level contrasts.",
        "",
        "## Receiver-only replacement: held-out double-new endpoint",
        "",
        "| Architecture | Mapping | Initialization | Initial | Final | Repair gain | Functional final | Recombined final |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for architecture in ("factorized", "holistic"):
        for mapping in ("identity", "reverse"):
            for initialization in ("fresh", "incumbent_receiver"):
                cell = summary["cells"][f"{architecture}/{mapping}/{initialization}"]
                def m(name):
                    return cell[name]["mean"]
                lines.append(
                    f"| {architecture} | {mapping} | {initialization} | {m('initial_natural'):.3f} | "
                    f"{m('final_natural'):.3f} | {m('repair_gain'):.3f} | "
                    f"{cell['functional_final_count_at_0_60']}/9 | {m('final_recombined'):.3f} |"
                )
    lines += ["", "## Paired contrasts", "",
              "`surface-by-initialization-interaction` is (reverse−identity among incumbent receivers) minus (reverse−identity among fresh receivers), paired by seed.",
              "",
              "| Contrast | Mean | Paired 95% t interval |",
              "|---|---:|---:|"]
    for name, contrast in summary["contrasts"].items():
        if "/surface-by-initialization-interaction/" not in name and "/incumbent_receiver:reverse-minus-identity/" not in name:
            continue
        lo, hi = contrast["ci95_t_df8"]
        lines.append(f"| `{name}` | {contrast['mean']:.3f} | [{lo:.3f}, {hi:.3f}] |")
    lines += ["", "## Interpretation boundary", "",
              "The mapping intervention permutes tabular attribute labels; it does not use images or a learned visual encoder. The factorized result concerns whether a copied receiver can re-ground a compositional message protocol after a stable label change. It is not evidence that agents invented natural language.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()
    summary = analyze(json.loads(Path(args.results).read_text()))
    Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    Path(args.md_out).write_text(markdown(summary))
    print(json.dumps({"status": "completed", "cells": len(summary["cells"])}, ensure_ascii=False))
