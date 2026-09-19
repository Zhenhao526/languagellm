"""Compact seed summaries for the incumbent-replacement stage."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import design


def _ci(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) < 2:
        return [float(x.mean()), float(x.mean())]
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    return [float(x.mean() - 1.96 * se), float(x.mean() + 1.96 * se)]


def _functional(values, threshold=0.60):
    return int(np.sum(np.asarray(values, dtype=np.float64) >= threshold))


def summarize(payload):
    rows = payload["transfers"]
    cells = {}
    for row in rows:
        key = (row["architecture"], row.get("surface_mapping", "identity"),
               row.get("receiver_initialization", "fresh"), row["support"])
        final = row["final"]
        cell = cells.setdefault(key, {"seed": [], "new_single": [], "new_double": [],
                                      "new_double_recombined": [], "old_seen": [], "old_combo": [],
                                      "initial_new_double": [], "incumbent": []})
        cell["seed"].append(int(row["seed"]))
        for kind, name in (("new_single", "new_single"), ("new_double", "new_double"),
                           ("old_seen", "old_seen"), ("old_combo", "old_combo")):
            cell[name].append(float(final[kind]["natural"]["team_return_mean"]))
        cell["new_double_recombined"].append(float(final["new_double"]["recombined"]["team_return_mean"]))
        cell["initial_new_double"].append(float(row.get("initial", final)["new_double"]["natural"]["team_return_mean"]))
        cell["incumbent"].append(str(row["incumbent_parameter_sha256"]))
    out = {}
    for (architecture, mapping, initialization, support), cell in sorted(cells.items()):
        endpoint = {}
        for name in ("old_seen", "old_combo", "new_single", "new_double", "new_double_recombined", "initial_new_double"):
            values = cell[name]
            endpoint[name] = {"mean": float(np.mean(values)), "ci95": _ci(values),
                              "functional": _functional(values), "seeds": len(values),
                              "values": values}
        out[f"{architecture}/{mapping}/{initialization}/{support}"] = {
            "architecture": architecture, "surface_mapping": mapping,
            "receiver_initialization": initialization, "support": support, "endpoint": endpoint,
            "incumbent_hash_count": len(set(cell["incumbent"])),
        }
        final_values = np.asarray(cell["new_double"], dtype=np.float64)
        initial_values = np.asarray(cell["initial_new_double"], dtype=np.float64)
        out[f"{architecture}/{mapping}/{initialization}/{support}"]["repair_gain"] = {
            "mean": float(np.mean(final_values - initial_values)),
            "ci95": _ci(final_values - initial_values),
            "values": (final_values - initial_values).tolist(),
        }
    contrasts = {}
    for architecture in design.ARCHITECTURES:
      for initialization in design.RECEIVER_INITIALIZATIONS:
       for mapping in design.OBJECT_SURFACE_MAPPINGS:
        def vals(support, name):
            return np.asarray(out[f"{architecture}/{mapping}/{initialization}/{support}"]["endpoint"][name]["values"], dtype=np.float64)
        pair = vals("expanded_single_pair", "new_double")
        for support in ("expanded_single_sender", "expanded_single_receiver", "expanded_single_none"):
            diff = pair - vals(support, "new_double")
            contrasts[f"{architecture}/{mapping}/{initialization}:pair-minus-{support}:new_double"] = {
                "mean": float(diff.mean()), "ci95": _ci(diff), "values": diff.tolist()
            }
       for support in design.TRANSFER_SUPPORTS:
        identity = np.asarray(out[f"{architecture}/identity/{initialization}/{support}"]["endpoint"]["new_double"]["values"], dtype=np.float64)
        reverse = np.asarray(out[f"{architecture}/reverse/{initialization}/{support}"]["endpoint"]["new_double"]["values"], dtype=np.float64)
        diff = reverse - identity
        contrasts[f"{architecture}/{initialization}:reverse-minus-identity/{support}:new_double"] = {
            "mean": float(diff.mean()), "ci95": _ci(diff), "values": diff.tolist()
        }
       for mapping in design.OBJECT_SURFACE_MAPPINGS:
        fresh = np.asarray(out[f"{architecture}/{mapping}/fresh/expanded_single_receiver"]["endpoint"]["new_double"]["values"], dtype=np.float64)
        incumbent = np.asarray(out[f"{architecture}/{mapping}/incumbent_receiver/expanded_single_receiver"]["endpoint"]["new_double"]["values"], dtype=np.float64)
        diff = incumbent - fresh
        contrasts[f"{architecture}/{mapping}:incumbent-init-minus-fresh-init/receiver:new_double"] = {
            "mean": float(diff.mean()), "ci95": _ci(diff), "values": diff.tolist()
        }
    return {"schema": "open_world_grounding_incumbent_analysis_v1",
            "transfers": len(rows), "cells": out, "contrasts": contrasts}


def markdown(summary):
    lines = ["# Incumbent-replacement transfer summary", "", f"Transfers: {summary['transfers']}.", "",
             "## Child cells", ""]
    for key, cell in summary["cells"].items():
        e = cell["endpoint"]
        lines.append(f"- {key}: old-seen {e['old_seen']['mean']:.3f}, old-combo {e['old_combo']['mean']:.3f}, "
                     f"new-single {e['new_single']['mean']:.3f}, new-double {e['new_double']['mean']:.3f} "
                     f"({e['new_double']['functional']}/{e['new_double']['seeds']}), "
                     f"recombined {e['new_double_recombined']['mean']:.3f}, "
                     f"initial {e['initial_new_double']['mean']:.3f}, "
                     f"gain {cell['repair_gain']['mean']:.3f}")
    lines += ["", "## Paired contrasts", ""]
    for key, c in summary["contrasts"].items():
        lines.append(f"- {key}: {c['mean']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}]")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--md-out", required=True)
    args = parser.parse_args()
    summary = summarize(json.loads(Path(args.results).read_text()))
    Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    Path(args.md_out).write_text(markdown(summary))
    print(json.dumps({"status": "completed", "cells": len(summary["cells"])}, ensure_ascii=False))
