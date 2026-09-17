"""Compact summaries for leave-one-goal-out recovery."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return [None, None]
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    critical = 2.04 if len(values) >= 30 else 2.13 if len(values) >= 15 else 2.365 if len(values) == 8 else 1.96
    half = critical * float(values.std(ddof=1)) / np.sqrt(len(values))
    return [float(values.mean() - half), float(values.mean() + half)]


def load(path):
    payload = json.loads(Path(path).read_text())
    out = {}
    for result in payload["results"]:
        key = (int(result["seed"]), result["condition"])
        if key in out:
            raise ValueError(f"duplicate {key}")
        out[key] = result
    return out


def value(result, goal_kind, mode="natural"):
    return float(result["final"][goal_kind][mode]["team_return_mean"])


def aggregate(results, parent_composable):
    rows = []
    for (seed, condition), result in sorted(results.items()):
        support, channel = condition.rsplit("_", 1)
        for goal_kind in ("all", "seen", "heldout"):
            for mode in ("natural", "permuted"):
                rows.append({"seed": seed, "condition": condition, "support": support, "channel": channel, "goal_kind": goal_kind, "mode": mode, "return": value(result, goal_kind, mode), "parent_composable": bool(parent_composable[str(seed)])})
    effects = {}
    seeds = sorted({seed for seed, _ in results})
    for channel in ("live", "silent"):
        for goal_kind in ("all", "seen", "heldout"):
            vals = [value(results[(seed, f"leave_one_out_{channel}")], goal_kind) - value(results[(seed, f"full_{channel}")], goal_kind) for seed in seeds]
            effects[f"leave_minus_full|{channel}|{goal_kind}"] = {"values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
    for support in ("full", "leave_one_out"):
        for goal_kind in ("all", "seen", "heldout"):
            vals = [value(results[(seed, f"{support}_live")], goal_kind) - value(results[(seed, f"{support}_silent")], goal_kind) for seed in seeds]
            effects[f"live_minus_silent|{support}|{goal_kind}"] = {"values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
    zero_shot = []
    for seed in seeds:
        result = results[(seed, "leave_one_out_live")]
        heldout = value(result, "heldout")
        zero_shot.append({"seed": seed, "parent_composable": bool(parent_composable[str(seed)]), "heldout": heldout, "passes": bool(heldout >= 0.60)})
    stratified = {}
    for status in (True, False):
        subset = [row for row in zero_shot if row["parent_composable"] is status]
        vals = [row["heldout"] for row in subset]
        stratified["parent_composable" if status else "parent_noncomposable"] = {"n": len(vals), "passes": int(sum(row["passes"] for row in subset)), "mean": float(np.mean(vals)) if vals else None, "ci95_t": ci(vals)}
    stratified_support = {}
    for status in (True, False):
        label = "parent_composable" if status else "parent_noncomposable"
        stratified_support[label] = {}
        group_seeds = [seed for seed in seeds if bool(parent_composable[str(seed)]) is status]
        for support in ("full", "leave_one_out"):
            for channel in ("live", "silent"):
                vals = [value(results[(seed, f"{support}_{channel}")], "heldout") for seed in group_seeds]
                stratified_support[label][f"{support}_{channel}"] = {"n": len(vals), "mean": float(np.mean(vals)) if vals else None, "passes": int(sum(x >= 0.60 for x in vals)), "ci95_t": ci(vals)}
    return {"schema": "heldout_composition_aggregate_v1", "runs": len(results), "rows": rows, "effects": effects, "zero_shot": zero_shot, "stratified": stratified, "stratified_support": stratified_support}


def write_markdown(path, data):
    lines = ["# Held-out goal-combination recovery", "", f"- runs: {data['runs']}", "- zero-shot rule: leave-one-out live heldout return ≥ 0.60", "", "| support | channel | all | seen | heldout |", "|---|---|---:|---:|---:|"]
    for support in ("full", "leave_one_out"):
        for channel in ("live", "silent"):
            values = {kind: np.mean([row["return"] for row in data["rows"] if row["support"] == support and row["channel"] == channel and row["goal_kind"] == kind and row["mode"] == "natural"]) for kind in ("all", "seen", "heldout")}
            lines.append(f"| `{support}` | `{channel}` | {values['all']:.3f} | {values['seen']:.3f} | {values['heldout']:.3f} |")
    lines += ["", "| comparison | mean difference | 95% CI |", "|---|---:|---:|"]
    for key in ("leave_minus_full|live|heldout", "live_minus_silent|leave_one_out|heldout", "live_minus_silent|full|heldout"):
        item = data["effects"][key]
        lines.append(f"| `{key}` | {item['mean']:.3f} | [{item['ci95_t'][0]:.3f}, {item['ci95_t'][1]:.3f}] |")
    lines += ["", "| parent stratum | n | heldout mean | passes |", "|---|---:|---:|---:|"]
    for key in ("parent_composable", "parent_noncomposable"):
        item = data["stratified"][key]
        mean = "NA" if item["mean"] is None else f"{item['mean']:.3f}"
        lines.append(f"| `{key}` | {item['n']} | {mean} | {item['passes']}/{item['n']} |")
    lines += ["", "The leave-one-out arm removes one target combination from child training while keeping the frozen parent sender able to emit its corresponding two-slot message. Passing the held-out criterion is evidence of behavioral zero-shot recovery, not evidence of open-ended language."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    cfg = json.loads((Path(args.prepared) / "prepared.json").read_text())
    data = aggregate(load(args.results), cfg["parent_composable"])
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    write_markdown(args.markdown, data)
    print(json.dumps({"status": "written", "runs": data["runs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
