"""Aggregate representation and held-out recovery effects."""
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
    critical = 2.04 if len(values) >= 30 else 2.365 if len(values) == 8 else 1.96
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


def value(result, kind, mode="natural"):
    return float(result["final"][kind][mode]["team_return_mean"])


def aggregate(results, parent_composable):
    seeds = sorted({seed for seed, _ in results})
    rows = []
    for (seed, condition), result in sorted(results.items()):
        representation = next(value for value in ("joint_history", "slot_local") if condition.startswith(value + "_"))
        support, channel = condition[len(representation) + 1 :].rsplit("_", 1)
        for kind in ("all", "seen", "heldout"):
            for mode in ("natural", "permuted"):
                rows.append({"seed": seed, "condition": condition, "representation": representation, "support": support, "channel": channel, "goal_kind": kind, "mode": mode, "return": value(result, kind, mode), "parent_composable": bool(parent_composable[str(seed)])})
    effects = {}
    for representation in ("joint_history", "slot_local"):
        for channel in ("live", "silent"):
            for kind in ("all", "seen", "heldout"):
                keys = [(seed, f"{representation}_leave_one_out_{channel}") for seed in seeds if (seed, f"{representation}_leave_one_out_{channel}") in results and (seed, f"{representation}_full_{channel}") in results]
                vals = [value(results[(seed, loo_key)], kind) - value(results[(seed, f"{representation}_full_{channel}")], kind) for seed, loo_key in keys]
                if vals:
                    effects[f"leave_minus_full|{representation}|{channel}|{kind}"] = {"values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
        for support in ("full", "leave_one_out"):
            for kind in ("all", "seen", "heldout"):
                keys = [(seed, f"{representation}_{support}_live") for seed in seeds if (seed, f"{representation}_{support}_live") in results and (seed, f"{representation}_{support}_silent") in results]
                vals = [value(results[(seed, live_key)], kind) - value(results[(seed, f"{representation}_{support}_silent")], kind) for seed, live_key in keys]
                if vals:
                    effects[f"live_minus_silent|{representation}|{support}|{kind}"] = {"values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
    representation_contrasts = {}
    for support in ("full", "leave_one_out"):
        for channel in ("live", "silent"):
            for kind in ("all", "seen", "heldout"):
                keys = [seed for seed in seeds if (seed, f"slot_local_{support}_{channel}") in results and (seed, f"joint_history_{support}_{channel}") in results]
                vals = [value(results[(seed, f"slot_local_{support}_{channel}")], kind) - value(results[(seed, f"joint_history_{support}_{channel}")], kind) for seed in keys]
                if vals:
                    representation_contrasts[f"slot_local_minus_joint|{support}|{channel}|{kind}"] = {"values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
    stratified = {}
    for status in (True, False):
        label = "parent_composable" if status else "parent_noncomposable"
        group_seeds = [seed for seed in seeds if bool(parent_composable[str(seed)]) is status]
        stratified[label] = {}
        for representation in ("joint_history", "slot_local"):
            for support in ("full", "leave_one_out"):
                vals = [value(results[(seed, f"{representation}_{support}_live")], "heldout") for seed in group_seeds]
                stratified[label][f"{representation}_{support}_live"] = {"n": len(vals), "mean": float(np.mean(vals)) if vals else None, "passes": int(sum(x >= 0.60 for x in vals)), "ci95_t": ci(vals)}
    return {"schema": "receiver_representation_aggregate_v1", "runs": len(results), "rows": rows, "effects": effects, "representation_contrasts": representation_contrasts, "stratified": stratified}


def write_md(path, data):
    lines = ["# Receiver representation and held-out composition", "", f"- runs: {data['runs']}", "- zero-shot rule: leave-one-out live heldout return ≥ 0.60", "", "| representation | support | channel | all | seen | heldout |", "|---|---|---|---:|---:|---:|"]
    for representation in ("joint_history", "slot_local"):
        for support in ("full", "leave_one_out"):
            for channel in ("live", "silent"):
                vals = {kind: np.mean([row["return"] for row in data["rows"] if row["representation"] == representation and row["support"] == support and row["channel"] == channel and row["goal_kind"] == kind and row["mode"] == "natural"]) for kind in ("all", "seen", "heldout")}
                lines.append(f"| `{representation}` | `{support}` | `{channel}` | {vals['all']:.3f} | {vals['seen']:.3f} | {vals['heldout']:.3f} |")
    lines += ["", "| contrast | mean | 95% CI |", "|---|---:|---:|"]
    for key in ("slot_local_minus_joint|leave_one_out|live|heldout", "slot_local_minus_joint|full|live|heldout", "leave_minus_full|joint_history|live|heldout", "leave_minus_full|slot_local|live|heldout"):
        item = data["representation_contrasts"].get(key) or data["effects"][key]
        lines.append(f"| `{key}` | {item['mean']:.3f} | [{item['ci95_t'][0]:.3f}, {item['ci95_t'][1]:.3f}] |")
    lines += ["", "| parent stratum | representation | full live | leave-one-out live | passes |", "|---|---|---:|---:|---:|"]
    for label in ("parent_composable", "parent_noncomposable"):
        for representation in ("joint_history", "slot_local"):
            full = data["stratified"][label][f"{representation}_full_live"]
            loo = data["stratified"][label][f"{representation}_leave_one_out_live"]
            full_mean = "NA" if full["mean"] is None else f"{full['mean']:.3f}"
            loo_mean = "NA" if loo["mean"] is None else f"{loo['mean']:.3f}"
            lines.append(f"| `{label}` | `{representation}` | {full_mean} | {loo_mean} | {loo['passes']}/{loo['n']} |")
    lines += ["", "`slot_local` changes only the receiver's state lookup: at each staged subtask it ignores the other slot. The comparison is therefore a test of representation-induced compositional generalization, not a change to the parent sender or the task payoff."]
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--results", required=True); parser.add_argument("--prepared", required=True); parser.add_argument("--out", required=True); parser.add_argument("--markdown", required=True)
    args = parser.parse_args(); cfg = json.loads((Path(args.prepared) / "prepared.json").read_text()); data = aggregate(load(args.results), cfg["parent_composable"]); Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n"); write_md(args.markdown, data); print(json.dumps({"status": "written", "runs": data["runs"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
