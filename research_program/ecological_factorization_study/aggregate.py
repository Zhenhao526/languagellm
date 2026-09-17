"""Compact analysis for the ecological factorization intervention."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import design


def ci(values):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return [None, None]
    if len(x) == 1:
        return [float(x[0]), float(x[0])]
    critical = {8: 2.365, 9: 2.306, 12: 2.201, 18: 2.11, 27: 2.056}.get(len(x), 2.0)
    half = critical * float(x.std(ddof=1)) / math.sqrt(len(x))
    return [float(x.mean() - half), float(x.mean() + half)]


def load(path):
    payload = json.loads(Path(path).read_text())
    parents = {(int(row["seed"]), row["form"], row["task"]): row for row in payload["parents"]}
    children = {(int(row["seed"]), row["condition"]): row for row in payload["children"]}
    return parents, children


def value(row, kind="heldout", mode="natural"):
    return float(row["final"][kind][mode]["team_return_mean"])


def _effect(values):
    return {"values": [float(v) for v in values], "mean": float(np.mean(values)) if values else None, "ci95_t": ci(values)}


def aggregate(path):
    parents, children = load(path)
    seeds = sorted({seed for seed, _ in children})
    rows = []
    for (seed, condition), row in sorted(children.items()):
        representation, form, task, support, channel = design.parse_child_condition(condition)
        for kind in ("all", "seen", "heldout"):
            for mode in ("natural", "permuted"):
                rows.append({"seed": seed, "condition": condition, "representation": representation, "form": form, "task": task, "support": support, "channel": channel, "goal_kind": kind, "mode": mode, "return": value(row, kind, mode)})
            if channel == "live" and form == "tri3":
                for mode in ("raw_recombined", "task_recombined"):
                    rows.append({"seed": seed, "condition": condition, "representation": representation, "form": form, "task": task, "support": support, "channel": channel, "goal_kind": kind, "mode": mode, "return": value(row, kind, mode)})
    groups = []
    for representation in design.REPRESENTATIONS:
        for form in design.FORMS:
            for task in design.TASKS:
                if representation == "slot_local" and form == "mono9":
                    continue
                for support in design.SUPPORTS:
                    for channel in design.CHANNELS:
                        for kind in ("all", "seen", "heldout"):
                            vals = [r["return"] for r in rows if r["representation"] == representation and r["form"] == form and r["task"] == task and r["support"] == support and r["channel"] == channel and r["goal_kind"] == kind and r["mode"] == "natural"]
                            if vals:
                                groups.append({"representation": representation, "form": form, "task": task, "support": support, "channel": channel, "goal_kind": kind, "n": len(vals), "mean": float(np.mean(vals)), "ci95_t": ci(vals), "passes": int(sum(v >= 0.60 for v in vals)) if support == "leave_one_out" and channel == "live" and kind == "heldout" else None})
    effects = {}
    for representation in design.REPRESENTATIONS:
        for form in design.FORMS:
            for task in design.TASKS:
                if representation == "slot_local" and form == "mono9":
                    continue
                for support in design.SUPPORTS:
                    for kind in ("all", "seen", "heldout"):
                        keys = [seed for seed in seeds if (seed, f"{representation}_{form}_{task}_{support}_live") in children and (seed, f"{representation}_{form}_{task}_{support}_silent") in children]
                        effects[f"live_minus_silent|{representation}|{form}|{task}|{support}|{kind}"] = _effect([value(children[(seed, f"{representation}_{form}_{task}_{support}_live")], kind) - value(children[(seed, f"{representation}_{form}_{task}_{support}_silent")], kind) for seed in keys])
                    for channel in design.CHANNELS:
                        keys = [seed for seed in seeds if (seed, f"{representation}_{form}_{task}_full_{channel}") in children and (seed, f"{representation}_{form}_{task}_leave_one_out_{channel}") in children]
                        for kind in ("all", "seen", "heldout"):
                            effects[f"leave_minus_full|{representation}|{form}|{task}|{channel}|{kind}"] = _effect([value(children[(seed, f"{representation}_{form}_{task}_leave_one_out_{channel}")], kind) - value(children[(seed, f"{representation}_{form}_{task}_full_{channel}")], kind) for seed in keys])
    for form in design.FORMS:
        for task in design.TASKS:
            for support in design.SUPPORTS:
                for channel in design.CHANNELS:
                    if form == "tri3":
                        for kind in ("all", "seen", "heldout"):
                            keys = [seed for seed in seeds if (seed, f"slot_local_tri3_{task}_{support}_{channel}") in children and (seed, f"joint_history_tri3_{task}_{support}_{channel}") in children]
                            effects[f"slot_local_minus_joint|tri3|{task}|{support}|{channel}|{kind}"] = _effect([value(children[(seed, f"slot_local_tri3_{task}_{support}_{channel}")], kind) - value(children[(seed, f"joint_history_tri3_{task}_{support}_{channel}")], kind) for seed in keys])
    for representation in ("joint_history", "slot_local"):
        for task in design.TASKS:
            for support in design.SUPPORTS:
                for channel in design.CHANNELS:
                    if representation == "slot_local":
                        forms = ("tri3",)
                    else:
                        forms = design.FORMS
                    if len(forms) == 2:
                        for kind in ("all", "seen", "heldout"):
                            keys = [seed for seed in seeds if (seed, f"{representation}_tri3_{task}_{support}_{channel}") in children and (seed, f"{representation}_mono9_{task}_{support}_{channel}") in children]
                            effects[f"tri3_minus_mono9|{representation}|{task}|{support}|{channel}|{kind}"] = _effect([value(children[(seed, f"{representation}_tri3_{task}_{support}_{channel}")], kind) - value(children[(seed, f"{representation}_mono9_{task}_{support}_{channel}")], kind) for seed in keys])
    for representation in design.REPRESENTATIONS:
        for form in design.FORMS:
            if representation == "slot_local" and form == "mono9":
                continue
            for support in design.SUPPORTS:
                for channel in design.CHANNELS:
                    for kind in ("all", "seen", "heldout"):
                        keys = [seed for seed in seeds if (seed, f"{representation}_{form}_factorized_{support}_{channel}") in children and (seed, f"{representation}_{form}_holistic_{support}_{channel}") in children]
                        effects[f"factorized_minus_holistic|{representation}|{form}|{support}|{channel}|{kind}"] = _effect([value(children[(seed, f"{representation}_{form}_factorized_{support}_{channel}")], kind) - value(children[(seed, f"{representation}_{form}_holistic_{support}_{channel}")], kind) for seed in keys])
    recombination = []
    for (seed, condition), row in sorted(children.items()):
        representation, form, task, support, channel = design.parse_child_condition(condition)
        if form != "tri3" or channel != "live":
            continue
        for kind in ("all", "seen", "heldout"):
            natural = value(row, kind, "natural")
            raw = value(row, kind, "raw_recombined")
            taskwise = value(row, kind, "task_recombined")
            recombination.append({"seed": seed, "representation": representation, "task": task, "support": support, "goal_kind": kind, "raw_gap": raw - natural, "task_gap": taskwise - natural})
    parent_groups = []
    for form in design.FORMS:
        for task in design.TASKS:
            vals = [value(row, "all") for (seed, f, t), row in parents.items() if f == form and t == task]
            parent_groups.append({"form": form, "task": task, "n": len(vals), "mean": float(np.mean(vals)) if vals else None, "ci95_t": ci(vals)})
    return {"schema": "ecological_factorization_aggregate_v1", "parents": len(parents), "children": len(children), "rows": rows, "groups": groups, "effects": effects, "recombination": recombination, "parent_groups": parent_groups, "rule": {"heldout_min": 0.60, "ceiling": environment_ceiling()}}


def environment_ceiling():
    return float(2 * design.STEPS_PER_SUBTASK * design.CORRECT_REWARD / design.HORIZON)


def write_md(path, data):
    lines = ["# Ecological factorization and novel-combination transfer", "", f"- parent runs: {data['parents']}", f"- child runs: {data['children']}", "- primary zero-shot metric: leave-one-out live heldout return", "", "## Parent performance", "", "| form | task | mean all-goal return |", "|---|---|---:|"]
    for row in data["parent_groups"]:
        if row["mean"] is not None:
            lines.append(f"| `{row['form']}` | `{row['task']}` | {row['mean']:.3f} [{row['ci95_t'][0]:.3f},{row['ci95_t'][1]:.3f}] |")
    lines += ["", "## Child held-out return", "", "| representation | form | task | support | channel | heldout | pass count |", "|---|---|---|---|---|---:|---:|"]
    for group in data["groups"]:
        if group["goal_kind"] == "heldout":
            p = "n/a" if group["passes"] is None else f"{group['passes']}/{group['n']}"
            lines.append(f"| `{group['representation']}` | `{group['form']}` | `{group['task']}` | `{group['support']}` | `{group['channel']}` | {group['mean']:.3f} [{group['ci95_t'][0]:.3f},{group['ci95_t'][1]:.3f}] | {p} |")
    lines += ["", "## Pre-registered contrasts", "", "| contrast | mean | 95% CI |", "|---|---:|---:|"]
    wanted = [
        "factorized_minus_holistic|slot_local|tri3|leave_one_out|live|heldout",
        "slot_local_minus_joint|tri3|factorized|leave_one_out|live|heldout",
        "tri3_minus_mono9|joint_history|factorized|leave_one_out|live|heldout",
        "leave_minus_full|slot_local|tri3|factorized|live|heldout",
    ]
    for key in wanted:
        item = data["effects"].get(key)
        if item and item["mean"] is not None:
            lines.append(f"| `{key}` | {item['mean']:.3f} | [{item['ci95_t'][0]:.3f},{item['ci95_t'][1]:.3f}] |")
    lines += ["", "## Recombination gaps", "", "| representation | task | support | raw factor gap | task-target gap |", "|---|---|---|---:|---:|"]
    for representation in design.REPRESENTATIONS:
        for task in design.TASKS:
            for support in design.SUPPORTS:
                vals = [x for x in data["recombination"] if x["representation"] == representation and x["task"] == task and x["support"] == support and x["goal_kind"] == "heldout"]
                if vals:
                    lines.append(f"| `{representation}` | `{task}` | `{support}` | {np.mean([x['raw_gap'] for x in vals]):.3f} | {np.mean([x['task_gap'] for x in vals]):.3f} |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--results", required=True); parser.add_argument("--out", required=True); parser.add_argument("--markdown", required=True); args = parser.parse_args()
    data = aggregate(args.results); Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n"); write_md(args.markdown, data); print(json.dumps({"status": "written", "parents": data["parents"], "children": data["children"]}, ensure_ascii=False))
