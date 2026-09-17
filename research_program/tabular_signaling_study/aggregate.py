"""Compact aggregation for tabular signaling executions.

The script consumes only ``execution/results.json`` files.  It never reads a
training trajectory to choose a checkpoint or a condition, and all effects are
paired by seed before confidence intervals are computed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
             7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
             13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
             18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def ci95(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return [None, None]
    mean = float(x.mean())
    if len(x) == 1:
        return [mean, mean]
    t = T_CRIT_95.get(len(x) - 1, 1.96)
    half = t * float(x.std(ddof=1)) / np.sqrt(len(x))
    return [mean - half, mean + half]


def load(paths):
    by_key = {}
    for path in paths:
        payload = json.loads(Path(path).read_text())
        for row in payload["results"]:
            key = (int(row["seed"]), row["condition"])
            if key in by_key:
                raise ValueError(f"duplicate seed/condition: {key}")
            by_key[key] = row
    return by_key


def flatten(by_key, split="heldout"):
    rows = []
    for (seed, condition), result in sorted(by_key.items()):
        f = result["final"][split]
        for mode in ("natural", "closed", "permuted"):
            value = f[mode]["team_return_mean"]
            rows.append({"seed": seed, "condition": condition, "split": split,
                         "mode": mode, "team_return": value,
                         "oracle": f[mode].get("oracle_team_return_mean"),
                         "normalized": f[mode].get("normalized_return_mean_on_oracle_positive"),
                         "mi": f[mode].get("message_type_mi", 0.0),
                         "message_entropy": f[mode].get("message_entropy", 0.0)})
    return rows


def summarize(rows):
    out = {}
    groups = {}
    for row in rows:
        groups.setdefault((row["condition"], row["mode"]), []).append(row)
    for (condition, mode), group in sorted(groups.items()):
        values = np.array([x["team_return"] for x in group], dtype=float)
        out[f"{condition}|{mode}"] = {
            "condition": condition, "mode": mode, "n": len(values),
            "mean": float(values.mean()), "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "ci95_t": ci95(values),
            "normalized_mean": float(np.nanmean([x["normalized"] for x in group])),
            "message_mi_mean": float(np.mean([x["mi"] for x in group])),
            "message_entropy_mean": float(np.mean([x["message_entropy"] for x in group])),
        }
    return out


def effects(by_key, split="heldout"):
    result = {}
    conditions = sorted({condition for _, condition in by_key})

    def paired(label, left, right, mode="natural"):
        common = sorted({s for s, c in by_key if c == left} & {s for s, c in by_key if c == right})
        values = []
        for seed in common:
            a = by_key[(seed, left)]["final"][split][mode]["team_return_mean"]
            b = by_key[(seed, right)]["final"][split][mode]["team_return_mean"]
            values.append(float(a - b))
        result[label] = {"left": left, "right": right, "mode": mode, "seeds": common,
                         "values": values, "mean": float(np.mean(values)) if values else None,
                         "ci95_t": ci95(values)}

    for condition in conditions:
        if condition.endswith("_live_persistent"):
            closed = condition.replace("_live_persistent", "_closed_persistent")
            if (0, condition) in by_key or any(c == closed for _, c in by_key):
                # The closed value is an evaluation control inside a live run;
                # this branch is retained only for future execution schemas.
                pass
    for _, condition in sorted(by_key):
        f = by_key[(next(s for s, c in by_key if c == condition), condition)]["final"][split]
        if "live" in condition:
            values = []
            for seed in sorted({s for s, c in by_key if c == condition}):
                fseed = by_key[(seed, condition)]["final"][split]
                values.append(float(fseed["natural"]["team_return_mean"] - fseed["closed"]["team_return_mean"]))
            result[f"{condition}|natural-minus-closed"] = {"condition": condition, "values": values,
                "mean": float(np.mean(values)), "ci95_t": ci95(values), "seeds": sorted({s for s, c in by_key if c == condition})}
            values = []
            for seed in sorted({s for s, c in by_key if c == condition}):
                fseed = by_key[(seed, condition)]["final"][split]
                values.append(float(fseed["natural"]["team_return_mean"] - fseed["permuted"]["team_return_mean"]))
            result[f"{condition}|natural-minus-permuted"] = {"condition": condition, "values": values,
                "mean": float(np.mean(values)), "ci95_t": ci95(values), "seeds": sorted({s for s, c in by_key if c == condition})}

    # Factor contrasts use natural returns and are computed only when both
    # cells are present for the same seed.  The labels are self-explanatory and
    # are useful for a preregistered interaction table.
    for condition in conditions:
        if "_PI_" in condition:
            fi = condition.replace("_PI_", "_FI_")
            if fi in conditions:
                paired(f"{fi}-minus-{condition}", fi, condition)
        if condition.startswith("recurrent_"):
            stateless = condition.replace("recurrent", "stateless", 1)
            if stateless in conditions:
                paired(f"{condition}-minus-{stateless}", condition, stateless)
        # A from-scratch silent arm is the clean channel-necessity control.
        # The in-run `closed` intervention is intentionally kept separate: in
        # FI it can be out of distribution because live training may have
        # learned to read the token even when the target is visible.
        if "_live_" in condition:
            silent = condition.replace("_live_", "_silent_")
            if silent in conditions:
                paired(f"{condition}-minus-{silent}|natural", condition, silent, mode="natural")
    return result


def write_markdown(path, data):
    lines = ["# Tabular signaling compact aggregation", "", f"- split: `{data['split']}`",
             f"- runs: {data['runs']}", "- confidence interval: paired-by-seed Student-t 95%", "",
             "| condition | natural | closed | permuted | Δ natural−closed | Δ natural−permuted | MI |", "|---|---:|---:|---:|---:|---:|---:|"]
    for condition in sorted({x["condition"] for x in data["rows"]}):
        vals = {x["mode"]: x for x in data["summary"].values() if x["condition"] == condition}
        e1 = data["effects"].get(f"{condition}|natural-minus-closed", {})
        e2 = data["effects"].get(f"{condition}|natural-minus-permuted", {})
        fmt = lambda x: "NA" if x is None else f"{x:.3f}"
        lines.append(f"| `{condition}` | {fmt(vals['natural']['mean'])} | {fmt(vals['closed']['mean'])} | {fmt(vals['permuted']['mean'])} | {fmt(e1.get('mean'))} | {fmt(e2.get('mean'))} | {fmt(vals['natural']['message_mi_mean'])} |")
    lines += ["", "The natural-minus-closed and natural-minus-permuted values are paired within seed. A nonzero token MI is descriptive and is not counted as causal evidence without the corresponding intervention.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("--markdown", required=True); ap.add_argument("--split", default="heldout")
    args = ap.parse_args(); by_key = load(args.results); rows = flatten(by_key, args.split)
    data = {"schema": "tabular_signaling_aggregate_v1", "split": args.split, "runs": len(by_key), "rows": rows,
            "summary": summarize(rows), "effects": effects(by_key, args.split)}
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    write_markdown(args.markdown, data)
    print(json.dumps({"status": "written", "runs": len(by_key), "out": args.out}, ensure_ascii=False))


if __name__ == "__main__":
    main()
