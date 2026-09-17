"""Paired analysis for the multi-partner sender-alignment study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import numpy as np
from . import design

T_CRIT_95_DF8 = 2.306004


def ci(values):
    x = np.asarray(values, dtype=np.float64)
    if len(x) == 0:
        return [None, None]
    mean = float(x.mean())
    if len(x) < 2:
        return [mean, mean]
    half = T_CRIT_95_DF8 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return [mean - half, mean + half]


def load(path):
    return json.loads(Path(path).read_text())


def metric(final, goal_kind, mode):
    return float(final[goal_kind][mode]["team_return_mean"])


def summarize(row):
    final = row["final"]
    initial = row.get("initial", {})
    heldout = metric(final, "heldout", "natural")
    initial_heldout = metric(initial, "heldout", "natural") if initial else None
    return {
        "seed": int(row["seed"]),
        "condition": row["condition"],
        "population_mode": row["population_mode"],
        "visibility": row["visibility"],
        "adaptation": row["adaptation"],
        "support": row["support"],
        "heldout_goal": int(row["heldout_goal"]),
        "heldout_natural": heldout,
        "initial_heldout_natural": initial_heldout,
        "learning_gain": heldout - initial_heldout if initial_heldout is not None else None,
        "heldout_silent": metric(final, "heldout", "silent"),
        "heldout_permuted": metric(final, "heldout", "permuted"),
        "all_natural": metric(final, "all", "natural"),
        "all_silent": metric(final, "all", "silent"),
        "all_permuted": metric(final, "all", "permuted"),
        "all_recombined": metric(final, "all", "raw_recombined"),
        "live_minus_silent": metric(final, "heldout", "natural") - metric(final, "heldout", "silent"),
        "natural_minus_permuted": metric(final, "heldout", "natural") - metric(final, "heldout", "permuted"),
        "all_natural_minus_permuted": metric(final, "all", "natural") - metric(final, "all", "permuted"),
        "sender_consistency": float(final["sender_consistency"]),
        "pairwise_min_hamming": int(final["pairwise_min_hamming"]),
        "functional": bool(heldout >= 0.60),
        "sender_codebook": final["sender_codebook"],
        "final_parameter_sha256": row["final_parameter_sha256"],
    }


def find(rows, **kwargs):
    for row in rows:
        if all(row.get(key) == value for key, value in kwargs.items()):
            return row
    return None


def paired(rows, left_kwargs, right_kwargs, value="heldout_natural"):
    diffs = []
    for seed in sorted({row["seed"] for row in rows}):
        left = find(rows, seed=seed, **left_kwargs)
        right = find(rows, seed=seed, **right_kwargs)
        if left is not None and right is not None:
            diffs.append(float(left[value]) - float(right[value]))
    return {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs else None, "ci95_t": ci(diffs), "values": diffs}


def add_contrast(out, name, result, **factors):
    if result["n"]:
        out.append({"name": name, **factors, **{k: v for k, v in result.items() if k != "values"}})


def group(rows, mode, visibility, adaptation, support):
    vals = [r for r in rows if r["population_mode"] == mode and r["visibility"] == visibility and r["adaptation"] == adaptation and r["support"] == support]
    if not vals:
        return None
    def mean(key): return float(np.mean([r[key] for r in vals]))
    return {
        "population_mode": mode,
        "visibility": visibility,
        "adaptation": adaptation,
        "support": support,
        "n": len(vals),
        "heldout_natural_mean": mean("heldout_natural"),
        "heldout_natural_ci95_t": ci([r["heldout_natural"] for r in vals]),
        "initial_heldout_natural_mean": mean("initial_heldout_natural") if all(r["initial_heldout_natural"] is not None for r in vals) else None,
        "learning_gain_mean": mean("learning_gain") if all(r["learning_gain"] is not None for r in vals) else None,
        "heldout_silent_mean": mean("heldout_silent"),
        "heldout_permuted_mean": mean("heldout_permuted"),
        "all_natural_mean": mean("all_natural"),
        "all_natural_minus_permuted_mean": mean("all_natural_minus_permuted"),
        "live_minus_silent_mean": mean("live_minus_silent"),
        "natural_minus_permuted_mean": mean("natural_minus_permuted"),
        "all_recombined_mean": mean("all_recombined"),
        "sender_consistency_mean": mean("sender_consistency"),
        "pairwise_min_hamming_mean": mean("pairwise_min_hamming"),
        "functional_count": int(sum(r["functional"] for r in vals)),
    }


def analyze(path):
    payload = load(path)
    parents = payload["parents"]
    children = payload["children"]
    parent_rows = []
    for row in parents:
        final = row["final"]
        parent_rows.append({
            "seed": int(row["seed"]),
            "population_mode": row["population_mode"],
            "all_natural": metric(final, "all", "natural"),
            "all_permuted": metric(final, "all", "permuted"),
            "all_natural_minus_permuted": metric(final, "all", "natural") - metric(final, "all", "permuted"),
            "heldout_natural": metric(final, "heldout", "natural"),
            "sender_consistency": float(final["sender_consistency"]),
            "sender_codebook": final["sender_codebook"],
            "pairwise_min_hamming": int(final["pairwise_min_hamming"]),
            "final_parameter_sha256": row["final_parameter_sha256"],
        })
    child_rows = [summarize(row) for row in children]
    groups = []
    for mode in design.POPULATION_MODES:
        for visibility in design.VISIBILITIES:
            for adaptation in design.ADAPTATIONS:
                for support in design.SUPPORTS:
                    g = group(child_rows, mode, visibility, adaptation, support)
                    if g is not None:
                        groups.append(g)
    contrasts = []
    for visibility in design.VISIBILITIES:
        for adaptation in design.ADAPTATIONS:
            for support in design.SUPPORTS:
                for value in ("heldout_natural", "sender_consistency", "all_natural_minus_permuted"):
                    result = paired(child_rows, {"population_mode": "heterogeneous", "visibility": visibility, "adaptation": adaptation, "support": support}, {"population_mode": "homogeneous", "visibility": visibility, "adaptation": adaptation, "support": support}, value=value)
                    add_contrast(contrasts, "heterogeneous_minus_homogeneous", result, value=value, visibility=visibility, adaptation=adaptation, support=support)
    for mode in design.POPULATION_MODES:
        for adaptation in design.ADAPTATIONS:
            for support in design.SUPPORTS:
                for value in ("heldout_natural", "sender_consistency"):
                    result = paired(child_rows, {"population_mode": mode, "visibility": "visible", "adaptation": adaptation, "support": support}, {"population_mode": mode, "visibility": "hidden", "adaptation": adaptation, "support": support}, value=value)
                    add_contrast(contrasts, "visible_minus_hidden", result, value=value, population_mode=mode, adaptation=adaptation, support=support)
    for mode in design.POPULATION_MODES:
        for visibility in design.VISIBILITIES:
            for support in design.SUPPORTS:
                for value in ("heldout_natural", "sender_consistency"):
                    result = paired(child_rows, {"population_mode": mode, "visibility": visibility, "adaptation": "coadapt", "support": support}, {"population_mode": mode, "visibility": visibility, "adaptation": "sender_only", "support": support}, value=value)
                    add_contrast(contrasts, "coadapt_minus_sender_only", result, value=value, population_mode=mode, visibility=visibility, support=support)
    for mode in design.POPULATION_MODES:
        for visibility in design.VISIBILITIES:
            for adaptation in design.ADAPTATIONS:
                result = paired(child_rows, {"population_mode": mode, "visibility": visibility, "adaptation": adaptation, "support": "leave_one_out"}, {"population_mode": mode, "visibility": visibility, "adaptation": adaptation, "support": "full"}, value="heldout_natural")
                add_contrast(contrasts, "leave_one_out_minus_full", result, population_mode=mode, visibility=visibility, adaptation=adaptation)
    parent_contrasts = []
    for value in ("all_natural", "all_natural_minus_permuted", "sender_consistency"):
        result = paired(parent_rows, {"population_mode": "heterogeneous"}, {"population_mode": "homogeneous"}, value=value)
        add_contrast(parent_contrasts, "heterogeneous_minus_homogeneous_parent", result, value=value)
    return {"schema": "partner_exposure_analysis_v1", "rule": {"functional_natural_min": 0.60, "primary": "fresh sender leave-one-out live held-out natural return", "interventions": ["population heterogeneity", "sender partner visibility", "coadaptation"]}, "parents": parent_rows, "parent_contrasts": parent_contrasts, "children": child_rows, "groups": groups, "contrasts": contrasts}


def fmt(x):
    return "—" if x is None else f"{x:.3f}"


def write_md(path, data):
    lines = [
        "# Multi-partner sender alignment under perceptual heterogeneity",
        "",
        "A parent population first learns a hidden-partner protocol. A fresh sender is then inserted while the four workers are retained. Every child batch exposes that sender to all four rotating partners; `hidden` and `visible` control whether the sender can condition on partner identity, while `sender_only` and `coadapt` control whether workers are frozen or updated together. `leave_one_out` masks one joint goal during adaptation.",
        "",
        "## Parent endpoint",
        "",
        "| population | all natural | all natural−permuted | sender consistency |",
        "|---|---:|---:|---:|",
    ]
    for row in data["parents"]:
        lines.append(f"| `{row['population_mode']}` | {row['all_natural']:.3f} | {row['all_natural_minus_permuted']:.3f} | {row['sender_consistency']:.3f} |")
    lines += ["", "Parent contrasts are paired by seed.", "", "| contrast | factors | mean | 95% CI | n |", "|---|---|---:|---|---:|"]
    for row in data["parent_contrasts"]:
        lo, hi = row["ci95_t"]
        factors = ", ".join(f"{k}={row[k]}" for k in ("value",) if k in row)
        lines.append(f"| `{row['name']}` | {factors} | {row['mean']:.3f} | [{lo:.3f}, {hi:.3f}] | {row['n']} |")
    lines += ["", "## Fresh sender endpoint", "", "| population | visibility | adaptation | support | initial | final | gain | silent | natural−permuted | consistency | functional |", "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["groups"]:
        lines.append(f"| `{row['population_mode']}` | `{row['visibility']}` | `{row['adaptation']}` | `{row['support']}` | {fmt(row['initial_heldout_natural_mean'])} | {row['heldout_natural_mean']:.3f} | {fmt(row['learning_gain_mean'])} | {row['heldout_silent_mean']:.3f} | {row['natural_minus_permuted_mean']:.3f} | {row['sender_consistency_mean']:.3f} | {row['functional_count']}/{row['n']} |")
    lines += ["", "## Paired contrasts", "", "| contrast | factors | mean | 95% CI | n |", "|---|---|---:|---|---:|"]
    for row in data["contrasts"]:
        lo, hi = row["ci95_t"]
        factor_keys = ("value", "population_mode", "visibility", "adaptation", "support")
        factors = ", ".join(f"{k}={row[k]}" for k in factor_keys if k in row)
        lines.append(f"| `{row['name']}` | {factors} | {row['mean']:.3f} | [{lo:.3f}, {hi:.3f}] | {row['n']} |")
    lines += ["", "The central test is whether exposure to multiple perceptual conventions changes a fresh sender's ability to recover a reusable message code. A hidden sender must use one codebook across all partners; a visible sender can maintain partner-specific mappings. This is a protocol-alignment mechanism test, not evidence of human syntax or open-ended language.", ""]
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    data = analyze(args.results)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    write_md(args.markdown, data)
    print(json.dumps({"status": "written", "parent_rows": len(data["parents"]), "child_rows": len(data["children"]), "groups": len(data["groups"])}, ensure_ascii=False))
