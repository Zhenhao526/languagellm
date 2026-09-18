"""Aggregate routed factor-sharing results with paired seed contrasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def ci(values):
    x = np.asarray(list(values), dtype=float)
    mean = float(x.mean()) if len(x) else float("nan")
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * sd / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return {"n": int(len(x)), "mean": mean, "sd": sd,
            "ci95": [mean - half, mean + half]}


def metric(row, kind, mode="natural"):
    return row["final"][kind][mode]["team_return_mean"]


def group(rows, keys):
    out = []
    groups = sorted({tuple(row[key] for key in keys) for row in rows})
    for group_key in groups:
        subset = [row for row in rows if tuple(row[key] for key in keys) == group_key]
        rec = {key: value for key, value in zip(keys, group_key)}
        rec["n"] = len(subset)
        kinds = tuple(kind for kind in ("all", "heldout_combo", "heldout_value")
                      if kind in subset[0]["final"])
        for kind in kinds:
            modes = ("natural", "permuted", "silent", "recombined") \
                if "recombined" in subset[0]["final"][kind] \
                else ("natural", "permuted", "silent")
            for mode in modes:
                rec[f"{kind}_{mode}"] = ci(metric(row, kind, mode) for row in subset)
            rec[f"{kind}_message_gap"] = ci(
                metric(row, kind, "natural") - metric(row, kind, "permuted")
                for row in subset)
            rec[f"{kind}_live_silent"] = ci(
                metric(row, kind, "natural") - metric(row, kind, "silent")
                for row in subset)
        rec["fresh_sender_consistency"] = ci(
            row["final"]["fresh_sender_consistency"] for row in subset)
        rec["community_codebook_hamming"] = ci(
            row["final"]["community_codebook_hamming"] for row in subset)
        factors = [row["final"]["factor_slot_unique_fraction"] for row in subset
                   if row["final"].get("factor_slot_unique_fraction") is not None]
        rec["factor_slot_unique_fraction"] = ci(factors) if factors else None
        routed = [row["final"] for row in subset
                  if row["final"].get("sender_route") is not None]
        if routed:
            rec["sender_route_best_alignment"] = ci(
                row["sender_route"]["best_alignment"] for row in routed)
            rec["sender_route_entropy"] = ci(
                row["sender_route"]["mean_entropy"] for row in routed)
            rec["receiver_route_best_alignment"] = ci(
                row["receiver_route"]["best_alignment"] for row in routed)
            rec["receiver_route_entropy"] = ci(
                row["receiver_route"]["mean_entropy"] for row in routed)
        else:
            rec["sender_route_best_alignment"] = None
            rec["sender_route_entropy"] = None
            rec["receiver_route_best_alignment"] = None
            rec["receiver_route_entropy"] = None
        rec["functional_combo_count"] = (
            sum(metric(row, "heldout_combo") >= 0.60 for row in subset)
            if "heldout_combo" in subset[0]["final"] else None)
        rec["functional_value_count"] = (
            sum(metric(row, "heldout_value") >= 0.60 for row in subset)
            if "heldout_value" in subset[0]["final"] else None)
        out.append(rec)
    return out


def _paired(by, seeds, left, right, kind):
    values = []
    for seed in seeds:
        if left in by.get(seed, {}) and right in by.get(seed, {}):
            values.append(metric(by[seed][left], kind) - metric(by[seed][right], kind))
    return ci(values)


def _has_pair(by, seeds, left, right):
    return any(left in by.get(seed, {}) and right in by.get(seed, {}) for seed in seeds)


def contrasts(children):
    keyed = {}
    for row in children:
        cell = (row["architecture"], row["population"], row["visibility"], row["support"])
        keyed.setdefault(int(row["seed"]), {})[cell] = row
    seeds = sorted(keyed)
    architectures = sorted({row["architecture"] for row in children})
    populations = sorted({row["population"] for row in children})
    visibilities = sorted({row["visibility"] for row in children})
    supports = sorted({row["support"] for row in children})
    out = []
    for population in populations:
        for visibility in visibilities:
            for support in supports:
                for architecture in architectures:
                    left = (architecture, population, visibility, support)
                    right = ("holistic", population, visibility, support)
                    if architecture == "holistic" or not _has_pair(keyed, seeds, left, right):
                        continue
                    vals = {kind: _paired(keyed, seeds, left, right, kind)
                            for kind in ("all", "heldout_combo", "heldout_value")}
                    out.append({"contrast": f"{architecture}_minus_holistic",
                                "population": population, "visibility": visibility,
                                "support": support, **vals})
                for left_index, left_arch in enumerate(architectures):
                    for right_arch in architectures[left_index + 1:]:
                        left = (left_arch, population, visibility, support)
                        right = (right_arch, population, visibility, support)
                        if "holistic" in (left_arch, right_arch) or not _has_pair(keyed, seeds, left, right):
                            continue
                        vals = {kind: _paired(keyed, seeds, left, right, kind)
                                for kind in ("all", "heldout_combo", "heldout_value")}
                        out.append({"contrast": f"{left_arch}_minus_{right_arch}",
                                    "population": population, "visibility": visibility,
                                    "support": support, **vals})
    for architecture in architectures:
        for visibility in visibilities:
            for support in supports:
                left = (architecture, "conflict", visibility, support)
                right = (architecture, "aligned", visibility, support)
                if not _has_pair(keyed, seeds, left, right):
                    continue
                vals = {kind: _paired(keyed, seeds, left, right, kind)
                        for kind in ("all", "heldout_combo", "heldout_value")}
                out.append({"contrast": "conflict_minus_aligned",
                            "architecture": architecture, "visibility": visibility,
                            "support": support, **vals})
    return out


def aggregate(payload):
    return {
        "schema": "routed_factorization_aggregate_v1",
        "parent_summary": group(payload["parents"], ["architecture", "population"]),
        "child_summary": group(payload["children"],
                                ["architecture", "population", "visibility", "support"]),
        "contrasts": contrasts(payload["children"]),
        "run_counts": {"parents": len(payload["parents"]), "children": len(payload["children"])},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    data = aggregate(json.loads(Path(args.results).read_text()))
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Routed factor-sharing aggregate", "",
             f"Parents: {data['run_counts']['parents']}; children: {data['run_counts']['children']}.",
             "", "## Child cells", ""]
    for row in data["child_summary"]:
        lines.append(
            f"- {row['architecture']}/{row['population']}/{row['visibility']}/{row['support']}: "
            f"all {row['all_natural']['mean']:.3f}, "
            f"combo {row['heldout_combo_natural']['mean']:.3f}, "
            f"value {row['heldout_value_natural']['mean']:.3f}, "
            f"combo functional {row['functional_combo_count']}/{row['n']}, "
            f"value functional {row['functional_value_count']}/{row['n']}")
        if row["sender_route_best_alignment"] is not None:
            lines.append(
                f"  routing sender best alignment "
                f"{row['sender_route_best_alignment']['mean']:.3f}; receiver "
                f"{row['receiver_route_best_alignment']['mean']:.3f}")
    lines += ["", "## Paired contrasts", ""]
    for row in data["contrasts"]:
        label = row.get("architecture", row.get("population", ""))
        lines.append(
            f"- {row['contrast']} {label}/{row['visibility']}/{row['support']}: "
            f"all {row['all']['mean']:.3f} [{row['all']['ci95'][0]:.3f}, {row['all']['ci95'][1]:.3f}], "
            f"combo {row['heldout_combo']['mean']:.3f}, value {row['heldout_value']['mean']:.3f}")
    Path(args.markdown).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
