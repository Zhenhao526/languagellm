"""Aggregate endpoint statistics and paired contrasts for community merging."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import design


def mean_sd_ci(values):
    x = np.asarray(values, dtype=np.float64)
    if not len(x):
        return {"n": 0, "mean": None, "sd": None, "ci95": [None, None]}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if len(x) > 1 else 0.0
    half = 1.96 * sd / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return {"n": int(len(x)), "mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def _metric(result, goal_kind, mode):
    return float(result["final"][goal_kind][mode]["team_return_mean"])


def _by_seed(rows):
    out = {}
    for row in rows:
        out[int(row["seed"])] = row
    return out


def paired(rows_a, rows_b, value_fn):
    a = _by_seed(rows_a); b = _by_seed(rows_b)
    seeds = sorted(set(a) & set(b))
    values = [float(value_fn(a[s]) - value_fn(b[s])) for s in seeds]
    return {"seeds": seeds, "values": values, **mean_sd_ci(values)}


def aggregate(payload):
    children = payload["children"]
    parents = payload["parents"]
    groups = defaultdict(list)
    for row in children:
        key = (row["population"], row["visibility"], row["adaptation"], row["support"], row["role"])
        groups[key].append(row)
    child_summary = []
    for key in sorted(groups):
        population, visibility, adaptation, support, role = key
        rows = groups[key]
        def stats(goal, mode):
            return mean_sd_ci([_metric(r, goal, mode) for r in rows])
        child_summary.append({
            "population": population,
            "visibility": visibility,
            "adaptation": adaptation,
            "support": support,
            "role": role,
            "n": len(rows),
            "all_natural": stats("all", "natural"),
            "all_permuted": stats("all", "permuted"),
            "all_silent": stats("all", "silent"),
            "all_message_gap": mean_sd_ci([_metric(r, "all", "natural") - _metric(r, "all", "permuted") for r in rows]),
            "all_live_silent": mean_sd_ci([_metric(r, "all", "natural") - _metric(r, "all", "silent") for r in rows]),
            "heldout_combo": stats("heldout_combo", "natural"),
            "heldout_value": stats("heldout_value", "natural"),
            "heldout_combo_recombined": stats("heldout_combo", "recombined"),
            "heldout_value_recombined": stats("heldout_value", "recombined"),
            "fresh_sender_consistency": mean_sd_ci([r["final"]["fresh_sender_consistency"] for r in rows]),
            "community_codebook_hamming": mean_sd_ci([r["final"]["community_codebook_hamming"] for r in rows]),
            "role_sender": mean_sd_ci([r["final"]["all"]["by_role"]["fresh_sender"]["team_return_mean"] for r in rows]),
            "role_receiver": mean_sd_ci([r["final"]["all"]["by_role"]["fresh_receiver"]["team_return_mean"] for r in rows]),
            "functional_combo_count": int(sum(_metric(r, "heldout_combo", "natural") >= 0.60 for r in rows)),
            "functional_value_count": int(sum(_metric(r, "heldout_value", "natural") >= 0.60 for r in rows)),
        })

    def select(population, visibility, adaptation, support, role):
        return [r for r in children if (r["population"], r["visibility"], r["adaptation"], r["support"], r["role"]) == (population, visibility, adaptation, support, role)]

    contrasts = []
    for visibility in design.VISIBILITIES:
        for adaptation in design.ADAPTATIONS:
            for support in design.SUPPORTS:
                for role in design.ROLES:
                    aligned = select("aligned", visibility, adaptation, support, role)
                    conflict = select("conflict", visibility, adaptation, support, role)
                    contrasts.append({"contrast": "conflict_minus_aligned", "visibility": visibility, "adaptation": adaptation, "support": support, "role": role, "heldout_combo": paired(conflict, aligned, lambda r: _metric(r, "heldout_combo", "natural")), "heldout_value": paired(conflict, aligned, lambda r: _metric(r, "heldout_value", "natural")), "message_gap": paired(conflict, aligned, lambda r: _metric(r, "all", "natural") - _metric(r, "all", "permuted"))})
                    if visibility == "visible":
                        hidden = select("conflict", "hidden", adaptation, support, role)
                        contrasts.append({"contrast": "visible_minus_hidden", "population": "conflict", "adaptation": adaptation, "support": support, "role": role, "heldout_combo": paired(conflict, hidden, lambda r: _metric(r, "heldout_combo", "natural")), "heldout_value": paired(conflict, hidden, lambda r: _metric(r, "heldout_value", "natural")), "consistency": paired(conflict, hidden, lambda r: r["final"]["fresh_sender_consistency"])})
                    if adaptation == "coadapt":
                        fresh_only = select("conflict", visibility, "fresh_only", support, role)
                        contrasts.append({"contrast": "coadapt_minus_fresh_only", "population": "conflict", "visibility": visibility, "support": support, "role": role, "heldout_combo": paired(conflict, fresh_only, lambda r: _metric(r, "heldout_combo", "natural")), "heldout_value": paired(conflict, fresh_only, lambda r: _metric(r, "heldout_value", "natural")), "community_hamming": paired(conflict, fresh_only, lambda r: r["final"]["community_codebook_hamming"])})
    parent_groups = defaultdict(list)
    for row in parents:
        parent_groups[(row["population"], row["community"])].append(row)
    parent_summary = []
    for key in sorted(parent_groups):
        pop, community = key
        rows = parent_groups[key]
        parent_summary.append({"population": pop, "community": int(community), "n": len(rows), "all_natural": mean_sd_ci([_metric(r, "all", "natural") for r in rows]), "all_permuted": mean_sd_ci([_metric(r, "all", "permuted") for r in rows]), "all_silent": mean_sd_ci([_metric(r, "all", "silent") for r in rows]), "message_gap": mean_sd_ci([_metric(r, "all", "natural") - _metric(r, "all", "permuted") for r in rows])})
    return {"schema": "community_merge_aggregate_v1", "parent_summary": parent_summary, "child_summary": child_summary, "contrasts": contrasts, "run_counts": {"parents": len(parents), "children": len(children)}}


def _fmt(stat):
    if stat["mean"] is None:
        return "NA"
    return f"{stat['mean']:.3f} [{stat['ci95'][0]:.3f}, {stat['ci95'][1]:.3f}]"


def markdown(aggregate):
    lines = ["# Community merge aggregate", "", f"parents={aggregate['run_counts']['parents']}, children={aggregate['run_counts']['children']}", "", "## Parent endpoints", "", "| population | community | all natural | all permuted | message gap |", "|---|---:|---:|---:|---:|"]
    for row in aggregate["parent_summary"]:
        lines.append(f"| {row['population']} | {row['community']} | {_fmt(row['all_natural'])} | {_fmt(row['all_permuted'])} | {_fmt(row['message_gap'])} |")
    lines += ["", "## Child endpoints", "", "| population | visibility | adaptation | support | role | combo natural | value natural | all message gap | fresh consistency | community hamming | functional combo/value |", "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in aggregate["child_summary"]:
        lines.append(f"| {row['population']} | {row['visibility']} | {row['adaptation']} | {row['support']} | {row['role']} | {_fmt(row['heldout_combo'])} | {_fmt(row['heldout_value'])} | {_fmt(row['all_message_gap'])} | {_fmt(row['fresh_sender_consistency'])} | {_fmt(row['community_codebook_hamming'])} | {row['functional_combo_count']}/{row['n']} ; {row['functional_value_count']}/{row['n']} |")
    lines += ["", "## Paired contrasts", "", "| contrast | factors | heldout combo | heldout value | extra |", "|---|---|---:|---:|---|"]
    for row in aggregate["contrasts"]:
        factors = ", ".join(f"{k}={v}" for k, v in row.items() if k in ("population", "visibility", "adaptation", "support", "role"))
        extra = ""
        for key in ("message_gap", "consistency", "community_hamming"):
            if key in row:
                extra = f"{key}: {_fmt(row[key])}"
        lines.append(f"| {row['contrast']} | {factors} | {_fmt(row.get('heldout_combo', {'mean': None}))} | {_fmt(row.get('heldout_value', {'mean': None}))} | {extra} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.results).read_text())
    agg = aggregate(payload)
    Path(args.out).write_text(json.dumps(agg, ensure_ascii=False, indent=2) + "\n")
    Path(args.markdown).write_text(markdown(agg))
    print(json.dumps({"status": "aggregated", "groups": len(agg["child_summary"]), "contrasts": len(agg["contrasts"])}, ensure_ascii=False))
