"""Aggregate open-world expansion endpoints and paired seed contrasts."""
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
    return {"n": int(len(x)), "mean": mean, "sd": sd, "ci95": [mean - half, mean + half]}


def metric(row, kind, mode="natural"):
    return row["final"][kind][mode]["team_return_mean"]


def group(rows):
    out = []
    keys = sorted({(r["architecture"], r["support"]) for r in rows})
    for architecture, support in keys:
        subset = [r for r in rows if r["architecture"] == architecture and r["support"] == support]
        rec = {"architecture": architecture, "support": support, "n": len(subset)}
        for kind in ("old_seen", "old_combo", "new_single", "new_double", "all"):
            rec[f"{kind}_natural"] = ci(metric(r, kind) for r in subset)
            rec[f"{kind}_recombined"] = ci(metric(r, kind, "recombined") for r in subset)
            rec[f"{kind}_message_gap"] = ci(metric(r, kind) - metric(r, kind, "permuted") for r in subset)
            rec[f"{kind}_live_silent"] = ci(metric(r, kind) - metric(r, kind, "silent") for r in subset)
            rec[f"{kind}_functional_count"] = sum(metric(r, kind) >= 0.60 for r in subset)
        for name in ("fresh_sender_consistency", "fresh_sender_consistency_old",
                     "fresh_sender_consistency_new_single", "fresh_sender_consistency_new_double",
                     "community_codebook_hamming"):
            rec[name] = ci(r["final"][name] for r in subset)
        out.append(rec)
    return out


def contrasts(rows):
    by = {(int(r["seed"]), r["architecture"], r["support"]): r for r in rows}
    seeds = sorted({int(r["seed"]) for r in rows})
    architectures = sorted({r["architecture"] for r in rows})
    supports = sorted({r["support"] for r in rows})
    out = []
    for support in supports:
        for i, left in enumerate(architectures):
            for right in architectures[i + 1:]:
                values = {}
                for kind in ("old_seen", "old_combo", "new_single", "new_double", "all"):
                    values[kind] = ci(
                        metric(by[(seed, left, support)], kind) - metric(by[(seed, right, support)], kind)
                        for seed in seeds if (seed, left, support) in by and (seed, right, support) in by
                    )
                out.append({"left": left, "right": right, "support": support, **values})
    return out


def aggregate(payload):
    return {
        "schema": "open_world_expansion_aggregate_v1",
        "child_summary": group(payload["children"]),
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
    lines = ["# Open-world expansion aggregate", "",
             f"Parents: {data['run_counts']['parents']}; children: {data['run_counts']['children']}.",
             "", "## Child cells", ""]
    for row in data["child_summary"]:
        lines.append(
            f"- {row['architecture']}/{row['support']}: "
            f"old-combo {row['old_combo_natural']['mean']:.3f} "
            f"({row['old_combo_functional_count']}/{row['n']}), "
            f"new-single {row['new_single_natural']['mean']:.3f} "
            f"({row['new_single_functional_count']}/{row['n']}), "
            f"new-double {row['new_double_natural']['mean']:.3f} "
            f"({row['new_double_functional_count']}/{row['n']}), "
            f"new-double recombined {row['new_double_recombined']['mean']:.3f}, "
            f"new consistency {row['fresh_sender_consistency_new_single']['mean']:.3f}")
    lines += ["", "## Paired contrasts", ""]
    for row in data["contrasts"]:
        c = row["new_single"]
        lines.append(
            f"- {row['left']} − {row['right']} / {row['support']}: "
            f"old-combo {row['old_combo']['mean']:.3f}, "
            f"new-single {c['mean']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}], "
            f"new-double {row['new_double']['mean']:.3f}")
    Path(args.markdown).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
