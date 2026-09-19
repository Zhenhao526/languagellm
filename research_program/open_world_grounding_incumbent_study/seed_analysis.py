"""Seed-level analysis for open-world expansion."""
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
    return float(row["final"][kind][mode]["team_return_mean"])


def analyse(payload):
    children = payload["children"]
    keyed = {(int(r["seed"]), r["architecture"], r["support"]): r for r in children}
    seeds = sorted({int(r["seed"]) for r in children})
    architectures = sorted({r["architecture"] for r in children})
    supports = sorted({r["support"] for r in children})
    summary = []
    for architecture in architectures:
        for support in supports:
            subset = [keyed[(seed, architecture, support)] for seed in seeds]
            rec = {"architecture": architecture, "support": support, "n": len(subset)}
            for kind in ("old_seen", "old_combo", "new_single", "new_double", "all"):
                rec[kind] = ci(metric(r, kind) for r in subset)
                rec[f"{kind}_recombined"] = ci(metric(r, kind, "recombined") for r in subset)
            for name in ("fresh_sender_consistency_old", "fresh_sender_consistency_new_single",
                         "fresh_sender_consistency_new_double", "community_codebook_hamming"):
                rec[name] = ci(r["final"][name] for r in subset)
            rec["old_combo_functional_count"] = sum(metric(r, "old_combo") >= 0.60 for r in subset)
            rec["new_single_functional_count"] = sum(metric(r, "new_single") >= 0.60 for r in subset)
            rec["new_double_functional_count"] = sum(metric(r, "new_double") >= 0.60 for r in subset)
            summary.append(rec)
    contrasts = []
    for support in supports:
        for left, right in (("factorized", "holistic"), ("tied_routed", "factorized"),
                            ("tied_routed", "holistic")):
            if not all((seed, left, support) in keyed and (seed, right, support) in keyed for seed in seeds):
                continue
            row = {"left": left, "right": right, "support": support}
            for kind in ("old_seen", "old_combo", "new_single", "new_double", "all"):
                row[kind] = ci(metric(keyed[(seed, left, support)], kind) -
                                metric(keyed[(seed, right, support)], kind) for seed in seeds)
            contrasts.append(row)
    return {"schema": "open_world_expansion_seed_analysis_v1", "condition_summary": summary,
            "contrasts": contrasts, "run_counts": {"children": len(children), "seeds": len(seeds)}}


def markdown(data):
    lines = ["# Open-world expansion seed analysis", "",
             f"Children: {data['run_counts']['children']}; seeds: {data['run_counts']['seeds']}.",
             "", "## New-single endpoint", ""]
    for r in data["condition_summary"]:
        lines.append(f"- {r['architecture']}/{r['support']}: "
                     f"old-combo {r['old_combo']['mean']:.3f} "
                     f"({r['old_combo_functional_count']}/{r['n']}), "
                     f"new-single {r['new_single']['mean']:.3f} "
                     f"({r['new_single_functional_count']}/{r['n']}), "
                     f"new-double {r['new_double']['mean']:.3f} "
                     f"({r['new_double_functional_count']}/{r['n']}), "
                     f"new-double recombined {r['new_double_recombined']['mean']:.3f}, "
                     f"new consistency {r['fresh_sender_consistency_new_single']['mean']:.3f}")
    lines += ["", "## Paired contrasts", ""]
    for r in data["contrasts"]:
        c = r["new_single"]
        lines.append(f"- {r['left']} − {r['right']} / {r['support']}: "
                     f"old-combo {r['old_combo']['mean']:.3f}, "
                     f"new-single {c['mean']:.3f} [{c['ci95'][0]:.3f}, {c['ci95'][1]:.3f}], "
                     f"new-double {r['new_double']['mean']:.3f}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()
    data = analyse(json.loads(Path(args.results).read_text()))
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    Path(args.markdown).write_text(markdown(data))


if __name__ == "__main__":
    main()
