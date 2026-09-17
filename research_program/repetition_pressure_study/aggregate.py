"""Compact analysis for emergent redundancy and noisy protocol repair."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import design


def ci(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        return [float(x.mean()), float(x.mean())] if len(x) else [None, None]
    h = 2.13145 * float(x.std(ddof=1)) / math.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def load(path):
    return json.loads(Path(path).read_text())


def summary(row):
    final = row["final"]
    new = final["new_worker"]["natural_at_train_noise"]["team_return_mean"]
    clean = final["new_worker"]["clean_natural"]["team_return_mean"]
    silent = final["new_worker"]["silent"]["team_return_mean"]
    incumbent = final["incumbent_workers_mean"]["natural_at_train_noise"]["team_return_mean"]
    permuted = final["new_worker"]["permuted_at_train_noise"]["team_return_mean"]
    return {
        "seed": int(row["seed"]),
        "condition": row["condition"],
        "form": row["form"],
        "adaptation": row["adaptation"],
        "noise_p": float(row["noise_p"]),
        "noise_key": row["noise_key"],
        "natural": float(new),
        "clean": float(clean),
        "silent": float(silent),
        "incumbent_natural": float(incumbent),
        "permuted": float(permuted),
        "live_minus_silent": float(new - silent),
        "noise_degradation": float(new - clean),
        "natural_minus_permuted": float(new - permuted),
        "pairwise_min_hamming": int(final["pairwise_min_hamming"]),
        "task_class_hamming": int(final["task_class_hamming"]),
        "within_class_hamming": int(final["within_class_hamming"]),
        "sender_codebook_distance_to_parent": row["sender_codebook_distance_to_parent"],
        "functional": bool(new >= 0.60),
        "error_correcting_candidate": bool(row["form"] == "triple2" and new >= 0.60 and final["task_class_hamming"] >= 2 and final["within_class_hamming"] == 0),
    }


def analyze(parents_path, results_path):
    parents = load(parents_path)["results"]
    children = load(results_path)["results"]
    rows = [summary(row) for row in children]
    parent_rows = [{"seed": int(row["seed"]), "form": row["form"], "natural": float(row["final"]["new_worker"]["natural_at_train_noise"]["team_return_mean"]), "pairwise_min_hamming": int(row["final"]["pairwise_min_hamming"]), "sender_codebook": row["final"]["sender_codebook"]} for row in parents]
    groups = []
    for form in design.FORMS:
        for adaptation in design.ADAPTATIONS:
            for noise_key, noise_p in design.NOISE_KEYS.items():
                values = [row for row in rows if row["form"] == form and row["adaptation"] == adaptation and row["noise_key"] == noise_key]
                groups.append({
                    "form": form,
                    "adaptation": adaptation,
                    "noise_key": noise_key,
                    "noise_p": float(noise_p),
                    "n": len(values),
                    "natural_mean": float(np.mean([row["natural"] for row in values])),
                    "natural_ci95_t": ci([row["natural"] for row in values]),
                    "clean_mean": float(np.mean([row["clean"] for row in values])),
                    "silent_mean": float(np.mean([row["silent"] for row in values])),
                    "live_minus_silent_mean": float(np.mean([row["live_minus_silent"] for row in values])),
                    "noise_degradation_mean": float(np.mean([row["noise_degradation"] for row in values])),
                    "incumbent_natural_mean": float(np.mean([row["incumbent_natural"] for row in values])),
                    "natural_minus_permuted_mean": float(np.mean([row["natural_minus_permuted"] for row in values])),
                    "min_hamming_mean": float(np.mean([row["pairwise_min_hamming"] for row in values])),
                    "functional_count": int(sum(row["functional"] for row in values)),
                    "error_correcting_candidate_count": int(sum(row["error_correcting_candidate"] for row in values)),
                })
    adaptation_effects = []
    for form in design.FORMS:
        for noise_key in design.NOISE_KEYS:
            pairs = []
            incumbent_pairs = []
            for seed in sorted({row["seed"] for row in rows}):
                coadapt = next((row for row in rows if row["seed"] == seed and row["form"] == form and row["noise_key"] == noise_key and row["adaptation"] == "coadapt"), None)
                worker_only = next((row for row in rows if row["seed"] == seed and row["form"] == form and row["noise_key"] == noise_key and row["adaptation"] == "worker_only"), None)
                if coadapt is None or worker_only is None:
                    continue
                pairs.append(coadapt["natural"] - worker_only["natural"])
                incumbent_pairs.append(coadapt["incumbent_natural"] - worker_only["incumbent_natural"])
            if not pairs:
                continue
            adaptation_effects.append({"form": form, "noise_key": noise_key, "coadapt_minus_worker_only_natural": float(np.mean(pairs)), "ci95_t": ci(pairs), "coadapt_minus_worker_only_incumbent": float(np.mean(incumbent_pairs)), "incumbent_ci95_t": ci(incumbent_pairs)})
    form_effects = []
    for noise_key in design.NOISE_KEYS:
        for adaptation in design.ADAPTATIONS:
            pairs = []
            for seed in sorted({row["seed"] for row in rows}):
                triple = next((row for row in rows if row["seed"] == seed and row["form"] == "triple2" and row["adaptation"] == adaptation and row["noise_key"] == noise_key), None)
                atomic = next((row for row in rows if row["seed"] == seed and row["form"] == "atomic8" and row["adaptation"] == adaptation and row["noise_key"] == noise_key), None)
                if triple is None or atomic is None:
                    continue
                pairs.append(triple["natural"] - atomic["natural"])
            if not pairs:
                continue
            form_effects.append({"noise_key": noise_key, "adaptation": adaptation, "triple2_minus_atomic8": float(np.mean(pairs)), "ci95_t": ci(pairs)})
    return {"schema": "shared_redundancy_analysis_v1", "rule": {"functional_natural_min": 0.60, "error_correcting_candidate": "triple2 natural >= 0.60, cross-parity Hamming distance >= 2 and within-parity Hamming distance = 0"}, "parents": parent_rows, "rows": rows, "groups": groups, "adaptation_effects": adaptation_effects, "form_effects": form_effects}


def write_md(path, data):
    groups = {(row["form"], row["adaptation"], row["noise_key"]): row for row in data["groups"]}
    lines = ["# Shared-parity redundancy pressure", "", "Both action stages require the same hidden parity. `triple2` and `atomic8` each have eight raw message states; `dual2` has four. The atomic corruption probability is matched to the probability that at least one of three binary coordinates flips.", "", "| form | adaptation | noise | natural | clean | silent | live−silent | class Hamming | within-class | functional | error-correcting candidate |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for form in design.FORMS:
        for adaptation in design.ADAPTATIONS:
            for noise_key in design.NOISE_KEYS:
                row = groups[(form, adaptation, noise_key)]
                lines.append(f"| `{form}` | `{adaptation}` | {row['noise_p']:.2f} | {row['natural_mean']:.3f} | {row['clean_mean']:.3f} | {row['silent_mean']:.3f} | {row['live_minus_silent_mean']:.3f} | {np.mean([v['task_class_hamming'] for v in data['rows'] if v['form']==form and v['adaptation']==adaptation and v['noise_key']==noise_key]):.2f} | {np.mean([v['within_class_hamming'] for v in data['rows'] if v['form']==form and v['adaptation']==adaptation and v['noise_key']==noise_key]):.2f} | {row['functional_count']}/{row['n']} | {row['error_correcting_candidate_count']}/{row['n']} |")
    lines += ["", "| form | noise | adaptation | triple2−atomic8 natural |", "|---|---:|---|---:|"]
    for row in data["form_effects"]:
        lines.append(f"| `triple2` vs `atomic8` | {design.NOISE_KEYS[row['noise_key']]:.2f} | `{row['adaptation']}` | {row['triple2_minus_atomic8']:.3f} [{row['ci95_t'][0]:.3f},{row['ci95_t'][1]:.3f}] |")
    lines += ["", "| form | noise | coadapt−worker_only new | coadapt−worker_only incumbent |", "|---|---:|---:|---:|"]
    for row in data["adaptation_effects"]:
        lines.append(f"| `{row['form']}` | {design.NOISE_KEYS[row['noise_key']]:.2f} | {row['coadapt_minus_worker_only_natural']:.3f} [{row['ci95_t'][0]:.3f},{row['ci95_t'][1]:.3f}] | {row['coadapt_minus_worker_only_incumbent']:.3f} [{row['incumbent_ci95_t'][0]:.3f},{row['incumbent_ci95_t'][1]:.3f}] |")
    lines += ["", "The analysis separates a code-space effect (`triple2` versus `atomic8`), a maintenance effect (`worker_only`) and a renegotiation effect (`coadapt`). It is a mechanism test, not a claim that any learned code is natural language."]
    Path(path).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--parents", required=True); parser.add_argument("--results", required=True); parser.add_argument("--out", required=True); parser.add_argument("--markdown", required=True); args = parser.parse_args()
    data = analyze(args.parents, args.results)
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    write_md(args.markdown, data)
    print(json.dumps({"status": "written", "parent_rows": len(data["parents"]), "rows": len(data["rows"]), "groups": len(data["groups"])}, ensure_ascii=False))
