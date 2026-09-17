"""Paired analysis for the incumbent-worker surface-remapping intervention."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import design


T_CRIT_95_DF8 = 2.306004  # nine paired seeds


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


def _metric(final, goal_kind, mode):
    return final[goal_kind][mode]["team_return_mean"]


def summarize(row):
    final = row["final"]
    initial = row.get("initial", {})
    heldout = float(_metric(final, "heldout", "natural"))
    initial_heldout = float(_metric(initial, "heldout", "natural")) if initial else None
    all_natural = float(_metric(final, "all", "natural"))
    all_permuted = float(_metric(final, "all", "permuted"))
    seen_natural = float(_metric(final, "seen", "natural"))
    seen_permuted = float(_metric(final, "seen", "permuted"))
    heldout_permuted = float(_metric(final, "heldout", "permuted"))
    silent_heldout = heldout_permuted if row["channel"] == "silent" else heldout
    return {
        "seed": int(row["seed"]),
        "condition": row["condition"],
        "initialization": row["initialization"],
        "representation": row["representation"],
        "form": row["form"],
        "task": row["task"],
        "mapping": row["mapping"],
        "support": row["support"],
        "channel": row["channel"],
        "heldout_goal": int(row["heldout_goal"]),
        "initial_heldout_natural": initial_heldout,
        "heldout_natural": heldout,
        "learning_gain": heldout - initial_heldout if initial_heldout is not None else None,
        "heldout_silent": silent_heldout,
        "heldout_permuted": heldout_permuted,
        "all_natural": all_natural,
        "all_permuted": all_permuted,
        "seen_natural": seen_natural,
        "seen_permuted": seen_permuted,
        "live_minus_silent": heldout - silent_heldout if row["channel"] == "live" else 0.0,
        "natural_minus_permuted": heldout - heldout_permuted if row["channel"] == "live" else 0.0,
        "all_natural_minus_permuted": all_natural - all_permuted if row["channel"] == "live" else 0.0,
        "seen_natural_minus_permuted": seen_natural - seen_permuted if row["channel"] == "live" else 0.0,
        "functional": bool(heldout >= 0.60),
        "sender_codebook": final["sender_codebook"],
        "pairwise_min_hamming": int(final["pairwise_min_hamming"]),
        "final_parameter_sha256": row["final_parameter_sha256"],
        "recombined": float(_metric(final, "heldout", "raw_recombined")) if "raw_recombined" in final["heldout"] else None,
    }


def _find(rows, **kwargs):
    for row in rows:
        if all(row.get(key) == value for key, value in kwargs.items()):
            return row
    return None


def _paired(rows, left_kwargs, right_kwargs, value):
    diffs = []
    seeds = sorted({row["seed"] for row in rows})
    for seed in seeds:
        left = _find(rows, seed=seed, **left_kwargs)
        right = _find(rows, seed=seed, **right_kwargs)
        if left is not None and right is not None:
            diffs.append(float(left[value]) - float(right[value]))
    return {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs else None, "ci95_t": ci(diffs), "values": diffs}


def analyze(path):
    payload = load(path)
    parents = payload["parents"]
    children = [summarize(row) for row in payload["children"]]
    parent_rows = []
    for row in parents:
        final = row["final"]
        parent_rows.append(
            {
                "seed": int(row["seed"]),
                "form": row["form"],
                "task": row["task"],
                "all_natural": float(_metric(final, "all", "natural")),
                "all_permuted": float(_metric(final, "all", "permuted")),
                "live_minus_permuted": float(_metric(final, "all", "natural") - _metric(final, "all", "permuted")),
                "heldout_natural": float(_metric(final, "heldout", "natural")),
                "sender_codebook": final["sender_codebook"],
                "pairwise_min_hamming": int(final["pairwise_min_hamming"]),
                "final_parameter_sha256": row["final_parameter_sha256"],
            }
        )

    groups = []
    for initialization in design.INITIALIZATIONS:
        for mapping in design.MAPPINGS:
            for channel in design.CHANNELS:
                values = [
                    row
                    for row in children
                    if row["initialization"] == initialization
                    and row["mapping"] == mapping
                    and row["channel"] == channel
                ]
                if not values:
                    continue
                initial = [row["initial_heldout_natural"] for row in values if row["initial_heldout_natural"] is not None]
                gains = [row["learning_gain"] for row in values if row["learning_gain"] is not None]
                groups.append(
                    {
                        "initialization": initialization,
                        "representation": "joint_history",
                        "form": "dual2",
                        "task": "factorized",
                        "mapping": mapping,
                        "support": "full",
                        "channel": channel,
                        "n": len(values),
                        "initial_heldout_natural_mean": float(np.mean(initial)) if initial else None,
                        "initial_heldout_natural_ci95_t": ci(initial),
                        "heldout_natural_mean": float(np.mean([row["heldout_natural"] for row in values])),
                        "heldout_natural_ci95_t": ci([row["heldout_natural"] for row in values]),
                        "learning_gain_mean": float(np.mean(gains)) if gains else None,
                        "learning_gain_ci95_t": ci(gains),
                        "all_natural_mean": float(np.mean([row["all_natural"] for row in values])),
                        "all_natural_minus_permuted_mean": float(np.mean([row["all_natural_minus_permuted"] for row in values])),
                        "seen_natural_minus_permuted_mean": float(np.mean([row["seen_natural_minus_permuted"] for row in values])),
                        "functional_count": int(sum(row["functional"] for row in values)),
                        "pairwise_min_hamming_mean": float(np.mean([row["pairwise_min_hamming"] for row in values])),
                    }
                )

    contrasts = []
    for initialization in design.INITIALIZATIONS:
        for channel in design.CHANNELS:
            for value, label in (("initial_heldout_natural", "initial_swap_minus_identity"), ("heldout_natural", "swap_minus_identity"), ("learning_gain", "gain_swap_minus_identity")):
                result = _paired(
                    children,
                    {"initialization": initialization, "mapping": "swap", "channel": channel},
                    {"initialization": initialization, "mapping": "identity", "channel": channel},
                    value,
                )
                if result["n"]:
                    contrasts.append({"name": label, "initialization": initialization, "channel": channel, **{k: v for k, v in result.items() if k != "values"}})
            result = _paired(
                children,
                {"initialization": initialization, "mapping": "identity", "channel": "live"},
                {"initialization": initialization, "mapping": "identity", "channel": "silent"},
                "all_natural",
            )
            if result["n"]:
                contrasts.append({"name": "live_minus_silent_all", "initialization": initialization, "mapping": "identity", **{k: v for k, v in result.items() if k != "values"}})
            result = _paired(
                children,
                {"initialization": initialization, "mapping": "swap", "channel": "live"},
                {"initialization": initialization, "mapping": "identity", "channel": "live"},
                "all_natural_minus_permuted",
            )
            if result["n"]:
                contrasts.append({"name": "communication_swap_minus_identity", "initialization": initialization, "channel": "live", **{k: v for k, v in result.items() if k != "values"}})
    for mapping in design.MAPPINGS:
        for channel in design.CHANNELS:
            for value, label in (("initial_heldout_natural", "incumbent_minus_fresh_initial"), ("heldout_natural", "incumbent_minus_fresh_final"), ("learning_gain", "incumbent_minus_fresh_gain")):
                result = _paired(
                    children,
                    {"initialization": "incumbent", "mapping": mapping, "channel": channel},
                    {"initialization": "fresh", "mapping": mapping, "channel": channel},
                    value,
                )
                if result["n"]:
                    contrasts.append({"name": label, "mapping": mapping, "channel": channel, **{k: v for k, v in result.items() if k != "values"}})

    return {
        "schema": "object_surface_transfer_analysis_v1",
        "rule": {"functional_natural_min": 0.60, "primary": "initial and final full-support live heldout return; repair gain"},
        "parents": parent_rows,
        "children": children,
        "groups": groups,
        "contrasts": contrasts,
    }


def write_md(path, data):
    lines = [
        "# Incumbent-worker surface transfer",
        "",
        "A parent learns a two-slot protocol on the identity object labels. The sender is then frozen. The `incumbent` child copies the parent worker for partner 0; the `fresh` child receives a new worker. Each child is evaluated under the original identity surface or a stable label swap, with live and silent communication controls.",
        "",
        "## Held-out transfer and repair",
        "",
        "| initialization | mapping | channel | initial | final | repair gain | 95% CI final | functional |",
        "|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in data["groups"]:
        initial = "—" if row["initial_heldout_natural_mean"] is None else f"{row['initial_heldout_natural_mean']:.3f}"
        gain = "—" if row["learning_gain_mean"] is None else f"{row['learning_gain_mean']:.3f}"
        lo, hi = row["heldout_natural_ci95_t"]
        lines.append(f"| `{row['initialization']}` | `{row['mapping']}` | `{row['channel']}` | {initial} | {row['heldout_natural_mean']:.3f} | {gain} | [{lo:.3f}, {hi:.3f}] | {row['functional_count']}/{row['n']} |")
    lines += [
        "",
        "## Paired contrasts",
        "",
        "| contrast | factors | mean difference | 95% CI | n |",
        "|---|---|---:|---|---:|",
    ]
    for row in data["contrasts"]:
        factors = ", ".join(f"{key}={value}" for key, value in row.items() if key in {"initialization", "mapping", "channel"})
        lo, hi = row["ci95_t"]
        lines.append(f"| `{row['name']}` | {factors} | {row['mean']:.3f} | [{lo:.3f}, {hi:.3f}] | {row['n']} |")
    lines += [
        "",
        "A positive `all_natural−permuted` value is the communication check: swapping messages across distinct goals changes the worker's behavior. The incumbent/swap initial contrast is the zero-shot cost of changing visible object labels after a protocol has been learned; the final contrast and repair gain measure reward-based local adaptation.",
    ]
    Path(path).write_text("\n".join(lines) + "\n")


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
