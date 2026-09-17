"""Compact, paired analysis for the object-surface remapping intervention."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from . import design


T_CRIT_95_DF8 = 2.306004  # t_(.975, 8), with nine paired seeds


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


def _results(payload, key):
    # The runner intentionally stores one paired payload.  Accepting a
    # `results` list as well keeps shard inspection convenient.
    if key in payload:
        return payload[key]
    if "results" in payload and key == "children":
        return payload["results"]
    raise KeyError(key)


def _metric(final, goal_kind, mode):
    return final[goal_kind][mode]["team_return_mean"]


def summarize(row):
    final = row["final"]
    natural = float(_metric(final, "heldout", "natural"))
    permuted = float(_metric(final, "heldout", "permuted"))
    silent = permuted if row["channel"] == "silent" else natural
    out = {
        "seed": int(row["seed"]),
        "condition": row["condition"],
        "representation": row["representation"],
        "form": row["form"],
        "task": row["task"],
        "mapping": row["mapping"],
        "support": row["support"],
        "channel": row["channel"],
        "heldout_goal": int(row["heldout_goal"]),
        "heldout_natural": natural,
        "heldout_silent": silent,
        "heldout_permuted": permuted,
        "all_natural": float(_metric(final, "all", "natural")),
        "seen_natural": float(_metric(final, "seen", "natural")),
        "live_minus_silent": natural - silent if row["channel"] == "live" else 0.0,
        "natural_minus_permuted": natural - permuted if row["channel"] == "live" else 0.0,
        "functional": bool(natural >= 0.60),
        "sender_codebook": final["sender_codebook"],
        "pairwise_min_hamming": int(final["pairwise_min_hamming"]),
        "sender_parameter_sha256": row["final_parameter_sha256"],
        "recombined": float(_metric(final, "heldout", "raw_recombined")) if "raw_recombined" in final["heldout"] else None,
    }
    return out


def _find(rows, **kwargs):
    for row in rows:
        if all(row.get(key) == value for key, value in kwargs.items()):
            return row
    return None


def _paired(rows, left_kwargs, right_kwargs, value="heldout_natural"):
    diffs = []
    for seed in sorted({row["seed"] for row in rows}):
        left = _find(rows, seed=seed, **left_kwargs)
        right = _find(rows, seed=seed, **right_kwargs)
        if left is not None and right is not None:
            diffs.append(float(left[value]) - float(right[value]))
    return {"n": len(diffs), "mean": float(np.mean(diffs)) if diffs else None, "ci95_t": ci(diffs), "values": diffs}


def analyze(path):
    payload = load(path)
    parents = _results(payload, "parents")
    children = _results(payload, "children")
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
    rows = [summarize(row) for row in children]

    groups = []
    for representation in design.REPRESENTATIONS:
        for form in design.FORMS:
            if representation == "slot_local" and form == "mono4":
                continue
            for mapping in design.MAPPINGS:
                for support in design.SUPPORTS:
                    for channel in design.CHANNELS:
                        values = [
                            row
                            for row in rows
                            if row["representation"] == representation
                            and row["form"] == form
                            and row["mapping"] == mapping
                            and row["support"] == support
                            and row["channel"] == channel
                        ]
                        if not values:
                            continue
                        natural = [row["heldout_natural"] for row in values]
                        recombined = [row["recombined"] for row in values if row["recombined"] is not None]
                        groups.append(
                            {
                                "representation": representation,
                                "form": form,
                                "mapping": mapping,
                                "support": support,
                                "channel": channel,
                                "n": len(values),
                                "heldout_natural_mean": float(np.mean(natural)),
                                "heldout_natural_ci95_t": ci(natural),
                                "all_natural_mean": float(np.mean([row["all_natural"] for row in values])),
                                "seen_natural_mean": float(np.mean([row["seen_natural"] for row in values])),
                                "live_minus_silent_mean": float(np.mean([row["live_minus_silent"] for row in values])),
                                "natural_minus_permuted_mean": float(np.mean([row["natural_minus_permuted"] for row in values])),
                                "recombined_mean": float(np.mean(recombined)) if recombined else None,
                                "functional_count": int(sum(row["functional"] for row in values)),
                                "pairwise_min_hamming_mean": float(np.mean([row["pairwise_min_hamming"] for row in values])),
                            }
                        )

    contrasts = []
    for representation in design.REPRESENTATIONS:
        for form in design.FORMS:
            if representation == "slot_local" and form == "mono4":
                continue
            for support in design.SUPPORTS:
                for channel in design.CHANNELS:
                    result = _paired(
                        rows,
                        {"representation": representation, "form": form, "mapping": "swap", "support": support, "channel": channel},
                        {"representation": representation, "form": form, "mapping": "identity", "support": support, "channel": channel},
                    )
                    if result["n"]:
                        contrasts.append({"name": "swap_minus_identity", "representation": representation, "form": form, "support": support, "channel": channel, **{k: v for k, v in result.items() if k != "values"}})

    for mapping in design.MAPPINGS:
        for support in design.SUPPORTS:
            for channel in design.CHANNELS:
                result = _paired(
                    rows,
                    {"representation": "joint_history", "form": "dual2", "mapping": mapping, "support": support, "channel": channel},
                    {"representation": "joint_history", "form": "mono4", "mapping": mapping, "support": support, "channel": channel},
                )
                if result["n"]:
                    contrasts.append({"name": "dual2_minus_mono4", "mapping": mapping, "support": support, "channel": channel, **{k: v for k, v in result.items() if k != "values"}})

    for mapping in design.MAPPINGS:
        for support in design.SUPPORTS:
            for channel in design.CHANNELS:
                result = _paired(
                    rows,
                    {"representation": "slot_local", "form": "dual2", "mapping": mapping, "support": support, "channel": channel},
                    {"representation": "joint_history", "form": "dual2", "mapping": mapping, "support": support, "channel": channel},
                )
                if result["n"]:
                    contrasts.append({"name": "slot_local_minus_joint_history", "mapping": mapping, "support": support, "channel": channel, **{k: v for k, v in result.items() if k != "values"}})

    for representation in design.REPRESENTATIONS:
        for form in design.FORMS:
            if representation == "slot_local" and form == "mono4":
                continue
            for mapping in design.MAPPINGS:
                for channel in design.CHANNELS:
                    result = _paired(
                        rows,
                        {"representation": representation, "form": form, "mapping": mapping, "support": "leave_one_out", "channel": channel},
                        {"representation": representation, "form": form, "mapping": mapping, "support": "full", "channel": channel},
                    )
                    if result["n"]:
                        contrasts.append({"name": "leave_one_out_minus_full", "representation": representation, "form": form, "mapping": mapping, "channel": channel, **{k: v for k, v in result.items() if k != "values"}})

    return {"schema": "object_surface_remap_analysis_v1", "rule": {"functional_natural_min": 0.60, "primary": "leave-one-out live heldout natural return"}, "parents": parent_rows, "children": rows, "groups": groups, "contrasts": contrasts}


def write_md(path, data):
    lines = [
        "# Object-surface remapping",
        "",
        "A parent learns on the identity mapping from visible surface label to semantic object type. A fresh worker receives the frozen sender codebook and learns under either the same mapping or a stable label swap. `dual2` and `mono4` have four complete message states; only `dual2` exposes two staged slots.",
        "",
        "## Held-out live endpoint",
        "",
        "| representation | form | mapping | support | channel | held-out natural | 95% CI | functional |",
        "|---|---|---|---|---|---:|---|---:|",
    ]
    for row in data["groups"]:
        lo, hi = row["heldout_natural_ci95_t"]
        lines.append(f"| `{row['representation']}` | `{row['form']}` | `{row['mapping']}` | `{row['support']}` | `{row['channel']}` | {row['heldout_natural_mean']:.3f} | [{lo:.3f}, {hi:.3f}] | {row['functional_count']}/{row['n']} |")
    lines += ["", "## Paired contrasts", "", "| contrast | factors | mean difference | 95% CI | n |", "|---|---|---:|---|---:|"]
    for row in data["contrasts"]:
        factors = ", ".join(f"{key}={value}" for key, value in row.items() if key in {"representation", "form", "mapping", "support", "channel"})
        lo, hi = row["ci95_t"]
        lines.append(f"| `{row['name']}` | {factors} | {row['mean']:.3f} | [{lo:.3f}, {hi:.3f}] | {row['n']} |")
    lines += ["", "`live−silent` and `natural−permuted` are communication checks. The swap intervention holds latent goals and semantic scenes fixed while changing only visible surface labels. This finite tabular study measures protocol transfer and local repair; it does not claim that the agents have human language."]
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
