"""Aggregate parent/child turnover runs with paired transmission effects."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from . import design

T95 = {7: 2.365, 15: 2.131}


def ci(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) == 0: return [None, None]
    if len(x) == 1: return [float(x[0]), float(x[0])]
    t = T95.get(len(x) - 1, 1.96)
    h = t * float(x.std(ddof=1)) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def load(path):
    x = json.loads(Path(path).read_text())
    parents = {int(r["seed"]): r for r in x["parents"]}
    children = {(int(r["seed"]), r["condition"]): r for r in x["children"]}
    if len(parents) != len(x["parents"]) or len(children) != len(x["children"]): raise ValueError("duplicate run")
    return parents, children


def new_return(r, mode="natural", split="heldout"):
    return float(r["final"][split]["new_agent"][mode]["team_return_mean"])


def curve_auc(r):
    xs = np.asarray([x["update"] for x in r["learning_curve"]], dtype=float)
    ys = np.asarray([x["new_agent_natural"] for x in r["learning_curve"]], dtype=float)
    return float(np.trapezoid(ys, xs) / max(xs[-1], 1.0))


def row_for(r):
    split = "heldout"; h = r["final"][split]
    out = {"seed": int(r["seed"]), "condition": r["condition"], "role": r["role"],
           "natural": new_return(r), "closed": new_return(r, "closed"),
           "permuted": new_return(r, "permuted"), "scrambled": new_return(r, "scrambled"),
           "learning_curve_auc": curve_auc(r),
           "protocol_fidelity": r["protocol_fidelity"],
           "new_agent_semantic_success": h["codebook"]["new_agent_semantic_success"]}
    return out


def effects(parents, children):
    out = {}
    for role in design.ROLES:
        for label, a_channel, b_channel in (("live-minus-silent", "live", "silent"), ("live-minus-scrambled", "live", "scrambled")):
            vals = []
            seeds = sorted({seed for seed, cond in children if cond == f"{role}_{a_channel}"} & {seed for seed, cond in children if cond == f"{role}_{b_channel}"})
            for seed in seeds:
                a = children[(seed, f"{role}_{a_channel}")]; b = children[(seed, f"{role}_{b_channel}")]
                vals.append(new_return(a) - new_return(b))
            out[f"{role}|{label}"] = {"role": role, "label": label, "values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
        seeds = sorted({seed for seed, cond in children if cond == f"{role}_live"} & {seed for seed, cond in children if cond == f"{role}_silent"})
        vals = [curve_auc(children[(seed, f"{role}_live")]) - curve_auc(children[(seed, f"{role}_silent")]) for seed in seeds]
        out[f"{role}|curve-auc-live-minus-silent"] = {"role": role, "label": "curve-auc-live-minus-silent", "values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
        vals = []
        seeds = sorted({seed for seed, cond in children if cond == f"{role}_live"} & set(parents))
        for seed in seeds:
            r = children[(seed, f"{role}_live")]; parent = parents[seed]
            if role == "worker":
                base = parent["final"]["heldout"]["workers"][str(design.TARGET_WORKER)]["natural"]["team_return_mean"]
            else:
                base = float(np.mean([v["natural"]["team_return_mean"] for v in parent["final"]["heldout"]["workers"].values()]))
            vals.append(new_return(r) - float(base))
        out[f"{role}|child-minus-parent"] = {"role": role, "label": "child-minus-parent", "values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
    return out


def write_md(path, data):
    lines = ["# Generation transmission compact aggregation", "", f"- parent runs: {data['parent_runs']}", f"- child runs: {data['child_runs']}", "- split: heldout", "", "| condition | role | natural | silent | scrambled | live−silent | live−scrambled | learning AUC | child semantic |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for role in design.ROLES:
        for ch in design.CHANNELS:
            cond = f"{role}_{ch}"; rows = [r for r in data["rows"] if r["condition"] == cond]
            if not rows: continue
            natural = float(np.mean([r["natural"] for r in rows]));
            def mean_cond(c):
                q = [r["natural"] for r in data["rows"] if r["condition"] == c]
                return float(np.mean(q)) if q else float("nan")
            silent = mean_cond(f"{role}_silent"); scrambled = mean_cond(f"{role}_scrambled"); live = mean_cond(f"{role}_live")
            auc = float(np.mean([r["learning_curve_auc"] for r in rows])); sem = float(np.mean([r["new_agent_semantic_success"] for r in rows]))
            fmt = lambda x: f"{x:.3f}"
            ds = "NA" if not np.isfinite(live) or not np.isfinite(silent) else fmt(live - silent)
            dr = "NA" if not np.isfinite(live) or not np.isfinite(scrambled) else fmt(live - scrambled)
            lines.append(f"| `{cond}` | {role} | {fmt(natural)} | {fmt(silent)} | {fmt(scrambled)} | {ds} | {dr} | {fmt(auc)} | {fmt(sem)} |")
    lines += ["", "The paired effects compare the same seed's child run after replacing the same parent component. AUC is the trapezoidal heldout natural-return curve divided by the 3000-update horizon.", "", "## Paired effects", "", "| role | effect | mean | 95% t CI |", "|---|---|---:|---:|"]
    for key, e in data["effects"].items():
        if e["label"] in ("live-minus-silent", "live-minus-scrambled", "curve-auc-live-minus-silent", "child-minus-parent"):
            lines.append(f"| {e['role']} | {e['label']} | {e['mean']:.4f} | [{e['ci95_t'][0]:.4f}, {e['ci95_t'][1]:.4f}] |")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", required=True); ap.add_argument("--markdown", required=True)
    args = ap.parse_args(); parents, children = load(args.results)
    rows = [row_for(children[k]) for k in sorted(children)]
    data = {"schema": "generation_transmission_aggregate_v1", "parent_runs": len(parents), "child_runs": len(children), "rows": rows, "effects": effects(parents, children)}
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8"); write_md(args.markdown, data)
    print(json.dumps({"status": "written", "parent_runs": len(parents), "child_runs": len(children)}, ensure_ascii=False))


if __name__ == "__main__": main()
