"""Aggregate population runs and recover partner-level codebook diagnostics."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from . import design, runner

T95 = {7: 2.365, 15: 2.131}


def ci(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) == 0: return [None, None]
    if len(x) == 1: return [float(x[0]), float(x[0])]
    t = T95.get(len(x) - 1, 1.96)
    h = t * float(x.std(ddof=1)) / np.sqrt(len(x))
    return [float(x.mean() - h), float(x.mean() + h)]


def load(path):
    payload = json.loads(Path(path).read_text()); out = {}
    for r in payload["results"]:
        key = (int(r["seed"]), r["condition"])
        if key in out: raise ValueError(f"duplicate {key}")
        out[key] = r
    return out


def worker_rows(by_key, split="heldout"):
    rows = []
    for (seed, condition), r in sorted(by_key.items()):
        worker_ids = ["0"] if r["partner_mode"] == "fixed" else sorted(r["final"][split]["workers"].keys(), key=int)
        for worker in worker_ids:
            modes = r["final"][split]["workers"][worker]
            for mode in ("natural", "closed", "permuted"):
                x = modes[mode]
                rows.append({"seed": seed, "condition": condition, "worker": int(worker), "mode": mode,
                             "return": x["team_return_mean"], "oracle": x.get("oracle_team_return_mean"),
                             "normalized": x.get("normalized_return_mean_on_oracle_positive"),
                             "mi": x.get("message_goal_mi"), "episodes": x.get("episodes")})
    return rows


def codebook_rows(by_key):
    rows = []
    for (seed, condition), r in sorted(by_key.items()):
        pm, pv, ch, sc = design.parse_condition(condition)
        run = Path(r.get("_run_dir", ""))
        if not run.is_dir():
            continue
        cp = run / f"checkpoint_{r['updates']:04d}.npz"
        if not cp.is_file():
            continue
        with np.load(cp) as z:
            p = {"sender_logits_hidden": z["sender_logits_hidden"], "sender_logits_visible": z["sender_logits_visible"], "worker_logits": z["worker_logits"]}
        c = runner.codebook(p, pm, pv, sc)
        rows.append({"seed": seed, "condition": condition, **c})
    return rows


def summarize(rows):
    g = {}
    for row in rows:
        g.setdefault((row["condition"], row["worker"], row["mode"]), []).append(row)
    out = {}
    for key, group in sorted(g.items()):
        x = np.array([a["return"] for a in group], dtype=float)
        out["|".join(map(str, key))] = {"condition": key[0], "worker": key[1], "mode": key[2], "n": len(x),
            "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0., "ci95_t": ci(x)}
    return out


def effects(by_key, split="heldout"):
    out = {}
    def run_return(r, worker, mode):
        x = r["final"][split]["workers"][str(worker)][mode]["team_return_mean"]
        return float(x) if x is not None else np.nan
    def cell_effect(cond, mode_a, mode_b, label):
        vals = []
        for seed in sorted({s for s, c in by_key if c == cond}):
            r = by_key[(seed, cond)]
            workers = [0] if r["partner_mode"] == "fixed" else list(range(design.WORKERS))
            a = np.nanmean([run_return(r, w, mode_a) for w in workers]); b = np.nanmean([run_return(r, w, mode_b) for w in workers])
            vals.append(float(a - b))
        out[f"{cond}|{label}"] = {"condition": cond, "label": label, "values": vals, "mean": float(np.nanmean(vals)), "ci95_t": ci(vals)}
    for cond in sorted({c for _, c in by_key}):
        r = by_key[next(k for k in by_key if k[1] == cond)]
        if r["channel"] == "live":
            cell_effect(cond, "natural", "closed", "natural-minus-closed")
            cell_effect(cond, "natural", "permuted", "natural-minus-permuted")
            silent = cond.replace("_live_", "_silent_")
            if silent in {c for _, c in by_key}:
                vals = []
                for seed in sorted({s for s, c in by_key if c == cond} & {s for s, c in by_key if c == silent}):
                    a = by_key[(seed, cond)]; b = by_key[(seed, silent)]
                    wa = [0] if a["partner_mode"] == "fixed" else list(range(design.WORKERS))
                    wb = [0] if b["partner_mode"] == "fixed" else list(range(design.WORKERS))
                    vals.append(float(np.nanmean([run_return(a, w, "natural") for w in wa]) - np.nanmean([run_return(b, w, "natural") for w in wb])))
                out[f"{cond}|natural-minus-silent"] = {"condition": cond, "label": "natural-minus-silent", "values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
    # Fixed-to-rotating contrasts use the trained worker 0 in fixed and the
    # mean across independent workers in rotating, preserving the per-seed pair.
    for pv in design.PARTNER_VISIBILITY:
        for ch in design.CHANNELS:
            for sc in design.SCARCITIES:
                a = f"rotating_{pv}_{ch}_{sc}"; b = f"fixed_{pv}_{ch}_{sc}"
                if all(any(c == x for _, c in by_key) for x in (a, b)):
                    vals = []
                    for seed in sorted({s for s, c in by_key if c == a} & {s for s, c in by_key if c == b}):
                        ra, rb = by_key[(seed, a)], by_key[(seed, b)]
                        va = np.nanmean([run_return(ra, w, "natural") for w in range(design.WORKERS)])
                        vb = run_return(rb, 0, "natural")
                        vals.append(float(va - vb))
                    out[f"{a}-minus-{b}|natural"] = {"label": "rotating-minus-fixed", "values": vals, "mean": float(np.mean(vals)), "ci95_t": ci(vals)}
    return out


def write_md(path, data):
    lines = ["# Population signaling compact aggregation", "", f"- runs: {data['runs']}", "- split: heldout", "", "| condition | active workers | natural | closed | permuted | Δ live−silent |", "|---|---:|---:|---:|---:|---:|"]
    conds = sorted({x["condition"] for x in data["rows"]})
    for cond in conds:
        rr = [x for x in data["rows"] if x["condition"] == cond and x["mode"] == "natural"]
        mode = {m: [x["return"] for x in data["rows"] if x["condition"] == cond and x["mode"] == m and (x["worker"] == 0 or cond.startswith("rotating_"))] for m in ("natural", "closed", "permuted")}
        eff = data["effects"].get(f"{cond}|natural-minus-silent", {}).get("mean")
        fmt = lambda x: "NA" if x is None else f"{x:.3f}"
        lines.append(f"| `{cond}` | {len(rr)} | {fmt(np.nanmean(mode['natural']))} | {fmt(np.nanmean(mode['closed']))} | {fmt(np.nanmean(mode['permuted']))} | {fmt(eff)} |")
    lines += ["", "Natural-minus-closed and natural-minus-permuted are within-run intervention effects; natural-minus-silent compares a live run with a from-scratch silent run. The partner alignment fields are descriptive readouts from final checkpoints.", ""]
    lines += ["## Codebook readout", "", "| condition | sender-token agreement | semantic success mean | semantic success minimum |", "|---|---:|---:|---:|"]
    for cond in conds:
        cc = [x for x in data["codebook"] if x["condition"] == cond]
        if not cc:
            continue
        fmt = lambda x: "NA" if x is None else f"{x:.3f}"
        lines.append(f"| `{cond}` | {fmt(np.mean([x['sender_token_agreement'] for x in cc]))} | {fmt(np.mean([x['semantic_success_mean'] for x in cc]))} | {fmt(np.mean([x['semantic_success_min'] for x in cc]))} |")
    Path(path).write_text("\n".join(lines), encoding="utf8")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--execution", required=True); ap.add_argument("--out", required=True); ap.add_argument("--markdown", required=True)
    args = ap.parse_args(); by_key = load(args.results)
    for r in by_key.values(): r["_run_dir"] = str(Path(args.execution) / f"seed_{r['seed']}_{r['condition']}")
    rows = worker_rows(by_key); codebooks = codebook_rows(by_key)
    codebook_summary = {}
    for cond in sorted({x["condition"] for x in codebooks}):
        g = [x for x in codebooks if x["condition"] == cond]
        codebook_summary[cond] = {"n": len(g),
            "sender_token_agreement_mean": float(np.mean([x["sender_token_agreement"] for x in g])),
            "sender_token_agreement_ci95_t": ci([x["sender_token_agreement"] for x in g]),
            "semantic_success_mean": float(np.mean([x["semantic_success_mean"] for x in g])),
            "semantic_success_ci95_t": ci([x["semantic_success_mean"] for x in g]),
            "semantic_success_min_mean": float(np.mean([x["semantic_success_min"] for x in g]))}
    data = {"schema": "population_signaling_aggregate_v1", "runs": len(by_key), "rows": rows, "summary": summarize(rows), "effects": effects(by_key), "codebook": codebooks, "codebook_summary": codebook_summary}
    Path(args.out).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf8"); write_md(args.markdown, data)
    print(json.dumps({"status": "written", "runs": len(by_key), "codebook_rows": len(data["codebook"])}, ensure_ascii=False))


if __name__ == "__main__": main()
