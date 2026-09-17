"""Read-only paired comparison of audited PL and FI alt-partner runs.

This is an exploratory analysis artifact.  It never loads checkpoints or runs
the model; it compares the completed JSON trajectories and endpoints for the
same seed, schedule, and live/silent cell.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

UPDATES = np.asarray([0, 100, 500, 1500, 3000, 6000], dtype=np.float64)
T15 = 2.1314495455597715
CELLS = (("static", False), ("static", True), ("rematched", False), ("rematched", True))
KEYS = ("target_pair_legal_rate", "physical_execution_rate", "q_rate",
        "conditional_q_rate", "proposal_legal_rate")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory(run, key):
    rows = sorted(run["trajectory"], key=lambda row: row["update"])
    require([row["update"] for row in rows] == UPDATES.astype(int).tolist(), "Trajectory checkpoints differ")
    return np.asarray([row["target_trajectory"][key] for row in rows], dtype=np.float64)


def summary(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (16,) and np.isfinite(x).all(), "Expected 16 finite paired values")
    mean = float(x.mean()); sd = float(x.std(ddof=1)); half = T15 * sd / math.sqrt(16)
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=sd / math.sqrt(16),
                ci95_lower=mean - half, ci95_upper=mean + half,
                positive_count=int((x > 0).sum()), zero_count=int((x == 0).sum()),
                negative_count=int((x < 0).sum()), t_critical=T15)


def main(pl_path, fi_path, output):
    pl_path = Path(pl_path).resolve(); fi_path = Path(fi_path).resolve(); output = Path(output).resolve()
    require(not output.exists(), "Never overwrite comparison output")
    pl_data = json.loads(pl_path.read_text()); fi_data = json.loads(fi_path.read_text())
    pl = {(r["seed"], r["schedule"], bool(r["live"])): r for r in pl_data["runs"]}
    fi = {(r["seed"], r["schedule"], bool(r["live"])): r for r in fi_data["runs"]}
    require(set(pl) == set(fi) and len(pl) == 64, "PL/FI grids are not the same 64 cells")
    seeds = sorted({key[0] for key in pl}); require(len(seeds) == 16, "Expected 16 paired seeds")
    for seed in seeds:
        initials = {pl[seed, schedule, live]["initial_parameter_sha256"]
                    for schedule, live in CELLS} | {fi[seed, schedule, live]["initial_parameter_sha256"]
                                                    for schedule, live in CELLS}
        require(len(initials) == 1, "Initial parameters are not paired across PL/FI")

    endpoint_differences = {}; auc_differences = {}; seed_rows = []
    for schedule, live in CELLS:
        label = f"{schedule}_{'live' if live else 'silent'}"
        endpoint_differences[label] = {}; auc_differences[label] = {}
        for key in KEYS:
            endpoint = []
            auc = []
            for seed in seeds:
                pr = pl[seed, schedule, live]; fr = fi[seed, schedule, live]
                p_end = float(pr["final"]["new_layouts"][key]); f_end = float(fr["final"]["new_layouts"][key])
                endpoint.append(f_end - p_end)
                auc.append(float(np.trapezoid(trajectory(fr, key) - trajectory(pr, key), UPDATES) / 6000.0))
            endpoint_differences[label][key] = summary(endpoint)
            auc_differences[label][key] = summary(auc)

    primary_by_seed = []
    for seed in seeds:
        p_cells = {cell: trajectory(pl[seed, cell[0], cell[1]], "target_pair_legal_rate") for cell in CELLS}
        f_cells = {cell: trajectory(fi[seed, cell[0], cell[1]], "target_pair_legal_rate") for cell in CELLS}
        p_int = (p_cells["rematched", True] - p_cells["rematched", False]) - (p_cells["static", True] - p_cells["static", False])
        f_int = (f_cells["rematched", True] - f_cells["rematched", False]) - (f_cells["static", True] - f_cells["static", False])
        p_auc = float(np.trapezoid(p_int - p_int[0], UPDATES) / 6000.0)
        f_auc = float(np.trapezoid(f_int - f_int[0], UPDATES) / 6000.0)
        primary_by_seed.append(dict(seed=seed, fi_minus_pl_primary_centered_AUC=f_auc - p_auc,
                                    fi_primary_centered_AUC=f_auc, pl_primary_centered_AUC=p_auc))
    primary = summary([row["fi_minus_pl_primary_centered_AUC"] for row in primary_by_seed])
    result = dict(status="completed_read_only_comparison", comparison="FI_minus_PL_same_seed_schedule_channel",
                  pl_source=str(pl_path), fi_source=str(fi_path), input_sha256={"pl": sha(pl_path), "fi": sha(fi_path)},
                  seeds=seeds, cells=[f"{s}_{'live' if live else 'silent'}" for s, live in CELLS],
                  endpoint_differences=endpoint_differences, centered_time_auc_differences=auc_differences,
                  primary_interaction_difference=primary, primary_by_seed=primary_by_seed,
                  no_model_calls=True, no_optimizer_updates=True,
                  interpretation="Exploratory paired information-layer comparison; not the preregistered within-information primary.")
    output.mkdir(parents=False, exist_ok=False)
    path = output / "comparison.json"; path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "receipt.json").write_text(json.dumps(dict(status="passed", output_sha256=sha(path), model_forwards=0,
                                                          optimizer_updates=0, no_model_calls=True), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(dict(status="completed_read_only_comparison", output=str(output), primary=primary), ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pl", required=True); parser.add_argument("--fi", required=True); parser.add_argument("--output", required=True)
    args = parser.parse_args(); main(args.pl, args.fi, args.output)
