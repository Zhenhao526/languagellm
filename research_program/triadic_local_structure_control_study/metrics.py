"""Predeclared metrics for local packet-structure controls."""
from __future__ import annotations

import numpy as np

SEEDS = tuple(range(60101, 60117))
RULES = ("strict", "reciprocal")
LIVES = (False, True)
STEPS = (0, 100, 500, 1500, 3000, 6000)
CONDITIONS = tuple(rule + "_PL_" + ("live" if live else "silent") for rule in RULES for live in LIVES)
MODES = ("single_slot_cycle", "rank_canonical", "equality_pattern_relabel")
T15 = 2.1314495455597715
PRIMARY = "local_structure_control_selectivity_ensemble"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (16,) and np.isfinite(x).all(), "Sixteen paired seed values")
    mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / 4.0; half = T15 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15,
                t_critical=T15, interval_level=.95, ci95_lower=mean-half, ci95_upper=mean+half)


def margins(probabilities, group, background, host, donor, listeners, candidates):
    p = np.asarray(probabilities, dtype=np.float64); group = np.asarray(group); background = np.asarray(background)
    host = np.asarray(host); donor = np.asarray(donor); listeners = np.asarray(listeners); candidates = np.asarray(candidates)
    n = len(p); ix = np.arange(n); cp = p[ix[:, None], listeners[:, None], candidates]
    target = cp[ix, donor]; cp[ix, donor] = -np.inf; margin = target - cp.max(axis=1)
    groups = int(group.max()) + 1; backgrounds = int(background.max()) + 1
    tensor = margin.reshape(groups, backgrounds, 4, 4); off = ~np.eye(4, dtype=bool)
    cells = tensor[:, :, off].reshape(groups, backgrounds, 12).mean(axis=-1); group_means = cells.mean(axis=-1)
    return dict(schema="content_response_margin_v1", M=float(group_means.mean()), margin_mean=float(margin.mean()),
                off_diagonal_margin_mean=float(margin[host != donor].mean()), group_means=group_means.tolist(),
                off_diagonal_cell_means=cells.tolist(), margins=margin.tolist())


def primary(records, baseline):
    by = {(r["seed"], r["condition"], r["mode"], r["update"]): r for r in records}
    result = {}
    for mode in MODES:
        values = [float(np.mean([float(baseline[seed, rule + "_PL_live", 6000]["M"]) -
                                  float(by[seed, rule + "_PL_live", mode, 6000]["metrics"]["M"])
                                  for rule in RULES])) for seed in SEEDS]
        st = statistics(values)
        result[mode] = dict(name=mode + "_selectivity", statistic="mean_seed_rule_averaged_same_minus_control_M",
                            independent_societies=16, statistics=st, mean_selectivity=st["mean"], by_seed=values,
                            definition="mean over strict/reciprocal of [same-group live M − local-control live M] at update6000; then mean over16 seeds",
                            limits="A positive value indicates sensitivity to this local control; it does not establish compositionality or a public lexicon.")
    values = [float(np.mean([result[mode]["by_seed"][i] for mode in MODES])) for i in range(16)]
    st = statistics(values)
    result["ensemble"] = dict(name=PRIMARY, statistic="mean_seed_rule_and_control_averaged_selectivity",
                               independent_societies=16, control_count=len(MODES), statistics=st,
                               mean_selectivity=st["mean"], by_seed=values,
                               definition="mean over three local controls and strict/reciprocal of [same-group live M − control live M] at update6000; then mean over16 seeds",
                               limits="The ensemble measures robustness to local packet controls; it does not establish compositionality or language origin.")
    return result
