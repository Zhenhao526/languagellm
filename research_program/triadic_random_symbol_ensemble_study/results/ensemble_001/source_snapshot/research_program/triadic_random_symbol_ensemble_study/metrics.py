"""Predeclared metrics for the random symbol-permutation ensemble."""
from __future__ import annotations

import numpy as np

SEEDS = tuple(range(60101, 60117))
RULES = ("strict", "reciprocal")
LIVES = (False, True)
STEPS = (0, 100, 500, 1500, 3000, 6000)
CONDITIONS = tuple(rule + "_PL_" + ("live" if live else "silent")
                   for rule in RULES for live in LIVES)
MODES = tuple(f"perm_{i:02d}" for i in range(6))
T15 = 2.1314495455597715
PRIMARY = "random_symbol_permutation_selectivity_ensemble"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (16,) and np.isfinite(x).all(), "Sixteen paired seed values")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / 4.0
    half = T15 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15,
                t_critical=T15, interval_level=.95,
                ci95_lower=mean - half, ci95_upper=mean + half)


def margins(probabilities, group, background, host, donor, listeners, candidates):
    """Four-choice margin M with equal group/background weighting."""
    p = np.asarray(probabilities, dtype=np.float64)
    group = np.asarray(group); background = np.asarray(background)
    host = np.asarray(host); donor = np.asarray(donor)
    listeners = np.asarray(listeners); candidates = np.asarray(candidates)
    n = len(p); ix = np.arange(n)
    cp = p[ix[:, None], listeners[:, None], candidates]
    target = cp[ix, donor]
    cp[ix, donor] = -np.inf
    margin = target - cp.max(axis=1)
    groups = int(group.max()) + 1; backgrounds = int(background.max()) + 1
    tensor = margin.reshape(groups, backgrounds, 4, 4)
    off = ~np.eye(4, dtype=bool)
    cells = tensor[:, :, off].reshape(groups, backgrounds, 12).mean(axis=-1)
    group_means = cells.mean(axis=-1)
    return dict(schema="content_response_margin_v1", M=float(group_means.mean()),
                margin_mean=float(margin.mean()),
                off_diagonal_margin_mean=float(margin[host != donor].mean()),
                group_means=group_means.tolist(),
                off_diagonal_cell_means=cells.tolist(), margins=margin.tolist())


def primary(records, baseline):
    """Natural live M minus each random-permutation live M at update 6000.

    Values are averaged over the two execution rules within each paired seed;
    the ensemble statistic then averages the six permutations within that seed.
    """
    by = {(r["seed"], r["condition"], r["mode"], r["update"]): r
          for r in records}
    result = {}
    per_mode = {}
    for mode in MODES:
        values = []
        for seed in SEEDS:
            values.append(float(np.mean([
                float(baseline[seed, rule + "_PL_live", 6000]["M"])
                - float(by[seed, rule + "_PL_live", mode, 6000]["metrics"]["M"])
                for rule in RULES])))
        st = statistics(values)
        per_mode[mode] = dict(
            name=mode + "_selectivity",
            statistic="mean_seed_rule_averaged_same_minus_permuted_M",
            independent_societies=16, statistics=st, mean_selectivity=st["mean"],
            by_seed=values,
            definition="mean over strict/reciprocal of [same-group live M − permuted live M] at update6000; then mean over16 seeds",
            limits="A positive value indicates sensitivity to this fixed permutation; it does not establish compositionality or a public lexicon.")
        result[mode] = per_mode[mode]

    ensemble_values = []
    for seed in SEEDS:
        ensemble_values.append(float(np.mean([
            per_mode[mode]["by_seed"][SEEDS.index(seed)] for mode in MODES])))
    st = statistics(ensemble_values)
    result["ensemble"] = dict(
        name=PRIMARY, statistic="mean_seed_rule_and_permutation_averaged_selectivity",
        independent_societies=16, permutation_count=len(MODES), statistics=st,
        mean_selectivity=st["mean"], by_seed=ensemble_values,
        definition="mean over six random symbol bijections and strict/reciprocal of [same-group live M − permuted live M] at update6000; then mean over16 seeds",
        limits="The ensemble measures robustness of code sensitivity across fixed bijections; it does not establish compositionality or language origin.")
    return result
