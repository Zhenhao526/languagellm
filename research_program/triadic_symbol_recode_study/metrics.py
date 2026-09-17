"""Four-choice metrics and predeclared summaries for symbol recoding."""
from __future__ import annotations

import numpy as np

SEEDS = tuple(range(60101, 60117))
RULES = ("strict", "reciprocal")
LIVES = (False, True)
STEPS = (0, 100, 500, 1500, 3000, 6000)
CONDITIONS = tuple(rule + "_PL_" + ("live" if live else "silent") for rule in RULES for live in LIVES)
MODES = ("global_symbol_permutation", "position_rotation", "position_reverse")
T15 = 2.1314495455597715


def require(ok, message):
    if not ok:
        raise ValueError(message)


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (16,) and np.isfinite(x).all(), "Sixteen paired seed values")
    mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / 4; half = T15 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15, t_critical=T15,
                interval_level=.95, ci95_lower=mean - half, ci95_upper=mean + half)


def margins(p, group, background, host, donor, listeners, candidates):
    p = np.asarray(p, dtype=np.float64); group = np.asarray(group); background = np.asarray(background)
    host = np.asarray(host); donor = np.asarray(donor); listeners = np.asarray(listeners); candidates = np.asarray(candidates)
    n = len(p); ix = np.arange(n); cp = p[ix[:, None], listeners[:, None], candidates]
    target = cp[ix, donor]; cp[ix, donor] = -np.inf; margin = target - cp.max(axis=1)
    G = int(group.max()) + 1; B = int(background.max()) + 1; tensor = margin.reshape(G, B, 4, 4); off = ~np.eye(4, dtype=bool)
    cell = tensor[:, :, off].reshape(G, B, 12).mean(-1); group_mean = cell.mean(-1)
    return dict(schema="content_response_margin_v1", M=float(group_mean.mean()), margin_mean=float(margin.mean()),
                off_diagonal_margin_mean=float(margin[host != donor].mean()), group_means=group_mean.tolist(),
                off_diagonal_cell_means=cell.tolist(), margins=margin.tolist())


def primary(records, baseline):
    """Return per-mode same-group natural M minus recoded live M at 6000."""
    by = {(r["seed"], r["condition"], r["mode"], r["update"]): r for r in records}
    result = {}
    for mode in MODES:
        values = []
        for seed in SEEDS:
            rule_values = []
            for rule in RULES:
                same = float(baseline[seed, rule + "_PL_live", 6000]["M"])
                control = by[seed, rule + "_PL_live", mode, 6000]["metrics"]["M"]
                rule_values.append(same - control)
            values.append(float(np.mean(rule_values)))
        st = statistics(values)
        result[mode] = dict(name=mode + "_selectivity", statistic="mean_seed_rule_averaged_same_minus_recoded_M",
                            independent_societies=16, statistics=st, mean_selectivity=st["mean"], by_seed=values,
                            definition="mean over strict/reciprocal of [same-group live M − recoded live M] at update6000; then mean over16 seeds",
                            limits="A positive value indicates sensitivity to the recoding; it does not establish compositionality or a public lexicon.")
    return result
