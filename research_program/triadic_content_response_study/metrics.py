"""Predeclared four-choice content margin and paired interaction."""
from __future__ import annotations

from itertools import product
import numpy as np

SEEDS = tuple(range(60101, 60117))
RULES = ("strict", "reciprocal")
LIVES = (False, True)
STEPS = (0, 100, 500, 1500, 3000, 6000)
CONDITIONS = tuple(rule + "_PL_" + ("live" if live else "silent") for rule in RULES for live in LIVES)
T15 = 2.1314495455597715
PRIMARY = "endpoint_four_choice_content_margin_rule_by_communication_interaction"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def statistics(values):
    x = np.asarray(values, dtype=np.float64)
    require(x.shape == (16,) and np.isfinite(x).all(), "Sixteen paired society values required")
    mean = float(x.mean()); sd = float(x.std(ddof=1)); se = sd / 4; half = T15 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15, t_critical=T15,
                interval_level=.95, ci95_lower=mean - half, ci95_upper=mean + half,
                interval_method="Approximate two-sided Student-t interval across16 paired initializations")


def probe_metrics(probabilities, group_index, background_index, host_endpoint, donor_endpoint,
                  listeners, candidate_receiver_actions):
    p = np.asarray(probabilities, dtype=np.float64)
    g = np.asarray(group_index, dtype=np.int64); b = np.asarray(background_index, dtype=np.int64)
    e = np.asarray(host_endpoint, dtype=np.int64); d = np.asarray(donor_endpoint, dtype=np.int64)
    listener = np.asarray(listeners, dtype=np.int64); c = np.asarray(candidate_receiver_actions, dtype=np.int64)
    n = len(p)
    require(p.shape == (n, 3, 17) and np.isfinite(p).all(), "Invalid all-action probabilities")
    require(g.shape == b.shape == e.shape == d.shape == listener.shape == (n,), "Probe row metadata shape")
    require(c.shape == (n, 4) and np.all((c >= 0) & (c < 17)), "Candidate action shape/domain")
    rows = np.arange(n)
    candidate_prob = p[rows[:, None], listener[:, None], c]
    target = candidate_prob[rows, d]
    others = candidate_prob.copy(); others[rows, d] = -np.inf
    margin = target - np.max(others, axis=1)
    require(np.isfinite(margin).all(), "Finite four-choice margins")
    G = int(g.max()) + 1; B = int(b.max()) + 1
    tensor = margin.reshape(G, B, 4, 4)
    off = ~np.eye(4, dtype=bool)
    cell_means = tensor[:, :, off].reshape(G, B, 12).mean(axis=-1)
    group_means = cell_means.mean(axis=1)
    layer_index = np.asarray(_layer_indices(G), dtype=np.int64)
    layer_means = np.asarray([group_means[layer_index == layer].mean() for layer in range(12)], dtype=np.float64)
    return dict(schema="content_response_margin_v1", worlds=int(n), groups=G, backgrounds=B,
                rows_per_group_background=16, off_diagonal_rows=int(np.sum(e != d)),
                margin_mean=float(margin.mean()), off_diagonal_margin_mean=float(margin[e != d].mean()),
                M=float(layer_means.mean()), group_means=group_means.tolist(), layer_means=layer_means.tolist(),
                layer_definition="six directed sender-listener pairs × kind/length; four groups per layer; equal layer weights",
                margin_definition="p_receiver(a_d|host e,packet d) − max_{v≠d} p_receiver(a_v|host e,packet d); only e≠d enters M",
                off_diagonal_cell_means=cell_means.tolist(), margins=margin.tolist())


def _layer_indices(group_count):
    # Static construction orders family, sender, listener, then third.  Rebuild
    # the researcher-side layer labels from the four-group records in runner.
    # The explicit labels are passed separately in primary; this helper is only
    # used for the fixed 48-group ordering.
    vals = []
    for family_index in range(4):
        family_layer = 0 if family_index < 2 else 1
        for sender in range(3):
            for listener in range(3):
                if sender != listener:
                    pair_index = sum(1 for a in range(3) for b in range(3)
                                     if a != b and (a, b) < (sender, listener))
                    vals.extend([pair_index * 2 + family_layer] * 2)
    # make_static appends all six directed pairs for each family and third;
    # the order above is not that append order, so runner overwrites this with
    # exact group-layer labels.  Retain a safe fallback for standalone use.
    if len(vals) == group_count:
        return vals
    return [i // 4 for i in range(group_count)]


def probe_metrics_with_layers(probabilities, group_index, background_index, host_endpoint, donor_endpoint,
                              listeners, candidate_receiver_actions, group_layer_index):
    result = probe_metrics(probabilities, group_index, background_index, host_endpoint, donor_endpoint,
                           listeners, candidate_receiver_actions)
    p = np.asarray(probabilities, dtype=np.float64); g = np.asarray(group_index, dtype=np.int64)
    b = np.asarray(background_index, dtype=np.int64); e = np.asarray(host_endpoint, dtype=np.int64)
    d = np.asarray(donor_endpoint, dtype=np.int64); l = np.asarray(listeners, dtype=np.int64)
    c = np.asarray(candidate_receiver_actions, dtype=np.int64); rows = np.arange(len(p))
    cp = p[rows[:, None], l[:, None], c]; m = cp[rows, d]; cp[rows, d] = -np.inf
    margins = m - cp.max(axis=1)
    G = int(g.max()) + 1; B = int(b.max()) + 1; tensor = margins.reshape(G, B, 4, 4)
    off = ~np.eye(4, dtype=bool); cell = tensor[:, :, off].reshape(G, B, 12).mean(-1); group = cell.mean(1)
    layers = np.asarray(group_layer_index, dtype=np.int64)
    layer_means = np.asarray([group[layers == i].mean() for i in range(12)], dtype=np.float64)
    result.update(M=float(layer_means.mean()), layer_means=layer_means.tolist(), group_means=group.tolist(),
                  off_diagonal_cell_means=cell.tolist(), margins=margins.tolist())
    return result


def primary(runs):
    require(len(runs) == 64, "Exactly 64 policy trajectories required")
    by = {(r["seed"], r["rule"], r["live"]): r for r in runs}
    require(set(by) == set(product(SEEDS, RULES, LIVES)), "Incomplete policy grid")
    rows = []
    for seed in SEEDS:
        cells = {}
        for rule, live in product(RULES, LIVES):
            run = by[seed, rule, live]
            require([x["update"] for x in run["trajectory"]] == list(STEPS), "Fixed six checkpoint order")
            values = [float(x["metrics"]["M"]) for x in run["trajectory"]]
            require(np.isfinite(values).all() if isinstance(values, np.ndarray) else all(np.isfinite(values)), "Finite M trajectory")
            cells[(rule, live)] = values
        endpoint = (cells["reciprocal", True][-1] - cells["reciprocal", False][-1]) - (cells["strict", True][-1] - cells["strict", False][-1])
        trajectories = {f"{r}_{'live' if l else 'silent'}": v for (r, l), v in cells.items()}
        rows.append(dict(seed=seed, cells=trajectories, endpoint_interaction=float(endpoint)))
    stats = statistics([r["endpoint_interaction"] for r in rows])
    return dict(name=PRIMARY, statistic="endpoint_M_interaction", independent_societies=16,
                policy_trajectories=64, checkpoints=list(STEPS), statistics=stats,
                mean_endpoint_interaction=stats["mean"], by_seed=rows,
                definition="[(M_Rlive−M_Rsilent)−(M_Slive−M_Ssilent)] at update6000; M equal-weights 12 semantic layers after within-group/background balancing",
                limits="Four candidate actions are researcher-side full-plan labels; positive M does not prove all17-action correctness, compositionality, a public lexicon, or language.")
