"""Frozen estimands for the factorial semantic-generalization study."""
from itertools import product
import numpy as np

SEEDS = tuple(range(62101, 62117))
REGIMES = ('factorial_holdout', 'saturated')
RULES = ('strict', 'reciprocal')
LIVES = (False, True)
UPDATES = (0, 100, 500, 1500, 3000, 6000)
HORIZON = 6000
PARTS = ('train', 'heldout_resource', 'heldout_layout', 'heldout_both')
TARGET = 'heldout_resource'
GROUP = 'heldout_changed_actor'
CONTROL_GROUP = 'seen_changed_actor'
PRIMARY = 'mean_regime_interaction_centered_native_Q_time_AUC'
T15_975 = 2.1314495455597715
MEASURES = {'native_Q': ('native', 'Q'), 'native_Q_excess': ('native', 'Q_excess'),
            'common_reciprocal_Q': ('common_reciprocal', 'Q'),
            'common_reciprocal_Q_excess': ('common_reciprocal', 'Q_excess')}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def society_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.shape == (16,) and np.isfinite(values).all(), 'Exactly16 finite society values required')
    mean = float(values.mean()); sd = float(values.std(ddof=1)); se = sd / 4
    half = T15_975 * se
    return dict(n=16, mean=mean, sample_sd=sd, standard_error=se, df=15,
                t_critical=T15_975, interval_level=.95, ci95_lower=mean-half,
                ci95_upper=mean+half,
                interval_method='Approximate two-sided Student-t interval across16 paired initializations; not a world-level interval or equivalence proof.')


def regime_interaction(cells):
    """Four live/silent cells in two regimes, then six-point centered AUC."""
    expected = {(regime, live) for regime in REGIMES for live in LIVES}
    require(set(cells) == expected, 'Complete four regime/channel curves required')
    curves = {key: np.asarray(value, dtype=np.float64) for key, value in cells.items()}
    require(all(v.shape == (6,) and np.isfinite(v).all() for v in curves.values()), 'Six finite points required')
    factorial_gain = curves['factorial_holdout', True] - curves['factorial_holdout', False]
    saturated_gain = curves['saturated', True] - curves['saturated', False]
    interaction = factorial_gain - saturated_gain
    centered = interaction - interaction[0]
    widths = np.diff(np.asarray(UPDATES, dtype=np.float64))
    integral = lambda x: float(np.sum(widths * (x[:-1] + x[1:]) * .5) / HORIZON)
    return dict(factorial_live_minus_silent=factorial_gain.tolist(),
                saturated_live_minus_silent=saturated_gain.tolist(),
                interaction=interaction.tolist(), centered_interaction=centered.tolist(),
                baseline_interaction=float(interaction[0]), raw_AUC=integral(interaction),
                centered_AUC=integral(centered), endpoint_interaction=float(interaction[-1]),
                endpoint_centered_interaction=float(centered[-1]))


def primary(runs, target=TARGET, group=GROUP):
    require(isinstance(runs, list) and len(runs) == 128, 'Exactly128 complete training runs required')
    by = {}
    for run in runs:
        key = (run['seed'], run['regime'], run['rule'], run['live'])
        require(key not in by and run['seed'] in SEEDS and run['regime'] in REGIMES and
                run['rule'] in RULES and type(run['live']) is bool, 'Unexpected or duplicate run identity')
        require([row['update'] for row in run['trajectory']] == list(UPDATES), 'Six checkpoints required')
        by[key] = run
    require(set(by) == {(s, g, r, l) for s in SEEDS for g in REGIMES for r in RULES for l in LIVES},
            'Incomplete128-cell grid')
    rows = []
    for seed in SEEDS:
        rule_rows = {}
        for rule in RULES:
            cells = {}
            for regime, live in product(REGIMES, LIVES):
                run = by[seed, regime, rule, live]
                values = [row['evaluation']['factor_response'][group]['Q'] for row in run['trajectory']]
                require(all(np.isfinite(v) and 0 <= v <= 1 for v in values), 'Invalid held-out response trajectory')
                cells[regime, live] = values
            rule_rows[rule] = regime_interaction(cells)
        averaged = float(np.mean([rule_rows[r]['centered_AUC'] for r in RULES]))
        rows.append(dict(seed=seed, by_rule=rule_rows, rule_mean_centered_AUC=averaged))
    stats = society_statistics([row['rule_mean_centered_AUC'] for row in rows])
    return dict(name=PRIMARY, target=target, edge_group=group, independent_societies=16,
                training_runs=128, checkpoints=list(UPDATES), horizon_updates=HORIZON,
                primary_measure='heldout_changed_actor_native_Q', primary_statistic='rule_mean_centered_AUC',
                statistics=stats, mean_centered_AUC=stats['mean'], by_seed=rows,
                rule_statistics={rule: society_statistics([row['by_rule'][rule]['centered_AUC'] for row in rows]) for rule in RULES},
                definition='For each seed and rule, [(factorial live−silent)−(saturated live−silent)] on held-out-resource changed-actor Q; subtract step0, trapezoid-integrate over six checkpoints /6000, then average strict and reciprocal and the16 seeds.',
                unit='Probability difference averaged over training time; multiply by100 for percentage points.',
                scope='The edge group is a researcher-side label for unseen resource predicates. This is a behavioral semantic-response estimand, not a language score or proof of compositionality.')


def _messages(messages, spec):
    raw = np.asarray(messages); n = int(spec['world_count']); b = len(spec['layouts']) * len(spec['private_sites']); needs = len(spec['needs'])
    require(spec.get('state_order') == 'need-major, then layout, then owner' and n == needs * b,
            'Complete need-major Cartesian specification required')
    require(raw.shape == (n, 2, 3, 4) and raw.dtype.kind in 'iu' and ((raw >= 0) & (raw < 8)).all(),
            'Complete legal packet panel required')
    codes = np.sum(raw.astype(np.int64, copy=False) * np.asarray([512, 64, 8, 1]), axis=-1)
    return raw, codes, needs, b


def adjusted_rand_index(left, right):
    left = np.asarray(left); right = np.asarray(right)
    require(left.ndim == 1 and left.shape == right.shape and len(left) > 0 and
            left.dtype.kind in 'iu' and right.dtype.kind in 'iu', 'ARI vectors')
    _, a, ca = np.unique(left, return_inverse=True, return_counts=True)
    _, b, cb = np.unique(right, return_inverse=True, return_counts=True)
    _, joint = np.unique(a.astype(np.int64) * len(cb) + b, return_counts=True)
    pairs = lambda counts: int(np.sum(counts * (counts - 1) // 2, dtype=np.int64))
    total = len(left) * (len(left) - 1) // 2; aa = pairs(ca); bb = pairs(cb); ab = pairs(joint)
    denominator = total * (aa + bb) - 2 * aa * bb
    return 1.0 if denominator == 0 else float(2 * (total * ab - aa * bb) / denominator)


def _entropy(codes):
    _, counts = np.unique(codes, return_counts=True); p = counts.astype(float) / len(codes)
    h = float(-np.sum(p * np.log2(p)))
    if h == 0: h = 0.0
    return dict(entropy_bits=h, effective_packet_count=float(2 ** h), observed_packet_count=len(counts), constant_code=len(counts) == 1)


def message_snapshot(messages, spec):
    _, codes, needs, b = _messages(messages, spec); owners = len(spec['private_sites']); panels = []
    for actor, window in product(range(3), range(2)):
        values = codes[:, window, actor].reshape(needs, b); backgrounds = []
        for background in range(b):
            backgrounds.append(dict(background_index=background, layout_index=background // owners,
                                    owner_index=background % owners, need_worlds=needs,
                                    **_entropy(values[:, background])))
        conditional = float(np.mean([row['entropy_bits'] for row in backgrounds]))
        panels.append(dict(actor=actor, window=window, conditional_entropy_bits=conditional,
                           conditional_effective_packet_count=float(2 ** conditional), backgrounds=backgrounds,
                           global_observed_packet_count=int(len(np.unique(values)))))
    return dict(schema='natural_message_snapshot_v1', worlds=spec['world_count'], need_worlds=needs,
                background_count=b, panel_order='actor,window', panels=panels,
                definition='Four-position base8 packet; H(M|public background) is equal-background mean Shannon entropy.',
                scope='Packet form only; entropy and ARI do not identify meaning or compositionality.')


def message_transition(before, after, spec):
    first, codes_a, needs, b = _messages(before, spec); last, codes_b, _, _ = _messages(after, spec)
    owners = len(spec['private_sites']); panels = []
    for actor, window in product(range(3), range(2)):
        a = codes_a[:, window, actor]; z = codes_b[:, window, actor]; abg = a.reshape(needs, b); zbg = z.reshape(needs, b); backgrounds = []
        for background in range(b):
            left, right = abg[:, background], zbg[:, background]; le, re = _entropy(left), _entropy(right)
            backgrounds.append(dict(background_index=background, layout_index=background // owners,
                                    owner_index=background % owners, ARI=adjusted_rand_index(left, right),
                                    observed_packets_before=le['observed_packet_count'], observed_packets_after=re['observed_packet_count'],
                                    constant_before=le['constant_code'], constant_after=re['constant_code'],
                                    entropy_bits_before=le['entropy_bits'], entropy_bits_after=re['entropy_bits']))
        panels.append(dict(actor=actor, window=window, packet_equal_rate=float(np.mean(a == z)),
                           token_hamming_fraction=float(np.mean(first[:, window, actor] != last[:, window, actor])),
                           global_ARI=adjusted_rand_index(a, z),
                           mean_background_ARI=float(np.mean([r['ARI'] for r in backgrounds])),
                           global_observed_packets_before=int(len(np.unique(a))), global_observed_packets_after=int(len(np.unique(z))),
                           global_constant_before=len(np.unique(a)) == 1, global_constant_after=len(np.unique(z)) == 1,
                           conditional_entropy_bits_before=float(np.mean([r['entropy_bits_before'] for r in backgrounds])),
                           conditional_entropy_bits_after=float(np.mean([r['entropy_bits_after'] for r in backgrounds])), backgrounds=backgrounds))
    return dict(schema='adjacent_natural_message_transition_v1', worlds=spec['world_count'], need_worlds=needs,
                background_count=b, panel_order='actor,window', panels=panels,
                definition='Adjacent natural packet partitions; no semantic relabeling.')

