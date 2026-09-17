"""Researcher-side edge groups for the factorial resource holdout.

The learner never receives these labels.  They are used only after a complete
ordered evaluation to separate responses involving an unseen object/length
conjunction from the ordinary one-attribute response measure.
"""
from itertools import product
import numpy as np

AXES = ('kind', 'length', 'destination')
STATE_ORDER = 'need-major, then layout, then owner'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def resource_id(need):
    return int(need) // 3


def heldout_edge_indices(cases, needs, heldout_resources=(5, 6)):
    """Return equal-stratum edge groups touching a held-out changed-agent need.

    `cases` is produced by the frozen single-need response builder.  The first
    group contains every edge for which the changed person's before or after
    resource predicate is held out.  The complement is retained as a useful
    within-test control.  Both groups are represented in the canonical nine
    person×axis strata; empty strata stay explicit and are never dropped.
    """
    ns = [tuple(map(int, row)) for row in needs]
    edges = [tuple(map(int, row)) for row in cases['edge_need_indices']]
    changed = list(map(int, cases['changed_person']))
    axis = list(map(int, cases['axis_index']))
    require(len(edges) == len(changed) == len(axis), 'Mismatched response edge arrays')
    held = set(map(int, heldout_resources))
    groups = {'heldout_changed_actor': [[] for _ in range(9)],
              'seen_changed_actor': [[] for _ in range(9)]}
    for edge_index, ((before, after), who, ax) in enumerate(zip(edges, changed, axis)):
        resource_before = resource_id(ns[before][who])
        resource_after = resource_id(ns[after][who])
        key = 'heldout_changed_actor' if resource_before in held or resource_after in held else 'seen_changed_actor'
        groups[key][who * 3 + ax].append(edge_index)
    # Training-only partitions can legitimately contain no held-out resource
    # endpoints (the factorial train arm is one such partition).  The target
    # partitions are checked for nonempty groups by the static preparation.
    return {key: value for key, value in groups.items()}


def subset_metrics(cases, executed_pair_indices, edge_indices):
    """Equal-nine-stratum Q and analytic shuffle for one edge subset."""
    values = np.asarray(executed_pair_indices)
    require(values.shape == (cases['world_count'],) and values.dtype.kind in 'iu',
            'Full executed-pair vector required')
    require(((values >= -1) & (values <= 2)).all(), 'Executed-pair domain')
    n = int(cases['need_worlds']); bcount = int(cases['n_backgrounds'])
    domains = values.astype(np.int8, copy=False).reshape(n, bcount)
    all_edge_count = len(cases['edge_need_indices'])
    global_counts = []
    for b in range(bcount):
        counts = np.bincount(domains[:, b] + 1, minlength=4)[1:]
        global_counts.append(counts)
    global_counts = np.asarray(global_counts, dtype=np.int64)
    strata = []
    totals = {'both_correct': 0, 'edge_count': 0}
    for stratum, ids in enumerate(edge_indices):
        ids = np.asarray(ids, dtype=np.int64)
        require(ids.ndim == 1 and len(ids) > 0 and ((ids >= 0) & (ids < all_edge_count)).all(),
                'Nonempty valid edge subset required')
        edge = np.asarray(cases['edge_need_indices'], dtype=np.int64)[ids]
        truth = np.asarray(cases['target_pairs'], dtype=np.int8)[ids]
        qs = []; shuffles = []; both = 0
        for b in range(bcount):
            actual = domains[edge, b]
            correct = actual[:, None] == truth
            hits = correct.all(axis=1)
            q = float(hits.mean())
            counts = global_counts[b]
            numerator = int(np.sum(counts[truth[:, 0]] * counts[truth[:, 1]], dtype=np.int64))
            denominator = n * (n - 1) * len(ids)
            shuffle = numerator / denominator
            qs.append(q); shuffles.append(shuffle); both += int(hits.sum())
        row = dict(changed_person=stratum // 3, axis_index=stratum % 3,
                   edge_count=int(len(ids)), state_edge_count=int(len(ids) * bcount),
                   Q=float(np.mean(qs)), Q_shuffle=float(np.mean(shuffles)),
                   Q_excess=float(np.mean(qs) - np.mean(shuffles)),
                   raw_both_correct=both)
        strata.append(row)
        totals['both_correct'] += both; totals['edge_count'] += len(ids) * bcount
    q = float(np.mean([row['Q'] for row in strata]))
    shuffle = float(np.mean([row['Q_shuffle'] for row in strata]))
    return dict(schema='factorial_edge_subset_metrics_v1', group_worlds=cases['world_count'],
                complete_nine_strata=True, strata=strata, Q=q, Q_shuffle=shuffle,
                Q_excess=q - shuffle, raw_counts=totals,
                weighting='Equal nine changed-person×axis strata; within each stratum equal edges and backgrounds.',
                scope='Edges are researcher-labelled after evaluation. Held-out group means the changed actor has resource 5 or 6 at one endpoint; no label enters observations or training.')
