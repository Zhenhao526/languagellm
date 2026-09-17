"""Select positions using all training natural messages, without validation outputs."""
from fractions import Fraction
import numpy as np

AXES = ('kind', 'length', 'destination')
TRAINING_WORLD_COUNT = 419904


def require(condition, message):
    if not condition:
        raise ValueError(message)


def packet_codes(packets):
    a = np.asarray(packets)
    require(a.ndim >= 2 and a.shape[-2:] == (2, 4) and a.dtype.kind in 'iu'
            and np.all((a >= 0) & (a < 8)), 'Expected legal two-window packets')
    flat = a.reshape(*a.shape[:-2], 8).astype(np.uint32)
    return (flat * (np.uint32(8) ** np.arange(8, dtype=np.uint32))).sum(-1, dtype=np.uint32)


def training_code_sets(messages):
    """All saved train endpoint worlds, not only rows referenced by discovery.

    The default is intentionally strict: a discovery-endpoint subset can omit a
    naturally emitted packet and would mislabel it as unseen during validation.
    """
    m = np.asarray(messages)
    require(m.shape == (TRAINING_WORLD_COUNT, 2, 3, 4),
            'Training code sets require all 419904 saved train endpoint worlds')
    return [np.unique(packet_codes(m[:, :, actor, :])) for actor in range(3)]


def _exact(value):
    return dict(numerator=value.numerator, denominator=value.denominator)


def selection_from_counts(change_counts, pair_counts):
    """Exact listener-stratified rates and specificity; floats are display only.

    change_counts[S,axis,L,position] and pair_counts[S,axis,L] are integers.
    Self-listener slots are unused zeroes; both other listeners must be present.
    Large-count pure fixtures can exercise near ties without allocating worlds.
    """
    change = np.asarray(change_counts)
    total = np.asarray(pair_counts)
    require(change.shape == (3, 3, 3, 8) and total.shape == (3, 3, 3)
            and change.dtype.kind in 'iu' and total.dtype.kind in 'iu',
            'Expected integer S×axis×L counts and S×axis×L×8 changes')
    require(np.all(total >= 0) and np.all(change >= 0)
            and np.all(change <= total[..., None]), 'Invalid count bounds')
    response = [[[Fraction(0) for _ in range(8)] for _ in range(3)] for _ in range(3)]
    for actor in range(3):
        require(np.all(total[actor, :, actor] == 0) and np.all(change[actor, :, actor] == 0),
                'Self-listener slots must be unused zeroes')
        for semantic in range(3):
            for hearer in range(3):
                if hearer == actor:
                    continue
                denominator = int(total[actor, semantic, hearer])
                require(denominator > 0, 'Missing discovery sender/listener/axis stratum')
                for position in range(8):
                    response[actor][semantic][position] += Fraction(
                        int(change[actor, semantic, hearer, position]), denominator) / 2
    scores, selected, maximizers = [], [], []
    for actor in range(3):
        actor_scores, actor_selected, actor_maximizers = [], [], []
        for semantic in range(3):
            values = [response[actor][semantic][position] - sum(
                (response[actor][other][position] for other in range(3) if other != semantic),
                Fraction(0)) / 2 for position in range(8)]
            maximum = max(values)
            tied = [position for position, value in enumerate(values) if value == maximum]
            actor_scores.append(values)
            actor_selected.append(tied[0])
            actor_maximizers.append(tied)
        scores.append(actor_scores)
        selected.append(actor_selected)
        maximizers.append(actor_maximizers)
    return dict(selected_positions=selected, maximizing_positions=maximizers,
                response_rates=[[[float(v) for v in row] for row in actor] for actor in response],
                specificity_scores=[[[float(v) for v in row] for row in actor] for actor in scores],
                exact_response_rates=[[[ _exact(v) for v in row] for row in actor] for actor in response],
                exact_specificity_scores=[[[ _exact(v) for v in row] for row in actor] for actor in scores],
                stratum_change_counts=change.tolist(), stratum_pair_counts=total.tolist(),
                stratum_index_order=['sender', 'axis', 'listener', 'position (changes only)'],
                axes=list(AXES), positions=[dict(index=i, window=i // 4, position=i % 4) for i in range(8)],
                arithmetic='Integer change counts; Fraction within each listener stratum, exact equal-listener mean, and exact specificity. Float fields are display only.',
                rule='Target-axis symbol-change rate minus the other two axis rates averaged; two listener strata equal. Exact rational argmax, smallest flattened index on a mathematical tie. No tolerance, positivity gate or forced distinct positions.',
                scope='Complete training content-pair/background support at the saved greedy endpoint; not a meaning assignment and not validation-based position selection.')


def select_positions(spec, messages):
    """Count changes on the complete discovery table, then use exact fractions.

    The supplied spec is the complete training content-pair/background table.
    The caller verifies partition identity/full natural-world count/source hashes.
    Small synthetic tables remain possible for pure arithmetic tests.
    """
    m = np.asarray(messages)
    require(m.ndim == 4 and m.shape[1:] == (2, 3, 4) and m.dtype.kind in 'iu'
            and np.all((m >= 0) & (m < 8)), 'Invalid training messages')
    endpoints = np.asarray(spec['endpoint_indices'])
    axis, sender, listener = (np.asarray(spec[k]) for k in ('axis', 'sender', 'listener'))
    require(endpoints.shape == (len(axis), 2) and endpoints.dtype.kind in 'iu'
            and np.all((endpoints >= 0) & (endpoints < len(m))), 'Invalid discovery endpoints')
    require(axis.ndim == 1 and sender.shape == listener.shape == axis.shape
            and all(v.dtype.kind in 'iu' and np.all((v >= 0) & (v < 3)) for v in (axis, sender, listener))
            and np.all(sender != listener), 'Invalid axis/actor indices')
    counts = np.zeros((3, 3, 3), dtype=np.int64)
    changes = np.zeros((3, 3, 3, 8), dtype=np.int64)
    for actor in range(3):
        for semantic in range(3):
            for hearer in range(3):
                if hearer == actor:
                    continue
                rows = np.flatnonzero((axis == semantic) & (sender == actor) & (listener == hearer))
                require(len(rows) > 0, 'Missing discovery sender/listener/axis stratum')
                pairs = endpoints[rows]
                differences = (m[pairs[:, 0], :, actor, :] != m[pairs[:, 1], :, actor, :]).reshape(-1, 8)
                changes[actor, semantic, hearer] = np.count_nonzero(differences, axis=0)
                counts[actor, semantic, hearer] = len(rows)
    return selection_from_counts(changes, counts)
