"""Private-need remapping for paired static/rematched training."""
from __future__ import annotations

from itertools import combinations, permutations, product

import numpy as np

from research_program.triadic_message_study import runner as core

PERMS = tuple(permutations(range(3)))  # new physical actor -> old need owner
JOINT_ACTIONS = np.asarray(core.base.JOINT_ACTIONS, dtype=np.int64)
STRUCTURAL_PLANS = tuple((i, j, site, destination)
                         for i, j in combinations(range(3), 2)
                         for site, destination in product(range(4), range(2)))
PLAN_INDEX = {plan: index for index, plan in enumerate(STRUCTURAL_PLANS)}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def permutation_indices(rng, batch_size):
    require(batch_size > 0, 'Batch must be nonempty')
    return rng.integers(0, len(PERMS), size=batch_size, dtype=np.int8)


def _plan_map(perm):
    result = []
    for i, j, site, destination in STRUCTURAL_PLANS:
        old_pair = tuple(sorted((perm[i], perm[j])))
        result.append(PLAN_INDEX[(*old_pair, site, destination)])
    return np.asarray(result, dtype=np.int64)


PLAN_MAPS = tuple(_plan_map(perm) for perm in PERMS)


def remap_batch(x, rewards, packed_states, permutation_ids, information):
    """Assign each sampled need row to a new physical actor.

    PL rows retain only the new actor's own need block.  FI rows retain all
    needs but reorder the shared three-block representation into physical
    actor order.  Layout, owners, self identity and reward columns are kept
    in lockstep.
    """
    x = np.asarray(x, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    states = np.asarray(packed_states)
    pids = np.asarray(permutation_ids, dtype=np.int8)
    B = len(x)
    require(information in ('PL', 'FI'), 'Unknown information condition')
    require(x.shape == (B, 3, 54) and rewards.shape == (B, 24)
            and states.shape == (B, 10), 'Invalid batch shapes')
    require(pids.shape == (B,) and np.all((pids >= 0) & (pids < len(PERMS))),
            'Invalid permutation ids')
    out_x = x.copy()
    out_states = states.copy()
    out_rewards = np.empty_like(rewards)
    for pidx, perm in enumerate(PERMS):
        rows = np.flatnonzero(pids == pidx)
        if not len(rows):
            continue
        if information == 'PL':
            out_x[rows, :, :21] = 0.0
            out_x[rows, :, 50:53] = 0.0
            for new_actor, old_actor in enumerate(perm):
                out_x[rows, new_actor, 7 * new_actor:7 * new_actor + 7] = \
                    x[rows, old_actor, 7 * old_actor:7 * old_actor + 7]
                out_x[rows, new_actor, 21:50] = x[rows, old_actor, 21:50]
                out_x[rows, new_actor, 50 + new_actor] = 1.0
            require(np.all(out_x[rows, :, 53] == 0), 'PL flag changed during remap')
        else:
            shared_needs = x[rows, 0, :21].copy()
            reordered = np.concatenate(
                [shared_needs[:, 7 * old:7 * old + 7] for old in perm], axis=1)
            for new_actor in range(3):
                out_x[rows, new_actor, :21] = reordered
                out_x[rows, new_actor, 21:50] = x[rows, 0, 21:50]
                out_x[rows, new_actor, 50:53] = 0.0
                out_x[rows, new_actor, 50 + new_actor] = 1.0
                out_x[rows, new_actor, 53] = 1.0
        out_states[rows, :3] = states[rows][:, list(perm)]
        out_rewards[rows] = rewards[rows][:, PLAN_MAPS[pidx]]
    require(np.isfinite(out_x).all() and np.isfinite(out_rewards).all(),
            'Remap produced nonfinite values')
    return out_x, out_rewards, out_states


def remap_histogram(permutation_ids):
    ids = np.asarray(permutation_ids, dtype=np.int8)
    require(ids.ndim == 1 and np.all((ids >= 0) & (ids < len(PERMS))),
            'Invalid permutation ids')
    return np.bincount(ids, minlength=len(PERMS)).astype(np.int64).tolist()


if __name__ == '__main__':
    print({'permutations': [list(p) for p in PERMS],
           'plan_maps': [m.tolist() for m in PLAN_MAPS]})
