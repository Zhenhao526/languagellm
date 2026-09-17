"""Batch remapping of private needs to physical agents for rematched training."""
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


def remap_batch(x, rewards, packed_states, permutation_ids):
    """Assign each sampled old need row to a new physical actor.

    The public layout and private-site ownership stay fixed. PL feature blocks
    are rebuilt so only the new actor's own need is visible. Reward columns are
    permuted by the corresponding structural pair map.
    """
    x = np.asarray(x, dtype=np.float64); rewards = np.asarray(rewards, dtype=np.float64)
    states = np.asarray(packed_states); pids = np.asarray(permutation_ids, dtype=np.int8)
    B = len(x)
    require(x.shape == (B, 3, 54) and rewards.shape == (B, 24) and states.shape == (B, 10), 'Invalid batch shapes')
    require(pids.shape == (B,) and np.all((pids >= 0) & (pids < len(PERMS))), 'Invalid permutation ids')
    out_x = x.copy(); out_x[:, :, :21] = 0.; out_x[:, :, 50:53] = 0.
    out_states = states.copy(); out_rewards = np.empty_like(rewards)
    for pidx, perm in enumerate(PERMS):
        mask = pids == pidx
        if not np.any(mask):
            continue
        rows = np.flatnonzero(mask)
        for new_actor, old_actor in enumerate(perm):
            out_x[rows, new_actor, 7 * new_actor:7 * new_actor + 7] = x[rows, old_actor, 7 * old_actor:7 * old_actor + 7]
            out_x[rows, new_actor, 21:50] = x[rows, old_actor, 21:50]
            out_x[rows, new_actor, 50 + new_actor] = 1.
        out_states[rows, :3] = states[rows][:, list(perm)]
        out_rewards[rows] = rewards[rows][:, PLAN_MAPS[pidx]]
    return out_x, out_rewards, out_states


def remap_histogram(permutation_ids):
    ids = np.asarray(permutation_ids, dtype=np.int8)
    require(ids.ndim == 1 and np.all((ids >= 0) & (ids < len(PERMS))), 'Invalid permutation ids')
    return np.bincount(ids, minlength=len(PERMS)).astype(np.int64).tolist()


if __name__ == '__main__':
    print({'permutations': [list(p) for p in PERMS], 'plan_maps': [m.tolist() for m in PLAN_MAPS]})
