"""Private-need remapping used by the independent confirmation package.

The implementation is shared with the earlier rematching study after the
mapping has been checked against the multi-legal reward table.  This wrapper
keeps the new experiment's import surface and frozen source manifest explicit.
"""
from __future__ import annotations

import numpy as np

from research_program.triadic_rematched_partner_confirmation_study import remap as _base

PERMS = _base.PERMS
PLAN_MAPS = _base.PLAN_MAPS


def require(ok, message):
    if not ok:
        raise ValueError(message)


def permutation_indices(rng, batch_size):
    return _base.permutation_indices(rng, batch_size)


def remap_batch(x, rewards, packed_states, permutation_ids):
    """Remap FI features, all shared need blocks, and reward columns in lockstep.

    A permutation is defined as ``new physical actor -> old need owner``.
    Full-information observations therefore keep every need block, but move
    the block for old owner ``j`` into the block for new actor ``i``.  The
    public layout/owner map is shared by all viewers and stays fixed.
    """
    x = np.asarray(x, dtype=np.float64); rewards = np.asarray(rewards, dtype=np.float64)
    states = np.asarray(packed_states); pids = np.asarray(permutation_ids, dtype=np.int8)
    B = len(x)
    require(x.shape == (B, 3, 54) and rewards.shape == (B, 24) and states.shape == (B, 10), 'Invalid batch shapes')
    require(pids.shape == (B,) and np.all((pids >= 0) & (pids < len(PERMS))), 'Invalid permutation ids')
    out_x = x.copy(); out_states = states.copy(); out_rewards = np.empty_like(rewards)
    for pidx, perm in enumerate(PERMS):
        mask = pids == pidx
        if not np.any(mask):
            continue
        rows = np.flatnonzero(mask)
        shared_needs = x[rows, 0, :21].copy()
        shared_public = x[rows, 0, 21:50].copy()
        reordered_needs = np.concatenate(
            [shared_needs[:, 7 * old_actor:7 * old_actor + 7] for old_actor in perm], axis=1)
        for new_actor in range(3):
            out_x[rows, new_actor, :21] = reordered_needs
            out_x[rows, new_actor, 21:50] = shared_public
            out_x[rows, new_actor, 50:53] = 0.0
            out_x[rows, new_actor, 50 + new_actor] = 1.0
            out_x[rows, new_actor, 53] = 1.0
        out_states[rows, :3] = states[rows][:, list(perm)]
        out_rewards[rows] = rewards[rows][:, PLAN_MAPS[pidx]]
    require(out_x.shape == (len(out_x), 3, 54) and out_rewards.shape == (len(out_x), 24)
            and out_states.shape == (len(out_x), 10), 'Remap shape changed')
    require(np.isfinite(out_x).all() and np.isfinite(out_rewards).all(), 'Remap produced nonfinite values')
    return out_x, out_rewards, out_states


def remap_histogram(permutation_ids):
    return _base.remap_histogram(permutation_ids)
