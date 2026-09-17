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
    """Remap PL features, need columns and reward columns in lockstep."""
    result = _base.remap_batch(x, rewards, packed_states, permutation_ids)
    out_x, out_rewards, out_states = result
    require(out_x.shape == (len(out_x), 3, 54) and out_rewards.shape == (len(out_x), 24)
            and out_states.shape == (len(out_x), 10), 'Remap shape changed')
    require(np.isfinite(out_x).all() and np.isfinite(out_rewards).all(), 'Remap produced nonfinite values')
    return result


def remap_histogram(permutation_ids):
    return _base.remap_histogram(permutation_ids)
