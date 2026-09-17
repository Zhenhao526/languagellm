"""Need remapping wrapper with an explicit local import surface."""
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
    out_x, out_rewards, out_states = _base.remap_batch(x, rewards, packed_states, permutation_ids)
    require(out_x.shape == (len(out_x), 3, 54) and out_rewards.shape == (len(out_x), 24)
            and out_states.shape == (len(out_x), 10), 'Remap shape changed')
    require(np.isfinite(out_x).all() and np.isfinite(out_rewards).all(), 'Remap produced nonfinite values')
    return out_x, out_rewards, out_states
