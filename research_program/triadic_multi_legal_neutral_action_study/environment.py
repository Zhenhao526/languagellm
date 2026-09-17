"""Settlement wrapper adding a symmetric explicit all-cancel outcome."""
from __future__ import annotations

import numpy as np

from research_program.triadic_reciprocal_execution_study import environment as strict_env

CANCEL_ACTION = 17


def require(ok, message):
    if not ok:
        raise ValueError(message)


def settle(states, actions, cancel_reward, rule='strict'):
    """Settle batches with wait=0, transport=1..16, cancel=17.

    A cancel is an explicit shared neutral commitment.  All three cancels
    yield ``cancel_reward`` and no physical execution.  Any mixed cancel
    blocks transport and yields zero.  Without cancels the audited strict
    settlement is called unchanged.
    """
    states = np.asarray(states)
    actions = np.asarray(actions)
    require(states.ndim == 2 and states.shape[1:] == (10,), 'States must be [N,10]')
    require(actions.shape == (len(states), 3) and actions.dtype.kind in 'iu'
            and np.all((actions >= 0) & (actions <= CANCEL_ACTION)), 'Actions must be in 0..17')
    cancel_reward = float(cancel_reward)
    require(np.isfinite(cancel_reward) and 0 <= cancel_reward <= 1, 'Invalid cancel reward')
    n = len(states)
    any_cancel = np.any(actions == CANCEL_ACTION, axis=1)
    all_cancel = np.all(actions == CANCEL_ACTION, axis=1)
    ordinary = ~any_cancel
    executed = np.zeros((n, 3), dtype=bool)
    satisfied = np.zeros((n, 3), dtype=bool)
    actual_pair = np.full(n, -1, dtype=np.int8)
    executed_site = np.full(n, -1, dtype=np.int8)
    executed_material = np.full(n, -1, dtype=np.int8)
    executed_destination = np.full(n, -1, dtype=np.int8)
    executed_roles = np.full((n, 3), -1, dtype=np.int8)
    ignored = np.zeros((n, 3), dtype=bool)
    unexecuted = (actions > 0) & (actions != CANCEL_ACTION)
    greedy_reward = np.zeros(n, dtype=np.float64)
    if ordinary.any():
        data = strict_env.settle(states[ordinary], actions[ordinary], rule)
        for key in ('executed', 'satisfied', 'actual_pair_index', 'executed_site', 'executed_material',
                    'executed_destination', 'executed_roles', 'ignored_proposal', 'unexecuted_proposal'):
            target = {'executed': executed, 'satisfied': satisfied, 'actual_pair_index': actual_pair,
                      'executed_site': executed_site, 'executed_material': executed_material,
                      'executed_destination': executed_destination, 'executed_roles': executed_roles,
                      'ignored_proposal': ignored, 'unexecuted_proposal': unexecuted}[key]
            target[ordinary] = data[key]
        greedy_reward[ordinary] = data['greedy_reward']
    greedy_reward[all_cancel] = cancel_reward
    cancel_count = (actions == CANCEL_ACTION).sum(axis=1).astype(np.int8)
    return dict(executed=executed, satisfied=satisfied, greedy_reward=greedy_reward,
                executed_roles=executed_roles, actual_pair_index=actual_pair,
                ignored_proposal=ignored, unexecuted_proposal=unexecuted,
                executed_site=executed_site, executed_material=executed_material,
                executed_destination=executed_destination, any_cancel=any_cancel,
                all_cancel=all_cancel, cancel_count=cancel_count,
                cancel_reward=float(cancel_reward), rule=rule)
