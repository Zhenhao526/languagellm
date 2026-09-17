"""Two collectors supply a shared camp with food and water each day.

Category IDs here belong to the simulator. A policy must receive image features
for its own two options, public inventory, and (optionally) a discrete message;
it must never receive ``private_kinds`` or any other simulator category ID.

There are 14 equally likely ordered scenes: all assignments of food/water to
four options except all-food and all-water. Every scene permits the pair to
collect one of each. Scenes are independent across steps and actions cannot
alter the scene sampler. Each step gathers first, caps storage, then consumes
one food and one water. Missing consumption and discarded overflow are resource
costs. Reward is ``1 - (shortage + overflow) / 2``; it is not a communication
bonus. The first-run default is capacity one and initially empty storage:
resources are consumed/discarded each day and cannot buffer later days. Thus
reward equals one for one-food/one-water collection, and zero otherwise.
Episodes end only at the fixed horizon, never at a resource shortage.

Larger storage capacities remain configurable for a later ecological control.
The earlier capacity-four, initial-(2,2) proposal admits a no-message policy
with perfect reward: each collector prefers the publicly less-stocked resource,
using opposite preferences at ties. Across all possible scenes, inventory stays
within (2,2), (3,1), and (1,3), with no shortage or overflow. See the exhaustive
audit in ``库存反例核查.json`` next to this file. That buffered condition cannot
demonstrate a communication benefit over optimal noncommunicating collectors.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np


FOOD = 0
WATER = 1
RESOURCE_NAMES = ("food", "water")
LOCAL_OPTIONS = np.asarray(list(itertools.product((FOOD, WATER), repeat=2)), dtype=np.int64)
FEASIBLE_SCENES = np.asarray(
    [
        scene
        for scene in itertools.product((FOOD, WATER), repeat=4)
        if len(set(scene)) == 2
    ],
    dtype=np.int64,
).reshape(14, 2, 2)
FEASIBLE_SCENES.flags.writeable = False
LOCAL_OPTIONS.flags.writeable = False


def _integers(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integers")
    return arr.astype(np.int64, copy=False)


def sample_scenes(rng: np.random.Generator, n: int | None = None) -> np.ndarray:
    """Return one ``(2, 2)`` scene or ``n`` scenes of shape ``(n, 2, 2)``."""
    if n is not None and (not isinstance(n, (int, np.integer)) or n < 0):
        raise ValueError("n must be a nonnegative integer or None")
    return FEASIBLE_SCENES[rng.integers(len(FEASIBLE_SCENES), size=n)].copy()


def transition(
    inventory: np.ndarray,
    private_kinds: np.ndarray,
    actions: np.ndarray,
    capacity: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Apply one step, either unbatched or with matching leading batch axes.

    Shapes are ``(..., 2)``, ``(..., 2, 2)``, and ``(..., 2)`` respectively.
    Actions are local option indices, not resource IDs. Returned arrays never
    alias the input arrays. Returns ``(next_inventory, reward, info)``.
    ``reward`` has the leading batch shape (scalar for
    an unbatched transition). ``inventory`` is the post-consumption inventory.
    ``consumed + shortage == [1, 1]`` and
    ``before + gathered == inventory + consumed + overflow``.
    Default capacity one leaves zero post-consumption inventory. Set capacity
    explicitly when studying a buffered condition.
    """
    if not isinstance(capacity, (int, np.integer)) or capacity < 1:
        raise ValueError("capacity must be a positive integer")
    before = _integers(inventory, "inventory")
    kinds = _integers(private_kinds, "private_kinds")
    chosen_actions = _integers(actions, "actions")
    if before.ndim < 1 or before.shape[-1] != 2:
        raise ValueError("inventory must have shape (..., 2)")
    if kinds.shape != before.shape[:-1] + (2, 2):
        raise ValueError("private_kinds must have matching shape (..., 2, 2)")
    if chosen_actions.shape != before.shape:
        raise ValueError("actions must have matching shape (..., 2)")
    if np.any((before < 0) | (before > capacity)):
        raise ValueError("inventory must lie between zero and capacity")
    if np.any((kinds < 0) | (kinds > 1)):
        raise ValueError("resource categories must be zero or one")
    if np.any((chosen_actions < 0) | (chosen_actions > 1)):
        raise ValueError("actions must be zero or one")
    selected = np.take_along_axis(kinds, chosen_actions[..., None], axis=-1)[..., 0]
    gathered = (selected[..., :, None] == np.arange(2)).sum(axis=-2)
    available = before + gathered
    overflow = np.maximum(available - capacity, 0)
    stored = np.minimum(available, capacity)
    consumed = np.minimum(stored, 1)
    shortage = 1 - consumed
    after = stored - consumed
    resource_cost = (shortage + overflow).sum(axis=-1)
    info = {
        "inventory_before": before.copy(),
        "actions": chosen_actions.copy(),
        "selected_kinds": selected.copy(),
        "gathered": gathered,
        "overflow": overflow,
        "stored_before_consumption": stored,
        "consumed": consumed,
        "shortage": shortage,
        "inventory": after,
        "resource_cost": resource_cost,
        "reward": 1.0 - resource_cost.astype(np.float64) / 2.0,
        "balanced_gathering": selected[..., 0] != selected[..., 1],
    }
    return after.copy(), info["reward"].copy(), info


class ResourceGatheringEnv:
    """Small NumPy environment; observation dictionaries are simulator-facing.

    ``reset(seed=...)`` returns inventory and both private scene labels so that
    the caller can render/sample photographs. ``observe_agent`` exposes only
    public numeric state and that agent's supplied image features. It accepts
    no partner image input. All policies use the same environment dynamics.
    """

    def __init__(
        self,
        seed: int | None = None,
        horizon: int = 16,
        initial_inventory: tuple[int, int] = (0, 0),
        capacity: int = 1,
    ) -> None:
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not isinstance(capacity, (int, np.integer)) or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        self.initial_inventory = _integers(initial_inventory, "initial_inventory").copy()
        if self.initial_inventory.shape != (2,):
            raise ValueError("initial_inventory must have shape (2,)")
        if np.any((self.initial_inventory < 0) | (self.initial_inventory > capacity)):
            raise ValueError("initial_inventory must lie between zero and capacity")
        self.capacity = int(capacity)
        self.horizon = int(horizon)
        self.rng = np.random.default_rng(seed)
        self.inventory = self.initial_inventory.copy()
        self.private_kinds = np.zeros((2, 2), dtype=np.int64)
        self.t = 0
        self._started = False

    def _observation(self) -> dict[str, Any]:
        return {
            "inventory": self.inventory.copy(),
            "private_kinds": self.private_kinds.copy(),
            "step": self.t,
            "remaining_steps": self.horizon - self.t,
        }

    def reset(
        self, seed: int | None = None, *, rng: np.random.Generator | None = None
    ) -> dict[str, Any]:
        if seed is not None and rng is not None:
            raise ValueError("pass seed or rng, not both")
        if rng is not None:
            self.rng = rng
        elif seed is not None:
            self.rng = np.random.default_rng(seed)
        self.inventory = self.initial_inventory.copy()
        self.private_kinds = sample_scenes(self.rng)
        self.t = 0
        self._started = True
        return self._observation()

    def observe_agent(self, agent_id: int, option_features: np.ndarray) -> dict[str, Any]:
        """Build a policy-safe observation from this agent's own image features.

        The caller supplies exactly its two private feature vectors. No category
        ID, partner option, or resource-consequence debug field is returned.
        Raw public inventory is returned; the policy may divide by capacity.
        """
        if not self._started:
            raise RuntimeError("call reset before observing")
        if agent_id not in (0, 1):
            raise ValueError("agent_id must be zero or one")
        features = np.asarray(option_features)
        if features.ndim != 2 or features.shape[0] != 2:
            raise ValueError("option_features must have shape (2, feature_dimension)")
        return {
            "inventory": self.inventory.copy(),
            "option_features": features.copy(),
            "remaining_steps": self.horizon - self.t,
        }

    def step(self, actions: np.ndarray) -> tuple[dict[str, Any], float, bool, dict[str, Any]]:
        if not self._started:
            raise RuntimeError("call reset before step")
        if self.t >= self.horizon:
            raise RuntimeError("episode is finished; call reset")
        next_inventory, reward, info = transition(
            self.inventory, self.private_kinds, actions, self.capacity
        )
        info["private_kinds"] = self.private_kinds.copy()
        self.inventory = next_inventory
        self.t += 1
        done = self.t == self.horizon
        if not done:
            self.private_kinds = sample_scenes(self.rng)
        info["step"] = self.t
        info["terminal"] = done
        return self._observation(), float(reward), done, info


def one_step_coordination_bounds() -> dict[str, float | int]:
    """Exact balanced-pick references and default unbuffered reward bounds.

    Enumerate every deterministic decentralized policy whose input is its own
    ordered categories. This gives an optimistic no-current-message bound:
    perfect visual categorization is granted. With IID scenes, common history
    or public randomness only mixes these policies and cannot improve 10/14.
    In the default capacity-one, initial-(0,0) environment, reward equals the
    balanced-pick indicator, so expected per-step reward is also bounded by 5/7.
    This implies an expected 16-step no-message return at most 16 * 5/7, with
    perfect local visual classification. Finite sampled returns may exceed it.
    With larger buffers public inventory changes the optimal gathering objective;
    10/14 must NOT be used as an upper bound on their inventory reward/survival.
    """
    local_indices = FEASIBLE_SCENES[..., 0] * 2 + FEASIBLE_SCENES[..., 1]
    best = 0
    for policy_a in itertools.product((0, 1), repeat=4):
        for policy_b in itertools.product((0, 1), repeat=4):
            acts = np.stack(
                (np.asarray(policy_a)[local_indices[:, 0]], np.asarray(policy_b)[local_indices[:, 1]]),
                axis=-1,
            )
            selected = np.take_along_axis(FEASIBLE_SCENES, acts[..., None], axis=-1)[..., 0]
            best = max(best, int(np.count_nonzero(selected[:, 0] != selected[:, 1])))
    random_success = 0
    for a, b in itertools.product((0, 1), repeat=2):
        random_success += int(np.count_nonzero(FEASIBLE_SCENES[:, 0, a] != FEASIBLE_SCENES[:, 1, b]))
    return {
        "number_of_scenes": len(FEASIBLE_SCENES),
        "uniform_random_balanced_probability": random_success / (4 * len(FEASIBLE_SCENES)),
        "best_decentralized_balanced_scenes": best,
        "best_decentralized_balanced_probability": best / len(FEASIBLE_SCENES),
        "full_information_balanced_probability": 1.0,
        "default_no_message_expected_reward_upper_bound": best / len(FEASIBLE_SCENES),
        "default_full_information_expected_reward": 1.0,
    }
