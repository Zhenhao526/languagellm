"""A three-agent collection world with an exact 25% no-message baseline.

A collects fiber, B collects fuel, and C observes both demands.  Each site
contains one item of each type; p_A/p_B independently swap its slot order.
Demand and position bits are sampled independently on every round.

This module deliberately contains no communication protocol or model code.
Only ``private_observation`` belongs in an agent's current observation.  World
fields, ``targetslots``, and the combined score are experimenter-side data.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from random import Random
from typing import Mapping


COLLECTORS = ("A", "B")
ROLES = (*COLLECTORS, "C")


def _require_bit(name: str, value: object) -> None:
    # Reject booleans and strings rather than silently accepting parsed output.
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"{name} must be the integer 0 or 1; got {value!r}")


@dataclass(frozen=True)
class World:
    """Four independent bits: demands d_a/d_b and position swaps p_a/p_b."""

    d_a: int
    d_b: int
    p_a: int
    p_b: int

    def __post_init__(self) -> None:
        for name in ("d_a", "d_b", "p_a", "p_b"):
            _require_bit(name, getattr(self, name))

    def private_observation(self, role: str) -> dict:
        """Return fresh data containing only this role's permitted information.

        Slot s at A contains F(s XOR p_a); B analogously contains G(s XOR p_b).
        C knows the two requested resource types but neither site's slot order.
        No current demand is disclosed to either collector.
        """
        if role == "C":
            return {
                "role": "C",
                "demand": {"fiber": f"F{self.d_a}", "fuel": f"G{self.d_b}"},
            }
        if role in COLLECTORS:
            prefix, position, resource_kind = (
                ("F", self.p_a, "fiber")
                if role == "A"
                else ("G", self.p_b, "fuel")
            )
            return {
                "role": role,
                "resource_kind": resource_kind,
                "slots": [
                    {"slot": slot, "resource": f"{prefix}{slot ^ position}"}
                    for slot in (0, 1)
                ],
            }
        raise ValueError(f"Unknown role {role!r}; expected one of {ROLES!r}")

    def targetslots(self) -> dict[str, int]:
        """Experimenter-only correct slot indices; never expose to collectors."""
        return {"A": self.d_a ^ self.p_a, "B": self.d_b ^ self.p_b}

    def score(self, actions: Mapping[str, int]) -> dict[str, bool]:
        """Score simultaneously committed actions, with strict action validation.

        The runner must distribute feedback separately: each collector may see
        its own correctness and overall success; C may see overall success.
        Passing this whole result to a collector would disclose partner feedback.
        """
        if set(actions) != set(COLLECTORS):
            raise ValueError("actions must contain exactly the keys 'A' and 'B'")
        for role in COLLECTORS:
            _require_bit(f"action {role}", actions[role])
        targets = self.targetslots()
        a_correct = actions["A"] == targets["A"]
        b_correct = actions["B"] == targets["B"]
        return {"A": a_correct, "B": b_correct, "overall": a_correct and b_correct}


def sample(rng: Random) -> World:
    """Sample all four bits iid; no rejection sampling or balanced blocks."""
    return World(*(rng.randrange(2) for _ in range(4)))


def enumerate_worlds() -> tuple[World, ...]:
    """All 16 equally likely states, in (d_a, d_b, p_a, p_b) lexical order."""
    return tuple(World(*bits) for bits in product((0, 1), repeat=4))
