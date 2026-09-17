"""Scientific-design and information-boundary checks; standard library only.

Run from the workspace root with:
    python3 -m unittest discover -s qwen_collect_pilot -p test_env.py -v
"""

from itertools import product
from random import Random
import unittest

try:
    from .env import World, enumerate_worlds, sample
except ImportError:
    from env import World, enumerate_worlds, sample


class WorldTests(unittest.TestCase):
    def test_exhaustive_world_space(self) -> None:
        worlds = enumerate_worlds()
        self.assertEqual(len(worlds), 16)
        self.assertEqual(len(set(worlds)), 16)
        for field in ("d_a", "d_b", "p_a", "p_b"):
            self.assertEqual(sum(getattr(world, field) for world in worlds), 8)

    def test_samples_use_four_fresh_independent_rng_draws(self) -> None:
        # A deterministic comparison checks the sampler, not a flaky frequency
        # threshold. Independence comes from four successive fair RNG draws.
        actual_rng, expected_rng = Random(20260914), Random(20260914)
        for _ in range(20):
            expected = World(*(expected_rng.randrange(2) for _ in range(4)))
            self.assertEqual(sample(actual_rng), expected)

    def test_role_observation_key_whitelists(self) -> None:
        for world in enumerate_worlds():
            for role in ("A", "B"):
                observation = world.private_observation(role)
                self.assertEqual(set(observation), {"role", "resource_kind", "slots"})
                self.assertEqual(observation["role"], role)
                self.assertEqual([item["slot"] for item in observation["slots"]], [0, 1])
                for item in observation["slots"]:
                    self.assertEqual(set(item), {"slot", "resource"})
                prefix = "F" if role == "A" else "G"
                self.assertEqual(
                    {item["resource"] for item in observation["slots"]},
                    {f"{prefix}0", f"{prefix}1"},
                )
            coordinator = world.private_observation("C")
            self.assertEqual(set(coordinator), {"role", "demand"})
            self.assertEqual(set(coordinator["demand"]), {"fiber", "fuel"})

    def test_collector_observations_do_not_depend_on_hidden_state(self) -> None:
        for own_position in (0, 1):
            reference_a = World(0, 0, own_position, 0).private_observation("A")
            reference_b = World(0, 0, 0, own_position).private_observation("B")
            for demand_a, demand_b, other_position in product((0, 1), repeat=3):
                self.assertEqual(
                    World(demand_a, demand_b, own_position, other_position)
                    .private_observation("A"),
                    reference_a,
                )
                self.assertEqual(
                    World(demand_a, demand_b, other_position, own_position)
                    .private_observation("B"),
                    reference_b,
                )

    def test_coordinator_observation_does_not_depend_on_positions(self) -> None:
        for demand_a, demand_b in product((0, 1), repeat=2):
            reference = World(demand_a, demand_b, 0, 0).private_observation("C")
            for position_a, position_b in product((0, 1), repeat=2):
                self.assertEqual(
                    World(demand_a, demand_b, position_a, position_b)
                    .private_observation("C"),
                    reference,
                )

    def test_complete_information_reaches_one_hundred_percent(self) -> None:
        for world in enumerate_worlds():
            actions = world.targetslots()
            self.assertEqual(world.score(actions), {"A": True, "B": True, "overall": True})
            for role, requested in (("A", f"F{world.d_a}"), ("B", f"G{world.d_b}")):
                selected = world.private_observation(role)["slots"][actions[role]]
                self.assertEqual(selected["resource"], requested)

    def test_every_no_communication_policy_scores_exactly_one_quarter(self) -> None:
        # The only variable local observation is the site's permutation bit.
        # A deterministic policy assigns a slot choice to each of its two values.
        # All 4 x 4 pairs achieve 4/16. Randomized and shared-randomness strategies
        # are mixtures of these policies and therefore have the same expectation.
        policies = tuple(product((0, 1), repeat=2))
        worlds = enumerate_worlds()
        for policy_a, policy_b in product(policies, repeat=2):
            successes = sum(
                world.score({"A": policy_a[world.p_a], "B": policy_b[world.p_b]})["overall"]
                for world in worlds
            )
            self.assertEqual(successes, 4, (policy_a, policy_b))

    def test_scores_require_both_collectors_to_be_correct(self) -> None:
        world = World(0, 1, 0, 0)
        self.assertEqual(world.score({"A": 0, "B": 0}), {"A": True, "B": False, "overall": False})
        self.assertEqual(world.score({"A": 1, "B": 1}), {"A": False, "B": True, "overall": False})
        self.assertEqual(world.score({"A": 1, "B": 0}), {"A": False, "B": False, "overall": False})

    def test_invalid_worlds_roles_and_actions_are_rejected(self) -> None:
        for invalid in (-1, 2, True, "0", 0.0, None):
            with self.assertRaises(ValueError):
                World(invalid, 0, 0, 0)
        world = World(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            world.private_observation("unknown")
        for actions in ({"A": 0}, {"A": 0, "B": 0, "C": 0}, {"A": "0", "B": 0}, {"A": True, "B": 0}):
            with self.assertRaises(ValueError):
                world.score(actions)


if __name__ == "__main__":
    unittest.main()
