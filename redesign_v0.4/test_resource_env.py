"""Consequence and information-boundary checks for the resource environment."""

import itertools
import unittest

import numpy as np

from resource_env import FEASIBLE_SCENES, ResourceGatheringEnv, one_step_coordination_bounds, sample_scenes, transition


class ResourceEnvironmentTests(unittest.TestCase):
    def test_each_scene_permits_complementary_collection(self):
        self.assertEqual(FEASIBLE_SCENES.shape, (14, 2, 2))
        self.assertEqual(len({tuple(s.flat) for s in FEASIBLE_SCENES}), 14)
        for scene in FEASIBLE_SCENES:
            self.assertTrue(any(scene[0, a] != scene[1, b] for a, b in itertools.product(range(2), repeat=2)))

    def test_exact_resource_accounting_every_inventory_scene_and_action(self):
        cases = list(itertools.product(itertools.product(range(5), repeat=2), range(14), itertools.product(range(2), repeat=2)))
        stock = np.asarray([x[0] for x in cases])
        kinds = FEASIBLE_SCENES[[x[1] for x in cases]]
        actions = np.asarray([x[2] for x in cases])
        next_inventory, reward, result = transition(stock, kinds, actions, capacity=4)
        np.testing.assert_array_equal(next_inventory, result["inventory"])
        np.testing.assert_array_equal(reward, result["reward"])
        np.testing.assert_array_equal(result["gathered"].sum(-1), 2)
        np.testing.assert_array_equal(stock + result["gathered"], result["inventory"] + result["consumed"] + result["overflow"])
        np.testing.assert_array_equal(result["consumed"] + result["shortage"], np.ones_like(stock))
        self.assertTrue(np.all((result["inventory"] >= 0) & (result["inventory"] <= 3)))
        np.testing.assert_allclose(result["reward"], 1 - (result["shortage"] + result["overflow"]).sum(-1) / 2)

    def test_choices_have_native_resource_consequences_for_either_agent(self):
        kinds = np.asarray([[0, 1], [0, 1]])
        baseline = transition([0, 0], kinds, [0, 0], capacity=4)[2]
        np.testing.assert_array_equal(baseline["shortage"], [0, 1])
        np.testing.assert_array_equal(baseline["inventory"], [1, 0])
        for actions in ([1, 0], [0, 1]):
            result = transition([0, 0], kinds, actions, capacity=4)[2]
            np.testing.assert_array_equal(result["shortage"], [0, 0])
            np.testing.assert_array_equal(result["inventory"], [0, 0])
            self.assertGreater(result["reward"], baseline["reward"])

    def test_overflow_is_discarded_before_consumption(self):
        result = transition([4, 0], [[0, 1], [0, 1]], [0, 0], capacity=4)[2]
        np.testing.assert_array_equal(result["overflow"], [2, 0])
        np.testing.assert_array_equal(result["shortage"], [0, 1])
        np.testing.assert_array_equal(result["inventory"], [3, 0])
        self.assertEqual(float(result["reward"]), -0.5)

    def test_full_information_keeps_camp_supplied(self):
        env = ResourceGatheringEnv(seed=401, horizon=100)
        obs = env.reset()
        for step in range(100):
            kinds = obs["private_kinds"]
            actions = next((a, b) for a, b in itertools.product(range(2), repeat=2) if kinds[0, a] != kinds[1, b])
            obs, reward, done, info = env.step(actions)
            self.assertEqual(reward, 1)
            np.testing.assert_array_equal(obs["inventory"], [0, 0])
            self.assertEqual(done, step == 99)
        with self.assertRaises(RuntimeError):
            env.step([0, 0])

    def test_shortages_do_not_terminate_and_scene_rng_is_action_independent(self):
        a, b = ResourceGatheringEnv(seed=7, horizon=16, initial_inventory=(0, 0)), ResourceGatheringEnv(seed=7, horizon=16)
        oa, ob = a.reset(), b.reset()
        for t in range(16):
            np.testing.assert_array_equal(oa["private_kinds"], ob["private_kinds"])
            oa, _, da, _ = a.step([0, 0])
            ob, _, db, _ = b.step([1, 1])
            self.assertEqual(da, t == 15)
            self.assertEqual(db, t == 15)

    def test_observation_copies_and_policy_input_contains_no_category_ids(self):
        env = ResourceGatheringEnv(seed=7)
        obs = env.reset()
        original = env.private_kinds.copy()
        obs["private_kinds"][:] = 9
        np.testing.assert_array_equal(env.private_kinds, original)
        features = np.zeros((2, 32), dtype=np.float32)
        safe = env.observe_agent(0, features)
        self.assertEqual(set(safe), {"inventory", "option_features", "remaining_steps"})
        safe["option_features"][:] = 5
        self.assertTrue(np.all(features == 0))

    def test_fixed_roles_are_imperfect_and_exact_reference_bounds(self):
        successes = 0
        for scene in FEASIBLE_SCENES:
            # Agent 0 prefers food; agent 1 prefers water, falling back locally.
            actions = [int(np.argmax(scene[0] == 0)), int(np.argmax(scene[1] == 1))]
            successes += scene[0, actions[0]] != scene[1, actions[1]]
        self.assertEqual(successes, 10)
        bounds = one_step_coordination_bounds()
        self.assertEqual(bounds["best_decentralized_balanced_scenes"], 10)
        self.assertAlmostEqual(bounds["uniform_random_balanced_probability"], 4 / 7)
        self.assertAlmostEqual(bounds["best_decentralized_balanced_probability"], 5 / 7)
        self.assertAlmostEqual(bounds["default_no_message_expected_reward_upper_bound"], 5 / 7)
        self.assertEqual(bounds["default_full_information_expected_reward"], 1)

    def test_default_perishable_resources_make_reward_equal_balanced_collection(self):
        for actions in itertools.product(range(2), repeat=2):
            after, rewards, info = transition(
                np.zeros((14, 2), dtype=np.int64),
                FEASIBLE_SCENES,
                np.tile(actions, (14, 1)),
            )
            np.testing.assert_array_equal(after, 0)
            np.testing.assert_array_equal(rewards, info["balanced_gathering"].astype(float))
            np.testing.assert_array_equal(info["shortage"].sum(-1), 1 - rewards)
            np.testing.assert_array_equal(info["overflow"].sum(-1), 1 - rewards)
        env = ResourceGatheringEnv()
        self.assertEqual(env.capacity, 1)
        np.testing.assert_array_equal(env.reset()["inventory"], [0, 0])

    def test_buffered_counterexample_has_no_message_perfect_reward(self):
        # Exhaust all one-step transitions of the reachable invariant state set.
        invariant = {(2, 2), (3, 1), (1, 3)}
        reached = set()
        for stock in invariant:
            for scene in FEASIBLE_SCENES:
                if stock[0] == stock[1]:
                    preferred = (0, 1)
                else:
                    preferred = (int(np.argmin(stock)),) * 2
                actions = [int(np.argmax(scene[i] == preferred[i])) for i in range(2)]
                after, reward, _ = transition(stock, scene, actions, capacity=4)
                self.assertEqual(reward, 1)
                self.assertIn(tuple(after), invariant)
                reached.add(tuple(after))
        self.assertEqual(reached, invariant)

    def test_reset_and_sampler_reproducibility(self):
        env = ResourceGatheringEnv(horizon=2)
        first = env.reset(seed=100)
        env.step([0, 0])
        second = env.reset(seed=100)
        np.testing.assert_array_equal(first["private_kinds"], second["private_kinds"])
        self.assertEqual(sample_scenes(np.random.default_rng(1), 0).shape, (0, 2, 2))

    def test_invalid_actions_and_states_are_rejected(self):
        for actions in ([2, 0], [0.0, 1.0], [0]):
            with self.assertRaises(ValueError):
                transition([2, 2], [[0, 1], [0, 1]], actions)
        with self.assertRaises(ValueError):
            transition([-1, 2], [[0, 1], [0, 1]], [0, 1])
        with self.assertRaises(RuntimeError):
            ResourceGatheringEnv().step([0, 0])


if __name__ == "__main__":
    unittest.main()
