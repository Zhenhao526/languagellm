"""Utility-kernel checks; no policy files, optimizer updates or formal training.

One scratch-network compatibility test calls each training-gradient function
once on B=2 artificial worlds: 18 forward calls, 72 module samples total.
All other forward/backward checks use explicit synthetic callbacks.
"""
from copy import deepcopy
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np

from research_program.triadic_partial_payoff_study import utility as u


def native_table(batch=2):
    native = np.zeros((batch, 24), dtype=np.float64)
    for row in range(batch):
        native[row, row % 24] = 1
        native[row, (row + 3) % 24] = 0.5
        native[row, (row + 8) % 24] = 0.5
    return native


class UtilityTests(unittest.TestCase):
    def test_mapping_support_and_no_input_mutation(self):
        native = native_table()
        before = native.copy()
        for alpha in u.ALPHAS:
            mapped = u.utility_table(native, alpha)
            np.testing.assert_array_equal(mapped > 0, native > 0)
            np.testing.assert_array_equal(mapped == 1, native == 1)
            np.testing.assert_array_equal(mapped[native == 0.5], alpha)
            self.assertFalse(np.shares_memory(mapped, native))
        np.testing.assert_array_equal(native, before)

    def test_invalid_support_shape_values_and_alpha(self):
        native = native_table()
        invalid = [np.empty((0, 24)), np.zeros((2, 23)), np.zeros((2, 24)),
                   np.ones((2, 24)), native.ravel()]
        for value in [0.1, np.nan, np.inf, -0.5, np.nextafter(0.5, 1.0)]:
            changed = native.copy(); changed[0, 3] = value; invalid.append(changed)
        for rewards in invalid:
            with self.subTest(rewards_shape=rewards.shape), self.assertRaises(ValueError):
                u.utility_table(rewards, 0.1)
        for alpha in [0, -0.1, 1, True, np.nan, np.inf, 1e-12, np.nextafter(.1, 1), .2]:
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                u.utility_table(native, alpha)
        for logits in [np.zeros((2, 3, 16)), np.zeros((1, 3, 17)),
                       np.full((2, 3, 17), np.inf), np.full((2, 3, 17), np.nan)]:
            with self.subTest(shape=logits.shape), self.assertRaises(ValueError):
                u.objective_terms(logits, native, .1)

    def test_uniform_fraction_and_all_17_actions(self):
        native = native_table()
        for alpha in u.ALPHAS:
            terms = u.objective_terms(np.zeros((2, 3, 17)), native, alpha)
            np.testing.assert_array_equal(terms['probabilities'], np.full((2, 3, 17), 1/17))
            np.testing.assert_allclose(terms['J'], (1 + 2*alpha) / 17**3, rtol=1e-15)
            np.testing.assert_allclose(terms['log_J'], np.log((1+2*alpha)/17**3), rtol=1e-15)
            np.testing.assert_allclose(terms['native_expected_reward'], 2/17**3, rtol=1e-15)
            np.testing.assert_allclose(terms['full_success_posterior_mass'], 1/(1+2*alpha), rtol=1e-15)
            np.testing.assert_allclose(terms['partial_success_posterior_mass'], 2*alpha/(1+2*alpha), rtol=1e-15)
            np.testing.assert_array_equal(terms['posterior_weights'][native == 0], 0)
            np.testing.assert_allclose(terms['log_J_logit_gradient'].sum(axis=-1), 0, atol=2e-16)
            # A never-rewarded action still carries the negative softmax term.
            actor = 0
            positive_actions = set(u.base.JOINT_ACTIONS[native[0] > 0, actor])
            unused = next(a for a in range(17) if a not in positive_actions)
            self.assertAlmostEqual(terms['log_J_logit_gradient'][0, actor, unused], -1/17)

    def test_all_joint_actions_exact_reference(self):
        rng = np.random.default_rng(41)
        logits = rng.normal(size=(2, 3, 17))
        native = native_table()
        for alpha in u.ALPHAS:
            terms = u.objective_terms(logits, native, alpha)
            table = {tuple(plan): index for index, plan in enumerate(u.base.JOINT_ACTIONS)}
            expected = np.zeros(2)
            for actions in product(range(17), repeat=3):
                index = table.get(actions)
                if index is not None:
                    probability = np.prod([terms['probabilities'][:, actor, actions[actor]] for actor in range(3)], axis=0)
                    reward = np.where(native[:, index] == .5, alpha, native[:, index])
                    expected += probability * reward
            np.testing.assert_allclose(terms['J'], expected, atol=0, rtol=3e-16)

    def test_finite_difference_log_and_expected_utility_gradients(self):
        rng = np.random.default_rng(271)
        logits = rng.normal(0, 1.1, size=(2, 3, 17))
        native = native_table()
        epsilon = 1e-5
        for alpha in u.ALPHAS:
            terms = u.objective_terms(logits, native, alpha)
            for index in np.ndindex(logits.shape):
                plus, minus = logits.copy(), logits.copy()
                plus[index] += epsilon; minus[index] -= epsilon
                a = u.objective_terms(plus, native, alpha)
                b = u.objective_terms(minus, native, alpha)
                row = index[0]
                self.assertAlmostEqual((a['log_J'][row]-b['log_J'][row])/(2*epsilon),
                                       terms['log_J_logit_gradient'][index], delta=2e-10)
                self.assertAlmostEqual((a['J'][row]-b['J'][row])/(2*epsilon),
                                       terms['mean_J_logit_gradient'][index], delta=2e-13)

    def test_half_utility_exact_old_objective_compatibility(self):
        rng = np.random.default_rng(44)
        logits = rng.normal(size=(2, 3, 17))
        native = native_table()
        old = u.core.coordination.objective_terms(logits, native)
        new = u.objective_terms(logits, native, .5)
        for key in old:
            np.testing.assert_array_equal(new[key], old[key], err_msg=key)
        np.testing.assert_array_equal(new['J'], new['native_expected_reward'])

    def test_underflow_log_remains_unfloored(self):
        logits = np.full((1, 3, 17), -1000.)
        logits[:, :, 0] = 0  # All-wait has zero native reward for every world.
        native = native_table(1)
        for alpha in u.ALPHAS:
            terms = u.objective_terms(logits, native, alpha)
            np.testing.assert_array_equal(terms['J'], 0)
            self.assertTrue(np.isfinite(terms['log_J']).all())
            self.assertLess(terms['log_J'][0], -1000)
            self.assertTrue(np.isfinite(terms['log_J_logit_gradient']).all())
            self.assertAlmostEqual(terms['posterior_weights'].sum(), 1)
            shifted_logits = logits.copy(); shifted_logits[:, :, 1:] -= 100
            farther = u.objective_terms(shifted_logits, native, alpha)
            self.assertLess(farther['log_J'][0], terms['log_J'][0] - 99)
        # A full-only world gives posterior exactly one despite float J=0.
        native[native == .5] = 0
        terms = u.objective_terms(logits, native, .1)
        np.testing.assert_array_equal(terms['full_success_posterior_mass'], 1)
        np.testing.assert_array_equal(terms['partial_success_posterior_mass'], 0)

    def test_fixed_policy_reweighting_identity(self):
        logits = np.random.default_rng(84).normal(size=(2, 3, 17))
        native = native_table()
        first, second = [u.objective_terms(logits, native, alpha) for alpha in u.ALPHAS]
        np.testing.assert_array_equal(first['probabilities'], second['probabilities'])
        np.testing.assert_array_equal(first['native_expected_reward'], second['native_expected_reward'])
        for terms, alpha in zip((first, second), u.ALPHAS):
            full, partial = terms['full_success_probability'], terms['partial_success_probability']
            np.testing.assert_allclose(terms['J'], full + alpha*partial, atol=0, rtol=3e-16)
            np.testing.assert_allclose(terms['native_expected_reward'], full + .5*partial, atol=0, rtol=3e-16)
            np.testing.assert_allclose(terms['full_success_posterior_mass'], full/terms['J'], rtol=2e-15)
        self.assertTrue((second['full_success_posterior_mass'] > first['full_success_posterior_mass']).all())


class TrainingTests(unittest.TestCase):
    def test_real_scratch_half_compatibility_all_nine_gradients(self):
        networks = u.core.make_networks(761)
        before = deepcopy(networks)
        rng = np.random.default_rng(719)
        observations = rng.normal(size=(2, 3, 54))
        uniforms = rng.random((2, 2, 2, 3, 4))
        native = native_table()
        old_gradients, old_row = u.core.training_gradients(networks, observations, native, True, uniforms, 17)
        new_gradients, new_row = u.training_gradients(networks, observations, native, True, uniforms, 17, .5)
        for index, (old, new) in enumerate(zip(old_gradients, new_gradients)):
            for key in old:
                np.testing.assert_array_equal(new[key], old[key], err_msg=f'{index}/{key}')
                np.testing.assert_array_equal(networks[index][key], before[index][key])
        translation = {'mean_J': 'mean_expected_utility', 'mean_log_J': 'mean_log_expected_utility',
                       'min_log_J': 'min_log_expected_utility', 'max_log_J': 'max_log_expected_utility',
                       'zero_float_J_states': 'zero_float_expected_utility_states'}
        for key, value in old_row.items():
            self.assertEqual(new_row[translation.get(key, key)], value, key)
        self.assertGreater(sum(np.linalg.norm(g['W3']) for i, g in enumerate(new_gradients) if i % 3 < 2), 0)
        self.assertAlmostEqual(new_row['mean_expected_utility'], new_row['mean_native_expected_reward'])

    def test_reward_and_posterior_do_not_enter_model_inputs(self):
        x = np.arange(2*3*54, dtype=np.float64).reshape(2, 3, 54) / 50
        native = native_table()
        changed = native.copy(); changed[changed == .5] = 0
        nets = [{'agent': a, 'module': m} for a in range(3) for m in range(3)]
        uniforms = np.random.default_rng(20).random((2, 2, 2, 3, 4))
        calls = []
        def forward(net, inputs):
            calls.append(inputs.copy())
            count = 17 if net['module'] == 2 else 32
            z = np.sin(inputs.sum(axis=1, keepdims=True) + np.arange(count)[None, :])
            return z, None
        with patch.object(u.base, 'actor_forward', side_effect=forward), \
             patch.object(u.base, 'actor_backward', side_effect=lambda net, cache, derivative: {'d': derivative.copy()}):
            _, row = u.training_gradients(nets, x, native, False, uniforms, 6000, .1)
            first_calls = calls[:]; calls.clear()
            u.training_gradients(nets, x, changed, False, uniforms, 6000, .1)
        self.assertEqual(len(calls), 9)
        for first, second in zip(first_calls, calls):
            np.testing.assert_array_equal(first, second)
        self.assertEqual([call.shape[-1] for call in calls], [54]*3 + [153]*3 + [252]*3)
        self.assertEqual(row['entropy_coefficient'], 0)
        self.assertAlmostEqual(row['mean_F'], row['mean_log_expected_utility'])
        self.assertAlmostEqual(row['receiver_loss'], -row['mean_F'])
        self.assertAlmostEqual(row['mean_full_success_posterior_mass'] + row['mean_partial_success_posterior_mass'], 1)

    def test_invalid_training_rejected_before_forward(self):
        x = np.zeros((2, 3, 54)); native = native_table()
        uniforms = np.zeros((2, 2, 2, 3, 4))
        common = dict(networks=[{}]*9, observations=x, native_rewards=native,
                      live=True, uniforms=uniforms, update=1, alpha=.1)
        invalid = [('observations', x[:, :, :53]), ('native_rewards', np.zeros_like(native)),
                   ('live', 'live'), ('update', 0), ('update', 1.5), ('update', True),
                   ('alpha', 0), ('uniforms', np.ones_like(uniforms)),
                   ('uniforms', np.full_like(uniforms, np.nan))]
        with patch.object(u.core, 'rollout', side_effect=AssertionError('invalid input reached model')):
            for key, value in invalid:
                kwargs = dict(common); kwargs[key] = value
                with self.subTest(key=key), self.assertRaises(ValueError):
                    u.training_gradients(**kwargs)


if __name__ == '__main__':
    unittest.main()
