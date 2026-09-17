"""No training: boundary, persistence and constructed-policy checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from research_program.triadic_learning_baseline import runner as r


class BoundaryTests(unittest.TestCase):
    def test_preparation_has_no_training_and_rejects_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prepared'
            with patch.object(r, 'train_seed', side_effect=AssertionError('training prohibited')):
                result = r.prepare(p)
                r.verify(p)
                before = r.sha(p / 'plan.json')
                with self.assertRaisesRegex(ValueError, 'overwrite'):
                    r.prepare(p)
                self.assertEqual(before, r.sha(p / 'plan.json'))
            self.assertEqual(result['status'], 'prepared_not_trained')
            self.assertFalse((p / 'execution').exists())
            self.assertEqual(result['partition_counts'],
                             dict(train=82836, new_needs=24732, new_layouts=27612, new_needs_and_layouts=8244))

    def test_tampered_split_rejected_before_execute_directory(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prepared'
            r.prepare(p)
            data = r.read(p / 'prepared.json')
            data['partitions']['train']['monitor_indices'][0] += 1
            (p / 'prepared.json').write_text(json.dumps(data))
            with patch.object(r, 'build_arrays', side_effect=AssertionError('must not build arrays')):
                with self.assertRaisesRegex(ValueError, 'Prepared split changed'):
                    r.execute(p)
            self.assertFalse((p / 'execution').exists())

    def test_tampered_snapshot_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prepared'
            r.prepare(p)
            source = p / 'source_snapshot/research_program/triadic_task/environment.py'
            source.write_text(source.read_text() + '\n# deliberate fixture corruption\n')
            with self.assertRaisesRegex(ValueError, 'snapshot changed'):
                r.verify(p)

    def test_started_execution_cannot_resume(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'prepared'
            r.prepare(p)
            (p / 'execution').mkdir()
            with patch.object(r, 'build_arrays', side_effect=AssertionError('must not read model input')):
                with self.assertRaisesRegex(ValueError, 'Never resume'):
                    r.execute(p)

    def test_encoder_rejects_researcher_information(self):
        state = r.env.State((2, 2, 7), (0, 3, 1, 2))
        views = {a: r.env.observe(state, a, shared_needs=True, full_information=True) for a in r.AGENTS}
        views['A']['correct_plan'] = {'A': 'wait'}
        with self.assertRaisesRegex(ValueError, 'Unexpected observation field'):
            r.encode_observations([views])

    def test_unseen_world_changes_do_not_change_local_actor_input(self):
        first = r.env.State((2, 2, 7), (0, 3, 1, 2))
        second = r.env.State((2, 8, 10), (0, 3, 2, 1))
        views = [{a: r.env.observe(s, a, shared_needs=False) for a in r.AGENTS} for s in (first, second)]
        self.assertEqual(views[0]['A'], views[1]['A'])
        encoded = r.encode_observations(views)
        np.testing.assert_array_equal(encoded[0, 0], encoded[1, 0])
        self.assertFalse(np.array_equal(encoded[0, 1], encoded[1, 1]))
        self.assertTrue(np.array_equal(encoded[:, 0, 7:21], np.zeros((2, 14))))

    def test_actor_parameters_and_initial_streams_independent(self):
        actors = [r.make_actor(np.random.SeedSequence([43101, i, 100])) for i in range(3)]
        self.assertEqual(sum(v.size for v in actors[0].values()), 8785)
        for i, j in ((0, 1), (0, 2), (1, 2)):
            for k in actors[i]:
                self.assertFalse(np.shares_memory(actors[i][k], actors[j][k]))
            self.assertFalse(np.array_equal(actors[i]['W1'], actors[j]['W1']))
        old = actors[1]['W1'].copy()
        actors[0]['W1'][0, 0] += 1
        np.testing.assert_array_equal(actors[1]['W1'], old)

    def test_entropy_definition_and_fixed_schedule(self):
        p = np.full((2, 3, 17), 1 / 17)
        entropy, gradient = r.entropy_and_logit_gradient(p, np.log(p))
        np.testing.assert_allclose(entropy, np.log(17), rtol=0, atol=1e-14)
        np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-16)
        self.assertEqual(r.entropy_coefficient(1), .001)
        self.assertAlmostEqual(r.entropy_coefficient(501), .0005)
        self.assertEqual(r.entropy_coefficient(1001), 0)
        self.assertEqual(r.entropy_coefficient(6000), 0)

    def test_all17_actions_stay_in_probability_denominator(self):
        state = r.env.State((2, 2, 7), (0, 3, 1, 2))
        rewards = r.reward_terms([state])
        p = np.full((1, 3, 17), 1 / 17)
        expected, gradient = r.expected_reward_and_logit_gradient(p, rewards)
        self.assertAlmostEqual(expected[0], rewards.sum() / 17**3, places=15)
        np.testing.assert_allclose(gradient.sum(-1), 0, rtol=0, atol=1e-17)
        self.assertEqual(r.JOINT_ACTIONS.shape, (24, 3))
        # These three simultaneously chosen transport actions cause overload.
        p.fill(0); p[:, :, 1] = 1
        zero, derivative = r.expected_reward_and_logit_gradient(p, rewards)
        self.assertEqual(zero.tolist(), [0.0])
        np.testing.assert_array_equal(derivative, np.zeros_like(derivative))

    def test_constructed_policy_keeps_discrete_and_exact_results_separate(self):
        # Analytic fixture, not study initialization or a task-performance pilot.
        state = r.env.State((2, 2, 7), (0, 3, 1, 2))
        observation = {a: r.env.observe(state, a, shared_needs=True, full_information=True) for a in r.AGENTS}
        arrays = {'states': [state], 'x': r.encode_observations([observation]),
                  'rewards': r.reward_terms([state]),
                  'packed_states': np.array([state.needs + state.layout + state.private_sites], dtype=np.int16)}
        targets = [
            {'kind': 'transport', 'site': 'S0', 'destination': 'R', 'partner': 'B'},
            {'kind': 'transport', 'site': 'S0', 'destination': 'R', 'partner': 'A'},
            {'kind': 'wait'},
        ]
        actors = []
        for i in range(3):
            actor = {f'W{k}': np.zeros((a, b)) for k, (a, b) in enumerate(((54,64),(64,64),(64,17)),1)}
            actor.update({f'b{k}': np.zeros(n) for k,n in enumerate((64,64,17),1)})
            actor['b3'][r.ACTIONS[i].index(targets[i])] = 4
            actors.append(actor)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'all_outputs.npz'
            result = r.evaluate(actors, arrays, save_path=path)
            with np.load(path, allow_pickle=False) as data:
                self.assertEqual(data['action_probabilities'].shape, (1,3,17))
                self.assertTrue((data['action_probabilities'] > 0).all())
                self.assertEqual(data['greedy_reward'].tolist(), [1.0])
                self.assertEqual(data['executed'].tolist(), [[True,True,False]])
                self.assertEqual(data['action_indices'].tolist(), [[r.ACTIONS[i].index(targets[i]) for i in range(3)]])
            self.assertEqual(result['greedy_full_success_rate'], 1)
            self.assertLess(result['exact_stochastic_expected_reward_mean'], 1)
            self.assertEqual(result['greedy_executed_pair_worlds'], {'AB':1,'AC':0,'BC':0})
            with self.assertRaisesRegex(ValueError, 'output exists'):
                r.evaluate(actors, arrays, save_path=path)


if __name__ == '__main__':
    unittest.main()
