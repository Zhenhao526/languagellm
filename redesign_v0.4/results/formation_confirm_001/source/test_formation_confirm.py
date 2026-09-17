"""Control semantics and equivalence checks before the new-seed batch."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import run_formation_confirm as training
import run_curriculum as old
from agents import draw
from resource_env import FEASIBLE_SCENES, transition


class FormationConfirmationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_predeclared_new_seeds_and_equal_learning_schedule(self):
        self.assertEqual(training.CONFIG['seeds'], list(range(1201, 1211)))
        self.assertEqual(training.CONFIG['evaluation_n'], old.CONFIG['evaluation_n'])
        for u in range(1, 1201):
            plans = [training.schedule(u, c) for c in training.CONDITIONS]
            self.assertTrue(all(p['aux_weight'] == 0 for p in plans))
            self.assertTrue(all(p['action_entropy_weight'] == (.05 if u <= 900 else 0.) for p in plans))
            self.assertEqual(plans[0]['task'], 'curriculum' if u <= 600 else 'full')
            self.assertEqual(plans[1]['task'], 'full')
            self.assertEqual(plans[2]['task'], plans[0]['task'])
            self.assertEqual(plans[3]['task'], plans[0]['task'])

    def test_blocked_delivers_zero_from_first_step_and_sampling_still_advances(self):
        rngs = [np.random.default_rng(701), np.random.default_rng(702)]
        sent = [draw(torch.randn(64, 5), rng)[0] for rng in rngs]
        state_after_send = [copy.deepcopy(r.bit_generator.state) for r in rngs]
        blocked = training.delivered_to_receivers(sent, 'course_blocked')
        self.assertTrue(all(torch.count_nonzero(v) == 0 for v in blocked))
        active = training.delivered_to_receivers(sent, 'course_communication')
        self.assertTrue(torch.equal(active[0], sent[1]))
        self.assertTrue(torch.equal(active[1], sent[0]))
        self.assertEqual(state_after_send, [r.bit_generator.state for r in rngs])
        for i in range(2):
            before = np.random.default_rng(701 + i).bit_generator.state
            self.assertNotEqual(state_after_send[i], before)

    def test_substitutable_all_actions_have_native_reward_one(self):
        scenes = np.repeat(FEASIBLE_SCENES, 4, axis=0)
        actions = np.tile(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), (len(FEASIBLE_SCENES), 1))
        _, balanced, _ = transition(np.zeros((len(scenes), 2), np.int64), scenes, actions, capacity=1)
        self.assertTrue(np.any(balanced == 0))
        np.testing.assert_array_equal(training.native_reward(balanced, 'course_substitutable'), 1)
        np.testing.assert_array_equal(training.native_reward(balanced, 'course_communication'), balanced)

    def test_native_blocked_stochastic_uses_blank_and_substitute_success_is_distinct(self):
        calls = []
        def fake(*args, **kwargs):
            calls.append((kwargs['mode'], kwargs['greedy']))
            value = .25 if kwargs['mode'] == 'blank' else .9
            if not kwargs['greedy']:
                value += .01
            return {'mean_reward_per_step': value}, []
        with patch.object(training, 'evaluate', side_effect=fake):
            blocked = training.evaluate_bundle([], None, 1, 'course_blocked', 16, 16)
            substitute = training.evaluate_bundle([], None, 1, 'course_substitutable', 16, 16)
        self.assertEqual(blocked['native_task']['full']['greedy_success'], .25)
        self.assertEqual(blocked['native_task']['full']['stochastic_success'], .26)
        self.assertEqual(substitute['native_task']['full']['greedy_success'], 1.)
        self.assertEqual(substitute['full']['normal']['mean_reward_per_step'], .9)
        self.assertIn(('blank', False), calls)

    def test_message_detachment_preserves_private_gradients(self):
        agents = training.make_agents(1201)
        features = torch.randn(32, 2, 2, 1024)
        public, _ = training.private_public(32, 16)
        reps = [a.observe(features[:, i], public) for i, a in enumerate(agents)]
        sends = [draw(a.send(reps[i][1]), np.random.default_rng(i)) for i, a in enumerate(agents)]
        received = training.delivered_to_receivers([s[0] for s in sends], 'course_communication')
        loss = agents[0].act(*reps[0], received[0]).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is None for p in agents[1].parameters()))
        self.assertTrue(all(p.grad is None for p in agents[0].sender.parameters()))
        self.assertTrue(any(p.grad is not None for p in agents[0].actor.parameters()))

    def test_actual_training_matches_original_baseline_and_paired_controls(self):
        config = copy.deepcopy(training.CONFIG)
        config.update(updates=4, course_updates=2, exploration_updates=3, batch_size=32,
                      checkpoints=[0, 2, 4], evaluation_n=32, checkpoint_evaluation_n=32, intervention_n=32)
        old_config = copy.deepcopy(old.CONFIG)
        old_config.update(course_updates=2, transition_updates=1, pure_reward_updates=1,
                          batch_size=32, checkpoints=[0, 2, 4], evaluation_n=32, checkpoint_evaluation_n=32)
        states = copy.deepcopy([a.state_dict() for a in training.make_agents(1201)])
        bank = training.ImageBank()
        def fake(*args, **kwargs):
            return {'mean_reward_per_step': .5}, []
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()), \
                patch.object(training, 'evaluate', side_effect=fake), \
                patch.object(training, 'message_intervention', return_value={}), \
                patch.object(old, 'evaluate', side_effect=fake), \
                patch.object(old, 'message_intervention', return_value={}), \
                patch.object(old, 'initialize', side_effect=lambda s: training.load_agents(s, states)):
            path = Path(folder)
            for condition in training.CONDITIONS:
                training.train(1201, condition, states, bank, path / condition, config)
            old.train(1201, 'baseline', bank, path / 'original', old_config)
            new_states = torch.load(path / 'course_communication/checkpoint_0004.pt', weights_only=True)
            old_states = torch.load(path / 'original/checkpoint_0004.pt', weights_only=True)
            for left, right in zip(new_states, old_states):
                for k in left:
                    torch.testing.assert_close(left[k], right[k], rtol=0, atol=0)
            metrics = {c: [json.loads(line) for line in (path / c / 'training_metrics.jsonl').read_text().splitlines()]
                       for c in training.CONDITIONS}
            course = metrics['course_communication']
            for condition in ('course_blocked', 'course_substitutable'):
                self.assertEqual([r['world_input_sha256'] for r in course],
                                 [r['world_input_sha256'] for r in metrics[condition]])
            self.assertEqual([r['world_input_sha256'] for r in course[2:]],
                             [r['world_input_sha256'] for r in metrics['direct_communication'][2:]])
            self.assertTrue(all(a['received_nonzero'] == 0 for r in metrics['course_blocked'] for a in r['agents']))
            self.assertTrue(all(r['native_training_success'] == 1 for r in metrics['course_substitutable']))
            for c in training.CONDITIONS:
                initial = torch.load(path / c / 'initial.pt', weights_only=True)
                self.assertEqual(training.state_digest(initial), training.state_digest(states))
                saved = torch.load(path / c / 'training_state_final.pt', weights_only=True)
                self.assertEqual(saved['completed_updates'], 4)
                self.assertEqual(saved['world_rng'], course[-1]['world_rng_after'])
            for k in range(4):
                for c in training.CONDITIONS[1:]:
                    self.assertEqual([a['policy_rng_after'] for a in course[k]['agents']],
                                     [a['policy_rng_after'] for a in metrics[c][k]['agents']])


if __name__ == '__main__':
    unittest.main()
