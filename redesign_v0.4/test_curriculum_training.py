"""Adversarial checks of curriculum learning boundaries and matched execution.

Synthetic inputs here are engineering probes, not additional behavioral runs.
"""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from agents import ResourceAgent
import run_curriculum as training


class CurriculumTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_public_only_symbol_code_gets_no_private_information_bonus(self):
        # Same private distribution in every sample of each public group, but
        # a different deterministic-looking token at the two public times.
        p0 = torch.tensor([.90, .025, .025, .025, .025])
        p1 = p0.roll(1)
        probabilities = torch.cat([p0.repeat(8, 1), p1.repeat(8, 1)])
        groups = torch.tensor([0] * 8 + [1] * 8)
        public_code = training.signalling_loss(probabilities, groups, math.log(5) / 2)
        constant_code = training.signalling_loss(p0.repeat(16, 1), groups, math.log(5) / 2)
        torch.testing.assert_close(public_code, constant_code)
        # An incorrectly pooled marginal would spuriously reward this code.
        incorrectly_pooled = training.signalling_loss(probabilities, torch.zeros(16), math.log(5) / 2)
        self.assertLess(float(incorrectly_pooled), float(public_code) - .3)

    def test_auxiliary_gradient_is_private_and_does_not_train_action_or_value_head(self):
        torch.manual_seed(421)
        agents = [ResourceAgent(), ResourceAgent()]
        frozen_features = torch.randn(32, 2, 2, 1024)
        public, groups = training.private_public(32, 16)
        _, local = agents[0].observe(frozen_features[:, 0], public)
        loss = training.signalling_loss(agents[0].send(local).softmax(-1), groups, math.log(5) / 2)
        loss.backward()
        for name in ('project', 'sender'):
            gradients = [p.grad for p in getattr(agents[0], name).parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        for name in ('actor', 'value'):
            self.assertTrue(all(p.grad is None for p in getattr(agents[0], name).parameters()))
        self.assertTrue(all(p.grad is None for p in agents[1].parameters()))
        self.assertIsNone(frozen_features.grad)

    def test_complete_withdrawal_and_same_course_and_exploration_for_both_conditions(self):
        config = training.CONFIG
        self.assertEqual(config['sender_max_entropy_coefficient'], 0)
        for update in range(1, 1201):
            baseline = training.schedule(update, 'baseline', config)
            signalling = training.schedule(update, 'signalling', config)
            self.assertEqual(baseline['stage'], signalling['stage'])
            self.assertEqual(baseline['task'], signalling['task'])
            self.assertEqual(baseline['action_entropy_weight'], signalling['action_entropy_weight'])
            self.assertEqual(baseline['aux_weight'], 0)
            if update > 900:
                self.assertEqual(signalling['aux_weight'], 0)
                self.assertEqual(signalling['action_entropy_weight'], 0)
                self.assertEqual(signalling['task'], 'full')
        weights = [training.schedule(u, 'signalling')['aux_weight'] for u in range(600, 901)]
        self.assertEqual(weights[0], .1)
        self.assertEqual(weights[-1], 0)
        np.testing.assert_allclose(np.diff(weights), -.1 / 300, rtol=0, atol=1e-15)

    def test_actual_trainer_preserves_paired_random_stream_and_zero_bias_equivalence(self):
        # Execute train(), including both sequential optimizer updates. Replace
        # only evaluation: scores have no role in learning or stopping here.
        config = copy.deepcopy(training.CONFIG)
        config.update(course_updates=2, transition_updates=1, pure_reward_updates=1,
                      batch_size=32, checkpoints=[0, 1, 2, 3, 4], evaluation_n=32,
                      checkpoint_evaluation_n=32)
        bank = training.ImageBank()

        def fake_evaluation(*args, **kwargs):
            return {'mean_reward_per_step': .5}, []

        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(training, 'evaluate', side_effect=fake_evaluation), \
                patch.object(training, 'message_intervention', return_value={}), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(temporary)
            training.train(101, 'baseline', bank, root / 'baseline', config)
            zero = copy.deepcopy(config)
            zero['signalling_coefficient'] = 0.
            training.train(101, 'signalling', bank, root / 'zero_bias', zero)
            training.train(101, 'signalling', bank, root / 'with_bias', config)

            saved = {name: torch.load(root / name / 'checkpoint_0004.pt', weights_only=True)
                     for name in ('baseline', 'zero_bias', 'with_bias')}
            for baseline, zero_bias in zip(saved['baseline'], saved['zero_bias']):
                for key in baseline:
                    torch.testing.assert_close(baseline[key], zero_bias[key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(left[key], right[key])
                                for left, right in zip(saved['baseline'], saved['with_bias'])
                                for key in left))

            metrics = {name: [json.loads(line) for line in
                             (root / name / 'training_metrics.jsonl').read_text().splitlines()]
                       for name in saved}
            for name in ('zero_bias', 'with_bias'):
                self.assertEqual([row['world_input_sha256'] for row in metrics['baseline']],
                                 [row['world_input_sha256'] for row in metrics[name]])
                initial = torch.load(root / name / 'initial.pt', weights_only=True)
                prepared = torch.load(training.ROOT / 'results/pilot_001/prepared_s101.pt', weights_only=True)
                for actual, expected in zip(initial, prepared):
                    for key in actual:
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for rows in metrics.values():
                self.assertEqual(rows[-1]['stage'], 'pure_reward')
                self.assertEqual(rows[-1]['aux_weight'], 0)
                self.assertEqual(rows[-1]['action_entropy_weight'], 0)


if __name__ == '__main__':
    unittest.main()
