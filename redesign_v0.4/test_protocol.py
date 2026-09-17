"""Information isolation checks, using synthetic inputs rather than photos.

These are engineering checks. Passing them is not evidence of communication
learning, visual understanding, or resource competence.
"""
import unittest

import numpy as np
import torch

from agents import ResourceAgent, draw
from run_pilot import shuffled_messages


class InformationBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)
        self.agents = [ResourceAgent(), ResourceAgent()]
        self.features = torch.randn(8, 2, 2, 1024)
        self.public = torch.zeros(8, 3)
        self.representations = [
            agent.observe(self.features[:, i], self.public)
            for i, agent in enumerate(self.agents)
        ]

    def test_agent_zero_update_cannot_update_partner_or_cached_vision(self):
        parameter_ids = [{id(p) for p in a.parameters()} for a in self.agents]
        self.assertFalse(parameter_ids[0] & parameter_ids[1])
        before_partner = {k: v.detach().clone() for k, v in self.agents[1].state_dict().items()}
        before_features = self.features.clone()
        rng = [np.random.default_rng(101), np.random.default_rng(202)]
        sent = [
            draw(agent.send(self.representations[i][1]), rng[i])
            for i, agent in enumerate(self.agents)
        ]
        actions = [
            draw(agent.act(*self.representations[i], sent[1 - i][0].detach()), rng[i])
            for i, agent in enumerate(self.agents)
        ]
        optimizer = torch.optim.Adam(self.agents[0].parameters(), lr=1e-3)
        loss = -(sent[0][1] + actions[0][1]).mean()
        optimizer.zero_grad()
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in self.agents[0].parameters()))
        self.assertTrue(all(p.grad is None for p in self.agents[1].parameters()))
        optimizer.step()
        for name, value in self.agents[1].state_dict().items():
            torch.testing.assert_close(value, before_partner[name], rtol=0, atol=0)
        self.assertIsNone(self.features.grad)
        torch.testing.assert_close(self.features, before_features, rtol=0, atol=0)
        for symbol, _, _ in sent:
            self.assertEqual(symbol.dtype, torch.int64)
            self.assertFalse(symbol.requires_grad)

    def test_actor_rejects_continuous_message_vectors(self):
        with self.assertRaises(AssertionError):
            self.agents[0].act(*self.representations[0], torch.zeros(8))
        with self.assertRaises(AssertionError):
            self.agents[0].act(*self.representations[0], torch.zeros(8, 5, requires_grad=True))

    def test_partner_feature_changes_do_not_change_local_sender_inputs(self):
        # The caller hands an agent only its own features. A changed private
        # partner observation cannot affect this sender's logits.
        original = self.agents[0].send(self.representations[0][1]).detach()
        changed = self.features.clone()
        changed[:, 1] = torch.randn_like(changed[:, 1]) * 100
        _, local = self.agents[0].observe(changed[:, 0], self.public)
        actual = self.agents[0].send(local).detach()
        torch.testing.assert_close(actual, original, rtol=0, atol=0)

    def test_shuffle_preserves_counts_separately_within_public_state_and_direction(self):
        inventory = np.repeat(np.asarray([[0, 0], [1, 2], [2, 1]]), 100, axis=0)
        messages = np.column_stack([np.arange(300) % 5, (np.arange(300) // 3) % 5])
        before = messages.copy()
        delivered = shuffled_messages(messages, inventory, np.random.default_rng(91))
        np.testing.assert_array_equal(messages, before)
        self.assertTrue(np.any(delivered != messages))
        for state in np.unique(inventory, axis=0):
            mask = (inventory == state).all(1)
            for sender in range(2):
                np.testing.assert_array_equal(
                    np.bincount(delivered[mask, sender], minlength=5),
                    np.bincount(messages[mask, sender], minlength=5),
                )
        # A reassignment across samples is not a fixed token substitution.
        self.assertTrue(any(len(np.unique(delivered[messages[:, 0] == symbol, 0])) > 1 for symbol in range(5)))


if __name__ == '__main__':
    unittest.main()
