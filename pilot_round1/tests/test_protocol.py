"""Information-boundary checks; no optimizer steps or training runs."""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from pilot_round1.env import VisualWorld
from pilot_round1.model import (
    Agent, DIM, EVENT_DIM, HISTORY_LENGTH, append_record, empty_history,
    private_record, sample_policy,
)
from pilot_round1.train import evaluate


class ProtocolBoundaryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(409)

    def test_sender_record_has_only_own_image_and_no_receiver_choice(self):
        own = torch.arange(DIM, dtype=torch.float32, requires_grad=True)
        record = private_record(0, own, 2, None, 0)
        self.assertEqual(record.shape, (EVENT_DIM,))
        self.assertFalse(record.requires_grad)
        torch.testing.assert_close(record[:2], torch.tensor([1., 0.]))
        torch.testing.assert_close(record[2:66], own.detach())
        self.assertEqual(int(torch.count_nonzero(record[66:258])), 0)
        torch.testing.assert_close(record[258:262], torch.tensor([1., 0., 0., 0.]))
        torch.testing.assert_close(record[262:266], torch.tensor([0., 0., 1., 0.]))
        self.assertEqual(int(torch.count_nonzero(record[266:271])), 0)
        self.assertEqual(record[271].item(), 1.)
        with self.assertRaises(ValueError):
            private_record(0, own, 2, 1, 0)

    def test_receiver_record_retains_visible_candidates_not_hidden_target(self):
        visible = torch.arange(4 * DIM, dtype=torch.float32).reshape(4, DIM)
        record = private_record(1, visible, 3, 2, 0)
        torch.testing.assert_close(record[:2], torch.tensor([0., 1.]))
        torch.testing.assert_close(record[2:258], visible.flatten())
        torch.testing.assert_close(record[258:262], torch.ones(4))
        torch.testing.assert_close(record[262:266], torch.tensor([0., 0., 0., 1.]))
        torch.testing.assert_close(record[266:270], torch.tensor([0., 0., 1., 0.]))
        self.assertEqual(record[270].item(), 0.)
        self.assertEqual(record[271].item(), 1.)

    def test_history_is_four_completed_records_without_old_graph(self):
        history = empty_history('cpu')
        for step in range(7):
            old = history.clone()
            event = torch.full((EVENT_DIM,), float(step), requires_grad=True)
            updated = append_record(history, event)
            torch.testing.assert_close(history, old)
            history = updated
        self.assertEqual(history.shape, (HISTORY_LENGTH, EVENT_DIM))
        torch.testing.assert_close(history[:, 0], torch.tensor([3., 4., 5., 6.]))
        self.assertFalse(history.requires_grad)
        self.assertIsNone(history.grad_fn)

    def test_h0_masks_history_for_both_roles(self):
        agent = Agent()
        target = torch.randn(1, DIM)
        candidates = torch.randn(1, 4, DIM)
        symbol = torch.tensor([1])
        zero = empty_history('cpu')[None]
        arbitrary = torch.randn_like(zero) * 100
        torch.testing.assert_close(agent.send(target, zero, False),
                                   agent.send(target, arbitrary, False), rtol=0, atol=0)
        torch.testing.assert_close(agent.receive(candidates, symbol, zero, False),
                                   agent.receive(candidates, symbol, arbitrary, False),
                                   rtol=0, atol=0)
        # Normal weight learning is still possible with all history slots masked.
        agent.send(target, arbitrary, False).sum().backward()
        self.assertTrue(any(p.grad is not None for p in agent.sender.parameters()))

    def test_sender_and_listener_losses_have_disjoint_gradients(self):
        sender, receiver = Agent(), Agent()
        sender.freeze_perception()
        receiver.freeze_perception()
        self.assertTrue(set(map(id, sender.parameters())).isdisjoint(
            set(map(id, receiver.parameters()))))
        hs = empty_history('cpu')[None]
        hr = empty_history('cpu')[None]
        target = torch.randn(1, DIM)
        candidates = torch.randn(1, 4, DIM)
        message, sender_lp, _ = sample_policy(sender.send(target, hs, True)[0], .31)
        self.assertIsInstance(message, int)
        _, receiver_lp, _ = sample_policy(
            receiver.receive(candidates, torch.tensor([message]), hr, True)[0], .72)
        (-.75 * receiver_lp).backward()
        self.assertTrue(all(p.grad is None for p in sender.parameters()))
        self.assertTrue(any(p.grad is not None for p in receiver.listener.parameters()))
        receiver.zero_grad(set_to_none=True)
        (-.75 * sender_lp).backward()
        self.assertTrue(all(p.grad is None for p in receiver.parameters()))
        self.assertTrue(any(p.grad is not None for p in sender.sender.parameters()))

    def test_evaluation_is_read_only_and_ignores_hidden_target_metadata(self):
        agents = [Agent(), Agent()]
        for agent in agents:
            agent.freeze_perception()
        histories = [torch.randn(HISTORY_LENGTH, EVENT_DIM) for _ in agents]
        histories_before = [h.clone() for h in histories]
        states_before = [{k: v.clone() for k, v in a.state_dict().items()} for a in agents]
        args = (agents, histories, 'H1', 'emergent', np.arange(4), 713, 'cpu')
        normal = evaluate(*args, n=16, intervention=False)

        class ChangedHiddenMetadata(VisualWorld):
            def sample(self, batch_size):
                batch = super().sample(batch_size)
                batch['target_ids'] = (batch['target_ids'] + 1) % 4
                batch['correct_indices'] = np.argmax(
                    batch['candidate_ids'] == batch['target_ids'][:, None], axis=1)
                return batch

        with patch('pilot_round1.train.VisualWorld', ChangedHiddenMetadata):
            altered = evaluate(*args, n=16, intervention=False)
        # Images and permitted histories stayed unchanged, so hidden label edits
        # may change scoring, but may not change message-to-action behavior.
        self.assertEqual(normal['response_counts'], altered['response_counts'])
        self.assertEqual(normal['used_symbols'], altered['used_symbols'])
        evaluate(*args, n=16, intervention=True)
        for agent, before in zip(agents, states_before):
            for key, value in agent.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            self.assertTrue(all(p.grad is None for p in agent.parameters()))
        for history, before in zip(histories, histories_before):
            torch.testing.assert_close(history, before, rtol=0, atol=0)

    def test_intervention_uses_independent_reference_and_matched_eligible_trials(self):
        seed = 821
        seen_seeds = []
        main_targets = []

        class ToyWorld:
            def __init__(self, world_seed):
                seen_seeds.append(world_seed)
                self.rng = np.random.default_rng(world_seed)
                self.is_reference = world_seed == seed + 424242

            def sample(self, size):
                # Only 0 and 1 occur in the reference sample. The evaluation
                # must exclude naturally emitted 2/3 from intervention eligibility.
                target = self.rng.integers(0, 2 if self.is_reference else 4, size)
                candidates = self.rng.permuted(np.tile(np.arange(4), (size, 1)), axis=1)
                if not self.is_reference:
                    main_targets.extend(target.tolist())
                return {
                    'target_images': target.astype(np.float32).reshape(size, 1, 1, 1),
                    'candidate_images': candidates.astype(np.float32).reshape(size, 4, 1, 1, 1),
                    'target_ids': target, 'candidate_ids': candidates,
                    'correct_indices': np.argmax(candidates == target[:, None], axis=1),
                }

        class ToyAgent:
            def encoder(self, images):
                codes = torch.nn.functional.one_hot(images[:, 0, 0, 0].long(), 4).float()
                return torch.nn.functional.pad(codes, (0, DIM - 4))

            def send(self, targets, history, enabled):
                return targets[:, :4]

            def receive(self, candidates, symbols, history, enabled):
                query = torch.nn.functional.one_hot(symbols, 4).float()
                return (candidates[:, :, :4] * query[:, None, :]).sum(-1)

        with patch('pilot_round1.train.VisualWorld', ToyWorld):
            result = evaluate([ToyAgent(), ToyAgent()], [empty_history('cpu')] * 2,
                              'H1', 'emergent', np.arange(4), seed, 'cpu',
                              n=64, intervention=True)
        self.assertEqual(seen_seeds, [seed + 424242, seed])
        self.assertEqual(result['reference_n'], 1024)
        self.assertEqual(result['reference_used_symbols'], [[0, 1], [0, 1]])
        self.assertEqual(result['reference_message_to_class'],
                         [[0, 1, None, None], [0, 1, None, None]])
        self.assertEqual(result['intervention_n'], sum(t < 2 for t in main_targets))
        self.assertGreater(result['intervention_n'], 0)
        self.assertLess(result['intervention_n'], 64)
        self.assertEqual(sum(result['direction_n']), 64)
        self.assertEqual(result['direction_accuracy'], [1., 1.])
        self.assertEqual(result['original_accuracy_on_intervened'], 1.)
        self.assertEqual(result['intervention_accuracy'], 0.)
        self.assertEqual(result['intervention_follow_rate'], 1.)


if __name__ == '__main__':
    unittest.main()
